"""Schedule-swap sensitivity probe for an LBMDiTJointDDTAgent checkpoint.

Answers: "how much does the action chunk depend on the state denoising
stream?" by running the inference Euler walk under three regimes that share
obs and action-noise, then comparing the resulting action chunks (and
each one's MSE against the dataset's ground-truth action).

Regimes (same obs, same act_0, same x_state_init across all):
  diagonal       — default: t_state == t_action walks lo -> hi, x_state
                   integrates with the trunk.
  noise_pinned   — t_state held at lo; x_state held at its initial randn
                   draw (never integrates). Action stream walks normally.
                   This is the agent's built-in ``action_only`` schedule.
  oracle_pinned  — t_state held at hi; x_state held at the oracle clean
                   value target_ln(encoder(goal_obs)). Action stream walks
                   normally. Upper bound on what a perfect state stream
                   could give the action head.

Reported metrics, per condition pair (X, Y):
  ||a_X - a_Y||_2 normalized by the action's own scale
  per-dim L2 of (a_X - a_Y)
And per condition vs the dataset's normalized ground truth:
  MSE(a_X, a_true), MAE(a_X, a_true), overall + per-dim.

Reading the output:
  * diag ~= noise_pinned and diag ~= oracle_pinned -> action head ignores
    the state slot entirely; joint denoising is decorative.
  * noise_pinned ~= oracle_pinned but != diag -> state slot is used, but
    *only* the noise schedule moves it; oracle vs noise inputs are
    interchangeable to the action head (state slot is acting as a learned
    register, not an information channel).
  * diag ~= oracle_pinned but != noise_pinned -> diagonal walk recovers
    something close to the oracle clean state; state denoising IS doing
    work for the action chunk.
  * If oracle_pinned has lower MSE-vs-truth than diag -> the diagonal walk
    is failing to converge to a useful clean state and a perfect state
    stream would noticeably improve actions.

Usage:
    python scripts/diagnose_joint_ddt_state_sensitivity.py \\
        --ckpt_path  logs/<exp>/<ts>/models/model_step_90000.pt \\
        --config_path <run>/wandb/run-*/files/config.yaml \\
        --num_samples 32 \\
        --num_steps 25
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

LARDP_PATH = Path(__file__).resolve().parents[1]
if str(LARDP_PATH) not in sys.path:
    sys.path.append(str(LARDP_PATH))

from mip.agent_lbmdit_joint_ddt import LBMDiTJointDDTAgent
from mip.dataset_utils import RotationTransformer
from mip.datasets.robomimic_dataset import make_idm_dataset
from mip.networks.lbmdit_joint import LBMDiTJoint


def _unnormalize_actions(
    act_normed: torch.Tensor, action_normalizer,
) -> np.ndarray:
    """Inverse the MinMax normalizer to recover physical action units.

    Returns a numpy array of shape ``act_normed.shape`` with:
      [..., 0:3] pos in meters (world frame),
      [..., 3:9] rot6d (first two rows of rotation matrix),
      [..., 9:10] gripper command in {0, 1} (continuous in practice).
    """
    a = act_normed.detach().cpu().numpy().astype(np.float32)
    flat = a.reshape(-1, a.shape[-1])
    flat_un = action_normalizer.unnormalize(flat)
    return flat_un.reshape(a.shape)


def _rot6d_to_matrix(rot6d_flat: np.ndarray) -> np.ndarray:
    """(N, 6) -> (N, 3, 3) rotation matrices."""
    rt = RotationTransformer(from_rep="rotation_6d", to_rep="matrix")
    return rt.forward(rot6d_flat)


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Per-element geodesic angle (in degrees) between two batches of (.., 3, 3)."""
    M = np.matmul(R1, np.swapaxes(R2, -1, -2))
    tr = np.einsum("...ii->...", M)
    cos_th = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def _abs_space_stats(a_norm: torch.Tensor, b_norm: torch.Tensor,
                     action_normalizer) -> dict:
    """Decompose two action chunks to physical units and return a delta summary.

    Returns:
        dict with keys
          pos_l2_mean_m, pos_l2_max_m         — per-step Euclidean position delta (m)
          rot_geo_mean_deg, rot_geo_max_deg   — per-step geodesic rotation delta (deg)
          gripper_l1_mean, gripper_disagree   — per-step gripper command delta and the
                                                fraction of steps where the binary
                                                commands (threshold 0.5) disagree.
    """
    A = _unnormalize_actions(a_norm, action_normalizer)
    B = _unnormalize_actions(b_norm, action_normalizer)

    pos_a, pos_b = A[..., :3], B[..., :3]
    pos_l2 = np.linalg.norm(pos_a - pos_b, axis=-1)         # (n, H)

    n, H, _ = A.shape
    R_a = _rot6d_to_matrix(A[..., 3:9].reshape(-1, 6)).reshape(n, H, 3, 3)
    R_b = _rot6d_to_matrix(B[..., 3:9].reshape(-1, 6)).reshape(n, H, 3, 3)
    rot_deg = _geodesic_deg(R_a, R_b)                       # (n, H)

    grip_l1 = np.abs(A[..., 9] - B[..., 9])                 # (n, H)
    bin_a = (A[..., 9] > 0.5).astype(np.int32)
    bin_b = (B[..., 9] > 0.5).astype(np.int32)
    grip_disagree = float((bin_a != bin_b).mean())

    return {
        "pos_l2_mean_m":   float(pos_l2.mean()),
        "pos_l2_max_m":    float(pos_l2.max()),
        "rot_geo_mean_deg": float(rot_deg.mean()),
        "rot_geo_max_deg":  float(rot_deg.max()),
        "gripper_l1_mean": float(grip_l1.mean()),
        "gripper_disagree": grip_disagree,
    }


@torch.no_grad()
def _sample_with_state_mode(
    agent: LBMDiTJointDDTAgent,
    *,
    obs: dict,
    goal_obs: dict,
    act_0: torch.Tensor,
    x_state_init: torch.Tensor,
    num_steps: int,
    state_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the joint Euler walk with a configurable state stream.

    Re-implements ``LBMDiTJointDDTAgent.sample`` but exposes:
      * the initial state-slot draw (so all regimes share the same randn),
      * an explicit ``state_mode`` controlling the (t_state, x_state) walk.

    Returns ``(x_action_final, x_state_final)``.
    """
    net = agent.net_ema
    encoder, target_ln = agent._eval_encoder_modules(use_ema=True)
    device = act_0.device
    B = act_0.shape[0]
    cfg_scale = agent._cfg_scale
    eps = agent._t_eps
    lo, hi = eps, 1.0 - eps

    z_t = target_ln(encoder(obs, None))
    obs_dim = z_t.shape[-1]
    assert x_state_init.shape == (B, 1, obs_dim), (
        f"x_state_init shape {tuple(x_state_init.shape)} != {(B, 1, obs_dim)}"
    )

    if state_mode == "diagonal":
        x_state = x_state_init.clone()
        t_state_grid = np.linspace(lo, hi, num_steps + 1)
    elif state_mode == "noise_pinned":
        # Same init as diagonal but never integrates: ds=0 every step.
        x_state = x_state_init.clone()
        t_state_grid = np.full(num_steps + 1, lo)
    elif state_mode == "oracle_pinned":
        # Replace x_state with the oracle clean target, hold at t=hi.
        x_state = target_ln(encoder(goal_obs, None))
        t_state_grid = np.full(num_steps + 1, hi)
    else:
        raise ValueError(f"Unknown state_mode: {state_mode!r}")

    t_action_grid = np.linspace(lo, hi, num_steps + 1)

    # Per-stream SD3 shift (no-op when alpha == 1; this ckpt trained at 1.0).
    t_state_grid = agent._apply_t_shift(t_state_grid, agent._shift_state)
    t_action_grid = agent._apply_t_shift(t_action_grid, agent._shift_action)

    x_action = act_0.clone()

    expert_idx = torch.full(
        (B,), LBMDiTJoint.EXPERT_IDX, device=device, dtype=torch.long,
    )
    null_idx = torch.full(
        (B,), LBMDiTJoint.NULL_IDX, device=device, dtype=torch.long,
    )

    for i in range(num_steps):
        ts_now = float(t_state_grid[i])
        ta_now = float(t_action_grid[i])
        ds = float(t_state_grid[i + 1] - ts_now)
        da = float(t_action_grid[i + 1] - ta_now)
        t_state_b = torch.full((B,), ts_now, device=device)
        t_action_b = torch.full((B,), ta_now, device=device)

        v_s_cond, v_a_cond, _ = net(
            x_state, x_action, t_state_b, t_action_b, z_t, expert_idx,
        )
        if cfg_scale > 0:
            v_s_un, v_a_un, _ = net(
                x_state, x_action, t_state_b, t_action_b, z_t, null_idx,
            )
            v_s = (1 + cfg_scale) * v_s_cond - cfg_scale * v_s_un
            v_a = (1 + cfg_scale) * v_a_cond - cfg_scale * v_a_un
        else:
            v_s, v_a = v_s_cond, v_a_cond

        x_state = x_state + v_s * ds
        x_action = x_action + v_a * da

    return x_action, x_state


def _per_dim_stats(name: str, x: torch.Tensor, dim_names: list[str]):
    print(f"  {name}:")
    abs_ = x.abs().mean(dim=(0, 1))
    sq_ = (x ** 2).mean(dim=(0, 1)).sqrt()
    for i, dn in enumerate(dim_names):
        print(f"    {dn:<8s}  MAE={abs_[i].item():.4f}  RMSE={sq_[i].item():.4f}")


def diagnose(args):
    cfg = OmegaConf.load(args.config_path)
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)
    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)

    device = torch.device(cfg.optimization.device)
    print(f"\n== device: {device} ==")
    print(f"   ckpt:   {args.ckpt_path}")
    print(f"   config: {args.config_path}")
    print(f"   schedule (trained): {cfg.optimization.joint_t_schedule}, "
          f"shift_state={cfg.optimization.joint_t_shift_state}, "
          f"shift_action={cfg.optimization.joint_t_shift_action}, "
          f"t_dist={cfg.optimization.joint_t_dist}")

    print("\n[1/3] Building agent + loading checkpoint...")
    agent = LBMDiTJointDDTAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()

    # ---- dataset ----
    norm_override = None
    if args.normalizer_path is not None:
        with open(args.normalizer_path, "rb") as f:
            norm_override = pickle.load(f)
        print(f"   using normalizer from: {args.normalizer_path}")
    elif args.dataset_path is not None:
        # When evaluating a heldout file, the heldout-only normalizer would
        # have a different action scale than the one the model trained on.
        # Re-derive it from the training HDF5 so the action units match.
        print("\n[2a/3] Re-deriving normalizer from training dataset (for "
              "heldout eval) ...")
        train_dataset = make_idm_dataset(cfg.task, mode="train")
        base_train = (
            train_dataset.datasets[0]
            if isinstance(train_dataset, torch.utils.data.ConcatDataset)
            else train_dataset
        )
        norm_override = base_train.normalizer
        del train_dataset, base_train
        print("   training normalizer captured; switching to heldout HDF5")

    if args.dataset_path is not None:
        OmegaConf.update(cfg, "task.dataset_paths", [args.dataset_path],
                         merge=False)
        OmegaConf.update(cfg, "task.dataset_path", None, merge=False)
        OmegaConf.update(cfg, "task.val_dataset_percentage", 0.0,
                         merge=False)
        print(f"\n[2/3] Loading heldout dataset: {args.dataset_path}")
    else:
        print("\n[2/3] Loading training dataset for in-distribution samples...")

    dataset = make_idm_dataset(cfg.task, mode="train", normalizer=norm_override)
    base_ds = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )
    action_normalizer = base_ds.normalizer["action"]
    n = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    print(f"   dataset size={len(dataset)}, sampling {n} indices")

    sample_keys = list(cfg.task.shape_meta["obs"].keys())
    obs_batch = {k: [] for k in sample_keys}
    goal_batch = {k: [] for k in sample_keys}
    act_batch_normed = []
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
            goal_batch[k].append(s["goal_obs"][k])
        act_batch_normed.append(s["action"][: cfg.task.horizon])

    obs_torch = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in obs_batch.items()
    }
    goal_torch = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in goal_batch.items()
    }
    act_true = torch.from_numpy(
        np.stack([t.numpy() for t in act_batch_normed])
    ).to(device)

    # ---- shared randomness across the three regimes ----
    print("\n[3/3] Running schedule-swap probe...")
    print(f"   num_steps={args.num_steps}  num_samples={n}")

    obs_dim = cfg.network.encoder_out_dim or cfg.network.emb_dim
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    act_0 = torch.randn(
        (n, cfg.task.horizon, cfg.task.act_dim), generator=g,
    ).to(device)
    x_state_init = torch.randn(
        (n, 1, obs_dim), generator=g,
    ).to(device)
    print(f"   act_0 stats: mean={act_0.mean().item():+.3f} "
          f"std={act_0.std().item():.3f}")
    print(f"   x_state_init stats: mean={x_state_init.mean().item():+.3f} "
          f"std={x_state_init.std().item():.3f}")

    results: dict[str, torch.Tensor] = {}
    states: dict[str, torch.Tensor] = {}
    for mode in ("diagonal", "noise_pinned", "oracle_pinned"):
        a, s = _sample_with_state_mode(
            agent,
            obs=obs_torch, goal_obs=goal_torch,
            act_0=act_0, x_state_init=x_state_init,
            num_steps=args.num_steps,
            state_mode=mode,
        )
        results[mode] = a
        states[mode] = s

    dim_names = ["pos_x", "pos_y", "pos_z",
                 "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
                 "gripper"]

    # -----------------------------------------------------------------------
    # 1. Pairwise action deltas — does the action chunk move when we swap
    #    the state stream's behavior?
    # -----------------------------------------------------------------------
    print("\n== (1) Pairwise action-chunk deltas (shared obs + act noise) ==")
    pairs = [
        ("diagonal",       "noise_pinned"),
        ("diagonal",       "oracle_pinned"),
        ("noise_pinned",   "oracle_pinned"),
    ]
    for a_name, b_name in pairs:
        d = results[a_name] - results[b_name]
        l2 = d.pow(2).mean().sqrt().item()
        mae = d.abs().mean().item()
        ref_scale = results[a_name].abs().mean().item() + 1e-12
        per_dim_rmse = d.pow(2).mean(dim=(0, 1)).sqrt()
        per_dim_mae = d.abs().mean(dim=(0, 1))
        print(f"\n  {a_name:<14s} vs {b_name:<14s}")
        print(f"    overall RMSE = {l2:.4f}   MAE = {mae:.4f}   "
              f"relative-to-|a_{a_name}|.mean() = {l2 / ref_scale:.3f}")
        for i, dn in enumerate(dim_names):
            print(f"    {dn:<8s}  ΔRMSE={per_dim_rmse[i].item():.4f}  "
                  f"ΔMAE={per_dim_mae[i].item():.4f}")

    # -----------------------------------------------------------------------
    # 1b. Absolute-space pairwise deltas: position (m), rotation (deg, geodesic),
    #     gripper (raw 0/1 command).
    # -----------------------------------------------------------------------
    print("\n== (1b) Pairwise action deltas in PHYSICAL units ==")
    print(f"  {'pair':<32s}{'pos Δ (m)':>22s}{'rot Δ (deg)':>22s}"
          f"{'gripper':>22s}")
    print(f"  {'':<32s}{'mean  /  max':>22s}{'mean  /  max':>22s}"
          f"{'L1  /  disagree%':>22s}")
    for a_name, b_name in pairs:
        st = _abs_space_stats(results[a_name], results[b_name], action_normalizer)
        print(
            f"  {a_name + ' vs ' + b_name:<32s}"
            f"{st['pos_l2_mean_m']*1000:>9.2f}mm/{st['pos_l2_max_m']*1000:>7.2f}mm "
            f"{st['rot_geo_mean_deg']:>9.3f}°/{st['rot_geo_max_deg']:>8.3f}°  "
            f"{st['gripper_l1_mean']:>8.4f}/{st['gripper_disagree']*100:>7.2f}%"
        )

    # -----------------------------------------------------------------------
    # 2. Each regime's MSE / MAE vs ground-truth normalized action — which
    #    state stream produces the most accurate actions on in-dist obs?
    # -----------------------------------------------------------------------
    print("\n== (2) Each regime vs dataset ground-truth normalized action ==")
    print(f"  {'regime':<16s}{'overall MSE':>14s}{'overall MAE':>14s}")
    for mode in ("diagonal", "noise_pinned", "oracle_pinned"):
        e = results[mode] - act_true
        mse = e.pow(2).mean().item()
        mae = e.abs().mean().item()
        print(f"  {mode:<16s}{mse:>14.4f}{mae:>14.4f}")
    print()
    print(f"  {'dim':<8s}" + "".join(
        f"{m+' MSE':>16s}" for m in ("diag", "noise", "oracle")
    ))
    for i, dn in enumerate(dim_names):
        row = f"  {dn:<8s}"
        for mode in ("diagonal", "noise_pinned", "oracle_pinned"):
            e = results[mode] - act_true
            row += f"{e[..., i].pow(2).mean().item():>16.4f}"
        print(row)

    # -----------------------------------------------------------------------
    # 2b. Each regime vs ground truth, in PHYSICAL units.
    # -----------------------------------------------------------------------
    print("\n== (2b) Each regime vs ground truth in PHYSICAL units ==")
    print(f"  {'regime':<16s}{'pos Δ (m)':>22s}{'rot Δ (deg)':>22s}"
          f"{'gripper':>22s}")
    print(f"  {'':<16s}{'mean  /  max':>22s}{'mean  /  max':>22s}"
          f"{'L1  /  disagree%':>22s}")
    for mode in ("diagonal", "noise_pinned", "oracle_pinned"):
        st = _abs_space_stats(results[mode], act_true, action_normalizer)
        print(
            f"  {mode:<16s}"
            f"{st['pos_l2_mean_m']*1000:>9.2f}mm/{st['pos_l2_max_m']*1000:>7.2f}mm "
            f"{st['rot_geo_mean_deg']:>9.3f}°/{st['rot_geo_max_deg']:>8.3f}°  "
            f"{st['gripper_l1_mean']:>8.4f}/{st['gripper_disagree']*100:>7.2f}%"
        )

    # -----------------------------------------------------------------------
    # 3. Diagonal's final state vs the oracle clean state — is the diagonal
    #    walk actually recovering the oracle clean state, or drifting?
    # -----------------------------------------------------------------------
    print("\n== (3) Diagonal final x_state vs oracle clean state ==")
    s_diag = states["diagonal"]                     # (n, 1, obs_dim)
    s_oracle = states["oracle_pinned"]              # oracle (constant, == target_ln(enc(goal)))
    d = s_diag - s_oracle
    rmse = d.pow(2).mean().sqrt().item()
    mae = d.abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        s_diag.flatten(1), s_oracle.flatten(1), dim=-1,
    ).mean().item()
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"mean_cosine(s_diag, s_oracle)={cos:.4f}")
    print(f"  ||s_diag||_2.mean={s_diag.flatten(1).norm(dim=-1).mean().item():.3f}   "
          f"||s_oracle||_2.mean={s_oracle.flatten(1).norm(dim=-1).mean().item():.3f}")

    # -----------------------------------------------------------------------
    # 4. Bottom-line interpretive summary.
    # -----------------------------------------------------------------------
    print("\n== bottom line ==")
    d_dn = (results["diagonal"] - results["noise_pinned"]).pow(2).mean().sqrt().item()
    d_do = (results["diagonal"] - results["oracle_pinned"]).pow(2).mean().sqrt().item()
    d_no = (results["noise_pinned"] - results["oracle_pinned"]).pow(2).mean().sqrt().item()
    mse_true = {
        m: (results[m] - act_true).pow(2).mean().item()
        for m in ("diagonal", "noise_pinned", "oracle_pinned")
    }
    print(f"  RMSE(action) diag<->noise   = {d_dn:.4f}")
    print(f"  RMSE(action) diag<->oracle  = {d_do:.4f}")
    print(f"  RMSE(action) noise<->oracle = {d_no:.4f}")
    print(f"  MSE-vs-truth  diag={mse_true['diagonal']:.4f}  "
          f"noise={mse_true['noise_pinned']:.4f}  "
          f"oracle={mse_true['oracle_pinned']:.4f}")
    print("  -> If d_no is ~0, the action head ignores x_state content (it's "
          "either using only its slot as a learned register, or the trunk's "
          "cross-token mixing isn't transmitting state info to action tokens).")
    print("  -> If d_dn ~= 0 but d_no >> 0, the diagonal walk's state stream "
          "isn't moving enough off the noise endpoint to matter.")
    print("  -> If MSE-vs-truth ranks oracle < diag << noise, the action head "
          "*does* use a clean state, but diagonal under-delivers it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--normalizer_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Optional heldout HDF5 path. When set, the "
                             "normalizer is re-derived from the original "
                             "training HDF5 (per cfg.task.dataset_paths) "
                             "so action units stay consistent.")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose(args)
