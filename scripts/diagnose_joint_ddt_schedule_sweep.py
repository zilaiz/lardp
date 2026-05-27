"""Inference t-schedule sweep for an LBMDiTJointDDTAgent checkpoint.

Companion to ``diagnose_joint_ddt_state_sensitivity.py``. Holds (obs, act_0,
x_state_init) constant and runs the joint Euler walk under multiple
(t_state, t_action) schedules so we can answer:

    "Does state_first or pyramid (with various lead offsets) produce more
     accurate action chunks than the default diagonal schedule, especially
     out-of-distribution?"

Schedules compared (mip's t convention: 0 = noise, 1 = data):
  diagonal              t_state == t_action throughout, both lo -> hi
  state_first           clean state first (lo -> hi in first half, then hold),
                        then clean action (hold at lo, then lo -> hi)
  pyramid (offset=p)    state leads action by p*num_steps Euler sub-steps;
                        offset=0 -> diagonal; offset~=0.5 -> state_first
  action_only           t_state pinned at lo (x_state never integrates);
                        included as a lower-bound reference

For each schedule we report (vs the dataset's ground-truth action, in
physical units): per-step mean / max position error (mm), per-step mean /
max geodesic rotation error (deg), per-step gripper L1 + binary-flip
fraction. Also reports the cosine between the final x_state token and the
oracle target_ln(encoder(goal_obs)) so you can see whether the schedule
actually lets the state denoiser land closer to the oracle.

Usage (heldout, recommended):
    python scripts/diagnose_joint_ddt_schedule_sweep.py \
        --ckpt_path  logs/<exp>/<ts>/models/model_step_90000.pt \
        --config_path outputs/<date>/<time>/.hydra/config.yaml \
        --dataset_path data/franka_coffee_pod_cog/image_heldout.hdf5 \
        --num_samples 64 \
        --num_steps 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

LARDP_PATH = Path(__file__).resolve().parents[1]
if str(LARDP_PATH) not in sys.path:
    sys.path.append(str(LARDP_PATH))

from mip.agent_lbmdit_joint_ddt import LBMDiTJointDDTAgent
from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
from mip.dataset_utils import RotationTransformer
from mip.datasets.robomimic_dataset import make_idm_dataset
from mip.networks.lbmdit_joint import LBMDiTJoint


_AGENT_CLS = {
    "lbmdit_joint_ddt": LBMDiTJointDDTAgent,
    "lbmdit_joint_pt":  LBMDiTJointPTAgent,
}


def _pick_agent_cls(cfg):
    net_type = getattr(cfg.network, "network_type", None)
    if net_type in _AGENT_CLS:
        return _AGENT_CLS[net_type]
    raise ValueError(
        f"Unknown joint network_type {net_type!r}; expected one of {list(_AGENT_CLS)}"
    )


# ----------------------- physical-unit helpers (shared) -----------------------


def _unnormalize_actions(act_normed: torch.Tensor, action_normalizer) -> np.ndarray:
    a = act_normed.detach().cpu().numpy().astype(np.float32)
    flat = a.reshape(-1, a.shape[-1])
    flat_un = action_normalizer.unnormalize(flat)
    return flat_un.reshape(a.shape)


def _rot6d_to_matrix(rot6d_flat: np.ndarray) -> np.ndarray:
    rt = RotationTransformer(from_rep="rotation_6d", to_rep="matrix")
    return rt.forward(rot6d_flat)


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    M = np.matmul(R1, np.swapaxes(R2, -1, -2))
    tr = np.einsum("...ii->...", M)
    cos_th = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def _abs_space_stats(a_norm: torch.Tensor, b_norm: torch.Tensor,
                     action_normalizer) -> dict:
    A = _unnormalize_actions(a_norm, action_normalizer)
    B = _unnormalize_actions(b_norm, action_normalizer)
    pos_l2 = np.linalg.norm(A[..., :3] - B[..., :3], axis=-1)
    n, H, _ = A.shape
    R_a = _rot6d_to_matrix(A[..., 3:9].reshape(-1, 6)).reshape(n, H, 3, 3)
    R_b = _rot6d_to_matrix(B[..., 3:9].reshape(-1, 6)).reshape(n, H, 3, 3)
    rot_deg = _geodesic_deg(R_a, R_b)
    grip_l1 = np.abs(A[..., 9] - B[..., 9])
    bin_a = (A[..., 9] > 0.5).astype(np.int32)
    bin_b = (B[..., 9] > 0.5).astype(np.int32)
    return {
        "pos_l2_mean_mm": float(pos_l2.mean()) * 1000.0,
        "pos_l2_max_mm":  float(pos_l2.max())  * 1000.0,
        "rot_geo_mean_deg": float(rot_deg.mean()),
        "rot_geo_max_deg":  float(rot_deg.max()),
        "gripper_l1_mean": float(grip_l1.mean()),
        "gripper_disagree_pct": float((bin_a != bin_b).mean()) * 100.0,
    }


# --------------------- schedule-aware joint Euler walk ----------------------


@torch.no_grad()
def _sample_with_pinning(
    agent: LBMDiTJointDDTAgent,
    *,
    obs: dict,
    goal_obs: dict,
    act_0: torch.Tensor,
    x_state_init: torch.Tensor,
    num_steps: int,
    pin_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the joint walk with the state slot pinned to a fixed content.

    pin_mode:
      "oracle_pinned" — x_state = target_ln(encoder(goal_obs)), t_state = hi
      "noise_pinned"  — x_state = x_state_init (randn), t_state = lo
    """
    net = agent.net_ema
    encoder, target_ln = agent._eval_encoder_modules(use_ema=True)
    device = act_0.device
    B = act_0.shape[0]
    cfg_scale = agent._cfg_scale
    eps = agent._t_eps
    lo, hi = eps, 1.0 - eps

    z_t = target_ln(encoder(obs, None))

    if pin_mode == "oracle_pinned":
        x_state = target_ln(encoder(goal_obs, None))
        t_state_grid = np.full(num_steps + 1, hi)
    elif pin_mode == "noise_pinned":
        x_state = x_state_init.clone()
        t_state_grid = np.full(num_steps + 1, lo)
    else:
        raise ValueError(pin_mode)

    t_action_grid = np.linspace(lo, hi, num_steps + 1)
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


@torch.no_grad()
def _sample_with_schedule(
    agent: LBMDiTJointDDTAgent,
    *,
    obs: dict,
    act_0: torch.Tensor,
    x_state_init: torch.Tensor,
    num_steps: int,
    schedule: str,
    pyramid_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint Euler walk with a configurable (t_state, t_action) schedule.

    Mirrors ``agent.sample`` for ``sample_mode == "stochastic"`` but accepts
    a shared ``x_state_init`` (so we can compare schedules with matched
    initial noise) and toggles the schedule via the agent's own
    ``_build_schedule`` so the SD3 shift / eps / pyramid logic stays
    consistent with training-time conventions.
    """
    net = agent.net_ema
    encoder, target_ln = agent._eval_encoder_modules(use_ema=True)
    device = act_0.device
    B = act_0.shape[0]
    cfg_scale = agent._cfg_scale

    z_t = target_ln(encoder(obs, None))
    obs_dim = z_t.shape[-1]
    assert x_state_init.shape == (B, 1, obs_dim)

    saved_schedule = agent._t_schedule
    saved_offset = agent._pyramid_offset
    try:
        agent._t_schedule = schedule
        agent._pyramid_offset = float(pyramid_offset)
        t_state_grid, t_action_grid = agent._build_schedule(num_steps)
    finally:
        agent._t_schedule = saved_schedule
        agent._pyramid_offset = saved_offset

    x_state = x_state_init.clone()
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


# ------------------------------ main probe ----------------------------------


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
    print(f"   trained schedule: {cfg.optimization.joint_t_schedule}")

    AgentCls = _pick_agent_cls(cfg)
    print(f"\n[1/3] Building agent ({AgentCls.__name__}) + loading checkpoint...")
    agent = AgentCls(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()

    # ---- dataset (training, or heldout with re-derived training normalizer) ----
    norm_override = None
    if args.dataset_path is not None:
        print("\n[2a/3] Re-deriving normalizer from training dataset...")
        train_dataset = make_idm_dataset(cfg.task, mode="train")
        base_train = (
            train_dataset.datasets[0]
            if isinstance(train_dataset, torch.utils.data.ConcatDataset)
            else train_dataset
        )
        norm_override = base_train.normalizer
        del train_dataset, base_train
        OmegaConf.update(cfg, "task.dataset_paths", [args.dataset_path],
                         merge=False)
        OmegaConf.update(cfg, "task.dataset_path", None, merge=False)
        OmegaConf.update(cfg, "task.val_dataset_percentage", 0.0, merge=False)
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

    # ---- shared randomness across all schedules ----
    print("\n[3/3] Sweeping inference schedules...")
    print(f"   num_steps={args.num_steps}  num_samples={n}")

    obs_dim = cfg.network.encoder_out_dim or cfg.network.emb_dim
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    act_0 = torch.randn(
        (n, cfg.task.horizon, cfg.task.act_dim), generator=g,
    ).to(device)
    x_state_init = torch.randn(
        (n, 1, obs_dim), generator=g,
    ).to(device)

    # Oracle clean state for the cosine readout
    with torch.no_grad():
        encoder, target_ln = agent._eval_encoder_modules(use_ema=True)
        s_oracle = target_ln(encoder(goal_torch, None))

    # ---- schedules to sweep ----
    # Each entry: (display name, kind, schedule-or-pinmode, pyramid_offset)
    # kind: "schedule" runs the joint walk under the given t-schedule;
    #       "pin" pins x_state to noise or oracle for the full walk.
    schedules: list[tuple[str, str, str, float]] = [
        ("diagonal",         "schedule", "diagonal",      0.0),
        ("pyramid_off=0.10", "schedule", "pyramid",       0.10),
        ("pyramid_off=0.25", "schedule", "pyramid",       0.25),
        ("pyramid_off=0.40", "schedule", "pyramid",       0.40),
        ("pyramid_off=0.50", "schedule", "pyramid",       0.50),
        ("state_first",      "schedule", "state_first",   0.0),
        ("action_only",      "schedule", "action_only",   0.0),
        ("noise_pinned",     "pin",      "noise_pinned",  0.0),
        ("oracle_pinned",    "pin",      "oracle_pinned", 0.0),
    ]

    per_schedule_action: dict[str, torch.Tensor] = {}
    per_schedule_state:  dict[str, torch.Tensor] = {}
    for name, kind, sched_or_pin, off in schedules:
        if kind == "schedule":
            a, s = _sample_with_schedule(
                agent,
                obs=obs_torch, act_0=act_0, x_state_init=x_state_init,
                num_steps=args.num_steps,
                schedule=sched_or_pin, pyramid_offset=off,
            )
        else:
            a, s = _sample_with_pinning(
                agent,
                obs=obs_torch, goal_obs=goal_torch,
                act_0=act_0, x_state_init=x_state_init,
                num_steps=args.num_steps,
                pin_mode=sched_or_pin,
            )
        per_schedule_action[name] = a
        per_schedule_state[name]  = s

    # -----------------------------------------------------------------------
    # (1) Each schedule vs ground truth, PHYSICAL units.
    # -----------------------------------------------------------------------
    print("\n== (1) Each schedule vs ground truth in PHYSICAL units ==")
    header = (
        f"  {'schedule':<20s}{'pos mean':>11s}{'pos max':>11s}"
        f"{'rot mean':>11s}{'rot max':>11s}"
        f"{'grip L1':>11s}{'flip %':>9s}{'cos(s,oracle)':>16s}"
    )
    print(header)
    rows = []
    for name, *_ in schedules:
        st = _abs_space_stats(per_schedule_action[name], act_true, action_normalizer)
        cos = torch.nn.functional.cosine_similarity(
            per_schedule_state[name].flatten(1),
            s_oracle.flatten(1), dim=-1,
        ).mean().item()
        rows.append((name, st, cos))
        print(
            f"  {name:<20s}"
            f"{st['pos_l2_mean_mm']:>9.2f}mm"
            f"{st['pos_l2_max_mm']:>9.2f}mm"
            f"{st['rot_geo_mean_deg']:>9.3f}°"
            f"{st['rot_geo_max_deg']:>9.3f}°"
            f"{st['gripper_l1_mean']:>11.4f}"
            f"{st['gripper_disagree_pct']:>8.2f}%"
            f"{cos:>16.4f}"
        )

    # -----------------------------------------------------------------------
    # (2) Pairwise action deltas vs the diagonal baseline (PHYSICAL units).
    # -----------------------------------------------------------------------
    print("\n== (2) Each schedule vs DIAGONAL baseline in PHYSICAL units ==")
    print(f"  {'schedule':<20s}{'pos mean':>11s}{'pos max':>11s}"
          f"{'rot mean':>11s}{'rot max':>11s}"
          f"{'grip L1':>11s}{'flip %':>9s}")
    a_diag = per_schedule_action["diagonal"]
    for name, *_ in schedules:
        if name == "diagonal":
            continue
        st = _abs_space_stats(per_schedule_action[name], a_diag, action_normalizer)
        print(
            f"  {name:<20s}"
            f"{st['pos_l2_mean_mm']:>9.2f}mm"
            f"{st['pos_l2_max_mm']:>9.2f}mm"
            f"{st['rot_geo_mean_deg']:>9.3f}°"
            f"{st['rot_geo_max_deg']:>9.3f}°"
            f"{st['gripper_l1_mean']:>11.4f}"
            f"{st['gripper_disagree_pct']:>8.2f}%"
        )

    # -----------------------------------------------------------------------
    # (3) Bottom-line ranking by (pos_mean, rot_mean, gripper_disagree).
    # -----------------------------------------------------------------------
    print("\n== (3) Best schedule per metric ==")
    by_pos    = sorted(rows, key=lambda r: r[1]["pos_l2_mean_mm"])
    by_rot    = sorted(rows, key=lambda r: r[1]["rot_geo_mean_deg"])
    by_grip   = sorted(rows, key=lambda r: r[1]["gripper_disagree_pct"])
    by_cos    = sorted(rows, key=lambda r: -r[2])
    print(f"  Lowest pos error : {by_pos[0][0]} "
          f"({by_pos[0][1]['pos_l2_mean_mm']:.2f} mm)")
    print(f"  Lowest rot error : {by_rot[0][0]} "
          f"({by_rot[0][1]['rot_geo_mean_deg']:.3f}°)")
    print(f"  Lowest grip flips: {by_grip[0][0]} "
          f"({by_grip[0][1]['gripper_disagree_pct']:.2f}%)")
    print(f"  Closest x_state to oracle: {by_cos[0][0]} "
          f"(cos={by_cos[0][2]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Optional heldout HDF5 path. When set, the "
                             "normalizer is re-derived from the original "
                             "training HDF5 so action units stay consistent.")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose(args)
