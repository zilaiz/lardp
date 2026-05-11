"""Offline goal-predictor eval on the Franka coffee-pod held-out expert split.

Validates a goal-predictor DiT checkpoint along two axes:

(1) **Embedding fidelity** — does the goal DiT produce a goal embedding
    close to the frozen IDM-encoder's embedding of the expert goal frame?
    Reports L2 / cosine-sim in raw encoder space, plus L2 in z-scored
    "normalized goal" space (the space the DiT was trained to match).

(2) **Action reachability** — given the predicted goal embedding, do
    actions sampled by the frozen IDM still reach the expert goal eef
    position? Reports world-frame L2 in meters at the goal step. Also
    reports the IDM ceiling: the same metric when the IDM is fed the
    expert goal embedding directly (bypassing the goal predictor) — gives
    you the gap between "best the IDM could ever do" and "what the goal
    predictor lets it do."

Procedure
---------
1. Load the saved IDM normalizer (sibling-of-IDM-checkpoint convention)
   and the goal stats (z-score mean/var) — both produced by the IDM
   training run.
2. Build `RobomimicImageIDMDataset` over the held-out HDF5 with the
   training-time normalizer. Same slicing as IDM eval.
3. For each goal-predictor checkpoint, instantiate
   `GoalPredictorDiTAgent` (which loads the frozen IDM internally) and
   load goal-DiT weights via `agent.load`. The frozen IDM is shared
   across the sweep — only the goal DiT changes.
4. Per batch, on EMA goal-DiT:
     a. Encode obs (To frames) and goal_obs (1 frame) with frozen IDM
        encoder. Z-score normalize the expert goal for the
        normalized-space metric.
     b. Run goal-DiT ODE on z_t to get a predicted normalized goal;
        denormalize to raw encoder space.
     c. Compute embedding fidelity metrics.
     d. Run IDM Euler ODE twice on the same noise:
        - cond stack with predicted goal  → action_pred_with_predicted
        - cond stack with expert  goal    → action_pred_with_expert
     e. Unnormalize actions, compare action[:, -1, :3] vs expert goal eef.

Usage
-----
    python scripts/eval_franka_goal_predictor_offline.py \\
        --gp-ckpt-dir logs/franka_coffee_pod_real_image_goal_predictor_dit_lbmidm_v2_256_seed0/2026_05_03_00_17_09/models \\
        --idm-ckpt    logs/franka_coffee_pod_real_image_flow_lbmidm_v2_256_seed0/2026_05_02_15_29_34/models/model_step_130000.pt \\
        --goal-stats  logs/franka_coffee_pod_real_image_flow_lbmidm_v2_256_seed0/2026_05_02_15_29_34/models/goal_stats_130000.pt \\
        --heldout-path data/franka_coffee_pod_cog/image_heldout.hdf5 \\
        --steps 10000,30000,50000,80000,130000,200000
"""

import argparse
import os
import pickle
import sys
import time
from glob import glob

import hydra
import loguru
import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, "/oscar/data/csun45/zzeng28/repo/lardp")

from mip.agent_goal_predictor_dit import GoalPredictorDiTAgent  # noqa: E402
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

CONFIG_DIR = "/oscar/data/csun45/zzeng28/repo/lardp/examples/configs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_cfg(idm_ckpt, goal_stats, batch_size, goal_dit_d_model=512):
    overrides = [
        "task=franka_coffee_pod_cog_image_gp",
        # goal_predictor_dit_v2 inherits from lbmidm_v2 — required because the
        # IDM was trained with the v2 architecture (with FDM head + decoupled
        # timestep_emb_dim). Using plain goal_predictor_dit yields shape
        # mismatches on net.time_mlp / adaLN_modulation when loading the IDM.
        "network=goal_predictor_dit_v2",
        # The GP at logs/.../2026_05_03_00_17_09 was trained with d_model=512
        # (the default null=enc_out_dim resolves to 256 — too small to match
        # the checkpoint's projector/blocks.mlp shapes). Override accordingly.
        f"network.goal_dit_d_model={goal_dit_d_model}",
        "optimization.loss_type=goal_predictor_dit",
        f"optimization.batch_size={batch_size}",
        f"optimization.idm_checkpoint_path={idm_ckpt}",
        f"optimization.goal_stats_path={goal_stats}",
        "optimization.auto_resume=False",
        "log.wandb_mode=disabled",
    ]
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = hydra.compose(config_name="main_franka", overrides=overrides)
    return cfg


def to_device_obs(batch_dict, slice_steps, device):
    out = {}
    for k, v in batch_dict.items():
        if slice_steps is not None:
            v = v[:, :slice_steps]
        out[k] = v.to(device, non_blocking=True)
    bs = next(iter(out.values())).shape[0]
    return TensorDict(out, batch_size=bs)


@torch.no_grad()
def goal_dit_ode(goal_flow_map, z_t, num_steps, generator):
    """Euler ODE in normalized goal space. Returns g in (B, 1, emb_dim)."""
    B, _, emb_dim = z_t.shape
    device = z_t.device
    g = torch.randn(
        (B, 1, emb_dim), device=device, dtype=z_t.dtype, generator=generator,
    )
    schedule = np.linspace(0, 1, num_steps + 1)
    for i in range(num_steps):
        s_val, t_val = schedule[i], schedule[i + 1]
        s = torch.full((B,), s_val, device=device)
        v = goal_flow_map.get_velocity(s, g, z_t)
        g = g + v * (t_val - s_val)
    return g  # normalized goal


@torch.no_grad()
def idm_ode_with_obs_emb(flow_map, obs_emb, act_0_shape, num_steps, sample_mode):
    """Euler ODE for IDM with a precomputed obs+goal stack (mirrors ode_sampler)."""
    B, Ta, act_dim = act_0_shape
    device = obs_emb.device
    if sample_mode == "stochastic":
        act_s = torch.randn((B, Ta, act_dim), device=device, dtype=obs_emb.dtype)
    else:
        act_s = torch.zeros((B, Ta, act_dim), device=device, dtype=obs_emb.dtype)
    schedule = np.linspace(0, 1, num_steps + 1)
    for i in range(num_steps):
        s_val, t_val = schedule[i], schedule[i + 1]
        s = torch.full((B,), s_val, device=device)
        b_s = flow_map.get_velocity(s, act_s, obs_emb)
        act_s = act_s + b_s * (t_val - s_val)
    return act_s


def discover_gp_checkpoints(ckpt_dir, ckpt, steps=None, sweep_stride=None):
    if ckpt is not None:
        return [(os.path.basename(ckpt), ckpt)]
    paths = sorted(glob(os.path.join(ckpt_dir, "model_step_*.pt")))
    items = []
    for p in paths:
        try:
            step = int(os.path.basename(p).split("_step_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        items.append((step, p))
    items.sort(key=lambda t: t[0])
    if steps:
        wanted = set(steps)
        avail = {s for s, _ in items}
        missing = sorted(wanted - avail)
        if missing:
            raise FileNotFoundError(
                f"Requested steps {missing} not found under {ckpt_dir} "
                f"(available: {sorted(avail)})"
            )
        items = [(s, p) for s, p in items if s in wanted]
    elif sweep_stride is not None and sweep_stride > 1:
        items = items[::sweep_stride]
    if not items:
        raise FileNotFoundError(f"No model_step_*.pt under {ckpt_dir}")
    return [(f"step_{s}", p) for s, p in items]


@torch.no_grad()
def evaluate_gp_checkpoint(
    agent: GoalPredictorDiTAgent,
    loader,
    normalizer,
    obs_steps: int,
    horizon: int,
    nfe_idm: int,
    nfe_goal: int,
    sample_mode: str,
    seed: int,
    max_batches: int | None,
):
    agent.eval()
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    act_norm = normalizer["action"]
    eef_norm = normalizer["obs"]["robot0_eef_pos"]

    emb_l2_list = []
    emb_cos_list = []
    emb_norm_l2_list = []
    pred_l2_list = []
    expert_l2_list = []
    pred_action_mse_list = []
    expert_action_mse_list = []
    # Direction-vs-magnitude diagnostics: are predicted-goal-conditioned
    # actions at least *moving the eef in the correct direction*? cos_sim
    # is between two 3D vectors (delta_pred, delta_expert) where
    #   delta_*    = predicted_final_eef_*       - current_eef
    #   delta_expert = expert_goal_eef           - current_eef
    # current_eef is obs["robot0_eef_pos"][:, -1, :] (last obs frame).
    pred_delta_cos_list = []
    expert_delta_cos_list = []
    pred_mag_ratio_list = []
    expert_mag_ratio_list = []
    delta_expert_norm_list = []  # how far the eef must travel (sanity)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        obs = to_device_obs(
            {k: v for k, v in batch["obs"].items()}, slice_steps=obs_steps, device=DEVICE,
        )
        goal_obs = to_device_obs(
            {k: v for k, v in batch["goal_obs"].items()}, slice_steps=None, device=DEVICE,
        )
        act_gt = batch["action"][:, :horizon, :].to(DEVICE, non_blocking=True)
        B = act_gt.shape[0]
        act_dim = act_gt.shape[-1]

        # 1) Encode with frozen IDM encoder.
        z_t = agent._inner_encoder(obs, None)                  # (B, To, emb_dim)
        z_goal_raw = agent._inner_encoder(goal_obs, None)       # (B, 1, emb_dim)
        expert_goal = z_goal_raw[:, 0, :]                       # (B, emb_dim) raw

        # 2) Goal-DiT ODE → predicted normalized goal → raw goal.
        pred_goal_norm = goal_dit_ode(
            agent.goal_flow_map_ema, z_t, nfe_goal, generator=g,
        )                                                       # (B, 1, emb_dim) normalized
        pred_goal_raw = agent._denormalize(pred_goal_norm)[:, 0, :]   # (B, emb_dim)

        # Embedding fidelity metrics.
        emb_l2 = torch.norm(pred_goal_raw - expert_goal, dim=-1)         # (B,)
        emb_cos = torch.nn.functional.cosine_similarity(
            pred_goal_raw, expert_goal, dim=-1,
        )                                                                # (B,)
        expert_goal_norm = agent._normalize(z_goal_raw)[:, 0, :]         # (B, emb_dim)
        emb_norm_l2 = torch.norm(
            pred_goal_norm[:, 0, :] - expert_goal_norm, dim=-1,
        )                                                                # (B,)
        emb_l2_list.append(emb_l2.cpu().numpy())
        emb_cos_list.append(emb_cos.cpu().numpy())
        emb_norm_l2_list.append(emb_norm_l2.cpu().numpy())

        # 3) IDM Euler ODE — same NFE — with predicted vs expert goal.
        cond_pred = torch.cat([z_t, pred_goal_raw.unsqueeze(1)], dim=1)
        cond_exp = torch.cat([z_t, expert_goal.unsqueeze(1)], dim=1)
        act_pred_norm = idm_ode_with_obs_emb(
            agent.flow_map, cond_pred, (B, horizon, act_dim), nfe_idm, sample_mode,
        )
        act_exp_norm = idm_ode_with_obs_emb(
            agent.flow_map, cond_exp, (B, horizon, act_dim), nfe_idm, sample_mode,
        )

        # Unnormalize.
        act_pred = act_norm.unnormalize(act_pred_norm.detach().cpu().numpy())
        act_exp = act_norm.unnormalize(act_exp_norm.detach().cpu().numpy())
        act_gt_un = act_norm.unnormalize(act_gt.detach().cpu().numpy())

        # Goal eef pos (world frame).
        goal_eef_norm = batch["goal_obs"]["robot0_eef_pos"][:, 0, :].cpu().numpy()
        goal_eef = eef_norm.unnormalize(goal_eef_norm)

        pred_l2 = np.linalg.norm(act_pred[:, -1, :3] - goal_eef, axis=-1)
        exp_l2 = np.linalg.norm(act_exp[:, -1, :3] - goal_eef, axis=-1)
        pred_l2_list.append(pred_l2)
        expert_l2_list.append(exp_l2)

        pred_action_mse_list.append(((act_pred - act_gt_un) ** 2).mean(axis=-1))
        expert_action_mse_list.append(((act_exp - act_gt_un) ** 2).mean(axis=-1))

        # ----- Direction diagnostics (in world-frame eef position space) -----
        # Current eef = last obs frame's robot0_eef_pos (unnormalized).
        cur_eef_norm = batch["obs"]["robot0_eef_pos"][:, obs_steps - 1, :].cpu().numpy()
        cur_eef = eef_norm.unnormalize(cur_eef_norm)                       # (B, 3)
        delta_expert = goal_eef - cur_eef                                  # (B, 3)
        delta_pred = act_pred[:, -1, :3] - cur_eef
        delta_exp_idm = act_exp[:, -1, :3] - cur_eef
        delta_expert_norm = np.linalg.norm(delta_expert, axis=-1)          # (B,)

        # Cosine similarity (guard against zero-norm with epsilon).
        eps = 1e-9
        def _cos(a, b):
            na = np.linalg.norm(a, axis=-1)
            nb = np.linalg.norm(b, axis=-1)
            return np.einsum("bi,bi->b", a, b) / np.maximum(na * nb, eps)
        pred_cos = _cos(delta_pred, delta_expert)
        exp_cos = _cos(delta_exp_idm, delta_expert)
        # Magnitude ratio (predicted travel / expert travel).
        pred_mag = np.linalg.norm(delta_pred, axis=-1) / np.maximum(delta_expert_norm, eps)
        exp_mag = np.linalg.norm(delta_exp_idm, axis=-1) / np.maximum(delta_expert_norm, eps)

        pred_delta_cos_list.append(pred_cos)
        expert_delta_cos_list.append(exp_cos)
        pred_mag_ratio_list.append(pred_mag)
        expert_mag_ratio_list.append(exp_mag)
        delta_expert_norm_list.append(delta_expert_norm)

    return {
        "emb_l2":             np.concatenate(emb_l2_list, axis=0),
        "emb_cos_sim":        np.concatenate(emb_cos_list, axis=0),
        "emb_norm_l2":        np.concatenate(emb_norm_l2_list, axis=0),
        "pred_goal_pos_l2":   np.concatenate(pred_l2_list, axis=0),
        "expert_goal_pos_l2": np.concatenate(expert_l2_list, axis=0),
        "pred_action_mse":    np.concatenate(pred_action_mse_list, axis=0),
        "expert_action_mse":  np.concatenate(expert_action_mse_list, axis=0),
        "pred_delta_cos":     np.concatenate(pred_delta_cos_list, axis=0),
        "expert_delta_cos":   np.concatenate(expert_delta_cos_list, axis=0),
        "pred_mag_ratio":     np.concatenate(pred_mag_ratio_list, axis=0),
        "expert_mag_ratio":   np.concatenate(expert_mag_ratio_list, axis=0),
        "delta_expert_norm":  np.concatenate(delta_expert_norm_list, axis=0),
    }


def summarize(m):
    # Filter direction metrics on samples where the expert delta is large
    # enough to give meaningful cos_sim (≥ 5mm). Tiny moves have undefined
    # direction; reporting a separate "non-trivial-move" subset keeps the
    # cos_sim metric interpretable.
    nontrivial = m["delta_expert_norm"] >= 0.005   # 5 mm
    n_nt = int(nontrivial.sum())
    pred_cos_nt = m["pred_delta_cos"][nontrivial] if n_nt else m["pred_delta_cos"]
    exp_cos_nt = m["expert_delta_cos"][nontrivial] if n_nt else m["expert_delta_cos"]
    pred_mag_nt = m["pred_mag_ratio"][nontrivial] if n_nt else m["pred_mag_ratio"]
    exp_mag_nt = m["expert_mag_ratio"][nontrivial] if n_nt else m["expert_mag_ratio"]
    return {
        "n":                       len(m["emb_l2"]),
        "n_nontrivial":            n_nt,
        "emb_l2_mean":             float(m["emb_l2"].mean()),
        "emb_l2_med":              float(np.median(m["emb_l2"])),
        "emb_cos_mean":            float(m["emb_cos_sim"].mean()),
        "emb_norm_l2_mean":        float(m["emb_norm_l2"].mean()),
        "pred_goal_l2_mean":       float(m["pred_goal_pos_l2"].mean()),
        "pred_goal_l2_med":        float(np.median(m["pred_goal_pos_l2"])),
        "pred_goal_l2_p95":        float(np.percentile(m["pred_goal_pos_l2"], 95)),
        "expert_goal_l2_mean":     float(m["expert_goal_pos_l2"].mean()),
        "expert_goal_l2_med":      float(np.median(m["expert_goal_pos_l2"])),
        "pred_act_mse_last_mean":  float(m["pred_action_mse"][:, -1].mean()),
        "expert_act_mse_last_mean": float(m["expert_action_mse"][:, -1].mean()),
        # Direction-only diagnostics on the non-trivial subset.
        "delta_expert_med_mm":     float(np.median(m["delta_expert_norm"]) * 1000.0),
        "pred_dir_cos_mean":       float(pred_cos_nt.mean()) if n_nt else float("nan"),
        "pred_dir_cos_med":        float(np.median(pred_cos_nt)) if n_nt else float("nan"),
        "pred_dir_pos_frac":       float((pred_cos_nt > 0).mean()) if n_nt else float("nan"),
        "expert_dir_cos_mean":     float(exp_cos_nt.mean()) if n_nt else float("nan"),
        "pred_mag_ratio_med":      float(np.median(pred_mag_nt)) if n_nt else float("nan"),
        "expert_mag_ratio_med":    float(np.median(exp_mag_nt)) if n_nt else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--gp-ckpt", type=str)
    g.add_argument("--gp-ckpt-dir", type=str)
    parser.add_argument("--idm-ckpt", type=str, required=True,
                        help="Frozen IDM checkpoint that the goal predictor "
                             "was trained on (typically model_step_130000.pt).")
    parser.add_argument("--goal-stats", type=str, required=True,
                        help="goal_stats_<step>.pt produced when training the "
                             "goal predictor.")
    parser.add_argument("--heldout-path", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, default=None,
                        help="Defaults to <idm-ckpt-dir>/normalizer.pkl.")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated GP step numbers to evaluate.")
    parser.add_argument("--sweep-stride", type=int, default=None)
    parser.add_argument("--nfe-idm", type=int, default=9)
    parser.add_argument("--nfe-goal", type=int, default=5)
    parser.add_argument("--sample-mode", type=str, default="zero",
                        choices=["zero", "stochastic"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-dit-d-model", type=int, default=512,
                        help="GP DiT d_model (must match the trained ckpt). "
                             "Default 512 matches the 2026_05_03 GP run.")
    parser.add_argument("--out-npz", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    limit_threads(1)
    torch.set_float32_matmul_precision("high")

    steps_list = [int(s) for s in args.steps.split(",")] if args.steps else None
    ckpt_list = discover_gp_checkpoints(
        args.gp_ckpt_dir, args.gp_ckpt, steps=steps_list,
        sweep_stride=args.sweep_stride,
    )

    norm_path = args.normalizer_path or os.path.join(
        os.path.dirname(args.idm_ckpt), "normalizer.pkl",
    )
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalizer not found: {norm_path}")
    with open(norm_path, "rb") as f:
        normalizer = pickle.load(f)
    loguru.logger.info(f"Loaded normalizer from {norm_path}")

    cfg = build_cfg(
        args.idm_ckpt, args.goal_stats, args.batch_size,
        goal_dit_d_model=args.goal_dit_d_model,
    )
    cfg.task.obs_dim = cfg.network.emb_dim
    obs_steps = cfg.task.obs_steps
    horizon = cfg.task.horizon
    act_steps = cfg.task.act_steps

    if not os.path.exists(args.heldout_path):
        raise FileNotFoundError(args.heldout_path)
    dataset = RobomimicImageIDMDataset(
        dataset_dir=os.path.expanduser(args.heldout_path),
        shape_meta=cfg.task.shape_meta,
        n_obs_steps=obs_steps,
        horizon=horizon,
        pad_before=obs_steps - 1,
        pad_after=act_steps - 1,
        abs_action=cfg.task.abs_action,
        val_dataset_percentage=0.0,
        mode="train",
        normalizer=normalizer,
    )
    loguru.logger.info(f"Held-out IDM dataset: {len(dataset)} samples")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=False,
    )

    # Build the GP agent ONCE; sweep only swaps goal_dit weights via agent.load.
    loguru.logger.info("Instantiating GoalPredictorDiTAgent (loads frozen IDM)...")
    agent = GoalPredictorDiTAgent(cfg)
    agent.eval()

    summaries = []
    raw_per_ckpt: dict[str, dict] = {}
    for label, path in ckpt_list:
        loguru.logger.info(f"Loading GP {label}: {path}")
        agent.load(path)
        agent.eval()
        t0 = time.time()
        metrics = evaluate_gp_checkpoint(
            agent, loader, normalizer, obs_steps, horizon,
            nfe_idm=args.nfe_idm, nfe_goal=args.nfe_goal,
            sample_mode=args.sample_mode, seed=args.seed,
            max_batches=args.max_batches,
        )
        dt = time.time() - t0
        s = summarize(metrics)
        s["label"] = label
        s["wall_s"] = dt
        summaries.append(s)
        raw_per_ckpt[label] = metrics
        loguru.logger.info(
            f"  n={s['n']} (nt={s['n_nontrivial']})  "
            f"emb_cos={s['emb_cos_mean']:.4f}  "
            f"pred_goal_l2={s['pred_goal_l2_mean']:.4f}  "
            f"exp_goal_l2={s['expert_goal_l2_mean']:.4f}  | "
            f"dir_cos(pred)={s['pred_dir_cos_mean']:.4f} "
            f"med={s['pred_dir_cos_med']:.4f} "
            f"pos%={s['pred_dir_pos_frac']:.3f}  "
            f"dir_cos(exp)={s['expert_dir_cos_mean']:.4f}  "
            f"mag_ratio(pred,exp)={s['pred_mag_ratio_med']:.3f}/{s['expert_mag_ratio_med']:.3f}  "
            f"({dt:.1f}s)"
        )

    # Summary table — split into two prints to keep each readable.
    print()
    print("=" * 110)
    print(f"Goal-predictor offline eval — {len(dataset)} samples, "
          f"NFE_goal={args.nfe_goal}, NFE_idm={args.nfe_idm}, sample_mode={args.sample_mode}")
    print("Embedding fidelity | Action reachability (meters)")
    print("=" * 110)
    print(
        f"{'checkpoint':<14s}  {'n':>5s}  "
        f"{'emb_l2':>8s}  {'emb_cos':>8s}  {'emb_nL2':>8s}  | "
        f"{'pred_l2':>8s}  {'pred_p95':>8s}  {'exp_l2':>8s}  {'exp_p95':>8s}"
    )
    for s in summaries:
        m = raw_per_ckpt[s["label"]]
        pred_p95 = float(np.percentile(m["pred_goal_pos_l2"], 95))
        exp_p95 = float(np.percentile(m["expert_goal_pos_l2"], 95))
        print(
            f"{s['label']:<14s}  {s['n']:>5d}  "
            f"{s['emb_l2_mean']:>8.4f}  {s['emb_cos_mean']:>8.4f}  {s['emb_norm_l2_mean']:>8.4f}  | "
            f"{s['pred_goal_l2_mean']:>8.4f}  {pred_p95:>8.4f}  "
            f"{s['expert_goal_l2_mean']:>8.4f}  {exp_p95:>8.4f}"
        )

    # Direction diagnostics — predicted-goal IDM action delta vs expert delta,
    # restricted to non-trivial moves (≥ 5mm). Also reports magnitude ratio.
    print()
    print("=" * 110)
    print("Direction diagnostics on non-trivial moves (delta_expert ≥ 5 mm in eef space)")
    print("dir_cos = cos_sim(action_pred[-1,:3] − cur_eef, expert_goal − cur_eef);  "
          "1 = perfect direction, 0 = orthogonal, <0 = wrong way.")
    print("mag_ratio = ‖action_pred[-1,:3] − cur_eef‖ / ‖expert_goal − cur_eef‖.")
    print("=" * 110)
    print(
        f"{'checkpoint':<14s}  {'n_nt':>5s}  {'Δ_med_mm':>8s}  | "
        f"{'pred_cos_mean':>13s}  {'pred_cos_med':>12s}  {'pred_pos%':>9s}  "
        f"{'exp_cos_mean':>12s}  | "
        f"{'pred_magR_med':>13s}  {'exp_magR_med':>12s}"
    )
    for s in summaries:
        print(
            f"{s['label']:<14s}  {s['n_nontrivial']:>5d}  {s['delta_expert_med_mm']:>8.2f}  | "
            f"{s['pred_dir_cos_mean']:>13.4f}  {s['pred_dir_cos_med']:>12.4f}  "
            f"{s['pred_dir_pos_frac']:>9.3f}  "
            f"{s['expert_dir_cos_mean']:>12.4f}  | "
            f"{s['pred_mag_ratio_med']:>13.3f}  {s['expert_mag_ratio_med']:>12.3f}"
        )

    # Best by predicted-goal action L2.
    best = min(summaries, key=lambda x: x["pred_goal_l2_mean"])
    print()
    print(f"Best GP by pred_goal_l2_mean: {best['label']}  ({best['pred_goal_l2_mean']:.4f} m)")

    if args.out_npz:
        out = {}
        for label, m in raw_per_ckpt.items():
            for k, v in m.items():
                out[f"{label}__{k}"] = v
        np.savez(args.out_npz, **out)
        print(f"\nRaw arrays saved to {args.out_npz}")


if __name__ == "__main__":
    main()
