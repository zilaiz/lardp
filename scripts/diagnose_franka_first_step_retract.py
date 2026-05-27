"""Detect 'small/quick retracting first step' in predicted action chunks.

For an absolute-action franka policy, the first action's position should be
close to (or slightly forward of) the current EEF position. A 'retracting'
prediction means action[0]'s position is *behind* the current EEF along the
chunk's overall motion direction — a small backward jerk before the chunk
resumes forward motion.

Metrics computed per predicted chunk:
  d0     = action[0, :3] − current_eef_pos                  (first-step disp)
  d_rest = mean(action[t, :3] − action[t-1, :3], t=1..H-1)  (avg subsequent disp)
  cos    = cosine(d0, d_rest)                                (>0 forward, <0 retract)

Reports, per policy (DP / joint_pt-diag / joint_pt-oracle):
  - mean cos(d0, d_rest) across samples
  - fraction of samples with cos < 0   (= 'retracting fraction')
  - fraction of samples with cos < -0.5 (= 'strong retract')
  - mean |d0| in mm
  - mean |d_rest| in mm
  - ratio |d0| / |d_rest|              (small if first step is shorter)
  - per-sample comparison vs ground-truth first-step direction
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

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_idm_dataset
from scripts.diagnose_joint_ddt_schedule_sweep import (
    _pick_agent_cls,
    _sample_with_pinning,
    _sample_with_schedule,
)


def _first_step_stats(
    act_norm: torch.Tensor,
    current_pos_m: np.ndarray,
    action_normalizer,
    label: str = "",
):
    """Compute first-step-direction stats for a predicted action chunk.

    Args:
        act_norm:        (n, H, 10) predicted action chunk in normalized space.
        current_pos_m:   (n, 3) current EEF position in meters (unnormalized).
        action_normalizer: MinMaxNormalizer for actions.
    """
    A = act_norm.detach().cpu().numpy().astype(np.float32)
    flat_un = action_normalizer.unnormalize(A.reshape(-1, 10)).reshape(A.shape)
    pos = flat_un[..., :3]                                # (n, H, 3) meters
    n, H, _ = pos.shape

    # First-step displacement: action[0].pos − current EEF pos
    d0 = pos[:, 0, :] - current_pos_m                      # (n, 3)

    # Subsequent per-step displacements
    d_per_step = pos[:, 1:, :] - pos[:, :-1, :]            # (n, H-1, 3)
    d_rest_mean = d_per_step.mean(axis=1)                  # (n, 3) mean direction
    d_rest_mag_mean = float(np.linalg.norm(d_per_step, axis=-1).mean()) * 1000  # mm

    d0_mag = np.linalg.norm(d0, axis=-1)                   # (n,)
    rest_mag = np.linalg.norm(d_rest_mean, axis=-1) + 1e-12  # (n,)

    # Cosine between first-step direction and chunk-rest direction
    dot = (d0 * d_rest_mean).sum(axis=-1)
    cos = dot / (d0_mag * rest_mag + 1e-12)                # (n,)

    retract_frac = float((cos < 0).mean()) * 100
    strong_retract_frac = float((cos < -0.5).mean()) * 100
    cos_mean = float(cos.mean())

    print(f"\n  [{label}]")
    print(f"    |d0|     mean = {d0_mag.mean()*1000:.2f} mm  "
          f"max = {d0_mag.max()*1000:.2f} mm")
    print(f"    |d_rest| mean (per-step magnitude) = {d_rest_mag_mean:.2f} mm")
    print(f"    |d0| / |d_rest| = {d0_mag.mean() / (d_rest_mag_mean / 1000 + 1e-12):.3f}")
    print(f"    cos(d0, d_rest_mean): mean = {cos_mean:+.3f}")
    print(f"    fraction of samples cos < 0    (retracting)        = {retract_frac:.2f}%")
    print(f"    fraction of samples cos < -0.5 (strong retract)    = {strong_retract_frac:.2f}%")

    return {
        "d0_mag_mm": d0_mag * 1000,
        "d_rest_mag_mm": d_rest_mag_mean,
        "cos": cos,
        "retract_frac": retract_frac,
        "strong_retract_frac": strong_retract_frac,
    }


def _vs_ground_truth(
    a_pred: torch.Tensor, a_true: torch.Tensor, current_pos_m: np.ndarray,
    action_normalizer, label: str = "",
):
    """Compare predicted first-step direction to ground-truth first-step direction."""
    P = a_pred.detach().cpu().numpy().astype(np.float32)
    T = a_true.detach().cpu().numpy().astype(np.float32)
    P_un = action_normalizer.unnormalize(P.reshape(-1, 10)).reshape(P.shape)[..., 0, :3]
    T_un = action_normalizer.unnormalize(T.reshape(-1, 10)).reshape(T.shape)[..., 0, :3]

    pred_d0 = P_un - current_pos_m                  # (n, 3)
    true_d0 = T_un - current_pos_m                  # (n, 3)

    pred_mag = np.linalg.norm(pred_d0, axis=-1)
    true_mag = np.linalg.norm(true_d0, axis=-1)

    dot = (pred_d0 * true_d0).sum(axis=-1)
    cos = dot / (pred_mag * true_mag + 1e-12)
    print(f"  [{label}] pred-d0 vs true-d0 direction:  "
          f"mean cos = {cos.mean():+.3f},  "
          f"frac < 0 (opposite direction) = {(cos < 0).mean()*100:.2f}%")


def diagnose(args):
    cfg = OmegaConf.load(args.config_path)
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)
    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)

    device = torch.device(cfg.optimization.device)

    norm_override = None
    if args.dataset_path is not None:
        print("[setup] Re-deriving normalizer from training dataset...")
        train_dataset = make_idm_dataset(cfg.task, mode="train")
        base_train = (
            train_dataset.datasets[0]
            if isinstance(train_dataset, torch.utils.data.ConcatDataset)
            else train_dataset
        )
        norm_override = base_train.normalizer
        del train_dataset, base_train
        OmegaConf.update(cfg, "task.dataset_paths", [args.dataset_path], merge=False)
        OmegaConf.update(cfg, "task.dataset_path", None, merge=False)
        OmegaConf.update(cfg, "task.val_dataset_percentage", 0.0, merge=False)
        print(f"[setup] Loading heldout: {args.dataset_path}")

    dataset = make_idm_dataset(cfg.task, mode="train", normalizer=norm_override)
    base_ds = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )
    action_normalizer = base_ds.normalizer["action"]
    obs_normalizer = base_ds.normalizer["obs"]

    n = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    print(f"[setup] dataset size={len(dataset)}, sampling {n} indices")

    sample_keys = list(cfg.task.shape_meta["obs"].keys())
    obs_batch = {k: [] for k in sample_keys}
    goal_batch = {k: [] for k in sample_keys}
    act_batch_normed = []
    current_pos_normed = []
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
            goal_batch[k].append(s["goal_obs"][k])
        act_batch_normed.append(s["action"][: cfg.task.horizon])
        # "current" EEF pos = last frame of obs window
        current_pos_normed.append(
            s["obs"]["robot0_eef_pos"][-1].numpy()
        )

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
    current_pos_m = obs_normalizer["robot0_eef_pos"].unnormalize(
        np.stack(current_pos_normed)
    )

    obs_dim = cfg.network.encoder_out_dim or cfg.network.emb_dim
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    act_0 = torch.randn(
        (n, cfg.task.horizon, cfg.task.act_dim), generator=g,
    ).to(device)
    x_state_init = torch.randn(
        (n, 1, obs_dim), generator=g,
    ).to(device)

    # ---- Baseline: ground-truth action chunk's first-step behavior ----
    print("\n[1] ground-truth action chunk first-step behavior:")
    _first_step_stats(act_true, current_pos_m, action_normalizer,
                      label="ground truth")

    # ---- The agent ----
    print(f"\n[2] joint_pt: {args.ckpt_path}")
    AgentCls = _pick_agent_cls(cfg)
    jpt = AgentCls(cfg)
    jpt.load(args.ckpt_path, load_optimizer=False)
    jpt.eval()

    for mode_name, kind, sched_or_pin in (
        ("diagonal",      "schedule", "diagonal"),
        ("oracle_pinned", "pin",      "oracle_pinned"),
    ):
        if kind == "schedule":
            a, _ = _sample_with_schedule(
                jpt,
                obs=obs_torch, act_0=act_0, x_state_init=x_state_init,
                num_steps=args.num_steps,
                schedule=sched_or_pin, pyramid_offset=0.0,
            )
        else:
            a, _ = _sample_with_pinning(
                jpt,
                obs=obs_torch, goal_obs=goal_torch,
                act_0=act_0, x_state_init=x_state_init,
                num_steps=args.num_steps,
                pin_mode=sched_or_pin,
            )
        _first_step_stats(a, current_pos_m, action_normalizer,
                          label=f"joint_pt {mode_name}")
        _vs_ground_truth(a, act_true, current_pos_m, action_normalizer,
                         label=f"joint_pt {mode_name}")
    del jpt
    torch.cuda.empty_cache()

    # ---- DP baseline for comparison ----
    if args.dp_ckpt and args.dp_config:
        print(f"\n[3] DP baseline: {args.dp_ckpt}")
        dp_cfg = OmegaConf.load(args.dp_config)
        OmegaConf.update(dp_cfg, "optimization.use_compile", False, merge=False)
        OmegaConf.update(dp_cfg, "optimization.use_cudagraphs", False, merge=False)
        if dp_cfg.task.obs_type == "image":
            OmegaConf.update(dp_cfg, "task.obs_dim", dp_cfg.network.emb_dim, merge=False)
        dp = TrainingAgent(dp_cfg)
        dp.load(args.dp_ckpt, load_optimizer=False)
        dp.eval()
        with torch.no_grad():
            dp_pred = dp.sample(
                act_0=act_0, obs=obs_torch,
                num_steps=args.num_steps, use_ema=True,
            )
        _first_step_stats(dp_pred, current_pos_m, action_normalizer,
                          label="DP")
        _vs_ground_truth(dp_pred, act_true, current_pos_m, action_normalizer,
                         label="DP")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="joint_pt checkpoint to inspect")
    parser.add_argument("--config_path", type=str, required=True,
                        help="joint_pt hydra config")
    parser.add_argument("--dp_ckpt", type=str, default=None,
                        help="optional DP baseline ckpt for comparison")
    parser.add_argument("--dp_config", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="heldout HDF5 (uses training normalizer)")
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose(args)
