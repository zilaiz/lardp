"""Compare DP vs joint_pt on tool_hang (robomimic, 7-dim DELTA actions).

Action format here is [pos_delta(3), axis_angle(3), gripper(1)] — different
from franka's [pos(3), rot6d(6), gripper(1)] absolute actions. Per-step
errors are reported as:
  pos delta L2 in mm,
  rotation L2 (axis_angle native — small-angle ≈ radians) in degrees via
    geodesic between rotation matrices,
  gripper L1 (continuous), and binary-flip at 0.

For joint_pt, additionally runs (diagonal vs state_first vs noise vs oracle)
to expose the diag→oracle gap.
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
from mip.dataset_utils import RotationTransformer
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset
from scripts.diagnose_joint_ddt_schedule_sweep import (
    _pick_agent_cls,
    _sample_with_pinning,
    _sample_with_schedule,
)


def _axisangle_to_matrix(aa: np.ndarray) -> np.ndarray:
    rt = RotationTransformer(from_rep="axis_angle", to_rep="matrix")
    return rt.forward(aa)


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    M = np.matmul(R1, np.swapaxes(R2, -1, -2))
    tr = np.einsum("...ii->...", M)
    cos_th = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def _delta_stats(a_norm: torch.Tensor, b_norm: torch.Tensor,
                 action_normalizer) -> dict:
    """Decompose two 7-dim delta-action chunks and return per-step physical stats."""
    A = action_normalizer.unnormalize(
        a_norm.detach().cpu().numpy().astype(np.float32).reshape(-1, 7)
    ).reshape(a_norm.shape)
    B = action_normalizer.unnormalize(
        b_norm.detach().cpu().numpy().astype(np.float32).reshape(-1, 7)
    ).reshape(a_norm.shape)
    # pos delta L2 in meters
    pos_l2 = np.linalg.norm(A[..., :3] - B[..., :3], axis=-1)
    # rotation as geodesic angle between rotation matrices
    n, H, _ = A.shape
    R_a = _axisangle_to_matrix(A[..., 3:6].reshape(-1, 3)).reshape(n, H, 3, 3)
    R_b = _axisangle_to_matrix(B[..., 3:6].reshape(-1, 3)).reshape(n, H, 3, 3)
    rot_deg = _geodesic_deg(R_a, R_b)
    # gripper
    grip_l1 = np.abs(A[..., 6] - B[..., 6])
    bin_a = (A[..., 6] > 0.0).astype(np.int32)
    bin_b = (B[..., 6] > 0.0).astype(np.int32)
    return {
        # Robomimic actions are in controller command units (roughly [-1, 1]),
        # NOT meters. Report raw L2 — caller should not interpret as mm.
        "pos_l2_mean_cmd": float(pos_l2.mean()),
        "pos_l2_max_cmd":  float(pos_l2.max()),
        "rot_geo_mean_deg": float(rot_deg.mean()),
        "rot_geo_max_deg":  float(rot_deg.max()),
        "gripper_l1_mean": float(grip_l1.mean()),
        "gripper_disagree_pct": float((bin_a != bin_b).mean()) * 100.0,
    }


def _build_dataset(cfg, args, mode):
    """Build a RobomimicImageIDMDataset for tool_hang in val mode.

    Use the IDM variant (not plain RobomimicImageDataset) so goal_obs is
    returned — joint_pt's oracle_pinned needs it. DP ignores goal_obs.
    """
    ds_path = OmegaConf.select(cfg, "task.dataset_path", default=None)
    if ds_path is None:
        paths = OmegaConf.select(cfg, "task.dataset_paths", default=None)
        if paths:
            ds_path = paths[0]
    return RobomimicImageIDMDataset(
        dataset_dir=ds_path,
        shape_meta=cfg.task.shape_meta,
        n_obs_steps=cfg.task.obs_steps,
        horizon=cfg.task.horizon,
        pad_before=cfg.task.obs_steps - 1,
        pad_after=cfg.task.act_steps - 1,
        abs_action=cfg.task.abs_action,
        val_dataset_percentage=cfg.task.val_dataset_percentage,
        mode=mode,
    )


def diagnose(args):
    # Both ckpts must share dataset semantics; use joint_pt's task config for
    # dataset construction (its val split is identical to DP's since they use
    # the same task.val_dataset_percentage and HDF5 path).
    cfg_jpt = OmegaConf.load(args.joint_pt_config)
    cfg_dp = OmegaConf.load(args.dp_config)
    for c in (cfg_jpt, cfg_dp):
        OmegaConf.update(c, "optimization.use_compile", False, merge=False)
        OmegaConf.update(c, "optimization.use_cudagraphs", False, merge=False)
        if not torch.cuda.is_available():
            OmegaConf.update(c, "optimization.device", "cpu", merge=False)
        if c.task.obs_type == "image":
            OmegaConf.update(c, "task.obs_dim", c.network.emb_dim, merge=False)

    def _resolve_path(cfg):
        p = OmegaConf.select(cfg, "task.dataset_path", default=None)
        if p:
            return p
        paths = OmegaConf.select(cfg, "task.dataset_paths", default=None)
        return paths[0] if paths else None
    jpt_path = _resolve_path(cfg_jpt)
    dp_path = _resolve_path(cfg_dp)
    assert jpt_path == dp_path, (
        f"Dataset mismatch:\n  joint_pt: {jpt_path}\n  dp: {dp_path}"
    )
    print(f"\n=== shared dataset: {jpt_path} ===")
    print(f"=== val_pct={cfg_jpt.task.val_dataset_percentage}  "
          f"abs_action={cfg_jpt.task.abs_action}  act_dim={cfg_jpt.task.act_dim} ===")

    device = torch.device(cfg_jpt.optimization.device)

    print("\n[1/3] Loading val dataset (heldout split)...")
    dataset = _build_dataset(cfg_jpt, args, mode="val")
    action_normalizer = dataset.normalizer["action"]
    n = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    print(f"   dataset size={len(dataset)}, sampling {n} indices")

    # Build obs + ground-truth action batch
    sample_keys = list(cfg_jpt.task.shape_meta["obs"].keys())
    obs_batch = {k: [] for k in sample_keys}
    goal_batch = {k: [] for k in sample_keys}
    act_batch_normed = []
    has_goal = False
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
            if "goal_obs" in s:
                goal_batch[k].append(s["goal_obs"][k])
        act_batch_normed.append(s["action"][: cfg_jpt.task.horizon])
        has_goal = "goal_obs" in s
    obs_torch = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in obs_batch.items()
    }
    act_true = torch.from_numpy(
        np.stack([t.numpy() for t in act_batch_normed])
    ).to(device)

    if has_goal:
        goal_torch = {
            k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
            for k, v in goal_batch.items()
        }
    else:
        # joint_pt needs goal_obs for oracle_pinned. RobomimicImageDataset
        # doesn't return one — fall back to the obs's last frame (degenerate;
        # oracle_pinned will be biased). Better: use IDM dataset variant
        # that returns goal_obs. Tool_hang_ph_image_gp doesn't have that.
        goal_torch = None
        print("   NOTE: dataset has no goal_obs — oracle_pinned will be skipped")

    # Shared randomness across DP and joint_pt
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    act_0 = torch.randn(
        (n, cfg_jpt.task.horizon, cfg_jpt.task.act_dim), generator=g,
    ).to(device)

    # ----- DP -----
    print(f"\n[2/3] DP sampling: {args.dp_ckpt}")
    dp = TrainingAgent(cfg_dp)
    dp.load(args.dp_ckpt, load_optimizer=False)
    dp.eval()
    with torch.no_grad():
        dp_pred = dp.sample(
            act_0=act_0, obs=obs_torch,
            num_steps=args.num_steps, use_ema=True,
        )
    dp_stats = _delta_stats(dp_pred, act_true, action_normalizer)
    print(f"   DP chunk pos Δ mean = {dp_stats['pos_l2_mean_cmd']:.4f}   "
          f"max = {dp_stats['pos_l2_max_cmd']:.4f}   (raw command units, [-1,1])")
    print(f"   DP chunk rot Δ mean = {dp_stats['rot_geo_mean_deg']:.3f}°   "
          f"max = {dp_stats['rot_geo_max_deg']:.3f}°")
    print(f"   DP gripper L1 = {dp_stats['gripper_l1_mean']:.4f}   "
          f"binary-flip = {dp_stats['gripper_disagree_pct']:.2f}%")
    del dp
    torch.cuda.empty_cache()

    # ----- joint_pt -----
    print(f"\n[3/3] joint_pt sampling: {args.joint_pt_ckpt}")
    AgentCls = _pick_agent_cls(cfg_jpt)
    jpt = AgentCls(cfg_jpt)
    jpt.load(args.joint_pt_ckpt, load_optimizer=False)
    jpt.eval()

    obs_dim = cfg_jpt.network.encoder_out_dim or cfg_jpt.network.emb_dim
    x_state_init = torch.randn(
        (n, 1, obs_dim), generator=g,
    ).to(device)

    modes = [
        ("diagonal",      "schedule", "diagonal"),
        ("state_first",   "schedule", "state_first"),
        ("noise_pinned",  "pin",      "noise_pinned"),
    ]
    if goal_torch is not None:
        modes.append(("oracle_pinned", "pin", "oracle_pinned"))

    print(f"\n   joint_pt modes: {[m[0] for m in modes]}")
    print(f"   {'mode':<14s}{'pos mean / max (cmd units)':>30s}"
          f"{'rot mean / max (deg)':>26s}{'grip L1 / flip%':>22s}")
    for name, kind, sched_or_pin in modes:
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
        st = _delta_stats(a, act_true, action_normalizer)
        print(
            f"   {name:<14s}"
            f"{st['pos_l2_mean_cmd']:>14.4f} / {st['pos_l2_max_cmd']:>9.4f}"
            f"{st['rot_geo_mean_deg']:>16.3f} / {st['rot_geo_max_deg']:>6.3f}"
            f"{st['gripper_l1_mean']:>12.4f} / {st['gripper_disagree_pct']:>5.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt", type=str, required=True)
    parser.add_argument("--dp_config", type=str, required=True)
    parser.add_argument("--joint_pt_ckpt", type=str, required=True)
    parser.add_argument("--joint_pt_config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose(args)
