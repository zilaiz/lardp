"""Last-action vs goal-pose accuracy for a vanilla DP (TrainingAgent) ckpt.

Companion to ``diagnose_joint_ddt_last_action_vs_goal.py`` — same
measurement, same heldout dataset, same physical units, but the policy is
``mip.agent.TrainingAgent`` (action-only flow matching, no state stream).

Used to baseline the joint_ddt's goal-pose targeting accuracy against a
plain DP trained on the same task.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

LARDP_PATH = Path(__file__).resolve().parents[1]
if str(LARDP_PATH) not in sys.path:
    sys.path.append(str(LARDP_PATH))

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_idm_dataset
from scripts.diagnose_joint_ddt_last_action_vs_goal import (
    _geodesic_deg,
    _last_action_vs_goal,
)


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

    print("\n[1/3] Building DP agent + loading checkpoint...")
    agent = TrainingAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()

    # ---- dataset (heldout, with re-derived training normalizer) ----
    norm_override = None
    if args.dataset_path is not None:
        print("\n[2a/3] Re-deriving normalizer from training dataset...")
        # The DP cfg uses `dataset_path` (string), not `dataset_paths` (list).
        # make_idm_dataset handles both via its fall-through logic.
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
        print(f"\n[2/3] Loading heldout dataset: {args.dataset_path}")
    else:
        print("\n[2/3] Loading training dataset...")

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
    print(f"   dataset size={len(dataset)}, sampling {n} indices")

    sample_keys = list(cfg.task.shape_meta["obs"].keys())
    obs_batch = {k: [] for k in sample_keys}
    act_batch_normed = []
    goal_pos_normed = []
    goal_quat_normed = []
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
        act_batch_normed.append(s["action"][: cfg.task.horizon])
        goal_pos_normed.append(s["goal_obs"]["robot0_eef_pos"][0].numpy())
        goal_quat_normed.append(s["goal_obs"]["robot0_eef_quat"][0].numpy())

    obs_torch = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in obs_batch.items()
    }
    act_true = torch.from_numpy(
        np.stack([t.numpy() for t in act_batch_normed])
    ).to(device)

    goal_pos_normed = np.stack(goal_pos_normed)
    goal_quat_normed = np.stack(goal_quat_normed)
    goal_pos_m = obs_normalizer["robot0_eef_pos"].unnormalize(goal_pos_normed)
    goal_quat = obs_normalizer["robot0_eef_quat"].unnormalize(goal_quat_normed)
    goal_R = Rotation.from_quat(goal_quat).as_matrix()

    print("\n[3/3] Sampling DP and comparing last action to goal_obs pose...")

    # Ground-truth sanity (recompute on this seed's batch so DP and DDT
    # reports are directly comparable).
    print("\n== ground-truth sanity ==")
    gt = _last_action_vs_goal(act_true, goal_pos_m, goal_R, action_normalizer)
    print(
        f"  act_true[:, H-1] pos Δ mean={gt['pos_l2_mean_mm']:.4f} mm  "
        f"max={gt['pos_l2_max_mm']:.4f} mm  "
        f"rot Δ mean={gt['rot_geo_mean_deg']:.4f}°  "
        f"max={gt['rot_geo_max_deg']:.4f}°"
    )

    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        act_0 = torch.randn(
            (n, cfg.task.horizon, cfg.task.act_dim), generator=g,
        ).to(device)
        a_pred = agent.sample(
            act_0=act_0, obs=obs_torch,
            num_steps=args.num_steps, use_ema=True,
        )

    st = _last_action_vs_goal(a_pred, goal_pos_m, goal_R, action_normalizer)
    print("\n== model: predicted last action vs goal_obs eef pose ==")
    print(f"  pos mean = {st['pos_l2_mean_mm']:.2f} mm   max = {st['pos_l2_max_mm']:.2f} mm")
    print(f"  rot mean = {st['rot_geo_mean_deg']:.3f}°  max = {st['rot_geo_max_deg']:.3f}°")

    # Also report whole-chunk mean for parity with the DDT report.
    A = action_normalizer.unnormalize(
        a_pred.detach().cpu().numpy().astype(np.float32).reshape(-1, 10)
    ).reshape(a_pred.shape)
    T = action_normalizer.unnormalize(
        act_true.detach().cpu().numpy().astype(np.float32).reshape(-1, 10)
    ).reshape(act_true.shape)
    pos_chunk = np.linalg.norm(A[..., :3] - T[..., :3], axis=-1)
    n_, H_, _ = A.shape
    from scripts.diagnose_joint_ddt_schedule_sweep import _rot6d_to_matrix
    R_a = _rot6d_to_matrix(A[..., 3:9].reshape(-1, 6)).reshape(n_, H_, 3, 3)
    R_t = _rot6d_to_matrix(T[..., 3:9].reshape(-1, 6)).reshape(n_, H_, 3, 3)
    rot_chunk = _geodesic_deg(R_a, R_t)
    grip_l1 = np.abs(A[..., 9] - T[..., 9])
    bin_a = (A[..., 9] > 0.5).astype(np.int32)
    bin_t = (T[..., 9] > 0.5).astype(np.int32)
    grip_flip = float((bin_a != bin_t).mean()) * 100.0
    print("\n== chunk-mean vs ground-truth action (parity with DDT report) ==")
    print(f"  pos: mean={pos_chunk.mean()*1000:.2f} mm  max={pos_chunk.max()*1000:.2f} mm")
    print(f"  rot: mean={rot_chunk.mean():.3f}°  max={rot_chunk.max():.3f}°")
    print(f"  gripper: L1 mean={grip_l1.mean():.4f}  binary-flip {grip_flip:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose(args)
