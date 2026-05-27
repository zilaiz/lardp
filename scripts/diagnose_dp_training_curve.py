"""Training-curve probe for a DP (TrainingAgent) run.

Companion to diagnose_joint_pt_training_curve.py. For a fixed dataset
batch and shared act_0, loads each --step ckpt and reports action
prediction in physical units:
  chunk pos / last-step pos (mm), chunk rot (deg), gripper flip %.

DP has no state stream, so no oracle / cosine readouts.
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
from scripts.diagnose_joint_ddt_schedule_sweep import (
    _abs_space_stats,
    _geodesic_deg,
    _rot6d_to_matrix,
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

    # ---- dataset ----
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
    else:
        norm_override = None

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
    act_batch_normed = []
    goal_pos_normed = []
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
        act_batch_normed.append(s["action"][: cfg.task.horizon])
        goal_pos_normed.append(s["goal_obs"]["robot0_eef_pos"][0].numpy())

    obs_torch = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in obs_batch.items()
    }
    act_true = torch.from_numpy(
        np.stack([t.numpy() for t in act_batch_normed])
    ).to(device)
    goal_pos_m = obs_normalizer["robot0_eef_pos"].unnormalize(
        np.stack(goal_pos_normed),
    )

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    act_0 = torch.randn(
        (n, cfg.task.horizon, cfg.task.act_dim), generator=g,
    ).to(device)

    print(f"\n[steps] sweeping {len(args.steps)} ckpts, "
          f"num_steps={args.num_steps}, num_samples={n}")
    print(
        f"\n  {'step':>7s}  {'chunk pos':>14s}{'last pos':>14s}"
        f"{'chunk rot':>11s}{'flip %':>9s}"
    )
    for step in args.steps:
        ckpt_path = Path(args.run_dir) / "models" / f"model_step_{step}.pt"
        if not ckpt_path.exists():
            print(f"  [skip] step {step}: ckpt not found")
            continue
        agent = TrainingAgent(cfg)
        agent.load(str(ckpt_path), load_optimizer=False)
        agent.eval()
        with torch.no_grad():
            a_pred = agent.sample(
                act_0=act_0, obs=obs_torch,
                num_steps=args.num_steps, use_ema=True,
            )
        chunk_st = _abs_space_stats(a_pred, act_true, action_normalizer)
        A = action_normalizer.unnormalize(
            a_pred.detach().cpu().numpy().astype(np.float32).reshape(-1, 10)
        ).reshape(a_pred.shape)
        last_pred_pos = A[:, -1, :3]
        last_pos_l2 = np.linalg.norm(last_pred_pos - goal_pos_m, axis=-1)
        last_pos_mean_mm = float(last_pos_l2.mean()) * 1000.0
        print(
            f"  {step:>7d}"
            f"{chunk_st['pos_l2_mean_mm']:>12.2f}mm"
            f"{last_pos_mean_mm:>12.2f}mm"
            f"{chunk_st['rot_geo_mean_deg']:>9.3f}°"
            f"{chunk_st['gripper_disagree_pct']:>8.2f}%"
        )
        del agent
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--steps", type=int, nargs="+",
                        default=[10000, 30000, 60000, 90000, 150000, 200000,
                                 290000, 300000])
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose(args)
