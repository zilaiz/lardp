"""Last-action vs goal-pose consistency check for joint_ddt.

Per ``examples/process_dataset/convert_franka_coffee_pod.py``:

    action[t] = pose_wrt_world[t+1]            (pos + rot)
    goal_obs  = pose_wrt_world[horizon]        (the frame at index = horizon)

Therefore for any sample, action[horizon-1] and goal_obs should encode the
same EEF pose (position + orientation). Gripper is NOT shifted by the
converter (``grasp[:-1]``), so it doesn't have to match — only pos / rot do.

This script measures, on the model's PREDICTED action chunks:

    (a) ground-truth sanity:  ‖ unnormalize(act_true[:, H-1, :3]) -
                                  unnormalize(goal_obs eef_pos) ‖
        should be ~0 (data-pipeline consistency).

    (b) model accuracy:        ‖ unnormalize(act_pred[:, H-1, :3]) -
                                  unnormalize(goal_obs eef_pos) ‖
        same metric for each regime (diagonal / state_first / oracle_pinned /
        noise_pinned). Rotation reported as geodesic degrees between
        pred-last-rot6d's matrix and the goal_obs eef_quat's matrix
        (scipy convention).

Reports per regime, in physical units:
    pos Δ (mm, mean / max), rot Δ (deg, mean / max), gripper-flip% at last
    action vs goal_obs robot0_gripper_qpos (also informational — gripper
    isn't expected to match goal_obs).
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

from mip.datasets.robomimic_dataset import make_idm_dataset
from scripts.diagnose_joint_ddt_schedule_sweep import (
    _pick_agent_cls,
    _rot6d_to_matrix,
    _sample_with_pinning,
    _sample_with_schedule,
)


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    M = np.matmul(R1, np.swapaxes(R2, -1, -2))
    tr = np.einsum("...ii->...", M)
    cos_th = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def _last_action_vs_goal(
    act_normed: torch.Tensor,
    goal_pos_m: np.ndarray,
    goal_R: np.ndarray,
    action_normalizer,
) -> dict:
    """Compare unnormalized last-action pose to goal_obs pose.

    Args:
        act_normed: (n, H, 10) normalized action chunks.
        goal_pos_m: (n, 3) goal-frame EEF position in meters.
        goal_R:     (n, 3, 3) goal-frame rotation matrices.
    """
    a = act_normed.detach().cpu().numpy().astype(np.float32)
    flat_un = action_normalizer.unnormalize(a.reshape(-1, 10)).reshape(a.shape)
    last = flat_un[:, -1, :]                                  # (n, 10)
    pred_pos = last[:, :3]
    pred_R = _rot6d_to_matrix(last[:, 3:9])                   # (n, 3, 3)

    pos_l2 = np.linalg.norm(pred_pos - goal_pos_m, axis=-1)   # (n,)
    rot_deg = _geodesic_deg(pred_R, goal_R)                   # (n,)
    return {
        "pos_l2_mean_mm": float(pos_l2.mean()) * 1000.0,
        "pos_l2_max_mm":  float(pos_l2.max())  * 1000.0,
        "rot_geo_mean_deg": float(rot_deg.mean()),
        "rot_geo_max_deg":  float(rot_deg.max()),
    }


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

    AgentCls = _pick_agent_cls(cfg)
    print(f"\n[1/3] Building agent ({AgentCls.__name__}) + loading checkpoint...")
    agent = AgentCls(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()

    # ---- dataset (heldout w/ training normalizer, or training) ----
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

    # Cache torch obs + numpy goal pose/quat
    sample_keys = list(cfg.task.shape_meta["obs"].keys())
    obs_batch = {k: [] for k in sample_keys}
    goal_batch = {k: [] for k in sample_keys}
    act_batch_normed = []
    goal_pos_normed = []
    goal_quat_normed = []
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
            goal_batch[k].append(s["goal_obs"][k])
        act_batch_normed.append(s["action"][: cfg.task.horizon])
        goal_pos_normed.append(s["goal_obs"]["robot0_eef_pos"][0].numpy())
        goal_quat_normed.append(s["goal_obs"]["robot0_eef_quat"][0].numpy())

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

    # Unnormalize goal pose to physical units
    goal_pos_normed = np.stack(goal_pos_normed)                   # (n, 3)
    goal_quat_normed = np.stack(goal_quat_normed)                 # (n, 4) xyzw
    goal_pos_m = obs_normalizer["robot0_eef_pos"].unnormalize(goal_pos_normed)
    goal_quat = obs_normalizer["robot0_eef_quat"].unnormalize(goal_quat_normed)
    goal_R = Rotation.from_quat(goal_quat).as_matrix()             # (n, 3, 3)

    # Shared (act_0, x_state_init) for all regimes
    obs_dim = cfg.network.encoder_out_dim or cfg.network.emb_dim
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    act_0 = torch.randn(
        (n, cfg.task.horizon, cfg.task.act_dim), generator=g,
    ).to(device)
    x_state_init = torch.randn(
        (n, 1, obs_dim), generator=g,
    ).to(device)

    print("\n[3/3] Sampling under each regime + comparing last action to "
          "goal_obs pose...")

    # ---- (a) ground-truth sanity: act_true[-1] should match goal_obs ----
    print("\n== ground-truth sanity ==")
    print("  act_true[:, H-1] (unnormalized) vs goal_obs eef_pose:")
    gt = _last_action_vs_goal(act_true, goal_pos_m, goal_R, action_normalizer)
    print(
        f"    pos Δ mean={gt['pos_l2_mean_mm']:.4f} mm  max={gt['pos_l2_max_mm']:.4f} mm"
    )
    print(
        f"    rot Δ mean={gt['rot_geo_mean_deg']:.4f}°  max={gt['rot_geo_max_deg']:.4f}°"
    )
    print("  (these should be ~0 — pure data-pipeline residual from MinMax precision)")

    # ---- (b) model accuracy across regimes ----
    print("\n== model: predicted last action vs goal_obs eef_pose ==")
    print(f"  {'regime':<16s}{'pos mean / max (mm)':>30s}{'rot mean / max (deg)':>30s}")

    modes = [
        ("diagonal",      "schedule", "diagonal",     0.0),
        ("state_first",   "schedule", "state_first",  0.0),
        ("noise_pinned",  "pin",      "noise_pinned", 0.0),
        ("oracle_pinned", "pin",      "oracle_pinned", 0.0),
    ]
    for name, kind, sched_or_pin, off in modes:
        if kind == "schedule":
            a, _ = _sample_with_schedule(
                agent,
                obs=obs_torch, act_0=act_0, x_state_init=x_state_init,
                num_steps=args.num_steps,
                schedule=sched_or_pin, pyramid_offset=off,
            )
        else:
            a, _ = _sample_with_pinning(
                agent,
                obs=obs_torch, goal_obs=goal_torch,
                act_0=act_0, x_state_init=x_state_init,
                num_steps=args.num_steps,
                pin_mode=sched_or_pin,
            )
        st = _last_action_vs_goal(a, goal_pos_m, goal_R, action_normalizer)
        print(
            f"  {name:<16s}"
            f"{st['pos_l2_mean_mm']:>16.2f} / {st['pos_l2_max_mm']:>9.2f}"
            f"{st['rot_geo_mean_deg']:>20.3f} / {st['rot_geo_max_deg']:>9.3f}"
        )


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
