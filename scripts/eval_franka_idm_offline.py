"""Offline IDM eval on the Franka coffee-pod held-out expert split.

Validates IDM checkpoints by running the IDM open-loop on real held-out
(obs, goal) pairs and asking: "does the predicted action chunk's last step
land the eef at the goal eef position?"

This bypasses the simulator entirely: the held-out HDF5 supplies oracle
state and oracle goal directly, and we compare against ground-truth eef
position at the goal frame (frame index `horizon` in the IDM dataset
slicing). Distances are reported in meters using the training-time
normalizer.

Procedure
---------
1. Read the IDM training-time normalizer from <ckpt_dir>/normalizer.pkl
   (the merged normalizer pickled by ``train_franka_idm_fdm.main``).
2. Load the held-out HDF5 (pre-converted via
   ``examples/process_dataset/convert_franka_coffee_pod.py``) using the
   same ``RobomimicImageIDMDataset`` slicing as training, with the loaded
   normalizer overridden so action/obs scales match training.
3. For each checkpoint (single or sweep), load via ``IDMFDMAgent.load``
   then for each batch:
     a. stack obs+goal → (B, To+1, ...) per key (matching
        ``train_franka_idm_fdm._stack_obs_goal``)
     b. encode + flow-match sample (``agent.sample`` with use_ema=True)
     c. unnormalize predicted actions
     d. metrics:
        - goal_pos_l2     : ||action_pred[:, -1, :3] − goal_eef_pos||₂  (meters)
        - bc_action_mse   : per-step MSE vs ground-truth (sanity check)
        - goal_rot6d_mse  : rot6d component MSE at the goal step
4. Aggregate (mean / median / P95) per checkpoint and rank.

Action-vs-goal alignment
------------------------
Per ``examples/process_dataset/convert_franka_coffee_pod.py``,
``action[t][:3] = pose_wrt_world[t+1, :3]`` (world-frame target eef pos at
step t+1). The IDM dataset slices ``action[:horizon]`` and goal frame at
index ``horizon``. So ``action[-1, :3]`` is the predicted eef pos at the
goal frame, directly comparable to ``goal_obs["robot0_eef_pos"][0, :3]``
after unnormalization.

Usage
-----
First, convert the held-out folder once:

    python examples/process_dataset/convert_franka_coffee_pod.py \\
        --input-dir /users/zzeng28/data/zzeng28/datasets/franka_coffee_pod_cog/expert/ \\
        --output-path data/franka_coffee_pod_cog/image_heldout.hdf5

Then sweep the checkpoint folder:

    python scripts/eval_franka_idm_offline.py \\
        --ckpt-dir logs/franka_coffee_pod_real_image_flow_lbmidm_v2_256_seed0/2026_05_02_15_29_34/models \\
        --heldout-path data/franka_coffee_pod_cog/image_heldout.hdf5

Or single-checkpoint:

    python scripts/eval_franka_idm_offline.py \\
        --ckpt logs/.../models/model_step_150000.pt \\
        --heldout-path data/franka_coffee_pod_cog/image_heldout.hdf5
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

sys.path.insert(0, "/oscar/data/csun45/zzeng28/repo/lardp")

from mip.agent_idm_fdm import IDMFDMAgent  # noqa: E402
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

CONFIG_DIR = "/oscar/data/csun45/zzeng28/repo/lardp/examples/configs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_cfg(extra_overrides=None):
    """Compose Hydra config matching train_franka_idm_fdm.main()."""
    overrides = [
        "task=franka_coffee_pod_cog_image_idm",
        "network=lbmidm_v2",
        "optimization.loss_type=flow",
        "optimization.batch_size=128",
        "optimization.auto_resume=False",
        "log.wandb_mode=disabled",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = hydra.compose(config_name="main_franka", overrides=overrides)
    return cfg


def stack_obs_goal(batch, obs_steps, device):
    """Stack obs (To frames) + goal (1 frame) per key into (B, To+1, ...) on device.

    Mirrors ``examples/train_franka_idm_fdm._stack_obs_goal`` so the
    encoder sees the exact same input shape as during training.
    """
    obs_batch = batch["obs"]
    goal_batch = batch["goal_obs"]
    out = {}
    for k in obs_batch:
        o = obs_batch[k][:, :obs_steps].to(device, non_blocking=True)
        g = goal_batch[k].to(device, non_blocking=True)
        out[k] = torch.cat([o, g], dim=1)
    return out


def discover_checkpoints(ckpt_dir, ckpt, sweep_stride=None, steps=None):
    """Resolve --ckpt / --ckpt-dir into a list of (label, path) pairs sorted by step.

    If `steps` (iterable of ints) is provided, restrict to those exact step
    numbers (e.g. [10000, 20000, 50000]).
    """
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
        available = {s for s, _ in items}
        missing = sorted(wanted - available)
        if missing:
            raise FileNotFoundError(
                f"Requested steps {missing} not found under {ckpt_dir} "
                f"(available: {sorted(available)})"
            )
        items = [(s, p) for s, p in items if s in wanted]
    elif sweep_stride is not None and sweep_stride > 1:
        items = items[::sweep_stride]
    if not items:
        raise FileNotFoundError(
            f"No model_step_*.pt found under {ckpt_dir}"
        )
    return [(f"step_{step}", path) for step, path in items]


def _rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Recover a rotation matrix R (..., 3, 3) from rot6d (..., 6).

    Mirrors the conversion in examples/process_dataset/convert_franka_coffee_pod.py:
    rot6d = first two rows of R. We Gram-Schmidt orthogonalize them in case the
    network output isn't perfectly on SO(3), then add the third row as cross
    product. Output rows are r1, r2, r3 (so R[..., 0, :] = first row, etc.)
    """
    a1, a2 = d6[..., :3], d6[..., 3:6]
    r1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(min=1e-9)
    a2_proj = a2 - (r1 * a2).sum(axis=-1, keepdims=True) * r1
    r2 = a2_proj / np.linalg.norm(a2_proj, axis=-1, keepdims=True).clip(min=1e-9)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=-2)  # (..., 3, 3) — rows


def _rotation_angle_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """Angle (degrees) of the relative rotation R_pred · R_gt.T."""
    RRt = np.einsum("...ij,...kj->...ik", R_pred, R_gt)
    tr = RRt[..., 0, 0] + RRt[..., 1, 1] + RRt[..., 2, 2]
    cos_th = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def _per_step_rotation_deg(R: np.ndarray) -> np.ndarray:
    """Per-step rotation magnitude across a chunk: ‖angle(R[i+1] · R[i].T)‖.

    Args:
        R: (..., T, 3, 3) rotation chunk.
    Returns:
        (..., T-1) per-step rotation magnitude in degrees.
    """
    R_next = R[..., 1:, :, :]
    R_prev = R[..., :-1, :, :]
    R_rel = np.einsum("...ij,...kj->...ik", R_next, R_prev)
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_th = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


@torch.no_grad()
def evaluate_checkpoint(
    agent: IDMFDMAgent,
    loader: torch.utils.data.DataLoader,
    normalizer,
    obs_steps: int,
    horizon: int,
    nfe: int,
    max_batches: int | None,
    seed: int,
):
    """Run the IDM offline-eval pass for a single (already-loaded) agent.

    Returns
    -------
    dict with stacked numpy arrays:
        goal_pos_l2     : (N,)         meters; ||pred_eef_at_goal − gt_eef_at_goal||₂
        action_mse      : (N, horizon) per-step squared error between predicted
                                        and ground-truth action vectors (unnormalized)
        goal_rot6d_mse  : (N,)         MSE on rot6d dims at the goal step
        goal_rot_angle_deg : (N,)      angular error of the predicted goal rotation,
                                        in degrees (recovered from rot6d via Gram-Schmidt)
        goal_gripper_l1 : (N,)         |pred_gripper − gt_gripper| at the goal step
        goal_gripper_disagree : (N,)   {0/1} whether round(pred) != round(gt) at goal
    """
    agent.eval()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    act_norm = normalizer["action"]
    eef_norm = normalizer["obs"]["robot0_eef_pos"]

    goal_l2_list = []
    action_mse_list = []
    rot6d_list = []
    rot_deg_list = []
    grip_l1_list = []
    grip_dis_list = []
    # Smoothness diagnostics: per-step (frame-to-frame) deltas across the
    # action chunk. If the IDM is "shortcut copying" goal info into a few
    # late steps, the predicted chunk will have a spike — typically at the
    # last step — that the ground-truth chunk doesn't have. Comparing
    # predicted-delta vs gt-delta distributions directly catches this.
    pred_pos_step_list = []  # (N, horizon-1)  mm
    gt_pos_step_list   = []  # (N, horizon-1)  mm
    pred_rot_step_list = []  # (N, horizon-1)  deg
    gt_rot_step_list   = []  # (N, horizon-1)  deg
    pred_grip_step_list = []  # (N, horizon-1) abs delta in {0, 1}
    gt_grip_step_list   = []  # (N, horizon-1)
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        obs_stack = stack_obs_goal(batch, obs_steps, DEVICE)
        B = next(iter(obs_stack.values())).shape[0]
        act_dim = batch["action"].shape[-1]

        act_0 = torch.randn(
            (B, horizon, act_dim), device=DEVICE, dtype=torch.float32, generator=g,
        )
        act_pred_norm = agent.sample(
            act_0=act_0, obs=obs_stack, num_steps=nfe, use_ema=True,
        )

        # Unnormalize predicted + ground-truth actions to physical units.
        act_pred = act_norm.unnormalize(act_pred_norm.detach().cpu().numpy())  # (B, Ta, 10)
        act_gt = act_norm.unnormalize(
            batch["action"][:, :horizon, :].detach().cpu().numpy(),
        )

        # Goal eef position (world frame, meters) — unnormalize the dataset's
        # already-normalized goal_obs.robot0_eef_pos.
        goal_eef_norm = batch["goal_obs"]["robot0_eef_pos"][:, 0, :].cpu().numpy()  # (B, 3)
        goal_eef = eef_norm.unnormalize(goal_eef_norm)

        # Position L2 at goal step.
        pred_eef_at_goal = act_pred[:, -1, :3]                      # (B, 3)
        l2 = np.linalg.norm(pred_eef_at_goal - goal_eef, axis=-1)   # (B,)
        goal_l2_list.append(l2)

        # Per-step action MSE (sanity check vs ground truth).
        per_step_mse = ((act_pred - act_gt) ** 2).mean(axis=-1)     # (B, Ta)
        action_mse_list.append(per_step_mse)

        # Goal-step rot6d MSE (raw, unitless, on the 6 rot6d entries).
        rot6d_mse = ((act_pred[:, -1, 3:9] - act_gt[:, -1, 3:9]) ** 2).mean(axis=-1)
        rot6d_list.append(rot6d_mse)

        # Goal-step rotation angle error (degrees), via rot6d → R recovery.
        R_pred = _rot6d_to_matrix(act_pred[:, -1, 3:9])
        R_gt = _rot6d_to_matrix(act_gt[:, -1, 3:9])
        rot_deg = _rotation_angle_deg(R_pred, R_gt)
        rot_deg_list.append(rot_deg)

        # Goal-step gripper: continuous L1 + binary disagreement.
        # Gripper is the binary `grasp` command from the conversion script:
        # 0=open, 1=closed. Predicted is float (action chunk pre-discretization).
        pred_grip = act_pred[:, -1, 9]
        gt_grip = act_gt[:, -1, 9]
        grip_l1 = np.abs(pred_grip - gt_grip)
        grip_dis = (np.round(pred_grip) != np.round(gt_grip)).astype(np.float32)
        grip_l1_list.append(grip_l1)
        grip_dis_list.append(grip_dis)

        # ----- Smoothness diagnostics (frame-to-frame chunk deltas) -----
        # Position step deltas in mm.
        pred_pos_step = np.linalg.norm(np.diff(act_pred[:, :, :3], axis=1), axis=-1) * 1000.0
        gt_pos_step   = np.linalg.norm(np.diff(act_gt[:, :, :3],  axis=1), axis=-1) * 1000.0
        pred_pos_step_list.append(pred_pos_step)
        gt_pos_step_list.append(gt_pos_step)
        # Rotation step deltas in degrees.
        R_pred_chunk = _rot6d_to_matrix(act_pred[:, :, 3:9])
        R_gt_chunk   = _rot6d_to_matrix(act_gt[:, :, 3:9])
        pred_rot_step = _per_step_rotation_deg(R_pred_chunk)
        gt_rot_step   = _per_step_rotation_deg(R_gt_chunk)
        pred_rot_step_list.append(pred_rot_step)
        gt_rot_step_list.append(gt_rot_step)
        # Gripper step deltas (discontinuity check).
        pred_grip_step = np.abs(np.diff(act_pred[:, :, 9], axis=1))
        gt_grip_step   = np.abs(np.diff(act_gt[:, :, 9],  axis=1))
        pred_grip_step_list.append(pred_grip_step)
        gt_grip_step_list.append(gt_grip_step)

    return {
        "goal_pos_l2":           np.concatenate(goal_l2_list, axis=0),
        "action_mse":            np.concatenate(action_mse_list, axis=0),
        "goal_rot6d_mse":        np.concatenate(rot6d_list, axis=0),
        "goal_rot_angle_deg":    np.concatenate(rot_deg_list, axis=0),
        "goal_gripper_l1":       np.concatenate(grip_l1_list, axis=0),
        "goal_gripper_disagree": np.concatenate(grip_dis_list, axis=0),
        # Per-step deltas across the chunk (smoothness): shape (N, horizon-1)
        "pred_pos_step_mm":      np.concatenate(pred_pos_step_list, axis=0),
        "gt_pos_step_mm":        np.concatenate(gt_pos_step_list,   axis=0),
        "pred_rot_step_deg":     np.concatenate(pred_rot_step_list, axis=0),
        "gt_rot_step_deg":       np.concatenate(gt_rot_step_list,   axis=0),
        "pred_grip_step":        np.concatenate(pred_grip_step_list, axis=0),
        "gt_grip_step":          np.concatenate(gt_grip_step_list,   axis=0),
    }


def summarize(metrics):
    l2 = metrics["goal_pos_l2"]
    a_mse = metrics["action_mse"]
    rot = metrics["goal_rot6d_mse"]
    rot_deg = metrics["goal_rot_angle_deg"]
    grip_l1 = metrics["goal_gripper_l1"]
    grip_dis = metrics["goal_gripper_disagree"]
    n = len(l2)
    return {
        "n": n,
        "goal_pos_l2_mean":      float(l2.mean()),
        "goal_pos_l2_median":    float(np.median(l2)),
        "goal_pos_l2_p95":       float(np.percentile(l2, 95)),
        "action_mse_mean":       float(a_mse.mean()),
        "action_mse_last":       float(a_mse[:, -1].mean()),
        "goal_rot6d_mse_mean":   float(rot.mean()),
        "goal_rot_deg_mean":     float(rot_deg.mean()),
        "goal_rot_deg_median":   float(np.median(rot_deg)),
        "goal_rot_deg_p95":      float(np.percentile(rot_deg, 95)),
        "goal_gripper_l1_mean":  float(grip_l1.mean()),
        "goal_gripper_disagree": float(grip_dis.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=str,
                   help="Single checkpoint .pt path.")
    g.add_argument("--ckpt-dir", type=str,
                   help="Folder of model_step_*.pt to sweep.")
    parser.add_argument("--heldout-path", type=str, required=True,
                        help="Held-out HDF5 produced by convert_franka_coffee_pod.py.")
    parser.add_argument("--normalizer-path", type=str, default=None,
                        help="Override path to normalizer.pkl (defaults to "
                             "<ckpt_dir>/normalizer.pkl).")
    parser.add_argument("--nfe", type=int, default=9,
                        help="Number of flow-matching ODE steps for sampling.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap batches per checkpoint (default: full pass).")
    parser.add_argument("--sweep-stride", type=int, default=None,
                        help="When sweeping, take every Nth checkpoint.")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated list of exact step numbers to "
                             "evaluate, e.g. '10000,20000,50000'. Mutually "
                             "exclusive with --sweep-stride.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the noise sample act_0.")
    parser.add_argument("--out-npz", type=str, default=None,
                        help="Optional .npz to dump raw arrays per checkpoint.")
    args = parser.parse_args()

    set_seed(args.seed)
    limit_threads(1)
    torch.set_float32_matmul_precision("high")

    # ---- resolve checkpoints + normalizer ----
    steps_list = None
    if args.steps:
        steps_list = [int(s) for s in args.steps.split(",") if s.strip()]
    ckpt_list = discover_checkpoints(
        args.ckpt_dir, args.ckpt, args.sweep_stride, steps=steps_list,
    )
    ckpt_dir = args.ckpt_dir or os.path.dirname(args.ckpt)
    norm_path = args.normalizer_path or os.path.join(ckpt_dir, "normalizer.pkl")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalizer not found at {norm_path}. Pass --normalizer-path "
            f"or ensure train_franka_idm_fdm saved one alongside checkpoints."
        )
    with open(norm_path, "rb") as f:
        normalizer = pickle.load(f)
    loguru.logger.info(f"Loaded normalizer from {norm_path}")

    # ---- config ----
    cfg = build_cfg(extra_overrides=[f"optimization.batch_size={args.batch_size}"])
    cfg.task.obs_dim = cfg.network.emb_dim  # mirror train_franka_idm_fdm.main
    obs_steps = cfg.task.obs_steps
    horizon = cfg.task.horizon
    act_steps = cfg.task.act_steps

    # ---- held-out dataset (no val split; use all episodes; share train normalizer) ----
    if not os.path.exists(args.heldout_path):
        raise FileNotFoundError(
            f"Held-out HDF5 not found: {args.heldout_path}. Convert via "
            f"examples/process_dataset/convert_franka_coffee_pod.py first."
        )
    dataset = RobomimicImageIDMDataset(
        dataset_dir=os.path.expanduser(args.heldout_path),
        shape_meta=cfg.task.shape_meta,
        n_obs_steps=obs_steps,
        horizon=horizon,
        pad_before=obs_steps - 1,
        pad_after=act_steps - 1,
        abs_action=cfg.task.abs_action,
        val_dataset_percentage=0.0,   # use all demos
        mode="train",
        normalizer=normalizer,
    )
    loguru.logger.info(f"Held-out IDM dataset: {len(dataset)} samples")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # ---- agent (single instance, reload weights per checkpoint) ----
    loguru.logger.info("Instantiating IDMFDMAgent...")
    agent = IDMFDMAgent(cfg)
    agent.eval()

    # ---- sweep ----
    summaries = []
    raw_per_ckpt: dict[str, dict] = {}
    for label, path in ckpt_list:
        loguru.logger.info(f"Loading {label}: {path}")
        agent.load(path)
        agent.eval()
        t0 = time.time()
        metrics = evaluate_checkpoint(
            agent, loader, normalizer, obs_steps, horizon,
            nfe=args.nfe, max_batches=args.max_batches, seed=args.seed,
        )
        dt = time.time() - t0
        s = summarize(metrics)
        s["label"] = label
        s["wall_s"] = dt
        summaries.append(s)
        raw_per_ckpt[label] = metrics
        loguru.logger.info(
            f"  n={s['n']}  "
            f"pos_l2 mean={s['goal_pos_l2_mean']*1000:.2f}mm  "
            f"med={s['goal_pos_l2_median']*1000:.2f}  "
            f"P95={s['goal_pos_l2_p95']*1000:.2f}  "
            f"rot mean={s['goal_rot_deg_mean']:.2f}° "
            f"med={s['goal_rot_deg_median']:.2f}° "
            f"P95={s['goal_rot_deg_p95']:.2f}°  "
            f"grip_L1={s['goal_gripper_l1_mean']:.4f} "
            f"disagree={s['goal_gripper_disagree']:.4f}  "
            f"({dt:.1f}s)"
        )

    # ---- summary table ----
    print()
    print("=" * 100)
    print(f"Offline IDM eval — {len(dataset)} held-out samples, NFE={args.nfe}")
    print("=" * 100)
    print(
        f"{'checkpoint':<14s}  {'n':>5s}  | "
        f"{'pos_mm_mean':>11s}  {'pos_mm_med':>10s}  {'pos_mm_p95':>10s}  | "
        f"{'rot°_mean':>9s}  {'rot°_med':>8s}  {'rot°_p95':>8s}  | "
        f"{'grip_L1':>7s}  {'grip_dis':>8s}"
    )
    for s in summaries:
        print(
            f"{s['label']:<14s}  {s['n']:>5d}  | "
            f"{s['goal_pos_l2_mean']*1000:>11.3f}  "
            f"{s['goal_pos_l2_median']*1000:>10.3f}  "
            f"{s['goal_pos_l2_p95']*1000:>10.3f}  | "
            f"{s['goal_rot_deg_mean']:>9.3f}  "
            f"{s['goal_rot_deg_median']:>8.3f}  "
            f"{s['goal_rot_deg_p95']:>8.3f}  | "
            f"{s['goal_gripper_l1_mean']:>7.4f}  "
            f"{s['goal_gripper_disagree']:>8.4f}"
        )

    # Best by mean L2
    best = min(summaries, key=lambda x: x["goal_pos_l2_mean"])
    print()
    print(f"Best checkpoint by goal_l2_mean: {best['label']}  ({best['goal_pos_l2_mean']:.4f} m)")

    if args.out_npz:
        out = {}
        for label, m in raw_per_ckpt.items():
            for k, v in m.items():
                out[f"{label}__{k}"] = v
        np.savez(args.out_npz, **out)
        print(f"\nRaw per-sample arrays dumped to {args.out_npz}")


if __name__ == "__main__":
    main()
