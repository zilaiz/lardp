#!/usr/bin/env python
"""Convert the Franka coffee-pod teleop dataset to a robomimic-style HDF5.

Input layout (per episode folder):
    <ep>/cam{1,2,3,4}/rgb/*.png           (480x270 RGB, one frame per state step)
    <ep>/cam{1,2,3,4}/depth{,_vis}        (IGNORED)
    <ep>/state/pose_wrt_world.npy         (T, 7)  [x, y, z, qx, qy, qz, qw]  WORLD frame
    <ep>/state/gripper_qpos.npy           (T,)    float32, observed finger opening
    <ep>/state/grasp.npy                  (T,)    {0, 1}, gripper command
    <ep>/state/joint_states.npy           (T, 9)  (unused here)
    <ep>/state/intervention.npy           (T,)    (unused here; sanity-checked only)

Output HDF5 layout (robomimic-style; one demo per episode):
    data/demo_i/
        actions                  (T-1, 10)  float32  = [pos(3), rot6d(6), gripper(1)]
        obs/left_cam_image       (T-1, 136, 136, 3) uint8   # cam1, crop 220x220 -> resize 136x136
        obs/right_cam_image      (T-1, 136, 136, 3) uint8   # cam2, crop 220x220 -> resize 136x136
        obs/front_cam_image      (T-1, 136, 136, 3) uint8   # cam3, crop 220x220 -> resize 136x136
        obs/wrist_cam_image      (T-1, 136, 136, 3) uint8   # cam4, resize 136x136 (no crop)
        obs/robot0_eef_pos       (T-1, 3)    float32   [x, y, z] at step t (current obs)
        obs/robot0_eef_quat      (T-1, 4)    float32   [qx, qy, qz, qw] at step t
        obs/robot0_gripper_qpos  (T-1, 1)    float32   finger opening at step t

Action semantics:
    action[t] is the command the policy should output given obs at step t.
    - Position / orientation: shift-by-one of the observed EEF pose
        pos    = pose_wrt_world[1:, :3]
        rot6d  = quat2rot6d(pose_wrt_world[1:, 3:7])        # xyzw -> R -> first 2 rows
      i.e., "go to where the EEF will be at t+1".
    - Gripper: grasp.npy is the command signal itself (leads gripper_qpos by ~3-4 frames);
        gripper = grasp[:-1]
      i.e., use grasp[t] as-is, no shift. We truncate to length T-1 to align with pos/rot6d.

Usage:
    uv run examples/process_dataset/convert_franka_coffee_pod.py \
        --input-dir /oscar/data/csun45/zzeng28/datasets/franka_coffee_pod_cog/demos \
        --output-path data/franka_coffee_pod_cog/image.hdf5
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


IMG_SIZE = 136                         # final square size; 136 leaves 8 px budget for
                                       # the encoder's 128x128 random crop augmentation
CROP_SIZE_STATIC = 220                 # center-crop for left/right/front cams (480x270 -> 220x220)
CAM_KEY_MAP = {
    "cam1": "left_cam_image",
    "cam2": "right_cam_image",
    "cam3": "front_cam_image",
    "cam4": "wrist_cam_image",
}
STATIC_CAMS = ("cam1", "cam2", "cam3")  # cams that get cropped before resize
WRIST_CAM = "cam4"                      # cam that gets resized directly


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternions in [qx, qy, qz, qw] order to (N, 6) rotation-6D.

    The 6D representation is the first two rows of the rotation matrix, flattened.
    Matches pytorch3d.transforms.matrix_to_rotation_6d convention — so
    pytorch3d.transforms.rotation_6d_to_matrix decodes this at inference.

    Quaternion double-cover (q vs -q) is automatically resolved by going through
    the rotation matrix, so no explicit canonicalization is needed.
    """
    R = Rotation.from_quat(quat_xyzw).as_matrix()         # (N, 3, 3)
    return R[:, :2, :].reshape(-1, 6).astype(np.float32)  # (N, 6)


def preprocess_frame(img: Image.Image, cam: str) -> np.ndarray:
    """Per-camera image preprocessing — returns (128, 128, 3) uint8.

    Static cams (left/right/front) get a SQUARE center-crop of side
    CROP_SIZE_STATIC from the 480x270 frame, then resize to 128x128.
    The wrist cam is resized directly with no crop.
    """
    if cam in STATIC_CAMS:
        w, h = img.size                         # 480 x 270
        assert (w, h) == (480, 270), f"unexpected size {img.size} for {cam}"
        s = min(CROP_SIZE_STATIC, h)            # can't exceed the shortest side
        left = (w - s) // 2
        top  = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))  # sxs square
    # cam4: direct resize, no crop (aspect ratio 480:270 stretched to 1:1 square)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)       # (128, 128, 3)


def load_cam_stack(ep_dir: Path, cam: str, n_frames: int) -> np.ndarray:
    """Load and preprocess all frames of one camera for one episode.

    Returns uint8 array of shape (n_frames, 128, 128, 3).
    """
    rgb_dir = ep_dir / cam / "rgb"
    files = sorted(os.listdir(rgb_dir))         # timestamps sort chronologically
    if len(files) != n_frames:
        raise ValueError(
            f"{ep_dir.name}/{cam}: {len(files)} frames but state reports {n_frames}"
        )
    out = np.empty((n_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, fname in enumerate(files):
        with Image.open(rgb_dir / fname) as im:
            im = im.convert("RGB")
            out[i] = preprocess_frame(im, cam)
    return out


def build_episode(ep_dir: Path) -> dict:
    """Build per-demo arrays for a single episode folder.

    Returns a dict with keys:
        actions             (T-1, 10) float32
        obs/<cam>_image     (T-1, 128, 128, 3) uint8   for each of the 4 cams
        obs/robot0_eef_pos  (T-1, 3) float32
        obs/robot0_eef_quat (T-1, 4) float32            [qx, qy, qz, qw]
        obs/robot0_gripper_qpos (T-1, 1) float32
        _meta: diagnostics (not written)
    """
    state_dir = ep_dir / "state"
    pose = np.load(state_dir / "pose_wrt_world.npy").astype(np.float32)   # (T, 7)
    grip = np.load(state_dir / "gripper_qpos.npy").astype(np.float32)     # (T,)
    grasp = np.load(state_dir / "grasp.npy").astype(np.float32)            # (T,) in {0,1}
    iv = np.load(state_dir / "intervention.npy")                          # (T,)

    T = pose.shape[0]
    assert grip.shape == (T,), f"{ep_dir.name}: gripper_qpos shape {grip.shape} vs T={T}"
    assert grasp.shape == (T,), f"{ep_dir.name}: grasp shape {grasp.shape} vs T={T}"
    assert iv.shape == (T,),    f"{ep_dir.name}: intervention shape {iv.shape} vs T={T}"
    if T < 2:
        raise ValueError(f"{ep_dir.name}: T={T} too short to produce action array")

    # Sanity check for intervention — warn if any frame had operator not in control
    if not np.all(iv == 1):
        n_out = int(np.sum(iv != 1))
        logger.warning(f"{ep_dir.name}: {n_out}/{T} frames with intervention!=1 "
                       "(kept, no masking applied)")

    # ---- Actions (T-1, 10) = [pos(3), rot6d(6), gripper(1)] ----
    pos_t1 = pose[1:, :3].astype(np.float32)                  # (T-1, 3)
    rot6d  = quat_xyzw_to_rot6d(pose[1:, 3:7])                # (T-1, 6)
    grip_a = grasp[:-1, None].astype(np.float32)              # (T-1, 1)  -- no shift
    actions = np.concatenate([pos_t1, rot6d, grip_a], axis=-1)

    # ---- Low-dim obs (T-1, ...) at current step t ----
    obs_pos  = pose[:-1, :3].astype(np.float32)                # (T-1, 3)
    obs_quat = pose[:-1, 3:7].astype(np.float32)               # (T-1, 4) [qx, qy, qz, qw]
    obs_gq   = grip[:-1, None].astype(np.float32)              # (T-1, 1)

    # ---- Images (T-1, 128, 128, 3) per cam ----
    cam_stacks = {}
    for cam, key in CAM_KEY_MAP.items():
        full = load_cam_stack(ep_dir, cam, n_frames=T)         # (T, 128, 128, 3)
        cam_stacks[key] = full[:-1]                            # truncate to T-1

    # ---- Diagnostics ----
    # rot6d continuity: max Euclidean step-to-step delta (big jumps => quat sign flips)
    rot_step = np.linalg.norm(np.diff(rot6d, axis=0), axis=-1) if len(rot6d) > 1 else np.zeros(0)
    meta = {
        "T": T,
        "n_out": T - 1,
        "act_stats": {
            "pos_min": pos_t1.min(0).tolist(),
            "pos_max": pos_t1.max(0).tolist(),
            "rot6d_min": rot6d.min(0).tolist(),
            "rot6d_max": rot6d.max(0).tolist(),
            "grip_unique": sorted(np.unique(grip_a).tolist()),
            "rot6d_max_step_jump": float(rot_step.max()) if rot_step.size else 0.0,
        },
    }

    return {
        "actions": actions,
        "obs": {
            **cam_stacks,
            "robot0_eef_pos": obs_pos,
            "robot0_eef_quat": obs_quat,
            "robot0_gripper_qpos": obs_gq,
        },
        "_meta": meta,
    }


def write_hdf5(output_path: Path, episodes: list[tuple[str, dict]]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        for i, (ep_name, ep) in enumerate(tqdm(episodes, desc="Writing HDF5")):
            demo = data_grp.create_group(f"demo_{i}")
            demo.attrs["source_episode"] = ep_name
            demo.create_dataset("actions", data=ep["actions"], compression="gzip",
                                compression_opts=4, chunks=True)
            obs_grp = demo.create_group("obs")
            for k, v in ep["obs"].items():
                # Image keys: chunk per-frame for lazy decoding; gzip to keep file compact.
                if v.ndim == 4 and v.dtype == np.uint8:
                    obs_grp.create_dataset(
                        k, data=v, compression="gzip", compression_opts=4,
                        chunks=(1,) + v.shape[1:],
                    )
                else:
                    obs_grp.create_dataset(
                        k, data=v, compression="gzip", compression_opts=4, chunks=True,
                    )
        f.attrs["n_demos"] = len(episodes)


def summarize(episodes: list[tuple[str, dict]]):
    if not episodes:
        return
    all_actions = np.concatenate([ep["actions"] for _, ep in episodes], axis=0)
    pos = all_actions[:, :3]; rot6d = all_actions[:, 3:9]; grip = all_actions[:, 9:]
    logger.info("=== Dataset action stats (over all demos) ===")
    logger.info(f"  n_demos         : {len(episodes)}")
    logger.info(f"  total steps     : {all_actions.shape[0]}")
    logger.info(f"  pos     min/max : {pos.min(0).round(4).tolist()} / {pos.max(0).round(4).tolist()}")
    logger.info(f"  rot6d   min/max : [{rot6d.min():.4f}, {rot6d.max():.4f}]")
    logger.info(f"  grip    unique  : {sorted(np.unique(grip).tolist())}")

    rot_jumps = [ep["_meta"]["act_stats"]["rot6d_max_step_jump"] for _, ep in episodes]
    logger.info(f"  rot6d max step-to-step jump across episodes: max={max(rot_jumps):.4f}, "
                f"median={float(np.median(rot_jumps)):.4f} "
                "(small values => quat ordering + continuity are consistent)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, required=True,
                   help="Path to demos/ folder containing episode_* subdirs.")
    p.add_argument("--output-path", type=str, required=True,
                   help="Output HDF5 path.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N episodes (for dry-runs).")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    ep_dirs = sorted([d for d in input_dir.iterdir()
                      if d.is_dir() and d.name.startswith("episode_")])
    if args.limit is not None:
        ep_dirs = ep_dirs[:args.limit]
    logger.info(f"Processing {len(ep_dirs)} episodes from {input_dir}")

    episodes = []
    for ep_dir in tqdm(ep_dirs, desc="Building episodes"):
        ep = build_episode(ep_dir)
        episodes.append((ep_dir.name, ep))
        m = ep["_meta"]["act_stats"]
        logger.debug(f"  {ep_dir.name}: T={ep['_meta']['T']} -> {ep['_meta']['n_out']} steps, "
                     f"rot6d max jump={m['rot6d_max_step_jump']:.3f}")

    summarize(episodes)
    write_hdf5(output_path, episodes)
    logger.info(f"Wrote {output_path}  ({output_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
