"""Visualize Franka coffee-pod data: per-demo 4-camera grid mp4 + action/state plot.

Usage:
    python scripts/visualize_franka_data.py \
        --files data/franka_coffee_pod_cog/image.hdf5 data/franka_coffee_pod_cog/image_play.hdf5 \
        --num-demos 5 --seed 0 --out viz/franka_coffee_pod_cog
"""

import argparse
import os
import random
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CAM_KEYS = ["front_cam_image", "wrist_cam_image"]


def upscale(img, scale):
    if scale == 1:
        return img
    H, W = img.shape[:2]
    return np.array(
        Image.fromarray(img).resize((W * scale, H * scale), Image.BICUBIC),
        dtype=np.uint8,
    )


def tile_row(images_per_cam, pad=4):
    """Place a list of (H, W, 3) frames side-by-side with padding."""
    H, W = images_per_cam[0].shape[:2]
    n = len(images_per_cam)
    canvas = np.full((H, n * W + (n - 1) * pad, 3), 32, dtype=np.uint8)
    for i, img in enumerate(images_per_cam):
        x = i * (W + pad)
        canvas[:, x : x + W] = img
    # Pad to even / macro_block_size=16 friendly dims to keep ffmpeg happy.
    gh, gw = canvas.shape[:2]
    pad_h = (16 - gh % 16) % 16
    pad_w = (16 - gw % 16) % 16
    if pad_h or pad_w:
        canvas = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=32)
    return canvas


def render_camera_grid_mp4(demo, out_path, fps=20, scale=4):
    cams = [demo["obs"][k] for k in CAM_KEYS]
    T = cams[0].shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=9, macro_block_size=1
    )
    try:
        for t in range(T):
            frames = [upscale(c[t], scale) for c in cams]
            writer.append_data(tile_row(frames))
    finally:
        writer.close()


def render_action_state_plot(demo, out_path, title):
    actions = demo["actions"][:]  # (T, 10)
    eef_pos = demo["obs"]["robot0_eef_pos"][:]
    eef_quat = demo["obs"]["robot0_eef_quat"][:]
    gripper = demo["obs"]["robot0_gripper_qpos"][:]
    T = actions.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    ax = axes[0]
    for i in range(actions.shape[1]):
        ax.plot(t, actions[:, i], label=f"a[{i}]", linewidth=0.9)
    ax.set_ylabel("action (10d)")
    ax.legend(ncol=5, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    ax = axes[1]
    for i, name in enumerate(["x", "y", "z"]):
        ax.plot(t, eef_pos[:, i], label=f"eef_{name}")
    ax.set_ylabel("eef_pos")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for i, name in enumerate(["x", "y", "z", "w"]):
        ax.plot(t, eef_quat[:, i], label=f"q_{name}")
    ax.set_ylabel("eef_quat")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, gripper[:, 0], color="black")
    ax.set_ylabel("gripper_qpos")
    ax.set_xlabel("timestep")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def visualize_file(hdf5_path, out_dir, num_demos, rng, scale=4):
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        chosen = rng.sample(demos, min(num_demos, len(demos)))
        chosen.sort(key=lambda s: int(s.split("_")[1]))
        print(f"[{hdf5_path.name}] picked: {chosen}")
        for name in chosen:
            demo = f["data"][name]
            T = demo["actions"].shape[0]
            stem = f"{hdf5_path.stem}__{name}"
            mp4_path = out_dir / f"{stem}.mp4"
            png_path = out_dir / f"{stem}_actions.png"
            print(f"  {name}: T={T}  ->  {mp4_path.name} + {png_path.name}")
            render_camera_grid_mp4(demo, mp4_path, scale=scale)
            render_action_state_plot(demo, png_path, title=f"{hdf5_path.name} / {name} (T={T})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--num-demos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="viz/franka_coffee_pod_cog")
    ap.add_argument("--scale", type=int, default=4, help="upscale factor for camera frames")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in args.files:
        visualize_file(Path(fp), out_dir, args.num_demos, rng, scale=args.scale)

    print(f"\nDone. Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
