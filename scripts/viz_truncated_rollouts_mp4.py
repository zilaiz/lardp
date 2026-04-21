"""Render 5 random successful demos per task as an MP4 (H.264).

Writes one MP4 per task: a horizontal strip of 5 demos played in lockstep.
Shorter demos are padded with their last frame so all 5 finish together.
Frames with reward > 0 get a red tint overlay so the first-success step is
visually obvious.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

PRIMARY_CAM = {
    "can": "agentview_image",
    "square": "agentview_image",
    "transport": "shouldercamera0_image",
    "tool_hang": "sideview_image",
}


def pick_successful_indices(f, n_pick, rng):
    data = f["data"]
    n = int(data.attrs["num_demos"])
    successful = [
        i for i in range(n)
        if float(np.sum(data[f"demo_{i}"]["rewards"][:])) > 0
    ]
    if not successful:
        return []
    n_pick = min(n_pick, len(successful))
    return sorted(rng.choice(successful, size=n_pick, replace=False).tolist())


def tint_success(frame: np.ndarray, is_success: bool) -> np.ndarray:
    if not is_success:
        return frame
    out = frame.astype(np.float32)
    out[..., 0] = np.clip(out[..., 0] * 0.6 + 255 * 0.4, 0, 255)
    out[..., 1] *= 0.6
    out[..., 2] *= 0.6
    return out.astype(np.uint8)


def render_task(src_path: Path, out_path: Path, n_demos: int, fps: int,
                upscale: int, seed: int):
    rng = np.random.default_rng(seed)
    task = src_path.parent.name
    cam_key = PRIMARY_CAM[task]

    with h5py.File(src_path, "r") as f:
        indices = pick_successful_indices(f, n_demos, rng)
        if not indices:
            print(f"[{task}] no successful demos")
            return

        demos_imgs = []
        demos_rew = []
        demos_len = []
        for idx in indices:
            d = f[f"data/demo_{idx}"]
            imgs = d[f"obs/{cam_key}"][:]
            rew = d["rewards"][:]
            demos_imgs.append(imgs)
            demos_rew.append(rew)
            demos_len.append(imgs.shape[0])

    T_max = max(demos_len)
    H, W = demos_imgs[0].shape[1:3]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for t in range(T_max):
            row = []
            for imgs, rew, L in zip(demos_imgs, demos_rew, demos_len):
                i = min(t, L - 1)
                frame = imgs[i]
                frame = tint_success(frame, rew[i] > 0)
                if upscale > 1:
                    frame = np.repeat(np.repeat(frame, upscale, axis=0),
                                      upscale, axis=1)
                row.append(frame)
            grid = np.concatenate(row, axis=1)
            writer.append_data(grid)
    finally:
        writer.close()
    print(f"[{task}] wrote {out_path} "
          f"(demos {indices}, T_max={T_max}, fps={fps})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/robomimic"))
    parser.add_argument(
        "--src-name", default="image_rollouts_clipped_truncate.hdf5",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("viz/truncated_rollouts"),
    )
    parser.add_argument("--n-demos", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tasks", nargs="+",
        default=["can", "square", "transport", "tool_hang"],
    )
    args = parser.parse_args()

    for task in args.tasks:
        src = args.root / task / args.src_name
        if not src.exists():
            print(f"[{task}] MISSING: {src}")
            continue
        out = args.out_dir / f"{task}.mp4"
        render_task(src, out, args.n_demos, args.fps, args.upscale, args.seed)


if __name__ == "__main__":
    main()
