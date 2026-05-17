"""Render random demos from collected play data as H.264 MP4s.

For each task in {can, square, transport, tool_hang} the script:

1. Opens ``data/robomimic/{task}/image_rollouts_clipped.hdf5``.
2. Randomly samples ``--n_demos`` demos (deterministic via ``--seed``).
3. Labels each demo as success/failed via ``sum(rewards) > 0`` and writes
   one MP4 per demo to
   ``viz/play_data_samples/{task}/{demo_name}_{success|failed}.mp4``
   with all camera views composed horizontally and a green (success) or
   red (failed) border so the label is visible in the rendered frame.

Usage:
    python scripts/viz_play_data_demos.py
    python scripts/viz_play_data_demos.py --n_demos 5 --fps 20 --upscale 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

TASKS = ["can", "square", "transport", "tool_hang"]


def list_image_keys(demo_group: h5py.Group) -> list[str]:
    return sorted(k for k in demo_group["obs"].keys() if "image" in k)


def compose_frame(demo_group: h5py.Group, t: int, image_keys: list[str]) -> np.ndarray:
    tiles = [np.asarray(demo_group["obs"][k][t]) for k in image_keys]
    return np.concatenate(tiles, axis=1)


def upscale(frame: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return frame
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def add_border(frame: np.ndarray, color: tuple[int, int, int], width: int) -> np.ndarray:
    if width <= 0:
        return frame
    out = frame.copy()
    out[:width, :] = color
    out[-width:, :] = color
    out[:, :width] = color
    out[:, -width:] = color
    return out


def is_success(demo_group: h5py.Group) -> bool:
    return float(np.sum(demo_group["rewards"][:])) > 0


def render_demo(
    demo_group: h5py.Group,
    out_path: Path,
    image_keys: list[str],
    fps: int,
    upscale_factor: int,
    border_color: tuple[int, int, int],
    border_width: int,
) -> int:
    T = demo_group["obs"][image_keys[0]].shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for t in range(T):
            frame = compose_frame(demo_group, t, image_keys)
            frame = upscale(frame, upscale_factor)
            frame = add_border(frame, border_color, border_width)
            writer.append_data(frame)
    finally:
        writer.close()
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".", type=str)
    parser.add_argument("--n_demos", default=4, type=int,
                        help="Number of demos to sample per task")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("--upscale", default=4, type=int,
                        help="Integer nearest-neighbor upscale factor (84x84 -> 84*f)")
    parser.add_argument("--out_dir", default="viz/play_data_samples", type=str)
    parser.add_argument("--tasks", default=",".join(TASKS), type=str)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = repo_root / args.out_dir
    rng = np.random.default_rng(args.seed)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for task in tasks:
        hdf5_path = repo_root / "data" / "robomimic" / task / "image_rollouts_clipped.hdf5"
        if not hdf5_path.exists():
            print(f"[{task}] SKIP — missing {hdf5_path}")
            continue
        with h5py.File(hdf5_path, "r") as f:
            demo_names = sorted(f["data"].keys(),
                                key=lambda s: int(s.split("_")[-1]))
            n_pick = min(args.n_demos, len(demo_names))
            chosen = rng.choice(len(demo_names), size=n_pick, replace=False)
            chosen = sorted(int(i) for i in chosen)
            sample_keys = list_image_keys(f["data"][demo_names[chosen[0]]])
            print(f"[{task}] {len(demo_names)} demos | "
                  f"image_keys={sample_keys} | sampling {chosen}")

            n_succ_total = sum(is_success(f["data"][d]) for d in demo_names)
            print(f"  total: {n_succ_total}/{len(demo_names)} successful")

            for idx in chosen:
                name = demo_names[idx]
                demo = f["data"][name]
                succ = is_success(demo)
                label = "success" if succ else "failed"
                color = (0, 200, 0) if succ else (220, 30, 30)
                out_path = out_root / task / f"{name}_{label}.mp4"
                image_keys = list_image_keys(demo)
                T = render_demo(
                    demo, out_path, image_keys,
                    fps=args.fps, upscale_factor=args.upscale,
                    border_color=color, border_width=4,
                )
                print(f"  [{task}] {name} [{label}]: T={T} -> "
                      f"{out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
