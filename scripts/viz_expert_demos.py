"""Render random expert (PH) demos as H.264 MP4s.

For each task in {can, square, transport, tool_hang} the script:

1. Opens ``data/robomimic/{task}/ph/image_v15_abs.hdf5``.
2. Randomly samples ``--n_demos`` demos (deterministic via ``--seed``).
3. Writes one MP4 per demo to
   ``viz/expert_demos/{task}/{demo_name}.mp4`` with all camera views
   composed horizontally.

Usage:
    python scripts/viz_expert_demos.py
    python scripts/viz_expert_demos.py --n_demos 5 --fps 20 --upscale 4
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
    tiles = []
    for k in image_keys:
        img = np.asarray(demo_group["obs"][k][t])
        if img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1) if img.max() <= 1.0 else img / 255.0
            img = (img * 255).astype(np.uint8)
        tiles.append(img)
    h = max(t.shape[0] for t in tiles)
    tiles = [
        np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) if t.shape[0] < h else t
        for t in tiles
    ]
    return np.concatenate(tiles, axis=1)


def upscale(frame: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return frame
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def render_demo(
    demo_group: h5py.Group,
    out_path: Path,
    image_keys: list[str],
    fps: int,
    upscale_factor: int,
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
    parser.add_argument("--upscale", default=4, type=int)
    parser.add_argument("--out_dir", default="viz/expert_demos", type=str)
    parser.add_argument("--tasks", default=",".join(TASKS), type=str)
    parser.add_argument(
        "--dataset_template",
        default="data/robomimic/{task}/ph/image_v15_abs.hdf5",
        type=str,
        help="Template path; {task} is substituted.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = repo_root / args.out_dir
    rng = np.random.default_rng(args.seed)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for task in tasks:
        hdf5_path = repo_root / args.dataset_template.format(task=task)
        if not hdf5_path.exists():
            print(f"[{task}] SKIP — missing {hdf5_path}")
            continue
        with h5py.File(hdf5_path, "r") as f:
            demo_names = sorted(f["data"].keys(),
                                key=lambda s: int(s.split("_")[-1]))
            n_pick = min(args.n_demos, len(demo_names))
            chosen = rng.choice(len(demo_names), size=n_pick, replace=False)
            chosen = sorted(int(i) for i in chosen)
            image_keys = list_image_keys(f["data"][demo_names[chosen[0]]])
            print(f"[{task}] {len(demo_names)} demos | "
                  f"image_keys={image_keys} | sampling {chosen}")

            for idx in chosen:
                name = demo_names[idx]
                demo = f["data"][name]
                out_path = out_root / task / f"{name}.mp4"
                T = render_demo(
                    demo, out_path, image_keys,
                    fps=args.fps, upscale_factor=args.upscale,
                )
                print(f"  [{task}] {name}: T={T} -> "
                      f"{out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
