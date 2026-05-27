"""Render Franka expert demos showing what the model sees after cropping.

For each demo we compose, per timestep, three columns per camera:

  [ raw 136x136 with 128 crop box | center-crop 128x128 (eval/deploy) | random-crop 128x128 (train) ]

Cameras stacked vertically. Random-crop offsets are sampled once per timestep
(matching how `CropRandomizer` re-samples per forward pass during training).

Source HDF5: data/franka_coffee_pod_cog/image.hdf5
Crop config: examples/configs/task/franka_base.yaml (crop_shape=[128,128]).

Usage:
    python scripts/viz_franka_cropped_demos.py
    python scripts/viz_franka_cropped_demos.py --n_demos 5 --cams front_cam_image,wrist_cam_image
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


def center_crop(img: np.ndarray, ch: int, cw: int) -> tuple[np.ndarray, int, int]:
    H, W = img.shape[:2]
    top = (H - ch) // 2
    left = (W - cw) // 2
    return img[top:top + ch, left:left + cw], top, left


def random_crop(img: np.ndarray, ch: int, cw: int, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    H, W = img.shape[:2]
    top = int(rng.integers(0, H - ch + 1))
    left = int(rng.integers(0, W - cw + 1))
    return img[top:top + ch, left:left + cw], top, left


def draw_box(img: np.ndarray, top: int, left: int, ch: int, cw: int,
             color: tuple[int, int, int], width: int = 1) -> np.ndarray:
    out = img.copy()
    bottom = top + ch - 1
    right = left + cw - 1
    out[top:top + width, left:right + 1] = color
    out[bottom - width + 1:bottom + 1, left:right + 1] = color
    out[top:bottom + 1, left:left + width] = color
    out[top:bottom + 1, right - width + 1:right + 1] = color
    return out


def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] >= target_h:
        return img
    pad = target_h - img.shape[0]
    top = pad // 2
    return np.pad(img, ((top, pad - top), (0, 0), (0, 0)))


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def render_demo(
    demo: h5py.Group,
    out_path: Path,
    cams: list[str],
    crop_h: int,
    crop_w: int,
    fps: int,
    upscale_factor: int,
    seed: int,
) -> int:
    T = demo["obs"][cams[0]].shape[0]
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for t in range(T):
            rows = []
            for cam in cams:
                raw = np.asarray(demo["obs"][cam][t])
                cc, ct, cl = center_crop(raw, crop_h, crop_w)
                rc, rt, rl = random_crop(raw, crop_h, crop_w, rng)
                raw_box = draw_box(raw, ct, cl, crop_h, crop_w, (0, 255, 0), 1)
                raw_box = draw_box(raw_box, rt, rl, crop_h, crop_w, (255, 200, 0), 1)
                tile_h = raw.shape[0]
                tiles = [raw_box, pad_to_height(cc, tile_h), pad_to_height(rc, tile_h)]
                rows.append(np.concatenate(tiles, axis=1))
            max_w = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0))) if r.shape[1] < max_w else r
                    for r in rows]
            frame = np.concatenate(rows, axis=0)
            frame = upscale(frame, upscale_factor)
            writer.append_data(frame)
    finally:
        writer.close()
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path,
                        default=Path("data/franka_coffee_pod_cog/image.hdf5"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("viz/franka_cropped"))
    parser.add_argument("--n_demos", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--upscale", type=int, default=3)
    parser.add_argument("--crop", type=int, nargs=2, default=(128, 128),
                        help="(H, W) crop size — must match training config.")
    parser.add_argument("--cams", type=str,
                        default="front_cam_image,wrist_cam_image",
                        help="Comma-separated obs keys. Default matches franka_base.yaml's selected cams.")
    args = parser.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    crop_h, crop_w = args.crop

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.src, "r") as f:
        data = f["data"]
        demo_names = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
        n_pick = min(args.n_demos, len(demo_names))
        chosen = sorted(int(i) for i in rng.choice(len(demo_names), size=n_pick, replace=False))
        print(f"[franka] {len(demo_names)} demos | cams={cams} | "
              f"crop={crop_h}x{crop_w} | sampling {chosen}")
        for idx in chosen:
            name = demo_names[idx]
            demo = data[name]
            out_path = args.out_dir / f"{name}_crop.mp4"
            T = render_demo(
                demo, out_path, cams, crop_h, crop_w,
                fps=args.fps, upscale_factor=args.upscale,
                seed=args.seed + idx,
            )
            print(f"  {name}: T={T} -> {out_path}")


if __name__ == "__main__":
    main()
