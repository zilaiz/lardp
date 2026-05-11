"""Render demos from data/franka_coffee_pod_cog/image.hdf5 as MP4.

One MP4 per demo. Each frame is the front and wrist camera views
concatenated horizontally [front | wrist].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

CAMS = ("front_cam_image", "wrist_cam_image")


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def render_demo(d, out_path: Path, fps: int, width: int, height: int):
    imgs = [d[f"obs/{c}"][:] for c in CAMS]
    T = imgs[0].shape[0]
    cam_w = width // 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for t in range(T):
            left = resize(imgs[0][t], cam_w, height)
            right = resize(imgs[1][t], cam_w, height)
            writer.append_data(np.concatenate([left, right], axis=1))
    finally:
        writer.close()
    print(f"wrote {out_path}  T={T}  size={width}x{height}  cams=front|wrist")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=Path,
        default=Path("data/franka_coffee_pod_cog/image.hdf5"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("viz/franka_coffee_pod"),
    )
    parser.add_argument(
        "--demos", type=int, nargs="*", default=None,
        help="Demo indices to render. Default = all demos.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=480,
                        help="Output frame width (split evenly between cams).")
    parser.add_argument("--height", type=int, default=270)
    args = parser.parse_args()

    with h5py.File(args.src, "r") as f:
        data = f["data"]
        if args.demos is None:
            ids = sorted(int(k.split("_")[1]) for k in data.keys()
                         if k.startswith("demo_"))
        else:
            ids = args.demos
        for i in ids:
            name = f"demo_{i}"
            if name not in data:
                print(f"[warn] missing {name}")
                continue
            out = args.out_dir / f"{name}.mp4"
            render_demo(data[name], out, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
