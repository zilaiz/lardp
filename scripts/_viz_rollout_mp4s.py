#!/usr/bin/env python
"""Encode every rollout episode's cam3 (front cam) to an individual mp4 at 3x
the native capture rate.

Native fps is derived per-episode from the timestamped filenames
(<sec>_<nanosec>.png); output fps = 3 * native so playback is 3x real time.
Output: dp_rollout_cam3_3x_mp4/<tag>_<episode>.mp4  (tag r1=rollout, r2=rollout_2).
"""

import os
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image

ROOT = "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout"
DIRS = [("r1", f"{ROOT}/dp_os1_h24_rollout"), ("r2", f"{ROOT}/dp_os1_h24_rollout_2")]
OUT = "/oscar/data/csun45/zzeng28/repo/lardp/dp_rollout_cam3_3x_mp4"
CAM = "cam3"
SPEEDUP = 3.0


def native_fps(files):
    def ts(f):
        s, ns = f[:-4].split("_")
        return int(s) + int(ns) / 1e9
    t = np.array([ts(f) for f in files])
    return 1.0 / float(np.median(np.diff(t)))


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for tag, d in DIRS:
        for ep in sorted(Path(d).iterdir()):
            if not (ep.is_dir() and ep.name.startswith("episode_")):
                continue
            rgb = ep / CAM / "rgb"
            files = sorted(os.listdir(rgb))
            fps = round(SPEEDUP * native_fps(files), 2)
            out = f"{OUT}/{tag}_{ep.name}.mp4"
            w = imageio.get_writer(
                out, fps=fps, codec="libx264", quality=8,
                macro_block_size=1, pixelformat="yuv420p",
                ffmpeg_params=["-crf", "20"],
            )
            for f in files:
                w.append_data(np.asarray(Image.open(rgb / f).convert("RGB")))
            w.close()
            n += 1
            print(f"[{n:2d}] {tag}_{ep.name}  T={len(files)}  fps={fps}", flush=True)
    total = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT) if f.endswith(".mp4"))
    print(f"\nDONE: {n} mp4s in {OUT}/  ({total/1e6:.1f} MB total)")


if __name__ == "__main__":
    main()
