#!/usr/bin/env python
"""Render one montage PER episode: N frames evenly sampled across the rollout,
front cam (cam3) + wrist cam (cam4) so the full short trajectory is visible.

Saves one PNG per episode for individual inspection.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout"
D1 = f"{ROOT}/dp_os1_h24_rollout"
D2 = f"{ROOT}/dp_os1_h24_rollout_2"

# (dataset_dir, episode_name, tag)
EPISODES = [
    (D2, "episode_20260613_202329_192", "SUSPECT"),
    (D1, "episode_20260611_164001_132", "SUSPECT"),
    (D1, "episode_20260611_165753_073", "SUSPECT"),
    (D1, "episode_20260611_164125_984", "SUSPECT"),
    # one normal short-from-home for reference
    (D1, "episode_20260611_163838_913", "NORMAL-short"),
]

N_FRAMES = 6          # evenly sampled across the episode
TW, TH = 480, 270     # native frame size


def load_idx(rgb_dir: Path, files, i):
    im = Image.open(rgb_dir / files[i]).convert("RGB")
    if im.size != (TW, TH):
        im = im.resize((TW, TH))
    return im


def strip(ep_dir: Path, cam: str, idxs, T):
    rgb = ep_dir / cam / "rgb"
    files = sorted(os.listdir(rgb))
    tiles = []
    for i in idxs:
        im = load_idx(rgb, files, i)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 120, 18], fill=(0, 0, 0))
        d.text((3, 3), f"{cam} f{i}/{T - 1}", fill=(0, 255, 0))
        tiles.append(im)
    return tiles


def main():
    for ed, name, tag in EPISODES:
        epd = Path(ed) / name
        T = len(os.listdir(epd / "cam3" / "rgb"))
        pose = np.load(epd / "state" / "pose_wrt_world.npy")
        z0 = float(pose[0, 2])
        idxs = np.linspace(0, T - 1, N_FRAMES).round().astype(int).tolist()

        front = strip(epd, "cam3", idxs, T)
        wrist = strip(epd, "cam4", idxs, T)

        ncol = N_FRAMES
        canvas = Image.new("RGB", (ncol * TW, 2 * TH + 22), (30, 30, 30))
        d = ImageDraw.Draw(canvas)
        d.text((5, 5), f"[{tag}] {name}  T={T}  start_z={z0:.3f} (home~0.377)",
               fill=(255, 255, 0))
        for c, im in enumerate(front):
            canvas.paste(im, (c * TW, 22))
        for c, im in enumerate(wrist):
            canvas.paste(im, (c * TW, 22 + TH))

        out = f"/oscar/data/csun45/zzeng28/repo/lardp/rollout_strip_{name}.png"
        canvas.save(out)
        print(f"saved {out}  T={T} idxs={idxs}")


if __name__ == "__main__":
    main()
