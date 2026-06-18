#!/usr/bin/env python
"""Render a montage of episode FIRST frames (front + wrist cam) to eyeball
whether any rollout begins from an intermediate state.

Top rows: the 4 short/far-start suspects.
Bottom rows: a few normal episodes for reference, plus first-vs-last of one
normal episode to show what 'home start' vs 'mid-task' looks like.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout"
D1 = f"{ROOT}/dp_os1_h24_rollout"
D2 = f"{ROOT}/dp_os1_h24_rollout_2"

SUSPECT = [
    (D2, "episode_20260613_202329_192"),
    (D1, "episode_20260611_164001_132"),
    (D1, "episode_20260611_165753_073"),
    (D1, "episode_20260611_164125_984"),
]
NORMAL = [
    (D1, "episode_20260611_163015_111"),
    (D1, "episode_20260611_163128_638"),
    (D2, "episode_20260613_202219_132"),
]

TILE = 200


def frame(ep_dir, cam, which="first"):
    rgb = Path(ep_dir) / cam / "rgb"
    files = sorted(os.listdir(rgb))
    fname = files[0] if which == "first" else files[-1]
    im = Image.open(rgb / fname).convert("RGB").resize((TILE, TILE))
    return im


def labeled(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, TILE, 14], fill=(0, 0, 0))
    d.text((2, 2), text, fill=(0, 255, 0))
    return im


def pos0(ep_dir, name):
    p = np.load(Path(ep_dir) / name / "state" / "pose_wrt_world.npy")
    return p[0, :3]


def main():
    rows = []
    # suspects: front + wrist first frame
    for ed, name in SUSPECT:
        epd = Path(ed) / name
        T = len(os.listdir(epd / "cam3" / "rgb"))
        p = pos0(ed, name).round(3).tolist()
        fr = labeled(frame(epd, "cam3", "first"), f"SUS {name[8:]} T{T}")
        wr = labeled(frame(epd, "cam4", "first"), f"z={p[2]} wrist")
        rows.append([fr, wr])
    # normals: front first
    for ed, name in NORMAL:
        epd = Path(ed) / name
        T = len(os.listdir(epd / "cam3" / "rgb"))
        fr = labeled(frame(epd, "cam3", "first"), f"OK {name[8:]} T{T}")
        wr = labeled(frame(epd, "cam4", "first"), "wrist")
        rows.append([fr, wr])
    # reference: first vs last of a normal episode (front)
    ed, name = NORMAL[0]
    epd = Path(ed) / name
    f0 = labeled(frame(epd, "cam3", "first"), "ref FIRST(home)")
    fN = labeled(frame(epd, "cam3", "last"), "ref LAST(end)")
    rows.append([f0, fN])

    ncol = 2
    nrow = len(rows)
    canvas = Image.new("RGB", (ncol * TILE, nrow * TILE), (40, 40, 40))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            canvas.paste(im, (c * TILE, r * TILE))
    out = "/oscar/data/csun45/zzeng28/repo/lardp/rollout_start_check.png"
    canvas.save(out)
    print(f"saved {out}  ({nrow} rows: 4 suspect + 3 normal + 1 first/last ref)")


if __name__ == "__main__":
    main()
