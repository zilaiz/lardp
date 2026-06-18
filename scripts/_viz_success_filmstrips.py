#!/usr/bin/env python
"""Render one 3x3 filmstrip per rollout episode for manual success labeling.

Success criterion (user): the robot sequentially places the WHITE coffee pod
THEN the green cog into the blue bowl (order matters).

Uses cam2 (RIGHT cam) — a near-top-down view where bowl contents AND objects
left on the mat are both clearly visible (cam3/front occludes the bowl interior
behind the rim). 9 frames evenly spaced, ROI-cropped to the work area and
upscaled. Each tile labeled f<idx> g<grasp> (g1 = gripper closed/holding) so
pick/place events and ordering are readable.

Use --only to render a subset (comma-separated episode-name substrings).
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout"
DIRS = [f"{ROOT}/dp_os1_h24_rollout", f"{ROOT}/dp_os1_h24_rollout_2"]
OUT = "/oscar/data/csun45/zzeng28/repo/lardp/success_strips"

CAM = "cam2"                          # right cam, near-top-down
N = 9
COLS = 3
CROP = (128, 34, 374, 224)           # cam2 bowl-centered ROI -> 246x190
UPSCALE = 2.0
_cw, _ch = CROP[2] - CROP[0], CROP[3] - CROP[1]
TW, TH = int(_cw * UPSCALE), int(_ch * UPSCALE)


def render(epd: Path, out_path: str):
    rgb = epd / CAM / "rgb"
    files = sorted(os.listdir(rgb))
    T = len(files)
    grasp = np.load(epd / "state" / "grasp.npy")
    g_changes = int(np.sum(np.abs(np.diff(grasp)) > 0))
    idxs = np.linspace(0, T - 1, N).round().astype(int).tolist()

    rows = (N + COLS - 1) // COLS
    canvas = Image.new("RGB", (COLS * TW, rows * TH + 20), (25, 25, 25))
    d = ImageDraw.Draw(canvas)
    d.text((5, 5), f"{epd.name}  T={T}  grasp_changes={g_changes}  [{CAM}]",
           fill=(255, 255, 0))

    for j, i in enumerate(idxs):
        im = Image.open(rgb / files[i]).convert("RGB").crop(CROP).resize((TW, TH))
        dd = ImageDraw.Draw(im)
        gi = int(grasp[min(i, len(grasp) - 1)])
        dd.rectangle([0, 0, 96, 16], fill=(0, 0, 0))
        dd.text((2, 2), f"f{i} g{gi}", fill=(0, 255, 0) if gi == 0 else (255, 120, 0))
        r, c = divmod(j, COLS)
        canvas.paste(im, (c * TW, 20 + r * TH))
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    subs = args.only.split(",") if args.only else None
    n = 0
    for di, d in enumerate(DIRS):
        tag = "r1" if di == 0 else "r2"
        for ep in sorted(Path(d).iterdir()):
            if not (ep.is_dir() and ep.name.startswith("episode_")):
                continue
            if subs and not any(s in ep.name for s in subs):
                continue
            render(ep, f"{OUT}/{tag}_{ep.name}.png")
            n += 1
    print(f"rendered {n} panels into {OUT}/")


if __name__ == "__main__":
    main()
