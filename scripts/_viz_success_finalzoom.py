#!/usr/bin/env python
"""High-res final-state bowl zoom for the success-candidate episodes.

The blue bowl sits at a fixed location (top-left of the cam2 frame), so we use a
FIXED crop on it (auto blue-blob detection latched onto the arm). For each
candidate we show two late frames (T-20 and T-1) zoomed on the bowl, so arm
occlusion in the very last frame doesn't hide the contents. 4 episodes per
montage, large panels.
"""

import os
from pathlib import Path

import numpy as np  # noqa: F401
from PIL import Image, ImageDraw

ROOT = "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout"
DIRS = {"r1": f"{ROOT}/dp_os1_h24_rollout", "r2": f"{ROOT}/dp_os1_h24_rollout_2"}
OUT = "/oscar/data/csun45/zzeng28/repo/lardp/success_strips"
CAM = "cam2"
CROP = (138, 30, 280, 150)           # fixed bowl ROI (center of cam2) -> 142x120
PANEL_W = 380
FRAME_OFFSETS = (25, 1)              # frames from end to show (T-25, T-1)

CANDIDATES = [
    ("r1", "episode_20260611_163015_111"),
    ("r1", "episode_20260611_164847_158"),
    ("r1", "episode_20260611_165054_235"),
    ("r1", "episode_20260611_165421_820"),
    ("r1", "episode_20260611_165536_578"),
    ("r1", "episode_20260611_165819_889"),
    ("r1", "episode_20260611_170000_177"),
    ("r1", "episode_20260611_170131_099"),
    ("r1", "episode_20260611_170302_434"),
    ("r1", "episode_20260611_170409_679"),
    ("r2", "episode_20260613_202340_880"),
    ("r2", "episode_20260613_204549_692"),
    ("r2", "episode_20260613_204825_523"),
    ("r2", "episode_20260613_213330_859"),
    ("r2", "episode_20260613_213600_219"),
    ("r2", "episode_20260613_214051_769"),
    ("r2", "episode_20260613_214149_155"),
    ("r2", "episode_20260613_214525_280"),
    ("r2", "episode_20260613_214755_434"),
    ("r2", "episode_20260613_215051_931"),
    ("r2", "episode_20260613_215315_976"),
    ("r2", "episode_20260613_220958_691"),
    ("r2", "episode_20260613_221246_838"),
    ("r2", "episode_20260613_221418_408"),
]

_cw, _ch = CROP[2] - CROP[0], CROP[3] - CROP[1]
PANEL_H = int(PANEL_W * _ch / _cw)


def panel(epd, idx, files):
    im = Image.open(epd / CAM / "rgb" / files[idx]).convert("RGB").crop(CROP)
    return im.resize((PANEL_W, PANEL_H))


def main():
    per = 4
    batches = [CANDIDATES[i:i + per] for i in range(0, len(CANDIDATES), per)]
    for bi, batch in enumerate(batches):
        cols = len(batch)
        rows = len(FRAME_OFFSETS)
        canvas = Image.new("RGB", (cols * PANEL_W, rows * PANEL_H + 18), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        for c, (tag, name) in enumerate(batch):
            epd = Path(DIRS[tag]) / name
            files = sorted(os.listdir(epd / CAM / "rgb"))
            T = len(files)
            d.text((c * PANEL_W + 3, 3), f"{tag}_{name[8:]}", fill=(255, 255, 0))
            for r, off in enumerate(FRAME_OFFSETS):
                idx = max(T - off, 0)
                p = panel(epd, idx, files)
                dd = ImageDraw.Draw(p)
                dd.rectangle([0, 0, 50, 14], fill=(0, 0, 0))
                dd.text((2, 1), f"f{idx}", fill=(0, 255, 255))
                canvas.paste(p, (c * PANEL_W, 18 + r * PANEL_H))
        out = f"{OUT}/_finalzoom_batch{bi}.png"
        canvas.save(out)
        print(f"saved {out} ({len(batch)} eps)  panel={PANEL_W}x{PANEL_H}")


if __name__ == "__main__":
    main()
