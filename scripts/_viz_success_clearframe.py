#!/usr/bin/env python
"""Definitive end-state view for success candidates.

For each candidate, scan the last 70 frames and pick the one where the ARM is
most retracted (minimum bright/white pixels in the arm-entry zone directly above
the bowl), giving the clearest unoccluded view of the bowl contents. Render a
wider crop (bowl + surrounding mat) so we can confirm BOTH the white pod AND the
green cog are inside, and the mat is clear. Montage 6 per image.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout"
DIRS = {"r1": f"{ROOT}/dp_os1_h24_rollout", "r2": f"{ROOT}/dp_os1_h24_rollout_2"}
OUT = "/oscar/data/csun45/zzeng28/repo/lardp/success_strips"
CAM = "cam2"

ARM_ZONE = (165, 8, 245, 52)         # x0,y0,x1,y1 region above bowl where arm enters
CROP = (140, 40, 280, 165)           # wider bowl+mat ROI -> 140x125
PANEL_W = 380
LASTK = 70

CANDIDATES = [
    ("r1", "episode_20260611_163015_111"), ("r1", "episode_20260611_164847_158"),
    ("r1", "episode_20260611_165054_235"), ("r1", "episode_20260611_165421_820"),
    ("r1", "episode_20260611_165536_578"), ("r1", "episode_20260611_165819_889"),
    ("r1", "episode_20260611_170000_177"), ("r1", "episode_20260611_170131_099"),
    ("r1", "episode_20260611_170302_434"), ("r1", "episode_20260611_170409_679"),
    ("r2", "episode_20260613_202340_880"), ("r2", "episode_20260613_204549_692"),
    ("r2", "episode_20260613_204825_523"), ("r2", "episode_20260613_213330_859"),
    ("r2", "episode_20260613_213600_219"), ("r2", "episode_20260613_214051_769"),
    ("r2", "episode_20260613_214149_155"), ("r2", "episode_20260613_214525_280"),
    ("r2", "episode_20260613_214755_434"), ("r2", "episode_20260613_215051_931"),
    ("r2", "episode_20260613_215315_976"), ("r2", "episode_20260613_220958_691"),
    ("r2", "episode_20260613_221246_838"), ("r2", "episode_20260613_221418_408"),
]

_cw, _ch = CROP[2] - CROP[0], CROP[3] - CROP[1]
PANEL_H = int(PANEL_W * _ch / _cw)


def arm_score(im):
    z = np.asarray(im.crop(ARM_ZONE)).astype(int)
    bright = (z[..., 0] > 150) & (z[..., 1] > 150) & (z[..., 2] > 150)
    return bright.mean()


def clearest_frame(epd, files):
    T = len(files)
    cand = range(max(T - LASTK, 0), T)
    best, best_s = T - 1, 1e9
    for i in cand:
        im = Image.open(epd / CAM / "rgb" / files[i]).convert("RGB")
        s = arm_score(im)
        if s < best_s:
            best_s, best = s, i
    return best


def main():
    per = 4
    batches = [CANDIDATES[i:i + per] for i in range(0, len(CANDIDATES), per)]
    for bi, batch in enumerate(batches):
        cols = len(batch)
        canvas = Image.new("RGB", (cols * PANEL_W, PANEL_H + 18), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        for c, (tag, name) in enumerate(batch):
            epd = Path(DIRS[tag]) / name
            files = sorted(os.listdir(epd / CAM / "rgb"))
            idx = clearest_frame(epd, files)
            p = Image.open(epd / CAM / "rgb" / files[idx]).convert("RGB").crop(CROP)
            p = p.resize((PANEL_W, PANEL_H))
            d.text((c * PANEL_W + 3, 3), f"{tag}_{name[8:]} f{idx}", fill=(255, 255, 0))
            canvas.paste(p, (c * PANEL_W, 18))
        out = f"{OUT}/_clearframe_batch{bi}.png"
        canvas.save(out)
        print(f"saved {out} ({len(batch)} eps)")


if __name__ == "__main__":
    main()
