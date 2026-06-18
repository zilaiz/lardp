#!/usr/bin/env python
"""Scan all rollout episodes for human presence via skin-color detection.

intervention.npy is constant 0 (autonomous rollouts), so it can't flag the
human-staging fragments. We instead detect skin pixels (YCbCr range) per frame.

Wrist cam (cam4) is the primary signal: it looks straight down at the dark mat
with no wooden border, so a hand reaching in produces a large, clean skin blob.
Front cam (cam3) is reported too but has tan wood borders => higher baseline.

Outputs:
  - calibration: skin fraction on known human frames (192, 073) vs a clean ep
  - per-episode ranking by wrist-cam max/mean skin fraction + #high frames
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout"
D1 = f"{ROOT}/dp_os1_h24_rollout"
D2 = f"{ROOT}/dp_os1_h24_rollout_2"
DIRS = [D1, D2]

KNOWN_BAD = {
    "episode_20260613_202329_192",
    "episode_20260611_164001_132",
    "episode_20260611_165753_073",
    "episode_20260611_164125_984",
}

SAMPLE = 30          # frames per episode (evenly spaced)
HIGH_THRESH = 0.04   # per-frame skin fraction counted as "high" (calibrated below)


def skin_fraction(im: Image.Image) -> float:
    """Fraction of pixels in the YCbCr skin range."""
    ycc = np.asarray(im.convert("YCbCr"), dtype=np.int16)
    Y, Cb, Cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    mask = (Y > 50) & (Cb >= 85) & (Cb <= 135) & (Cr >= 135) & (Cr <= 180)
    return float(mask.mean())


def episode_skin(epd: Path, cam: str, idxs):
    rgb = epd / cam / "rgb"
    files = sorted(os.listdir(rgb))
    fr = []
    for i in idxs:
        with Image.open(rgb / files[i]) as im:
            fr.append(skin_fraction(im.convert("RGB")))
    return np.array(fr)


def sampled_idxs(epd: Path):
    T = len(os.listdir(epd / "cam3" / "rgb"))
    n = min(SAMPLE, T)
    return np.linspace(0, T - 1, n).round().astype(int).tolist(), T


def main():
    # ---- calibration ----
    print("=== calibration (skin fraction) ===")
    cal = [
        (D2, "episode_20260613_202329_192", "HUMAN-known"),
        (D1, "episode_20260611_165753_073", "HUMAN-known"),
        (D1, "episode_20260611_163015_111", "CLEAN-ref"),
        (D2, "episode_20260613_202219_132", "CLEAN-ref"),
    ]
    for ed, name, tag in cal:
        epd = Path(ed) / name
        idxs, T = sampled_idxs(epd)
        w = episode_skin(epd, "cam4", idxs)
        f = episode_skin(epd, "cam3", idxs)
        print(f"  {tag:<12} {name[8:]}: wrist max={w.max():.3f} mean={w.mean():.3f} | "
              f"front max={f.max():.3f} mean={f.mean():.3f}")

    # ---- full scan ----
    print(f"\n=== full scan (wrist cam cam4, {SAMPLE} frames/ep, HIGH>{HIGH_THRESH}) ===")
    rows = []
    for ed in DIRS:
        for ep in sorted(Path(ed).iterdir()):
            if not (ep.is_dir() and ep.name.startswith("episode_")):
                continue
            idxs, T = sampled_idxs(ep)
            w = episode_skin(ep, "cam4", idxs)
            f = episode_skin(ep, "cam3", idxs)
            rows.append({
                "dir": Path(ed).name, "name": ep.name, "T": T,
                "w_max": w.max(), "w_mean": w.mean(),
                "w_hi": int((w > HIGH_THRESH).sum()),
                "f_max": f.max(), "f_mean": f.mean(),
                "known": ep.name in KNOWN_BAD,
            })

    rows.sort(key=lambda r: -r["w_max"])
    hdr = f"{'dir':<22}{'episode':<32}{'T':>5}{'wMAX':>7}{'wMEAN':>7}{'wHI':>5}{'fMAX':>7}  flag"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        flag = "KNOWN-BAD" if r["known"] else ("<-- HUMAN?" if r["w_max"] > HIGH_THRESH else "")
        print(f"{r['dir']:<22}{r['name']:<32}{r['T']:>5}{r['w_max']:>7.3f}"
              f"{r['w_mean']:>7.3f}{r['w_hi']:>5}{r['f_max']:>7.3f}  {flag}")


if __name__ == "__main__":
    main()
