#!/usr/bin/env python
"""For each candidate episode, densely scan EVERY frame's wrist-cam skin
fraction, then render the top-K highest-skin frames (wrist + front) labeled
with frame index / total and skin %, so we can confirm human presence and
WHERE in the episode it occurs (start / mid / end).
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout"
D1 = f"{ROOT}/dp_os1_h24_rollout"
D2 = f"{ROOT}/dp_os1_h24_rollout_2"

# candidates from the skin scan (wMAX desc), excluding the 4 known-bad
CANDIDATES = [
    (D1, "episode_20260611_163718_879"),  # 0.257 wHI5
    (D2, "episode_20260613_204825_523"),  # 0.207 wHI3
    (D1, "episode_20260611_165421_820"),  # 0.120 wHI2
    (D2, "episode_20260613_213937_990"),  # 0.094 wHI3
    (D1, "episode_20260611_165536_578"),  # 0.088 wHI1
    (D2, "episode_20260613_215051_931"),  # 0.083 wHI1
    (D2, "episode_20260613_221418_408"),  # 0.082 wHI1
    (D2, "episode_20260613_214755_434"),  # 0.064 wHI1
]

K = 6           # top-K skin frames to show
TW, TH = 480, 270


def skin_fraction_arr(arr_ycc) -> float:
    Y, Cb, Cr = arr_ycc[..., 0], arr_ycc[..., 1], arr_ycc[..., 2]
    mask = (Y > 50) & (Cb >= 85) & (Cb <= 135) & (Cr >= 135) & (Cr <= 180)
    return float(mask.mean())


def main():
    for ed, name in CANDIDATES:
        epd = Path(ed) / name
        wrist_dir = epd / "cam4" / "rgb"
        front_dir = epd / "cam3" / "rgb"
        wfiles = sorted(os.listdir(wrist_dir))
        ffiles = sorted(os.listdir(front_dir))
        T = len(wfiles)

        # dense per-frame wrist skin fraction
        fr = np.empty(T)
        for i, fn in enumerate(wfiles):
            with Image.open(wrist_dir / fn) as im:
                fr[i] = skin_fraction_arr(np.asarray(im.convert("YCbCr"), dtype=np.int16))

        top = np.argsort(-fr)[:K]
        top = sorted(top.tolist())     # chronological order in the strip

        canvas = Image.new("RGB", (K * TW, 2 * TH + 22), (30, 30, 30))
        d = ImageDraw.Draw(canvas)
        peak_pct = int(round(100 * fr.max()))
        loc = f"peak@f{int(fr.argmax())}/{T - 1}"
        d.text((5, 5), f"{name}  T={T}  wrist-skin peak={peak_pct}%  {loc}  "
                       f"(top-{K} skin frames, chronological)", fill=(255, 255, 0))

        for c, i in enumerate(top):
            with Image.open(wrist_dir / wfiles[i]) as im:
                w = im.convert("RGB").resize((TW, TH))
            j = min(i, len(ffiles) - 1)
            with Image.open(front_dir / ffiles[j]) as im:
                f = im.convert("RGB").resize((TW, TH))
            for tile, lab in ((w, f"wrist f{i} {int(round(100*fr[i]))}%"), (f, f"front f{i}")):
                dd = ImageDraw.Draw(tile)
                dd.rectangle([0, 0, 150, 18], fill=(0, 0, 0))
                dd.text((3, 3), lab, fill=(0, 255, 0))
            canvas.paste(w, (c * TW, 22))
            canvas.paste(f, (c * TW, 22 + TH))

        out = f"/oscar/data/csun45/zzeng28/repo/lardp/human_peak_{name}.png"
        canvas.save(out)
        print(f"saved {out}  T={T} peak={peak_pct}% @f{int(fr.argmax())} top={top}")


if __name__ == "__main__":
    main()
