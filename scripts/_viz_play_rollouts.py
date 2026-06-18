"""Render PushT play rollouts to MP4 (one rollout per file).

Reads the collected rollout HDF5 read-only, samples N demos, and writes each as
an upscaled MP4 with a step/coverage overlay. Coverage shown = stored per-step
reward = clip(coverage/0.95, 0, 1); peak over the episode = the mean_success
metric for that demo.
"""
import os
import numpy as np
import cv2
import h5py
import imageio

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

ROLLOUT = "data/pusht/image_rollouts.hdf5"
OUT_DIR = "pusht_rollout_videos"
N_SAMPLE = 20
UPSCALE = 4          # 96 -> 384
FPS = 15
SEED = 0


def overlay(frame, demo_idx, t, cov, peak):
    """frame: (H, W, 3) uint8 RGB. Draw small text (dark, readable on light bg)."""
    img = frame.copy()
    txt = [f"demo {demo_idx}", f"t={t:3d}  cov={cov:.2f}", f"peak={peak:.2f}"]
    for i, s in enumerate(txt):
        cv2.putText(img, s, (4, 14 + 14*i), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(ROLLOUT, "r", locking=False) as f:
        d = f["data"]
        n = int(d.attrs["num_demos"])
        # per-demo peak coverage (for filenames / representativeness)
        peak = np.array([float(d[f"demo_{i}"]["rewards"][:].max()) for i in range(n)])
        rng = np.random.default_rng(SEED)
        picks = sorted(rng.choice(n, size=min(N_SAMPLE, n), replace=False).tolist())
        print(f"{n} demos total; sampling {len(picks)} (seed {SEED})")
        for j, idx in enumerate(picks):
            g = d[f"demo_{idx}"]
            imgs = g["obs"]["image"][:]          # (T, 96, 96, 3) uint8 RGB (HWC)
            rew = g["rewards"][:]
            T = imgs.shape[0]
            run_peak = 0.0
            frames = []
            for t in range(T):
                fr = imgs[t]
                if fr.shape[-1] != 3:            # safety: if stored CHW
                    fr = np.moveaxis(fr, 0, -1)
                fr = cv2.resize(fr, (96*UPSCALE, 96*UPSCALE),
                                interpolation=cv2.INTER_NEAREST)
                cov = float(rew[t]) if t < len(rew) else 0.0
                run_peak = max(run_peak, cov)
                frames.append(overlay(fr, idx, t, cov, run_peak))
            score = peak[idx]
            out = os.path.join(OUT_DIR, f"play_{j:02d}_demo{idx}_score{score:.2f}_T{T}.mp4")
            imageio.mimwrite(out, frames, fps=FPS, codec="libx264",
                             quality=8, macro_block_size=16)
            print(f"  wrote {out}  (T={T}, peak_cov={score:.2f})")
    print(f"\nDone. {len(picks)} mp4s in {OUT_DIR}/  (peak_cov spread: "
          f"min {peak[picks].min():.2f} / med {np.median(peak[picks]):.2f} / max {peak[picks].max():.2f})")


if __name__ == "__main__":
    main()
