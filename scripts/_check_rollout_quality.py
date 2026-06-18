#!/usr/bin/env python
"""Quick quality audit of Franka rollout episode folders.

For each episode under the given dirs, reports:
  - frame count (pose length vs rgb file count, cross-checked)
  - initial / final EEF pose (world frame)
  - distance of the initial pose from the cohort median start pose
  - intervention / grasp summary

Flags:
  - extremely short rollouts (length outliers)
  - episodes whose FIRST frame pose is far from the typical start
    (candidate "recorded from an intermediate state")
  - frame-count mismatches between state and any camera
"""

import os
from pathlib import Path

import numpy as np

DIRS = [
    "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout/dp_os1_h24_rollout",
    "/oscar/data/csun45/zzeng28/datasets/dp_os1_h24_rollout/dp_os1_h24_rollout_2",
]
CAMS = ("cam1", "cam2", "cam3", "cam4")


def episode_info(ep_dir: Path) -> dict:
    sd = ep_dir / "state"
    pose = np.load(sd / "pose_wrt_world.npy").astype(np.float64)   # (T,7)
    grasp = np.load(sd / "grasp.npy").astype(np.float64)
    grip = np.load(sd / "gripper_qpos.npy").astype(np.float64)
    iv = np.load(sd / "intervention.npy")
    T = pose.shape[0]
    cam_counts = {}
    for c in CAMS:
        rgb = ep_dir / c / "rgb"
        cam_counts[c] = len(os.listdir(rgb)) if rgb.is_dir() else -1
    return {
        "name": ep_dir.name,
        "T": T,
        "pose0": pose[0],
        "poseN": pose[-1],
        "pos0": pose[0, :3],
        "posN": pose[-1, :3],
        "cam_counts": cam_counts,
        "iv_nonone": int(np.sum(iv != 1)),
        "grasp_unique": sorted(np.unique(grasp).tolist()),
        "grasp_changes": int(np.sum(np.abs(np.diff(grasp)) > 0)),
        "grip_range": (float(grip.min()), float(grip.max())),
    }


def main():
    eps = []
    for d in DIRS:
        dp = Path(d)
        for ep in sorted(dp.iterdir()):
            if ep.is_dir() and ep.name.startswith("episode_"):
                info = episode_info(ep)
                info["dir"] = dp.name
                eps.append(info)

    Ts = np.array([e["T"] for e in eps])
    pos0 = np.stack([e["pos0"] for e in eps])          # (N,3)
    med_pos0 = np.median(pos0, axis=0)
    start_dist = np.linalg.norm(pos0 - med_pos0, axis=1)

    # thresholds
    short_thresh = max(50, int(np.median(Ts) * 0.4))
    start_thresh = float(np.median(start_dist) + 3 * (np.median(np.abs(start_dist - np.median(start_dist))) + 1e-6))

    print(f"\n=== {len(eps)} episodes total ===")
    print(f"length: min={Ts.min()} max={Ts.max()} mean={Ts.mean():.0f} median={int(np.median(Ts))}")
    print(f"median start pos (world): {med_pos0.round(4).tolist()}")
    print(f"start-dist: median={np.median(start_dist):.4f} max={start_dist.max():.4f}")
    print(f"SHORT flag if T < {short_thresh}; FAR-START flag if dist > {start_thresh:.4f}\n")

    hdr = f"{'dir':<24}{'episode':<32}{'T':>5}{'startdist':>11}{'camOK':>7}{'ivNo1':>7}{'grspChg':>9}  flags"
    print(hdr)
    print("-" * len(hdr))
    for e, sd_ in zip(eps, start_dist):
        cam_ok = all(c == e["T"] for c in e["cam_counts"].values())
        flags = []
        if e["T"] < short_thresh:
            flags.append("SHORT")
        if sd_ > start_thresh:
            flags.append("FAR-START")
        if not cam_ok:
            flags.append(f"CAMMISMATCH{e['cam_counts']}")
        print(f"{e['dir']:<24}{e['name']:<32}{e['T']:>5}{sd_:>11.4f}"
              f"{('Y' if cam_ok else 'N'):>7}{e['iv_nonone']:>7}{e['grasp_changes']:>9}"
              f"  {' '.join(flags)}")

    # sorted views to eyeball outliers
    print("\n--- shortest 8 ---")
    for e in sorted(eps, key=lambda x: x["T"])[:8]:
        print(f"  {e['dir']}/{e['name']}  T={e['T']}")
    print("\n--- farthest-start 8 ---")
    order = np.argsort(-start_dist)
    for i in order[:8]:
        e = eps[i]
        print(f"  {e['dir']}/{e['name']}  dist={start_dist[i]:.4f}  pos0={e['pos0'].round(4).tolist()}")


if __name__ == "__main__":
    main()
