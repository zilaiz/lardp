"""Temporal smoothness of the DP/EO/EP representations along trajectories.

Encodes consecutive timesteps of expert and rollout trajectories and measures how
smoothly each representation evolves. Paired with physical motion so smoothness
from a (useless) collapsed/saturated rep is distinguishable from meaningful smoothness.

Per (setting, source), over sampled demos:
  step       mean ||Δh_t||                (standardized-embedding units; per-step move)
  cos_consec mean cos(Δh_t, Δh_{t-1})     directional consistency (1=momentum, 0=jitter)
  accel      mean||h_{t+1}-2h_t+h_{t-1}|| / mean||Δh_t||   (lower = smoother)
  latphys_r  corr(||Δh_t||, ||Δstate_t||) how well latent motion tracks physical motion
  jerk_frac  frac of steps with ||Δh_t|| > 3x median       (spikes / discontinuities)

Embeddings standardized per (setting,source) by that set's own mean/std so step
sizes are comparable across models. Reuses repr_extract for encoders/normalizers.

Usage: python scripts/repr_temporal.py --task can
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch

import scripts.repr_extract as X

N_DEMOS = 25
ROLL_MAXLEN = 300   # cap rollout demo length for balance/speed


def _set_keys(task):
    cfg = X._cfg(task, "EP")
    sm = cfg.task.shape_meta.obs
    X.RGB = [k for k, v in sm.items() if v.get("type") == "rgb"]
    X.LOWDIM = [k for k, v in sm.items() if v.get("type", "low_dim") == "low_dim"]
    return cfg


@torch.no_grad()
def encode_traj(enc, ln, f, demo, normd, To, bs=128, maxlen=None):
    """Return (h[Teff,256], state[Teff,S]) for consecutive windows of one demo."""
    g = f["data"][demo]
    obs = g["obs"]
    T = g["actions"].shape[0]
    if maxlen:
        T = min(T, maxlen)
    imgs = {k: obs[k][:T] for k in X.RGB}                       # (T,84,84,3)
    lows = {k: obs[k][:T].astype(np.float32) for k in X.LOWDIM}  # (T,d)
    state = np.concatenate([lows[k] for k in X.LOWDIM], axis=1)  # (T,S) raw physical
    Teff = T - To + 1
    H = []
    for b in range(0, Teff, bs):
        ts = range(b, min(b + bs, Teff))
        ob = {}
        for k in X.RGB:
            win = np.stack([imgs[k][t:t + To] for t in ts])     # (B,To,84,84,3)
            ob[k] = torch.from_numpy(X._img_norm(win)).to(X.DEVICE)
        for k in X.LOWDIM:
            win = np.stack([lows[k][t:t + To] for t in ts])      # (B,To,d)
            ob[k] = torch.from_numpy(normd[k].normalize(win)).to(X.DEVICE)
        h = enc(ob, None)[:, -1, :]
        H.append(h.cpu().numpy())
    return np.concatenate(H), state[To - 1:]


def traj_metrics(H, S):
    """Smoothness metrics for one standardized trajectory H (Teff,D), state S."""
    dH = np.diff(H, axis=0)                     # (T-1,D)
    step = np.linalg.norm(dH, axis=1)           # (T-1,)
    dS = np.linalg.norm(np.diff(S, axis=0), axis=1)
    # directional consistency between consecutive steps
    a, b = dH[:-1], dH[1:]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    accel = np.linalg.norm(b - a, axis=1)       # ||2nd diff||
    return step, cos, accel, dS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    task = args.task
    cfg = _set_keys(task)
    exp_p, roll_p = cfg.task.dataset_paths[0], cfg.task.dataset_paths[1]
    To = int(cfg.task.obs_steps)
    N_exp = X.fit_lowdim_normalizers(exp_p)
    N_mix = X.merge_lowdim(N_exp, X.fit_lowdim_normalizers(roll_p))
    norm_of = {"DP": N_exp, "EO": N_exp, "EP": N_mix}

    out = {"task": task, "settings": {}}
    for s in ["DP", "EO", "EP"]:
        enc, ln = X.build_encoder(task, s)
        rec = {}
        for src, path, maxlen in [("exp", exp_p, None), ("roll", roll_p, ROLL_MAXLEN)]:
            with h5py.File(path, "r") as f:
                demos = list(f["data"].keys())[:N_DEMOS]
                trajs = [encode_traj(enc, ln, f, dm, norm_of[s], To, maxlen=maxlen)
                         for dm in demos]
            # standardize embeddings by pooled mean/std of this (setting,source)
            allH = np.concatenate([h for h, _ in trajs])
            mu, sd = allH.mean(0, keepdims=True), allH.std(0, keepdims=True) + 1e-6
            steps, coss, accels, dSs = [], [], [], []
            for h, st in trajs:
                if len(h) < 3:
                    continue
                hz = (h - mu) / sd
                step, cos, accel, dS = traj_metrics(hz, st)
                steps.append(step); coss.append(cos); accels.append(accel); dSs.append(dS)
            step = np.concatenate(steps); cos = np.concatenate(coss)
            accel = np.concatenate(accels)
            # align dS to accel/cos length (they use step[:-1])
            dS_step = np.concatenate(dSs)
            r = {
                "step": float(np.mean(step)),
                "cos_consec": float(np.mean(cos)),
                "accel_ratio": float(np.mean(accel) / (np.mean(step) + 1e-9)),
                "latphys_r": float(np.corrcoef(step, dS_step)[0, 1]),
                "jerk_frac": float(np.mean(step > 3 * np.median(step))),
            }
            rec[src] = r
        out["settings"][s] = rec
        del enc, ln
        torch.cuda.empty_cache()

    json.dump(out, open(f"{X.OUT_ROOT}/{task}/temporal.json", "w"), indent=2)
    for src in ["exp", "roll"]:
        print(f"\n===== {task} [{src}] temporal smoothness =====")
        print(f"{'metric':<14}{'DP':>10}{'EO':>10}{'EP':>10}")
        for m in ["step", "cos_consec", "accel_ratio", "latphys_r", "jerk_frac"]:
            print(f"{m:<14}" + "".join(f"{out['settings'][s][src][m]:>10.3f}"
                                        for s in ["DP", "EO", "EP"]))
    print(f"\nsaved {X.OUT_ROOT}/{task}/temporal.json")


if __name__ == "__main__":
    main()
