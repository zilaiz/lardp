"""Expanded probe suite: linear + MLP, ground-truth + latent forward targets,
full action chunk as input and as decode target.

Reads repr_analysis/<task>/features.npz. For each setting (DP/EO/EP) and source
(expert held-out, rollout held-out) computes held-out R^2 for:

  action decode      X=emb          -> Y=first action (A)       [actdec_first]
                     X=emb          -> Y=full chunk (H*A)        [actdec_chunk]
  forward (latent)   X=emb+chunk    -> Y=encoder(goal) (256)     [fwd_goalemb]
  forward (physical) X=emb          -> Y=next state (S)          [antic_nxt]   (no action)
                     X=emb+chunk    -> Y=next state (S)          [fwd_nxt]
                     X=emb+chunk    -> Y=d_state (S)             [fwd_dstate]

each as linear ridge (alpha=1) AND a 2-hidden-layer MLP (best val R^2).
emb = standardized raw last-frame embedding (DP raw==ln). All R^2 in
standardized-target space; X standardized per train split inside each probe.

Usage: python scripts/repr_probe2.py --task can
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

OUT_ROOT = "/oscar/data/csun45/zzeng28/repo/lardp/repr_analysis"
SETTINGS = ["DP", "EO", "EP"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def split(n, seed):
    p = np.random.RandomState(seed).permutation(n)
    k = int(0.7 * n)
    return p[:k], p[k:]


def _std(tr, *arrs):
    mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True) + 1e-6
    return [(a - mu) / sd for a in arrs]


def _r2(Ytrue, Ypred):
    ss_res = ((Ytrue - Ypred) ** 2).sum()
    ss_tot = ((Ytrue - Ytrue.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-9))


def lin_r2(Xtr, Ytr, Xva, Yva, alpha=1.0):
    Xtr, Xva = _std(Xtr, Xtr, Xva)
    Ytr_s, Yva_s = _std(Ytr, Ytr, Yva)
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ Ytr_s)
    return _r2(Yva_s, Xva @ W)


def mlp_r2(Xtr, Ytr, Xva, Yva, epochs=400, h=512):
    Xtr, Xva = _std(Xtr, Xtr, Xva)
    Ytr_s, Yva_s = _std(Ytr, Ytr, Yva)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(Ytr_s, dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], h), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(h, Ytr.shape[1]),
    ).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best = -1e9
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(xt), yt)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                pv = net(xv).cpu().numpy()
            best = max(best, _r2(Yva_s, pv))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--no-mlp", action="store_true")
    args = ap.parse_args()
    task = args.task
    d = np.load(f"{OUT_ROOT}/{task}/features.npz")

    out = {"task": task, "settings": {}}
    for s in SETTINGS:
        He, Hr = d[f"{s}_exp_raw"], d[f"{s}_roll_raw"]
        mu, sd = He.mean(0, keepdims=True), He.std(0, keepdims=True) + 1e-6
        He, Hr = (He - mu) / sd, (Hr - mu) / sd
        Hge, Hgr = (d[f"{s}_exp_goal"] - mu) / sd, (d[f"{s}_roll_goal"] - mu) / sd
        r = {}
        for src, H, Hg, cur, nxt, act in [
            ("exp", He, Hge, d["exp_cur"], d["exp_nxt"], d["exp_act"]),
            ("roll", Hr, Hgr, d["roll_cur"], d["roll_nxt"], d["roll_act"]),
        ]:
            n = len(H)
            tr, va = split(n, 0 if src == "exp" else 1)
            chunk = act.reshape(n, -1)               # (N, H*A)
            a0 = act[:, 0, :]
            dstate = nxt - cur
            EC = np.concatenate([H, chunk], axis=1)  # emb + chunk
            probes = {
                "actdec_first": (H, a0),
                "actdec_chunk": (H, chunk),
                "fwd_goalemb":  (EC, Hg),
                "antic_nxt":    (H, nxt),
                "fwd_nxt":      (EC, nxt),
                "fwd_dstate":   (EC, dstate),
            }
            for name, (X, Y) in probes.items():
                r[f"{src}/{name}/lin"] = lin_r2(X[tr], Y[tr], X[va], Y[va])
                if not args.no_mlp:
                    r[f"{src}/{name}/mlp"] = mlp_r2(X[tr], Y[tr], X[va], Y[va])
        out["settings"][s] = r

    json.dump(out, open(f"{OUT_ROOT}/{task}/probes2.json", "w"), indent=2)

    # print expert table
    names = ["actdec_first", "actdec_chunk", "fwd_goalemb", "antic_nxt", "fwd_nxt", "fwd_dstate"]
    for src in ["exp", "roll"]:
        print(f"\n===== {task}  [{src} eval]   (lin | mlp) =====")
        print(f"{'probe':<16}{'DP':>16}{'EO':>16}{'EP':>16}")
        for nm in names:
            row = f"{nm:<16}"
            for s in SETTINGS:
                lin = out["settings"][s][f"{src}/{nm}/lin"]
                mlp = out["settings"][s].get(f"{src}/{nm}/mlp", float('nan'))
                row += f"  {lin:>5.2f}|{mlp:>5.2f}   "
            print(row)
    print(f"\nsaved {OUT_ROOT}/{task}/probes2.json")


if __name__ == "__main__":
    main()
