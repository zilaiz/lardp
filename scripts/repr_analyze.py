"""Representation-quality analysis for the DP/EO/EP encoders (per task).

Reads repr_analysis/<task>/features.npz (from repr_extract.py) and computes,
for each setting in {DP, EO, EP}:

  1. Geometry      effective rank + participation ratio of raw features (exp/roll)
  2. CKA           linear CKA between settings, on expert vs rollout states
  3. Probes        ridge R^2 for decision-relevant targets:
                     next-pos(3), next-state(9), d-state(9), first-action(7)
                   in-distribution (expert train->val) AND transfer (expert->rollout)
  4. Fwd residual  linear latent forward model  h_goal ~ A h_last + B action  (R^2)
  5. Typicality    Mahalanobis + kNN distance of rollout features to the expert
                   feature manifold; expert-vs-rollout linear-probe AUC

All probe/CKA/typicality metrics use raw features standardized by each model's
own expert-set mean/std (so LN-vs-none scale differences don't confound).
Geometry (effective rank) uses unstandardized raw features (per the target_std
LN-artifact lesson).

Usage: python scripts/repr_analyze.py --task can
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

OUT_ROOT = "/oscar/data/csun45/zzeng28/repo/lardp/repr_analysis"
SETTINGS = ["DP", "EO", "EP"]


# ---------------------------------------------------------------- helpers
def standardizer(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    return mu, sd


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=1.0):
    """Closed-form ridge; return variance-weighted R^2 over all target dims."""
    # standardize Y on train
    ym, ys = Ytr.mean(0, keepdims=True), Ytr.std(0, keepdims=True) + 1e-6
    Ytr_s, Yte_s = (Ytr - ym) / ys, (Yte - ym) / ys
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    W = np.linalg.solve(A, Xtr.T @ Ytr_s)
    pred = Xte @ W
    ss_res = ((Yte_s - pred) ** 2).sum()
    ss_tot = ((Yte_s - Yte_s.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-9))


def effective_rank(X):
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / (len(X) - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    s = ev.sum()
    if s <= 0:
        return 0.0, 0.0
    p = ev / s
    p = p[p > 1e-12]
    erank = float(np.exp(-(p * np.log(p)).sum()))
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())  # participation ratio
    return erank, pr


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    nx = np.linalg.norm(Xc.T @ Xc, "fro")
    ny = np.linalg.norm(Yc.T @ Yc, "fro")
    return float(hsic / (nx * ny + 1e-12))


def mahalanobis_typicality(H_exp, H_roll):
    mu = H_exp.mean(0)
    cov = np.cov(H_exp.T) + 1e-3 * np.eye(H_exp.shape[1])
    P = np.linalg.inv(cov)
    d_roll = H_roll - mu
    md = np.sqrt(np.einsum("ij,jk,ik->i", d_roll, P, d_roll))
    d_exp = H_exp - mu
    md_exp = np.sqrt(np.einsum("ij,jk,ik->i", d_exp, P, d_exp))
    # normalize by the expert self-distance so models are comparable
    return float(np.median(md) / (np.median(md_exp) + 1e-9))


def knn_dist(H_exp, H_roll, k=10):
    # mean distance of each rollout point to its k nearest expert points
    from scipy.spatial import cKDTree
    tree = cKDTree(H_exp)
    d, _ = tree.query(H_roll, k=k)
    d_self = cKDTree(H_exp).query(H_exp, k=k + 1)[0][:, 1:]
    return float(np.median(d.mean(1)) / (np.median(d_self.mean(1)) + 1e-9))


def expert_roll_auc(H_exp, H_roll):
    """AUC of a linear probe separating expert vs rollout (ridge-logit-free)."""
    X = np.concatenate([H_exp, H_roll])
    y = np.concatenate([np.zeros(len(H_exp)), np.ones(len(H_roll))])
    # split
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    d = X.shape[1]
    w = np.linalg.solve(X[tr].T @ X[tr] + 1.0 * np.eye(d), X[tr].T @ (y[tr] - 0.5))
    s = X[te] @ w
    # AUC
    yt = y[te]
    pos, neg = s[yt == 1], s[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    auc = (pos[:, None] > neg[None, :]).mean()
    return float(auc)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    task = args.task
    d = np.load(f"{OUT_ROOT}/{task}/features.npz")
    To, horizon = int(d["meta"][0]), int(d["meta"][1])

    e_cur, e_nxt, e_act = d["exp_cur"], d["exp_nxt"], d["exp_act"]
    r_cur, r_nxt, r_act = d["roll_cur"], d["roll_nxt"], d["roll_act"]
    # targets
    tgt = {
        "next_pos(3)": (e_nxt[:, :3], r_nxt[:, :3]),
        "next_state(9)": (e_nxt, r_nxt),
        "d_state(9)": (e_nxt - e_cur, r_nxt - r_cur),
        "first_action(7)": (e_act[:, 0, :], r_act[:, 0, :]),
        "cur_state(9)": (e_cur, r_cur),
    }

    rng = np.random.RandomState(0)
    nexp = len(e_cur)
    perm = rng.permutation(nexp)
    ntr = int(0.7 * nexp)
    tr, va = perm[:ntr], perm[ntr:]

    res = {"task": task, "settings": {}}
    # standardized raw features per setting
    feats = {}
    for s in SETTINGS:
        He, Hr = d[f"{s}_exp_raw"], d[f"{s}_roll_raw"]
        mu, sd = standardizer(He)
        feats[s] = {"exp": (He - mu) / sd, "roll": (Hr - mu) / sd,
                    "exp_goal": (d[f"{s}_exp_goal"] - mu) / sd,
                    "roll_goal": (d[f"{s}_roll_goal"] - mu) / sd,
                    "raw_exp": He, "raw_roll": Hr}

    for s in SETTINGS:
        He, Hr = feats[s]["exp"], feats[s]["roll"]
        Hg = feats[s]["exp_goal"]
        r = {}
        # 1. geometry (unstandardized raw)
        r["erank_exp"], r["pr_exp"] = effective_rank(feats[s]["raw_exp"])
        r["erank_roll"], r["pr_roll"] = effective_rank(feats[s]["raw_roll"])
        # rollout train/val split for rollout-in-distribution probes
        nr = len(Hr)
        rp = np.random.RandomState(1).permutation(nr)
        rtr, rva = rp[:int(0.7 * nr)], rp[int(0.7 * nr):]
        # 3. probes
        for name, (ye, yr) in tgt.items():
            r[f"probe_id::{name}"] = ridge_r2(He[tr], ye[tr], He[va], ye[va])
            r[f"probe_tx::{name}"] = ridge_r2(He, ye, Hr, yr)          # expert->rollout
            r[f"probe_rid::{name}"] = ridge_r2(Hr[rtr], yr[rtr], Hr[rva], yr[rva])  # rollout in-dist
        # 4. forward residual: h_goal ~ [h_last, action] ; expert AND rollout
        Hgr = feats[s]["roll_goal"]
        Xe = np.concatenate([He, e_act[:, 0, :]], axis=1)
        Xr = np.concatenate([Hr, r_act[:, 0, :]], axis=1)
        r["fwd_r2_exp"] = ridge_r2(Xe[tr], Hg[tr], Xe[va], Hg[va])
        r["fwd_r2_roll"] = ridge_r2(Xr[rtr], Hgr[rtr], Xr[rva], Hgr[rva])
        # 5. typicality of rollout vs expert manifold
        r["maha_roll"] = mahalanobis_typicality(He, Hr)
        r["knn_roll"] = knn_dist(He, Hr)
        r["exp_roll_auc"] = expert_roll_auc(He, Hr)
        res["settings"][s] = r

    # 2. CKA between settings (exp and roll)
    res["cka"] = {}
    for a, b in [("DP", "EO"), ("DP", "EP"), ("EO", "EP")]:
        res["cka"][f"{a}-{b}_exp"] = linear_cka(feats[a]["exp"], feats[b]["exp"])
        res["cka"][f"{a}-{b}_roll"] = linear_cka(feats[a]["roll"], feats[b]["roll"])

    os.makedirs(f"{OUT_ROOT}/{task}", exist_ok=True)
    json.dump(res, open(f"{OUT_ROOT}/{task}/metrics.json", "w"), indent=2)

    # ---- pretty print ----
    print(f"\n================= {task} =================")
    print(f"{'metric':<26}{'DP':>10}{'EO':>10}{'EP':>10}")
    rows = [
        ("erank_exp", "erank_exp"), ("erank_roll", "erank_roll"),
        ("probe_id next_pos", "probe_id::next_pos(3)"),
        ("probe_id next_state", "probe_id::next_state(9)"),
        ("probe_id d_state", "probe_id::d_state(9)"),
        ("probe_id first_action", "probe_id::first_action(7)"),
        ("probe_TX d_state", "probe_tx::d_state(9)"),
        ("probe_Rid next_pos", "probe_rid::next_pos(3)"),
        ("probe_Rid d_state", "probe_rid::d_state(9)"),
        ("probe_Rid first_action", "probe_rid::first_action(7)"),
        ("fwd_r2_exp", "fwd_r2_exp"),
        ("fwd_r2_roll", "fwd_r2_roll"),
        ("maha_roll (norm)", "maha_roll"),
        ("knn_roll (norm)", "knn_roll"),
        ("exp_roll_auc", "exp_roll_auc"),
    ]
    for label, key in rows:
        vals = [res["settings"][s][key] for s in SETTINGS]
        print(f"{label:<26}" + "".join(f"{v:>10.3f}" for v in vals))
    print("CKA:  " + "  ".join(f"{k}={v:.3f}" for k, v in res["cka"].items()))
    print(f"saved {OUT_ROOT}/{task}/metrics.json")


if __name__ == "__main__":
    main()
