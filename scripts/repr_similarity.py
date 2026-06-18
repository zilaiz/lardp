"""How different are the DP/EO/EP encoders' embeddings of the SAME observations?

Reads features.npz (three encoders, identical windows). Per pair, computes:
  cos_raw     mean per-sample cosine(A_i, B_i)             (basis-dependent!)
  cos_cent    cosine after subtracting each encoder's mean (removes common offset)
  l2_unit     mean ||A_i/|A_i| - B_i/|B_i|||               (basis-dependent!)
  xpred_R2    ridge R^2 predicting B from A AND A from B, averaged (basis-INVARIANT:
              how much of one embedding is LINEARLY recoverable from the other)
  cka         linear CKA (basis-invariant similarity)

The raw cos/l2 are reported because they were asked for, but independently-trained
encoders live in arbitrary rotated/scaled bases, so cos_raw can be ~0 even when the
encoders carry identical information. xpred_R2 and cka are the meaningful "how
different" numbers.

Usage: python scripts/repr_similarity.py [--extra EO_s2e]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

OUT_ROOT = "/oscar/data/csun45/zzeng28/repo/lardp/repr_analysis"
TASKS = ["can", "square", "transport", "tool_hang"]


def xpred_r2(A, B, alpha=1.0):
    """ridge R^2 predicting B from A (variance-weighted over B dims)."""
    Am, As = A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6
    Bm, Bs = B.mean(0, keepdims=True), B.std(0, keepdims=True) + 1e-6
    A, B = (A - Am) / As, (B - Bm) / Bs
    n = len(A); k = int(0.7 * n)
    idx = np.random.RandomState(0).permutation(n); tr, va = idx[:k], idx[k:]
    d = A.shape[1]
    W = np.linalg.solve(A[tr].T @ A[tr] + alpha * np.eye(d), A[tr].T @ B[tr])
    pred = A[va] @ W
    ss_res = ((B[va] - pred) ** 2).sum()
    ss_tot = ((B[va] - B[va].mean(0)) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-9))


def cka(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    return float(np.linalg.norm(Yc.T @ Xc, "fro") ** 2 /
                 (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro") + 1e-12))


def pair_metrics(A, B):
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    cos_raw = float((a * b).sum(1).mean())
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    ac = Ac / (np.linalg.norm(Ac, axis=1, keepdims=True) + 1e-9)
    bc = Bc / (np.linalg.norm(Bc, axis=1, keepdims=True) + 1e-9)
    cos_cent = float((ac * bc).sum(1).mean())
    l2_unit = float(np.linalg.norm(a - b, axis=1).mean())
    xp = 0.5 * (xpred_r2(A, B) + xpred_r2(B, A))
    return {"cos_raw": cos_raw, "cos_cent": cos_cent, "l2_unit": l2_unit,
            "xpred_R2": xp, "cka": cka(A, B)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["DP", "EO", "EP"])
    ap.add_argument("--feat", default="raw", choices=["raw", "ln"])
    args = ap.parse_args()
    S = args.settings
    feat = args.feat  # 'ln' => post-LN for EO/EP (LN as part of encoder); DP _ln==_raw
    pairs = [(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S))]

    print(f"[features: {feat}]")
    for src in ["exp", "roll"]:
        print(f"\n############ {src.upper()} embeddings: pairwise similarity (mean over tasks) ############")
        print(f"{'pair':<10}{'cos_raw':>9}{'cos_cent':>10}{'l2_unit':>9}{'xpred_R2':>10}{'cka':>7}")
        agg = {p: {m: [] for m in ["cos_raw", "cos_cent", "l2_unit", "xpred_R2", "cka"]} for p in pairs}
        for t in TASKS:
            d = np.load(f"{OUT_ROOT}/{t}/features.npz")
            emb = {s: d[f"{s}_{src}_{feat}"] for s in S}
            for p in pairs:
                m = pair_metrics(emb[p[0]], emb[p[1]])
                for k, v in m.items():
                    agg[p][k].append(v)
        for p in pairs:
            r = {k: np.mean(v) for k, v in agg[p].items()}
            print(f"{p[0]+'-'+p[1]:<10}{r['cos_raw']:>9.3f}{r['cos_cent']:>10.3f}"
                  f"{r['l2_unit']:>9.3f}{r['xpred_R2']:>10.3f}{r['cka']:>7.3f}")


if __name__ == "__main__":
    main()
