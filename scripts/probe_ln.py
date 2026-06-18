"""Robustness check: re-run the key probes on POST-LN representations (EO/EP).

Mirrors the expert action-chunk + physical-next-state MLP probes from
repr_probe2, but on the `_ln` features (target_ln applied) instead of `_raw`.
DP has no LN so `_ln == _raw`. If conclusions match the raw version, the
before-vs-after-LN choice doesn't matter.
"""
import numpy as np
from scripts.repr_probe2 import mlp_r2, split

OUT_ROOT = "/oscar/data/csun45/zzeng28/repo/lardp/repr_analysis"
TASKS = ["can", "square", "transport", "tool_hang"]
SETTINGS = ["DP", "EO", "EP"]

print("POST-LN expert probes (MLP R^2): action chunk / next-state   [raw in brackets]")
print(f"{'task':<11}{'DP':>18}{'EO':>18}{'EP':>18}")
for t in TASKS:
    d = np.load(f"{OUT_ROOT}/{t}/features.npz")
    n = len(d["exp_cur"]); tr, va = split(n, 0)
    chunk = d["exp_act"].reshape(n, -1); nxt = d["exp_nxt"]
    row = f"{t:<11}"
    for s in SETTINGS:
        for feat in ["ln", "raw"]:
            H = d[f"{s}_exp_{feat}"]
            H = (H - H.mean(0)) / (H.std(0) + 1e-6)
            if feat == "ln":
                a_ln = mlp_r2(H[tr], chunk[tr], H[va], chunk[va])
                n_ln = mlp_r2(H[tr], nxt[tr], H[va], nxt[va])
            else:
                a_raw = mlp_r2(H[tr], chunk[tr], H[va], chunk[va])
                n_raw = mlp_r2(H[tr], nxt[tr], H[va], nxt[va])
        row += f"  {a_ln:.2f}/{n_ln:.2f}[{a_raw:.2f}/{n_raw:.2f}]"
    print(row)
