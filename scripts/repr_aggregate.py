"""Aggregate repr metrics across the 4 robomimic tasks and tie to closed-loop.

Reads repr_analysis/<task>/metrics.json for can/square/transport/tool_hang,
prints a per-metric DP/EO/EP table (per task + mean), and reports how well each
representation metric's DP/EO/EP ordering agrees with the closed-loop success
ordering (ms_25, per-seed-first avg-of-last-10).
"""
from __future__ import annotations

import json

import numpy as np

OUT_ROOT = "/oscar/data/csun45/zzeng28/repo/lardp/repr_analysis"
TASKS = ["can", "square", "transport", "tool_hang"]
SETTINGS = ["DP", "EO", "EP"]

# Closed-loop success (ms_25, per-seed-first, avg-of-last-10, %), from the
# wandb aggregation session (one number per setting per task).
CLOSED_LOOP = {
    "can":       {"DP": 70.4, "EO": 73.5, "EP": 75.6},
    "square":    {"DP": 49.0, "EO": 50.6, "EP": 54.2},
    "transport": {"DP": 74.0, "EO": 74.8, "EP": 72.5},
    "tool_hang": {"DP": 29.9, "EO": 32.0, "EP": 37.6},
}

KEYS = [
    ("erank_exp", "erank_exp", +1),
    ("erank_roll", "erank_roll", +1),
    ("probe_id::next_pos(3)", "probe_id next_pos", +1),
    ("probe_id::d_state(9)", "probe_id d_state", +1),
    ("probe_id::first_action(7)", "probe_id first_action", +1),
    ("probe_rid::next_pos(3)", "probe_Rid next_pos", +1),
    ("probe_rid::d_state(9)", "probe_Rid d_state", +1),
    ("probe_rid::first_action(7)", "probe_Rid first_action", +1),
    ("fwd_r2_exp", "fwd_r2_exp", +1),
    ("fwd_r2_roll", "fwd_r2_roll", +1),
    ("maha_roll", "maha_roll", 0),
    ("knn_roll", "knn_roll", 0),
]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def main():
    M = {t: json.load(open(f"{OUT_ROOT}/{t}/metrics.json")) for t in TASKS}

    print("\n############ CLOSED-LOOP (ms_25 avg, %) ############")
    print(f"{'task':<12}{'DP':>8}{'EO':>8}{'EP':>8}   ordering")
    for t in TASKS:
        cl = CLOSED_LOOP[t]
        order = " < ".join(sorted(SETTINGS, key=lambda s: cl[s]))
        print(f"{t:<12}" + "".join(f"{cl[s]:>8.1f}" for s in SETTINGS) + f"   {order}")

    # Closed loop is dominated by TASK difficulty, so cross-task correlation is
    # the wrong lens. The question is WITHIN each task: does the metric order
    # DP/EO/EP the way closed-loop success does? Report per-task ordering
    # agreement (mean within-task Spearman over the 3 settings), plus how often
    # the metric reproduces the two closed-loop facts: EP-best (3/4 tasks) and
    # EO>=DP (4/4 tasks).
    print("\n############ REPRESENTATION METRICS (per task: DP/EO/EP, + mean) ############")
    summary = []
    for key, label, sign in KEYS:
        per_task = {t: [M[t]["settings"][s].get(key, np.nan) for s in SETTINGS] for t in TASKS}
        means = [np.mean([per_task[t][i] for t in TASKS]) for i in range(3)]
        flip = -1.0 if sign < 0 else 1.0  # lower-is-better metrics
        wt_rhos, ep_best, eo_ge_dp = [], 0, 0
        for t in TASKS:
            mt = [flip * per_task[t][i] for i in range(3)]
            cl = [CLOSED_LOOP[t][s] for s in SETTINGS]
            wt_rhos.append(spearman(mt, cl))
            if np.argmax(mt) == 2:      # EP highest under the metric
                ep_best += 1
            if mt[1] >= mt[0]:          # EO >= DP under the metric
                eo_ge_dp += 1
        wt = float(np.mean(wt_rhos))
        line = f"{label:<22}"
        for t in TASKS:
            line += " " + "/".join(f"{per_task[t][i]:.2f}" for i in range(3))
        line += f"  | mean {('/'.join(f'{m:.2f}' for m in means))}"
        line += f"  wt_rho={wt:+.2f} EPbest={ep_best}/4 EO>=DP={eo_ge_dp}/4"
        print(line)
        summary.append((label, wt, ep_best, eo_ge_dp))

    print("\n  [closed-loop reference: EP-best=3/4 (not transport), EO>=DP=4/4]")
    print("\n############ within-task closed-loop alignment (sorted) ############")
    for label, wt, eb, ed in sorted(summary, key=lambda x: -x[1]):
        print(f"  {label:<22} wt_rho={wt:+.2f}  EP-best={eb}/4  EO>=DP={ed}/4")


if __name__ == "__main__":
    main()
