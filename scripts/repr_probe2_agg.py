"""Aggregate the expanded probe suite (probes2.json) across the 4 tasks.

Prints, per probe and per eval source, the DP/EO/EP mean over tasks (linear and
MLP), the max pairwise gap (how similar/different the three reps are), and which
setting wins. Separates CLEAN targets (optimal-action / physical-state) from
CONFOUNDED ones (suboptimal-action / behavior-transition / latent-self-consistency).
"""
from __future__ import annotations

import json

import numpy as np

OUT_ROOT = "/oscar/data/csun45/zzeng28/repo/lardp/repr_analysis"
TASKS = ["can", "square", "transport", "tool_hang"]
SETTINGS = ["DP", "EO", "EP"]
PROBES = ["actdec_first", "actdec_chunk", "fwd_goalemb", "antic_nxt", "fwd_nxt", "fwd_dstate"]
# what each probe means for decision quality on each source
CLEAN_ON_EXPERT = {"actdec_first", "actdec_chunk", "antic_nxt", "fwd_nxt", "fwd_dstate"}
# on rollout, physical-state targets are clean; action/transition/latent are confounded
CLEAN_ON_ROLL = {"antic_nxt", "fwd_nxt"}


def main():
    M = {t: json.load(open(f"{OUT_ROOT}/{t}/probes2.json")) for t in TASKS}
    for src in ["exp", "roll"]:
        clean = CLEAN_ON_EXPERT if src == "exp" else CLEAN_ON_ROLL
        print(f"\n################ {src.upper()} EVAL  (mean over 4 tasks; lin / mlp) ################")
        print(f"{'probe':<14}{'tag':<8}{'DP':>13}{'EO':>13}{'EP':>13}   maxgap(mlp)")
        for nm in PROBES:
            for kind in ["lin", "mlp"]:
                vals = [np.mean([M[t]["settings"][s][f"{src}/{nm}/{kind}"] for t in TASKS])
                        for s in SETTINGS]
                if kind == "lin":
                    lin_vals = vals
                    continue
                tag = "clean" if nm in clean else "CONF"
                gap = max(vals) - min(vals)
                row = f"{nm:<14}{tag:<8}"
                for i, s in enumerate(SETTINGS):
                    row += f"  {lin_vals[i]:>4.2f}/{vals[i]:>4.2f} "
                row += f"   {gap:+.2f}"
                print(row)


if __name__ == "__main__":
    main()
