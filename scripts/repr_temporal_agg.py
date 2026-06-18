"""Aggregate temporal smoothness (temporal.json) across the 4 tasks."""
import json
import numpy as np

OUT_ROOT = "/oscar/data/csun45/zzeng28/repo/lardp/repr_analysis"
TASKS = ["can", "square", "transport", "tool_hang"]
SETTINGS = ["DP", "EO", "EP"]
METRICS = [
    ("step", "lower=smaller move"),
    ("cos_consec", "higher=smoother"),
    ("accel_ratio", "lower=smoother"),
    ("latphys_r", "higher=meaningful"),
    ("jerk_frac", "lower=fewer spikes"),
]


def main():
    M = {t: json.load(open(f"{OUT_ROOT}/{t}/temporal.json")) for t in TASKS}
    for src in ["exp", "roll"]:
        print(f"\n###### {src.upper()} temporal smoothness (mean over 4 tasks) ######")
        print(f"{'metric':<14}{'hint':<20}{'DP':>9}{'EO':>9}{'EP':>9}")
        for m, hint in METRICS:
            vals = [np.mean([M[t]["settings"][s][src][m] for t in TASKS]) for s in SETTINGS]
            print(f"{m:<14}{hint:<20}" + "".join(f"{v:>9.3f}" for v in vals))


if __name__ == "__main__":
    main()
