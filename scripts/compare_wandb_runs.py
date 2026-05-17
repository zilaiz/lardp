import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


def fetch_history(api, run_path, metric):
    run = api.run(run_path)
    rows = run.history(keys=[metric], pandas=True, samples=100000)
    rows = rows.dropna(subset=[metric]).sort_values("_step").reset_index(drop=True)
    return run, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--curve",
        action="append",
        default=None,
        help="Triplet 'run_path|metric|label'. Repeatable.",
    )
    p.add_argument("--out", default="viz/compare_mean_success.png")
    args = p.parse_args()

    if not args.curve:
        args.curve = [
            "brown-palm/lbmdit/af0e355e-8aee-4c77-958f-a7df3142192d|eval/mean_success_9|DP",
            "brown-palm/lbmdit_joint/96f481bc-76ee-4f97-a72e-3ea7a1795519|eval/mean_success_50|Joint",
        ]

    api = wandb.Api()
    curves = []
    for spec in args.curve:
        run_path, metric, label = spec.split("|")
        run, hist = fetch_history(api, run_path, metric)
        curves.append((run, metric, label, hist))

    task_name = curves[0][0].config.get("task", {}).get("env_name", "unknown")
    max_step = min(c[3]["_step"].max() for c in curves)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, metric, label, hist in curves:
        h = hist[hist["_step"] <= max_step]
        ax.plot(h["_step"], h[metric], label=f"{label} ({metric})", linewidth=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean success")
    ax.set_title(f"mean success rate ({task_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    for run, metric, label, hist in curves:
        print(
            f"{label}: steps {hist['_step'].min()}–{hist['_step'].max()} ({len(hist)} pts)"
        )
    print(f"cropped to step <= {max_step}")


if __name__ == "__main__":
    main()
