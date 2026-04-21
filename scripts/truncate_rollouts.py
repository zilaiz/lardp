"""Truncate post-success tails in play-data rollouts.

For each task, reads image_rollouts_clipped.hdf5 and writes a new file
image_rollouts_clipped_truncate.hdf5 where:
  - successful demos (any positive reward) are truncated so the last recorded
    step is the first positive-reward step (T_dst = first_pos_idx + 1);
  - unsuccessful demos are copied unchanged.

Demo indexing and obs key set are preserved from the source file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

TASKS = ("can", "square", "transport", "tool_hang")


def truncate_file(src_path: Path, dst_path: Path) -> dict:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    n_truncated = 0
    n_unchanged_success = 0
    n_failed = 0
    total_src_steps = 0
    total_dst_steps = 0
    trunc_amounts = []

    with h5py.File(src_path, "r") as fin, h5py.File(dst_path, "w") as fout:
        data_in = fin["data"]
        n_in = int(data_in.attrs["num_demos"])
        data_out = fout.create_group("data")

        for i in range(n_in):
            demo_in = data_in[f"demo_{i}"]
            rewards = demo_in["rewards"][:]
            T_src = rewards.shape[0]
            total_src_steps += T_src

            pos_idx = np.where(rewards > 0)[0]
            if len(pos_idx) == 0:
                T_dst = T_src
                n_failed += 1
            else:
                T_dst = int(pos_idx[0]) + 1
                if T_dst < T_src:
                    n_truncated += 1
                    trunc_amounts.append(T_src - T_dst)
                else:
                    n_unchanged_success += 1

            demo_out = data_out.create_group(f"demo_{i}")
            obs_out = demo_out.create_group("obs")
            obs_in = demo_in["obs"]
            for key in obs_in.keys():
                obs_out.create_dataset(key, data=obs_in[key][:T_dst])

            demo_out.create_dataset("actions", data=demo_in["actions"][:T_dst])
            demo_out.create_dataset("rewards", data=rewards[:T_dst])
            demo_out.create_dataset("dones", data=demo_in["dones"][:T_dst])

            demo_out.attrs["num_samples"] = int(T_dst)
            total_dst_steps += T_dst

        data_out.attrs["num_demos"] = n_in
        data_out.attrs["total"] = total_dst_steps

    return {
        "num_in": n_in,
        "truncated": n_truncated,
        "unchanged_success": n_unchanged_success,
        "failed": n_failed,
        "steps_in": total_src_steps,
        "steps_out": total_dst_steps,
        "mean_trunc": float(np.mean(trunc_amounts)) if trunc_amounts else 0.0,
        "max_trunc": int(np.max(trunc_amounts)) if trunc_amounts else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("data/robomimic"),
        help="Dataset root. Looks for <root>/<task>/<src-name>.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument(
        "--src-name", default="image_rollouts_clipped.hdf5",
        help="Source filename within each task directory.",
    )
    parser.add_argument(
        "--dst-name", default="image_rollouts_clipped_truncate.hdf5",
        help="Destination filename within each task directory.",
    )
    args = parser.parse_args()

    any_missing = False
    for task in args.tasks:
        src = args.root / task / args.src_name
        dst = args.root / task / args.dst_name
        if not src.exists():
            print(f"[{task}] MISSING: {src}", file=sys.stderr)
            any_missing = True
            continue
        stats = truncate_file(src, dst)
        print(
            f"[{task}] {src.name} -> {dst.name} | "
            f"demos {stats['num_in']} "
            f"(truncated {stats['truncated']}, unchanged_success "
            f"{stats['unchanged_success']}, failed {stats['failed']}) | "
            f"steps {stats['steps_in']} -> {stats['steps_out']} "
            f"({100 * stats['steps_out'] / max(stats['steps_in'], 1):.1f}%) | "
            f"mean trunc amount {stats['mean_trunc']:.1f} "
            f"(max {stats['max_trunc']})"
        )
    if any_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
