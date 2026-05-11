"""Export the action + low-dim observation normalizer for a Franka run.

Reads the train-time HDF5 + shape_meta from a hydra-saved config and writes
a small .npz file containing per-key MinMax statistics:

    action_min, action_max, action_range
    obs__<key>_min, obs__<key>_max, obs__<key>_range
    keys (string array of low-dim obs keys, in deterministic order)

Loaded by `interface_example.py` so the inference server can skip the HDF5
scan and start instantly.

Usage:
    python scripts/export_franka_normalizer.py \
        --config_path outputs/2026-04-21/02-56-39/.hydra/config.yaml \
        --out checkpoints/franka_coffee_pod_cog_lbmdit_normalizer.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from mip.dataset_utils import MinMaxNormalizer


def build_normalizer_arrays(config_path: str) -> dict:
    cfg = OmegaConf.load(config_path)
    task = cfg.task

    dataset_path = Path(str(task.dataset_path)).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"task.dataset_path missing: {dataset_path}")

    lowdim_keys: list[str] = sorted(
        k for k, v in task.shape_meta["obs"].items()
        if v.get("type", "low_dim") == "low_dim"
    )
    val_pct = float(getattr(task, "val_dataset_percentage", 0.0))

    print(f"Reading: {dataset_path}")
    with h5py.File(str(dataset_path), "r") as f:
        demos = f["data"]
        total = len(demos)
        train_count = total - int(total * val_pct) if val_pct > 0 else total
        demo_indices = list(range(train_count))
        print(f"  demos: train={train_count}/{total} (val_pct={val_pct})")

        def _concat(group_path: str) -> np.ndarray:
            chunks = [
                demos[f"demo_{i}"][group_path][:].astype(np.float32)
                for i in demo_indices
            ]
            return np.concatenate(chunks, axis=0)

        action = _concat("actions")
        action_norm = MinMaxNormalizer(action)
        print(f"  action: shape={action.shape} "
              f"min={action_norm.min} max={action_norm.max}")

        out = {
            "action_min":   action_norm.min.astype(np.float32),
            "action_max":   action_norm.max.astype(np.float32),
            "action_range": action_norm.range.astype(np.float32),
            "keys":         np.array(lowdim_keys, dtype=object),
        }
        for key in lowdim_keys:
            arr = _concat(f"obs/{key}")
            n = MinMaxNormalizer(arr)
            out[f"obs__{key}_min"]   = n.min.astype(np.float32)
            out[f"obs__{key}_max"]   = n.max.astype(np.float32)
            out[f"obs__{key}_range"] = n.range.astype(np.float32)
            print(f"  obs/{key}: shape={arr.shape} min={n.min} max={n.max}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the run's .hydra/config.yaml")
    parser.add_argument("--out", type=str, required=True,
                        help="Where to write the .npz")
    args = parser.parse_args()

    arrays = build_normalizer_arrays(args.config_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"\nWrote normalizer to {out_path}")


if __name__ == "__main__":
    main()
