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

from mip.dataset_utils import MinMaxNormalizer, QuantileNormalizer


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

    # When delta_action_anchor='current_obs' the dataset transforms raw
    # 10-dim absolute actions to 7-dim chunk-relative deltas at load time
    # and refits the action normalizer on those deltas. The deploy-time
    # stats must match — fit MinMax on transformed deltas instead of raw.
    delta_anchor = getattr(task, "delta_action_anchor", None)
    if delta_anchor is not None and delta_anchor != "current_obs":
        raise ValueError(
            f"Only delta_action_anchor='current_obs' is supported; "
            f"got {delta_anchor!r}"
        )
    # Match the run's training-time action normalizer. Default "minmax": a
    # saved config WITHOUT this field is a pre-quantile run (trained MinMax),
    # so exporting MinMax keeps deploy stats consistent with the model. New
    # delta runs record delta_action_normalizer="quantile" in their config.
    action_norm_type = getattr(task, "delta_action_normalizer", "minmax")

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

        if delta_anchor == "current_obs":
            # Mirrors FrankaImageDataset._compute_chunk_relative_deltas_for_normalizer:
            # iterate over valid chunk starts per episode, anchor at the last obs
            # frame (k + To - 1), call to_delta, then MinMax-fit on the concat.
            from mip.franka_delta_transform import to_delta

            H = int(task.horizon)
            To = int(task.obs_steps)
            actions_raw = _concat("actions")                          # (T, 10)
            eef_pos_raw = _concat("obs/robot0_eef_pos")               # (T, 3)
            eef_quat_raw = _concat("obs/robot0_eef_quat")             # (T, 4) xyzw

            # episode_ends as cumulative offsets across the train demos only.
            ep_lengths = [
                int(demos[f"demo_{i}"]["actions"].shape[0])
                for i in demo_indices
            ]
            episode_ends = np.cumsum(ep_lengths).tolist()

            all_deltas: list[np.ndarray] = []
            prev_end = 0
            for ep_end in episode_ends:
                k_max = min(ep_end - H, ep_end - To) + 1
                for k in range(prev_end, k_max):
                    anchor_pos = eef_pos_raw[k + To - 1]
                    anchor_quat = eef_quat_raw[k + To - 1]
                    all_deltas.append(
                        to_delta(actions_raw[k : k + H], anchor_pos, anchor_quat)
                    )
                prev_end = ep_end
            if not all_deltas:
                raise RuntimeError(
                    f"No valid chunks for delta normalizer "
                    f"(H={H}, To={To}, n_eps={len(episode_ends)})"
                )
            action = np.concatenate(all_deltas, axis=0)               # (N, 7)
            print(f"  action mode=delta (H={H}, To={To}): "
                  f"transformed {actions_raw.shape[0]} raw steps -> "
                  f"{action.shape[0]} delta chunks * H rows")
        else:
            action = _concat("actions")
            print(f"  action mode=absolute: shape={action.shape}")

        # Absolute actions always use MinMax (full-range channels). Delta
        # actions honor the run's configured normalizer (quantile by default
        # for new runs; see action_norm_type above).
        if delta_anchor == "current_obs" and action_norm_type == "quantile":
            action_norm = QuantileNormalizer(action)
            print("  action normalizer=quantile (q01/q99)")
        else:
            action_norm = MinMaxNormalizer(action)
            print("  action normalizer=minmax")
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
