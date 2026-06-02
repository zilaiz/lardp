"""Franka real-robot image dataset.

Companion to `examples/process_dataset/convert_franka_coffee_pod.py`, which
produces a robomimic-style HDF5:

    data/demo_i/
        actions                  (T, 10)  float32   [pos(3), rot6d(6), gripper(1)]
        obs/<cam>_image          (T, H, W, 3)  uint8
        obs/robot0_eef_pos       (T, 3)   float32
        obs/robot0_eef_quat      (T, 4)   float32   [qx, qy, qz, qw]
        obs/robot0_gripper_qpos  (T, 1)   float32

This module loads that HDF5 into a zarr-backed ReplayBuffer for fast random
access and exposes `FrankaImageDataset` for training. It deliberately does
NOT reimplement robomimic's rotation-transformer / action-conversion path:
actions are pre-expanded to their final 10-dim form at conversion time, so
the dataset just MinMax-normalizes them.
"""

import concurrent.futures
import multiprocessing
import os
from collections import defaultdict

import h5py
import numpy as np
import torch
import zarr
from loguru import logger
from tqdm import tqdm

from mip.dataset_utils import (
    ImageNormalizer,
    MinMaxNormalizer,
    QuantileNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset
from mip.datasets.imagecodecs import register_codecs

register_codecs()


class FrankaImageDataset(BaseDataset):
    """Image-observation dataset for real-robot Franka teleop.

    Args:
        dataset_path: Path to the HDF5 produced by the Franka conversion script.
        shape_meta: dict with keys "action" and "obs.<key>.{shape,type}". Only
            keys listed here are loaded from the HDF5 (supports config-time
            selection of a camera subset).
        n_obs_steps: How many obs frames to actually materialize per sample
            (optimization — the horizon window is `horizon` long but obs is
            only needed for the first `n_obs_steps` frames).
        horizon: Full action chunk length sampled per item.
        pad_before, pad_after: SequenceSampler padding at episode boundaries.
        val_dataset_percentage: Fraction of demos reserved for validation,
            split deterministically by demo index. For `mode="train"` we use
            the first `(1 - val_pct)` of demos; `mode="val"` gets the rest.
        mode: "train" or "val".
        normalizer: Optional pre-computed normalizer (e.g., to share across
            multiple datasets). If None, computed from this dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        shape_meta: dict,
        n_obs_steps: int | None = None,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        val_dataset_percentage: float = 0.0,
        mode: str = "train",
        normalizer: dict | None = None,
        delta_action_anchor: str | None = None,
        delta_action_normalizer: str = "quantile",
    ):
        super().__init__()
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode
        self.delta_action_normalizer = delta_action_normalizer

        # Parse obs keys from shape_meta
        rgb_keys: list[str] = []
        lowdim_keys: list[str] = []
        for key, attr in shape_meta["obs"].items():
            t = attr.get("type", "low_dim")
            if t == "rgb":
                rgb_keys.append(key)
            elif t == "low_dim":
                lowdim_keys.append(key)
            else:
                raise ValueError(f"Unknown obs type '{t}' for key {key!r}")

        self.replay_buffer = _convert_franka_to_replay(
            store=zarr.storage.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            val_dataset_percentage=val_dataset_percentage,
            mode=mode,
        )

        # `key_first_k` restricts obs keys to their first n_obs_steps frames
        # when pulling a sample — this avoids decoding full horizon-length
        # image slices we'd only throw away.
        key_first_k: dict[str, int] = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.delta_action_anchor = delta_action_anchor
        if self.delta_action_anchor is not None:
            if self.delta_action_anchor != "current_obs":
                raise ValueError(
                    f"Only delta_action_anchor='current_obs' is supported; "
                    f"got {self.delta_action_anchor!r}"
                )
            if self.n_obs_steps is None:
                raise ValueError(
                    "delta_action_anchor='current_obs' requires n_obs_steps "
                    "to be set (the anchor is the last obs frame)."
                )
            for k in ("robot0_eef_pos", "robot0_eef_quat"):
                if k not in self.lowdim_keys:
                    raise ValueError(
                        f"delta_action_anchor='current_obs' requires "
                        f"'{k}' in lowdim obs keys; got {self.lowdim_keys}"
                    )

        self.normalizer = normalizer if normalizer is not None else self.get_normalizer()

    def _compute_chunk_relative_deltas_for_normalizer(self) -> np.ndarray:
        """Stack delta-transformed actions across all in-episode chunks.

        Mirrors the helper in RobomimicImageDataset — used to fit the action
        normalizer on delta-distributed data when ``delta_action_anchor`` is
        set. Reads ``replay_buffer`` directly so we don't pay the per-sample
        image-decode cost during normalizer fitting.
        """
        from mip.franka_delta_transform import to_delta

        actions = np.asarray(self.replay_buffer["action"][:])           # (T, 10)
        eef_pos = np.asarray(self.replay_buffer["robot0_eef_pos"][:])   # (T, 3)
        eef_quat = np.asarray(self.replay_buffer["robot0_eef_quat"][:]) # (T, 4) xyzw
        episode_ends = np.asarray(self.replay_buffer.episode_ends[:])

        H = self.horizon
        To = self.n_obs_steps

        all_deltas: list[np.ndarray] = []
        prev_end = 0
        for ep_end in episode_ends:
            k_max = min(ep_end - H, ep_end - To) + 1
            for k in range(prev_end, k_max):
                anchor_pos = eef_pos[k + To - 1]
                anchor_quat = eef_quat[k + To - 1]
                abs_actions = actions[k : k + H]
                all_deltas.append(
                    to_delta(abs_actions, anchor_pos, anchor_quat)
                )
            prev_end = ep_end
        if not all_deltas:
            raise RuntimeError(
                "No valid chunks found for delta normalizer fitting "
                f"(H={H}, To={To}, episode_ends={episode_ends})"
            )
        return np.concatenate(all_deltas, axis=0)

    def get_normalizer(self) -> dict:
        normalizer: dict = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        if self.delta_action_anchor == "current_obs":
            delta_actions = self._compute_chunk_relative_deltas_for_normalizer()
            if getattr(self, "delta_action_normalizer", "quantile") == "quantile":
                normalizer["action"] = QuantileNormalizer(delta_actions)
            else:
                normalizer["action"] = MinMaxNormalizer(delta_actions)
        else:
            normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def __str__(self) -> str:
        return (
            f"FrankaImageDataset(mode={self.mode}, "
            f"keys={list(self.replay_buffer.keys())}, "
            f"steps={self.replay_buffer.n_steps}, "
            f"episodes={self.replay_buffer.n_episodes})"
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict:
        sample = self.sampler.sample_sequence(idx)

        # Only use the first n_obs_steps frames of obs (the rest is NaN-filled
        # by the sampler's key_first_k optimization).
        T_slice = slice(self.n_obs_steps)

        obs_dict: dict[str, np.ndarray] = {}
        for key in self.rgb_keys:
            # (T, H, W, C) uint8 -> (T, C, H, W) float32 in [0, 1]
            img = np.moveaxis(sample[key][T_slice], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = self.normalizer["obs"][key].normalize(img)
            del sample[key]

        # Capture raw (un-normalized) anchor BEFORE the lowdim loop deletes
        # robot0_eef_pos / quat. The captured arrays are used by the delta
        # transform below.
        anchor_pos = None
        anchor_quat = None
        if self.delta_action_anchor == "current_obs":
            anchor_pos = sample["robot0_eef_pos"][self.n_obs_steps - 1].astype(np.float32)
            anchor_quat = sample["robot0_eef_quat"][self.n_obs_steps - 1].astype(np.float32)

        for key in self.lowdim_keys:
            arr = sample[key][T_slice].astype(np.float32)
            obs_dict[key] = self.normalizer["obs"][key].normalize(arr)
            del sample[key]

        action = sample["action"].astype(np.float32)
        if self.delta_action_anchor == "current_obs":
            from mip.franka_delta_transform import to_delta
            action = to_delta(action, anchor_pos, anchor_quat)
        action = self.normalizer["action"].normalize(action)

        return {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
        }


def _convert_franka_to_replay(
    store,
    shape_meta: dict,
    dataset_path: str,
    val_dataset_percentage: float = 0.0,
    mode: str = "train",
    n_workers: int | None = None,
    max_inflight_tasks: int | None = None,
) -> ReplayBuffer:
    """Load the Franka HDF5 into a zarr-backed ReplayBuffer.

    Compared to robomimic's loader, this is trimmed to what the Franka
    pipeline actually needs: no state key, no mask group, no action conversion,
    no rotation transformer, no reward-based filtering. Images are copied into
    zarr via a thread pool (same idea as the robomimic loader).
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # Parse obs keys from shape_meta
    rgb_keys: list[str] = []
    lowdim_keys: list[str] = []
    for key, attr in shape_meta["obs"].items():
        t = attr.get("type", "low_dim")
        if t == "rgb":
            rgb_keys.append(key)
        elif t == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(os.path.expanduser(dataset_path), "r") as file:
        demos = file["data"]
        total_demos = len(demos)

        # Deterministic train/val split by demo index
        if val_dataset_percentage > 0.0:
            val_count = int(total_demos * val_dataset_percentage)
            train_count = total_demos - val_count
            if mode == "train":
                demo_indices = list(range(train_count))
            elif mode == "val":
                demo_indices = list(range(train_count, total_demos))
            else:
                raise ValueError(f"Invalid mode: {mode!r}")
        else:
            demo_indices = list(range(total_demos))
        logger.info(
            f"FrankaDataset[{mode}]: using {len(demo_indices)}/{total_demos} demos"
        )

        # Episode-end indices (cumulative)
        episode_ends: list[int] = []
        prev_end = 0
        for i in demo_indices:
            length = demos[f"demo_{i}"]["actions"].shape[0]
            prev_end += length
            episode_ends.append(prev_end)
        n_steps = episode_ends[-1] if episode_ends else 0
        episode_starts = [0] + episode_ends[:-1]
        meta_group.create_array(
            name="episode_ends",
            data=np.array(episode_ends, dtype=np.int64),
            compressor=None,
            overwrite=True,
        )

        # ---- Low-dim keys + action: concat across demos into dense arrays ----
        for key in tqdm(lowdim_keys + ["action"], desc=f"Loading {mode} lowdim"):
            src_path = f"obs/{key}" if key != "action" else "actions"
            chunks: list[np.ndarray] = []
            for i in demo_indices:
                chunks.append(demos[f"demo_{i}"][src_path][:].astype(np.float32))
            arr = np.concatenate(chunks, axis=0) if chunks else np.empty(0, dtype=np.float32)
            expected_shape = tuple(
                shape_meta["action"]["shape"] if key == "action"
                else shape_meta["obs"][key]["shape"]
            )
            assert arr.shape == (n_steps,) + expected_shape, (
                f"{key}: got {arr.shape}, expected ({n_steps},)+{expected_shape}"
            )
            data_group.create_array(
                name=key, data=arr, chunks=arr.shape,
                compressor=None, overwrite=True,
            )

        # ---- RGB keys: threaded per-frame copy into zarr (T, H, W, C) uint8 ----
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]  # confirm decode round-trip
                return True
            except Exception:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys),
            desc=f"Loading {mode} image data",
            mininterval=1.0,
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures: set[concurrent.futures.Future] = set()
                for key in rgb_keys:
                    c, h, w = tuple(shape_meta["obs"][key]["shape"])
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=None,
                        dtype=np.uint8,
                    )
                    for demo_list_idx, episode_idx in enumerate(demo_indices):
                        hdf5_arr = demos[f"demo_{episode_idx}"]["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                done, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for fut in done:
                                    if not fut.result():
                                        raise RuntimeError(f"Failed to copy {key}")
                                pbar.update(len(done))
                            zarr_idx = episode_starts[demo_list_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                done, futures = concurrent.futures.wait(futures)
                for fut in done:
                    if not fut.result():
                        raise RuntimeError("Failed to copy image (final flush)")
                pbar.update(len(done))

    return ReplayBuffer(root)


def make_franka_dataset(task_config, mode: str = "train") -> FrankaImageDataset:
    """Factory: build a FrankaImageDataset from a Hydra/OmegaConf task config."""
    dataset_path = task_config.dataset_path
    if dataset_path is None:
        raise ValueError("task.dataset_path must be provided for Franka training")
    dataset_path = os.path.expanduser(dataset_path)
    logger.info(f"Loading Franka dataset from {dataset_path}")

    return FrankaImageDataset(
        dataset_path=dataset_path,
        shape_meta=task_config.shape_meta,
        n_obs_steps=task_config.obs_steps,
        horizon=task_config.horizon,
        pad_before=task_config.obs_steps - 1,
        pad_after=task_config.act_steps - 1,
        val_dataset_percentage=task_config.val_dataset_percentage,
        mode=mode,
        delta_action_anchor=getattr(task_config, "delta_action_anchor", None),
        delta_action_normalizer=getattr(
            task_config, "delta_action_normalizer", "quantile"
        ),
    )
