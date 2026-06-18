"""PushT dataset.

Author: Chaoyi Pan
Date: 2025-10-15
"""

import os
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from mip.dataset_utils import (
    ImageNormalizer,
    MinMaxNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset

# ---------------------------------------------------------------------------
# Demo-subset helpers (train/val split + rollout mixing)
# ---------------------------------------------------------------------------


def _split_demo_indices(n_episodes, val_dataset_percentage, mode):
    """Deterministic train/val demo split, matching the robomimic convention.

    With ``val_dataset_percentage = p > 0`` the last ``int(n * p)`` demos are
    held out for validation and the first ``n - int(n * p)`` are used for
    training. This is the knob that controls how much expert data the policy
    actually sees (e.g. ``p = 0.9`` -> train on the first 10% of demos). ``p =
    0`` uses every demo.
    """
    if val_dataset_percentage and val_dataset_percentage > 0.0:
        val_count = int(n_episodes * val_dataset_percentage)
        train_count = n_episodes - val_count
        if mode == "train":
            return list(range(train_count))
        elif mode == "val":
            return list(range(train_count, n_episodes))
        raise ValueError(f"Invalid mode: {mode!r}. Must be 'train' or 'val'.")
    return list(range(n_episodes))


def _subset_replay_buffer(replay_buffer, demo_indices):
    """Return a new in-memory (numpy) ReplayBuffer with only ``demo_indices``."""
    subset = ReplayBuffer.create_empty_numpy()
    for i in demo_indices:
        subset.add_episode(replay_buffer.get_episode(i, copy=True))
    return subset


def _maybe_split_replay_buffer(replay_buffer, val_dataset_percentage, mode):
    """Subset a replay buffer to its train/val demos; no-op when using all."""
    n = replay_buffer.n_episodes
    demo_indices = _split_demo_indices(n, val_dataset_percentage, mode)
    if len(demo_indices) == n:
        return replay_buffer
    logger.info(
        f"PushT demo split (mode={mode}, val_pct={val_dataset_percentage}): "
        f"using {len(demo_indices)}/{n} demos"
    )
    return _subset_replay_buffer(replay_buffer, demo_indices)


def _take_rollout_fraction(replay_buffer, use_fraction):
    """Keep the first round(n * use_fraction) rollout demos (collection order).

    Fraction KEPT (inverse of val_dataset_percentage's held-out semantics).
    1.0 / None -> use all. Lets you control the play:expert ratio without
    pre-slicing the HDF5.
    """
    if use_fraction is None or use_fraction >= 1.0:
        return replay_buffer
    if not (0.0 < use_fraction < 1.0):
        raise ValueError(
            f"rollout_use_fraction must be in (0, 1]; got {use_fraction}"
        )
    n = replay_buffer.n_episodes
    keep = max(1, int(round(n * use_fraction)))
    if keep >= n:
        return replay_buffer
    logger.info(
        f"Rollout subsample: using first {keep}/{n} demos "
        f"(rollout_use_fraction={use_fraction})"
    )
    return _subset_replay_buffer(replay_buffer, list(range(keep)))


def load_pusht_rollout_replay_buffer(hdf5_path):
    """Load a RolloutRecorder HDF5 (robomimic format) into a PushT ReplayBuffer.

    The recorder stores, per demo, ``obs/{image, agent_pos}``, ``actions``,
    ``rewards`` and ``dones`` (images already HWC uint8). We map these onto the
    PushT zarr schema consumed by ``PushTImageDataset``: ``image -> 'img'``
    (T, H, W, C uint8), ``agent_pos -> 'state'`` (T, 2; the dataset only ever
    reads ``state[:, :2]``), ``actions -> 'action'`` (raw, unnormalized — same
    space as the expert zarr actions).
    """
    rb = ReplayBuffer.create_empty_numpy()
    with h5py.File(os.path.expanduser(hdf5_path), "r") as f:
        if "data" not in f:
            raise ValueError(f"No 'data' group in rollout HDF5: {hdf5_path}")
        data = f["data"]
        n_demos = int(data.attrs.get("num_demos", len(data.keys())))
        for i in range(n_demos):
            key = f"demo_{i}"
            if key not in data:
                continue
            demo = data[key]
            obs = demo["obs"]
            img = obs["image"][:]  # (T, H, W, C) uint8
            agent_pos = obs["agent_pos"][:].astype(np.float32)  # (T, 2)
            action = demo["actions"][:].astype(np.float32)  # (T, 2)
            rb.add_episode({"img": img, "state": agent_pos, "action": action})
    logger.info(
        f"Loaded {rb.n_episodes} rollout demos ({rb.n_steps} steps) "
        f"from {hdf5_path}"
    )
    return rb


class PushTMixedDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset over PushT sources that exposes the primary normalizer.

    Mirrors ``MultiImageDataset`` on the robomimic side: eval reads
    ``dataset.normalizer``, so the concat surfaces the primary (expert)
    dataset's normalizer, which every source already shares.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.normalizer = datasets[0].normalizer


def download_pusht_dataset(
    dataset_filename: str = "pusht/pusht_cchi_v7_replay.zarr.zip",
    repo_id: str = "ChaoyiPan/mip-dataset",
) -> str:
    """Download PushT dataset from HuggingFace and extract locally.

    Args:
        dataset_filename: Filename in the HuggingFace dataset repo (should be a .zip file)
        repo_id: HuggingFace repository ID

    Returns:
        Path to the extracted zarr dataset
    """
    logger.info(f"Downloading PushT dataset from {repo_id}/{dataset_filename}")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=dataset_filename,
        repo_type="dataset",
    )
    logger.info(f"Downloaded zip file to: {zip_path}")

    # Extract the zip file in the same directory
    zip_path_obj = Path(zip_path)
    extract_dir = zip_path_obj.parent

    logger.info(f"Extracting dataset to {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Find the extracted zarr directory
    # Assuming the zip contains a single .zarr directory
    zarr_name = zip_path_obj.stem  # Remove .zip extension
    if zarr_name.endswith(".zarr"):
        zarr_path = extract_dir / zarr_name
    else:
        # Look for .zarr directories
        zarr_dirs = list(extract_dir.glob("*.zarr"))
        if not zarr_dirs:
            raise FileNotFoundError(
                f"No .zarr directory found after extracting {zip_path}"
            )
        zarr_path = zarr_dirs[0]

    logger.info(f"Extracted dataset to: {zarr_path}")
    return str(zarr_path)


def _resolve_pusht_dataset_path(task_config):
    """Resolve the expert source path. A local ``dataset_path`` wins and is read
    directly — no HuggingFace lookup. We check truthiness (not hasattr): the
    fields always exist on the dataclass, so hasattr is always True and can't
    distinguish "set" from "null". Only fall back to the HF repo when no local
    path is given.
    """
    local_path = getattr(task_config, "dataset_path", None)
    repo = getattr(task_config, "dataset_repo", None)
    filename = getattr(task_config, "dataset_filename", None)
    if local_path:
        dataset_path = os.path.expanduser(local_path)
        logger.info(f"Loading PushT dataset from local path: {dataset_path}")
        return dataset_path
    if repo and filename:
        # Auto-download from HuggingFace (handles zip extraction)
        if filename.endswith(".zip"):
            return download_pusht_dataset(dataset_filename=filename, repo_id=repo)
        # Direct download (backward compatibility for non-zip files)
        logger.info(f"Downloading dataset from {repo}/{filename}")
        dataset_path = hf_hub_download(
            repo_id=repo, filename=filename, repo_type="dataset",
        )
        logger.info(f"Downloaded dataset to: {dataset_path}")
        return dataset_path
    raise ValueError(
        "Either dataset_path (local) or dataset_repo/dataset_filename "
        "(HuggingFace) must be provided"
    )


def make_dataset(task_config, mode="train"):
    """Create PushT dataset based on configuration.

    Args:
        task_config: Task configuration with dataset parameters
        mode: Dataset mode ("train"/"val") for the expert train/val demo split

    Returns:
        Dataset instance for PushT task
    """
    dataset_path = _resolve_pusht_dataset_path(task_config)

    val_pct = getattr(task_config, "val_dataset_percentage", 0.0)

    if task_config.obs_type == "state":
        return PushTStateDataset(
            dataset_path=dataset_path,
            horizon=task_config.horizon,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
            val_dataset_percentage=val_pct,
            mode=mode,
        )
    elif task_config.obs_type == "keypoint":
        return PushTKeypointDataset(
            dataset_path=dataset_path,
            horizon=task_config.horizon,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
            val_dataset_percentage=val_pct,
            mode=mode,
        )
    elif task_config.obs_type == "image":
        # Primary (expert) dataset from the zarr, with the train/val demo split
        # controlling how much expert data is used.
        expert_ds = PushTImageDataset(
            dataset_path=dataset_path,
            shape_meta=task_config.shape_meta,
            n_obs_steps=task_config.obs_steps,
            horizon=task_config.horizon,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
            val_dataset_percentage=val_pct,
            mode=mode,
            optimality_label=0,
        )

        rollout_paths = getattr(task_config, "rollout_dataset_paths", None)
        if not rollout_paths:
            return expert_ds

        # Mixed dataset: expert (zarr) + collected rollouts (HDF5). Rollouts
        # share the expert normalizer (their actions/obs live in the same raw
        # space); rollout_use_fraction controls how many rollout demos are kept
        # (the val split is expert-only). They are tagged optimality=1
        # (null/play); the expert is 0.
        roll_frac = getattr(task_config, "rollout_use_fraction", 1.0)
        datasets = [expert_ds]
        for path in rollout_paths:
            rollout_rb = _take_rollout_fraction(
                load_pusht_rollout_replay_buffer(path), roll_frac
            )
            datasets.append(
                PushTImageDataset(
                    replay_buffer=rollout_rb,
                    shape_meta=task_config.shape_meta,
                    n_obs_steps=task_config.obs_steps,
                    horizon=task_config.horizon,
                    pad_before=task_config.obs_steps - 1,
                    pad_after=task_config.act_steps - 1,
                    normalizer=expert_ds.normalizer,
                    optimality_label=1,
                )
            )
        logger.info(
            "PushT mixed dataset sample counts "
            f"(expert + rollouts): {[len(d) for d in datasets]}"
        )
        return PushTMixedDataset(datasets)
    else:
        raise ValueError(f"Invalid observation type: {task_config.obs_type}")


class PushTStateDataset(BaseDataset):
    """PushT dataset with state observations.

    State observation contains: [agent_x, agent_y, block_x, block_y, block_angle]
    """

    def __init__(
        self,
        dataset_path,
        obs_keys=("state", "action"),
        horizon=1,
        pad_before=0,
        pad_after=0,
        val_dataset_percentage=0.0,
        mode="train",
    ):
        super().__init__()
        replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=obs_keys)
        self.replay_buffer = _maybe_split_replay_buffer(
            replay_buffer, val_dataset_percentage, mode
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.normalizer = self.get_normalizer()

    def get_normalizer(self, **kwargs):
        state_normalizer = MinMaxNormalizer(self.replay_buffer["state"][:])
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        return {
            "obs": {"state": state_normalizer},
            "action": action_normalizer,
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        state = sample["state"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        data = {
            "obs": {"state": state},
            "action": action,
        }
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


class PushTKeypointDataset(BaseDataset):
    """PushT dataset with keypoint observations.

    Keypoint observation contains 9 keypoints (18 values) plus agent position (2 values).
    """

    def __init__(
        self,
        dataset_path,
        obs_keys=("keypoint", "state", "action"),
        horizon=1,
        pad_before=0,
        pad_after=0,
        val_dataset_percentage=0.0,
        mode="train",
    ):
        super().__init__()
        replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=obs_keys)
        self.replay_buffer = _maybe_split_replay_buffer(
            replay_buffer, val_dataset_percentage, mode
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.normalizer = self.get_normalizer()

    def get_normalizer(self, **kwargs):
        agent_pos_normalizer = MinMaxNormalizer(self.replay_buffer["state"][:, :2])
        keypoint_normalizer = MinMaxNormalizer(self.replay_buffer["keypoint"][:])
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        return {
            "obs": {
                "keypoint": keypoint_normalizer,
                "agent_pos": agent_pos_normalizer,
            },
            "action": action_normalizer,
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # keypoint: (T, 9, 2) -> (T, 18)
        data_size = sample["keypoint"].shape[0]
        keypoint = (
            sample["keypoint"]
            .reshape(-1, sample["keypoint"].shape[-1])
            .astype(np.float32)
        )
        keypoint = self.normalizer["obs"]["keypoint"].normalize(keypoint)
        keypoint = keypoint.reshape(data_size, -1)

        # agent_pos: (T, 2)
        agent_pos = sample["state"][:, :2].astype(np.float32)
        agent_pos = self.normalizer["obs"]["agent_pos"].normalize(agent_pos)

        # action: (T, 2)
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        data = {
            "obs": {
                "keypoint": keypoint,
                "agent_pos": agent_pos,
            },
            "action": action,
        }
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


class PushTImageDataset(BaseDataset):
    """PushT dataset with image observations.

    Image observation contains RGB image plus agent position.
    """

    def __init__(
        self,
        dataset_path=None,
        shape_meta: dict = None,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        replay_buffer=None,
        normalizer=None,
        val_dataset_percentage=0.0,
        mode="train",
        optimality_label=0,
    ):
        super().__init__()
        # Source is either a zarr path (expert) or a prebuilt replay buffer
        # (e.g. rollouts loaded from HDF5 via load_pusht_rollout_replay_buffer).
        if replay_buffer is None:
            replay_buffer = ReplayBuffer.copy_from_path(
                dataset_path, keys=["img", "state", "action"]
            )
        self.replay_buffer = _maybe_split_replay_buffer(
            replay_buffer, val_dataset_percentage, mode
        )
        self.optimality_label = int(optimality_label)

        # Parse shape_meta to get rgb and lowdim keys
        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                rgb_keys.append(key)
            elif type_ == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            # only take first k obs from images
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

        # A shared normalizer (e.g. the expert's, passed to rollout sources in
        # a mixed dataset) takes precedence over computing a fresh one.
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = self.get_normalizer()

    def get_normalizer(self, **kwargs):
        normalizer = defaultdict(dict)
        # For PushT, we map 'img' to 'image' and state[:,:2] to 'agent_pos'
        normalizer["obs"]["image"] = ImageNormalizer()
        normalizer["obs"]["agent_pos"] = MinMaxNormalizer(
            self.replay_buffer["state"][..., :2]
        )
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict = {}

        # Image: (T, H, W, C) -> (T, C, H, W)
        image = np.moveaxis(sample["img"][T_slice], -1, 1).astype(np.float32) / 255.0
        del sample["img"]
        obs_dict["image"] = self.normalizer["obs"]["image"].normalize(image)

        # Agent position: (T, 2)
        agent_pos = sample["state"][T_slice, :2].astype(np.float32)
        obs_dict["agent_pos"] = self.normalizer["obs"]["agent_pos"].normalize(agent_pos)

        # Action: (T, 2)
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
            # Per-sample optimality tag (0=expert, 1=rollout/play). Ignored by
            # the vanilla DP loop; carried for CFG-aware downstream consumers.
            "optimality": torch.tensor(self.optimality_label, dtype=torch.long),
        }
        return torch_data


class PushTImageGoalDataset(BaseDataset):
    """PushT image dataset that also returns a goal observation.

    Used by the joint state+action pipeline (LBMDiTJointPTAgent). Each sample
    carries the current obs (first ``n_obs_steps`` frames), the goal obs (the
    single frame at index ``horizon`` — i.e. right after the action chunk, the
    next-state target the FM state head regresses), the action chunk (first
    ``horizon`` frames), and an optimality label.

    Differs from PushTImageDataset: ``sequence_length = horizon + 1`` and no
    ``key_first_k`` truncation, since we need real frames at both
    ``[0:n_obs_steps]`` and ``[horizon]``. Mirrors RobomimicImageIDMDataset but
    without the (unused-by-the-joint-trunk) intermediate frame.
    """

    def __init__(
        self,
        dataset_path=None,
        shape_meta: dict = None,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        replay_buffer=None,
        normalizer=None,
        val_dataset_percentage=0.0,
        mode="train",
        optimality_label=0,
    ):
        super().__init__()
        if replay_buffer is None:
            replay_buffer = ReplayBuffer.copy_from_path(
                dataset_path, keys=["img", "state", "action"]
            )
        self.replay_buffer = _maybe_split_replay_buffer(
            replay_buffer, val_dataset_percentage, mode
        )
        self.optimality_label = int(optimality_label)

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                rgb_keys.append(key)
            elif type_ == "low_dim":
                lowdim_keys.append(key)

        assert horizon > n_obs_steps, (
            "goal dataset needs horizon > n_obs_steps so the goal frame at "
            f"index horizon lies after the obs window; got horizon={horizon}, "
            f"n_obs_steps={n_obs_steps}"
        )
        # No key_first_k — need frames at [0:n_obs_steps] AND [horizon]. The
        # extra +1 on sequence_length / pad_after exposes the goal frame.
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon + 1,
            pad_before=pad_before,
            pad_after=pad_after + 1,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = self.get_normalizer()

    def get_normalizer(self, **kwargs):
        normalizer = defaultdict(dict)
        normalizer["obs"]["image"] = ImageNormalizer()
        normalizer["obs"]["agent_pos"] = MinMaxNormalizer(
            self.replay_buffer["state"][..., :2]
        )
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def _img_frames(self, arr):
        # (T, H, W, C) uint8/float -> (T, C, H, W) float in [0, 1], normalized.
        img = np.moveaxis(arr, -1, 1).astype(np.float32) / 255.0
        return self.normalizer["obs"]["image"].normalize(img)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # Current obs: first n_obs_steps frames.
        obs_dict = {
            "image": self._img_frames(sample["img"][: self.n_obs_steps]),
            "agent_pos": self.normalizer["obs"]["agent_pos"].normalize(
                sample["state"][: self.n_obs_steps, :2].astype(np.float32)
            ),
        }
        # Goal obs: single frame at index horizon (right after the action chunk).
        goal_dict = {
            "image": self._img_frames(
                sample["img"][self.horizon : self.horizon + 1]
            ),
            "agent_pos": self.normalizer["obs"]["agent_pos"].normalize(
                sample["state"][self.horizon : self.horizon + 1, :2].astype(
                    np.float32
                )
            ),
        }
        # Action: first horizon frames.
        action = sample["action"][: self.horizon].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        return {
            "obs": dict_apply(obs_dict, torch.tensor),
            "goal_obs": dict_apply(goal_dict, torch.tensor),
            "action": torch.tensor(action),
            "optimality": torch.tensor(self.optimality_label, dtype=torch.long),
        }


def make_pusht_goal_dataset(task_config, mode="train"):
    """Goal-providing PushT dataset for the joint state+action pipeline.

    Expert demos come from the local zarr (optimality=0). Optional collected
    rollouts (``task_config.rollout_dataset_paths``, robomimic-format HDF5) are
    mixed in as optimality=1, sharing the expert normalizer and using all their
    demos (the val split is expert-only). Returns a single PushTImageGoalDataset
    when there are no rollouts, else a PushTMixedDataset. Image obs only.
    """
    if task_config.obs_type != "image":
        raise NotImplementedError(
            "make_pusht_goal_dataset only supports image obs "
            f"(got {task_config.obs_type}); the joint pipeline is image-only."
        )

    dataset_path = _resolve_pusht_dataset_path(task_config)
    val_pct = getattr(task_config, "val_dataset_percentage", 0.0)
    common = {
        "shape_meta": task_config.shape_meta,
        "n_obs_steps": task_config.obs_steps,
        "horizon": task_config.horizon,
        "pad_before": task_config.obs_steps - 1,
        "pad_after": task_config.act_steps - 1,
    }

    expert_ds = PushTImageGoalDataset(
        dataset_path=dataset_path,
        val_dataset_percentage=val_pct,
        mode=mode,
        optimality_label=0,
        **common,
    )

    rollout_paths = getattr(task_config, "rollout_dataset_paths", None)
    if not rollout_paths:
        return expert_ds

    roll_frac = getattr(task_config, "rollout_use_fraction", 1.0)
    datasets = [expert_ds]
    for path in rollout_paths:
        rollout_rb = _take_rollout_fraction(
            load_pusht_rollout_replay_buffer(path), roll_frac
        )
        datasets.append(
            PushTImageGoalDataset(
                replay_buffer=rollout_rb,
                normalizer=expert_ds.normalizer,
                optimality_label=1,
                **common,
            )
        )
    logger.info(
        "PushT joint goal dataset sample counts "
        f"(expert + rollouts): {[len(d) for d in datasets]}"
    )
    return PushTMixedDataset(datasets)
