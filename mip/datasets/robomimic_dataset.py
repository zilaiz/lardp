"""Robomimic state dataset.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import concurrent.futures
import os
from collections import defaultdict

import h5py
import numpy as np
import torch
import zarr
from huggingface_hub import hf_hub_download
from loguru import logger
from tqdm import tqdm

from mip.dataset_utils import (
    ImageNormalizer,
    MinMaxNormalizer,
    QuantileNormalizer,
    ReplayBuffer,
    RotationTransformer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset
from mip.datasets.imagecodecs import register_codecs

register_codecs()


class MultiImageDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset that exposes normalizer and undo_transform_action from the primary dataset."""

    def __init__(self, datasets):
        super().__init__(datasets)
        self.normalizer = datasets[0].normalizer
        self.undo_transform_action = datasets[0].undo_transform_action


def _resolve_dataset_path(task_config):
    """Resolve a single dataset path from config (HuggingFace or local)."""
    if hasattr(task_config, "dataset_repo") and hasattr(
        task_config, "dataset_filename"
    ):
        logger.info(
            f"Downloading dataset from {task_config.dataset_repo}/{task_config.dataset_filename}"
        )
        dataset_path = hf_hub_download(
            repo_id=task_config.dataset_repo,
            filename=task_config.dataset_filename,
            repo_type="dataset",
        )
        logger.info(f"Downloaded dataset to: {dataset_path}")
        return dataset_path
    elif hasattr(task_config, "dataset_path") and task_config.dataset_path:
        dataset_path = os.path.expanduser(task_config.dataset_path)
        logger.info(f"Loading dataset from {dataset_path}")
        return dataset_path
    return None


def make_dataset(task_config, mode="train"):
    # Check for multi-dataset image training
    if (
        task_config.dataset_paths is not None
        and task_config.obs_type == "image"
        and task_config.latent_type is None
    ):
        return _make_multi_image_dataset(task_config, mode)

    # Single dataset path resolution
    dataset_path = _resolve_dataset_path(task_config)
    if dataset_path is None:
        raise ValueError(
            "Either dataset_repo/dataset_filename or dataset_path must be provided"
        )

    if task_config.env_name in ["can", "lift", "square", "tool_hang", "transport"]:
        if task_config.obs_type == "state":
            return RobomimicDataset(
                dataset_path,
                horizon=task_config.horizon,
                obs_keys=task_config.obs_keys,
                pad_before=task_config.obs_steps - 1,
                pad_after=task_config.act_steps - 1,
                abs_action=task_config.abs_action,
                mode=mode,
                val_dataset_percentage=task_config.val_dataset_percentage,
            )
        elif task_config.obs_type == "image":
            latent_type = task_config.latent_type
            use_precomputed = task_config.use_precomputed
            camera_keys = task_config.camera_keys
            if latent_type == 'lam':
                lam_frame_skips = task_config.lam_frame_skips
                lam_latent_type = task_config.lam_latent_type
                if lam_latent_type is not None and use_precomputed:
                    return RobomimicImageLAMDataset(
                        dataset_path,
                        horizon=task_config.horizon,
                        shape_meta=task_config.shape_meta,
                        n_obs_steps=task_config.obs_steps,
                        pad_before=task_config.obs_steps - 1,
                        pad_after=task_config.act_steps - 1,
                        abs_action=task_config.abs_action,
                        val_dataset_percentage=task_config.val_dataset_percentage,
                        mode=mode,
                        camera_keys=camera_keys,
                        lam_frame_skips=lam_frame_skips,
                        lam_latent_type=lam_latent_type,
                    )
            elif latent_type == 'dino':
                dino_model = task_config.dino_model
                dino_types = task_config.dino_types
                if dino_model is not None and use_precomputed:
                    return RobomimicImageDINODataset(
                        dataset_path,
                        horizon=task_config.horizon,
                        shape_meta=task_config.shape_meta,
                        n_obs_steps=task_config.obs_steps,
                        pad_before=task_config.obs_steps - 1,
                        pad_after=task_config.act_steps - 1,
                        abs_action=task_config.abs_action,
                        val_dataset_percentage=task_config.val_dataset_percentage,
                        mode=mode,
                        dino_model=dino_model,
                        dino_types=dino_types,
                        camera_keys=camera_keys
                    )
            elif latent_type is not None:
                return RobomimicImageREPADataset(
                    dataset_path,
                    horizon=task_config.horizon,
                    shape_meta=task_config.shape_meta,
                    n_obs_steps=task_config.obs_steps,
                    pad_before=task_config.obs_steps - 1,
                    pad_after=task_config.act_steps - 1,
                    abs_action=task_config.abs_action,
                    val_dataset_percentage=task_config.val_dataset_percentage,
                    mode=mode,
                    camera_keys=camera_keys,
                )
            return RobomimicImageDataset(
                dataset_path,
                horizon=task_config.horizon,
                shape_meta=task_config.shape_meta,
                n_obs_steps=task_config.obs_steps,
                pad_before=task_config.obs_steps - 1,
                pad_after=task_config.act_steps - 1,
                abs_action=task_config.abs_action,
                val_dataset_percentage=task_config.val_dataset_percentage,
                mode=mode,
            )
        else:
            raise ValueError(f"Invalid observation type: {task_config.obs_type}")
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


def _make_multi_image_dataset(task_config, mode="train"):
    """Create image dataset(s) from one or more HDF5 files.

    Uses task_config.dataset_paths (list). The first dataset's normalizer is
    shared with all subsequent datasets. Returns a MultiImageDataset if multiple
    paths, or a single RobomimicImageDataset otherwise.
    """
    paths = [os.path.expanduser(p) for p in task_config.dataset_paths]
    logger.info(f"Multi-dataset paths: {paths}")

    filter_success = getattr(task_config, "filter_success", False)

    common_kwargs = dict(
        shape_meta=task_config.shape_meta,
        n_obs_steps=task_config.obs_steps,
        horizon=task_config.horizon,
        pad_before=task_config.obs_steps - 1,
        pad_after=task_config.act_steps - 1,
        abs_action=task_config.abs_action,
        mode=mode,
        delta_action_anchor=getattr(task_config, "delta_action_anchor", None),
        delta_action_normalizer=getattr(
            task_config, "delta_action_normalizer", "quantile"
        ),
    )

    # Primary (expert) dataset with val split
    datasets = []
    primary_ds = RobomimicImageDataset(
        dataset_dir=paths[0],
        val_dataset_percentage=task_config.val_dataset_percentage,
        **common_kwargs,
    )
    datasets.append(primary_ds)
    logger.info(f"Primary dataset: {len(primary_ds)} samples")

    # Secondary (rollout) datasets — each with own normalizer first
    for path in paths[1:]:
        ds = RobomimicImageDataset(
            dataset_dir=path,
            val_dataset_percentage=0.0,
            filter_success=filter_success,
            **common_kwargs,
        )
        datasets.append(ds)
        logger.info(f"Secondary dataset: {len(ds)} samples")

    if len(datasets) == 1:
        return datasets[0]

    # Merge normalizers: take global min/max across all datasets
    merged_normalizer = _merge_normalizers([ds.normalizer for ds in datasets])
    for ds in datasets:
        ds.normalizer = merged_normalizer
    logger.info("Merged normalizers across all image datasets")

    return MultiImageDataset(datasets)


class RobomimicDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_keys=("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        use_key_state_for_val: bool = False,
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_dir) as file:
            demos = file["data"]
            total_demos = len(demos)

            # Calculate split indices
            if val_dataset_percentage > 0.0:
                val_count = int(total_demos * val_dataset_percentage)
                train_count = total_demos - val_count

                # Use deterministic split based on indices
                if mode == "train":
                    demo_indices = list(range(train_count))
                elif mode == "val":
                    demo_indices = list(range(train_count, total_demos))
                else:
                    raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
            else:
                # Use all data for training when no validation split
                demo_indices = list(range(total_demos))

            if use_key_state_for_val:
                import robomimic.utils.env_utils as EnvUtils
                import robomimic.utils.file_utils as FileUtils
                import robomimic.utils.obs_utils as ObsUtils

                # Initialize observation utilities with dummy spec
                dummy_spec = {
                    "obs": {
                        "low_dim": ["robot0_eef_pos"],
                        "rgb": [],
                    },
                }
                ObsUtils.initialize_obs_utils_with_obs_specs(
                    obs_modality_specs=dummy_spec
                )

                # Create environment from dataset metadata
                env_meta = FileUtils.get_env_metadata_from_dataset(
                    dataset_path=dataset_dir
                )
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta, render=False, render_offscreen=False
                )

                # Check if this is a robosuite environment
                is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

            for i in tqdm(demo_indices, desc=f"Loading {mode} hdf5 to ReplayBuffer"):
                demo = demos[f"demo_{i}"]

                if use_key_state_for_val:
                    states = demo["states"][:]
                    # Prepare initial state for environment reset
                    initial_state = {"states": states[0]}
                    if is_robosuite_env:
                        initial_state["model"] = demo.attrs["model_file"]
                        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

                    # Reset environment to initial state
                    env.reset_to(initial_state)

                    # Evaluate key states in the trajectory
                    for _j, state in enumerate(states):
                        env.reset_to({"states": state})

                        # Get distance between frame and stand (example evaluation metric)
                        frame_site_name = "frame_tip_site"
                        stand_site_name = "stand_mount_site"

                        frame_site_pos = env.sim.data.site_xpos[
                            env.obj_site_id[frame_site_name]
                        ]
                        stand_site_pos = env.sim.data.site_xpos[
                            env.obj_site_id[stand_site_name]
                        ]
                        distance = np.linalg.norm(frame_site_pos - stand_site_pos)
                        logger.debug(distance)
                    exit()

                episode = _data_to_obs(
                    raw_obs=demo["obs"],
                    raw_actions=demo["actions"][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=self.rotation_transformer,
                )
                self.replay_buffer.add_episode(episode)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.abs_action = abs_action
        self.normalizer = self.get_normalizer()

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def get_normalizer(self):
        if self.abs_action:
            state_normalizer = MinMaxNormalizer(
                self.replay_buffer["obs"][:]
            )  # (N, obs_dim)
            action_normalizer = MinMaxNormalizer(
                self.replay_buffer["action"][:]
            )  # (N, action_dim)
        else:
            state_normalizer = MinMaxNormalizer(
                self.replay_buffer["obs"][:]
            )  # (N, obs_dim)
            action_normalizer = MinMaxNormalizer(
                self.replay_buffer["action"][:]
            )  # (N, action_dim)
        return {"obs": {"state": state_normalizer}, "action": action_normalizer}

    def sample_to_data(self, sample):
        state = sample["obs"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)
        data = {
            "obs": {"state": state},
            "action": action,
        }
        return data

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
    obs = np.concatenate([raw_obs[key] for key in obs_keys], axis=-1).astype(np.float32)

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)

    data = {"obs": obs, "action": raw_actions}
    return data


class RobomimicImageDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        normalizer=None,
        filter_success=False,
        delta_action_anchor: str | None = None,
        delta_action_normalizer: str = "quantile",
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode
        # Only consulted on the delta-action (current_obs) branch of
        # get_normalizer(); robomimic abs/relative tasks never set
        # delta_action_anchor, so this leaves their MinMax path untouched.
        self.delta_action_normalizer = delta_action_normalizer

        self.replay_buffer = _convert_robomimic_to_replay(
            store=zarr.storage.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            abs_action=abs_action,
            rotation_transformer=self.rotation_transformer,
            val_dataset_percentage=val_dataset_percentage,
            mode=mode,
            filter_success=filter_success,
        )

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
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
        self.abs_action = abs_action
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

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = self.get_normalizer()

    def _compute_chunk_relative_deltas_for_normalizer(self) -> np.ndarray:
        """Stack delta-transformed actions across all in-episode chunks.

        Used to fit the action normalizer when ``delta_action_anchor`` is set.
        Accesses ``replay_buffer`` directly (skips the sampler's image fetches)
        and ignores ``pad_before``/``pad_after`` — for MinMax fitting we just
        need representative coverage of the in-distribution delta magnitudes.
        """
        from mip.franka_delta_transform import to_delta

        actions = np.asarray(self.replay_buffer["action"][:])           # (T, 10)
        eef_pos = np.asarray(self.replay_buffer["robot0_eef_pos"][:])   # (T, 3)
        eef_quat = np.asarray(self.replay_buffer["robot0_eef_quat"][:]) # (T, 4) xyzw
        episode_ends = np.asarray(self.replay_buffer.episode_ends[:])

        H = self.horizon
        To = self.n_obs_steps if self.n_obs_steps is not None else 1

        all_deltas: list[np.ndarray] = []
        prev_end = 0
        for ep_end in episode_ends:
            # Valid in-episode chunk starts k: need k+To-1 < ep_end (anchor in
            # episode) AND k+H <= ep_end (full action chunk in episode).
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

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        if self.delta_action_anchor == "current_obs":
            # Fit on delta-transformed actions so action samples live in a
            # well-scaled [-1, 1] range matching the delta distribution.
            # Quantile (default) is robust to the heavy-tailed, sub-degree
            # rotation deltas that MinMax would crush into a thin band.
            delta_actions = self._compute_chunk_relative_deltas_for_normalizer()
            if getattr(self, "delta_action_normalizer", "quantile") == "quantile":
                normalizer["action"] = QuantileNormalizer(delta_actions)
            else:
                normalizer["action"] = MinMaxNormalizer(delta_actions)
        else:
            normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])

        return normalizer

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # obs
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                np.moveaxis(sample[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            # T,C,H,W
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        # Capture raw (un-normalized) eef pose at the last obs frame BEFORE
        # we normalize lowdim obs — needed as the delta-action anchor.
        anchor_pos = None
        anchor_quat = None
        if self.delta_action_anchor == "current_obs":
            anchor_pos = sample["robot0_eef_pos"][self.n_obs_steps - 1].astype(np.float32)
            anchor_quat = sample["robot0_eef_quat"][self.n_obs_steps - 1].astype(np.float32)

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        # action
        action = sample["action"].astype(np.float32)
        if self.delta_action_anchor == "current_obs":
            from mip.franka_delta_transform import to_delta
            action = to_delta(action, anchor_pos, anchor_quat)
        action = self.normalizer["action"].normalize(action)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
        }
        return torch_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


class RobomimicImageIDMDataset(RobomimicImageDataset):
    """Image dataset for inverse dynamics model training.

    Returns current obs (To frames), goal obs (1 frame after action chunk),
    and action sequence. Supports sharing a normalizer from another dataset
    (e.g., expert normalizer for rollout data).
    """

    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        normalizer=None,
        filter_success=False,
        optimality_label: int = 0,
        delta_action_anchor: str | None = None,
        delta_action_normalizer: str = "quantile",
    ):
        # We need to override the parent's __init__ because:
        # 1. sequence_length must be horizon+1 (extra frame for goal)
        # 2. pad_after must be act_steps (= pad_after+1 from caller) to allow goal frame
        # 3. No key_first_k optimization (need obs at both [0:obs_steps] and [horizon])
        BaseDataset.__init__(self)
        # Optimality slot carried into every batch sample. Read by joint /
        # CFG-aware agents (e.g. LBMDiTJointAgent): 0 = expert, 1 = null/play.
        # Default 0 keeps single-source training silent.
        self.optimality_label = int(optimality_label)
        # Action-normalizer choice for the delta (current_obs) branch of
        # get_normalizer(); ignored when delta_action_anchor is None.
        self.delta_action_normalizer = delta_action_normalizer
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode

        self.replay_buffer = _convert_robomimic_to_replay(
            store=zarr.storage.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            abs_action=abs_action,
            rotation_transformer=self.rotation_transformer,
            val_dataset_percentage=val_dataset_percentage,
            mode=mode,
            filter_success=filter_success,
        )

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        # No key_first_k — we need obs at [0:n_obs_steps] AND [horizon]
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon + 1,  # +1 for goal frame
            pad_before=pad_before,
            pad_after=pad_after + 1,  # +1 to allow goal frame at end
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        assert self.horizon > self.n_obs_steps, (
            "IDM dataset needs horizon > n_obs_steps so an intermediate frame "
            "strictly between To_1 and goal exists."
        )
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

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = self.get_normalizer()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # Current obs: first n_obs_steps frames
        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(sample[key][: self.n_obs_steps], -1, 1).astype(np.float32)
                / 255.0
            )
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][: self.n_obs_steps].astype(np.float32)
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        # Goal obs: frame at index `horizon` (right after the action chunk)
        goal_dict = {}
        for key in self.rgb_keys:
            goal_dict[key] = (
                np.moveaxis(
                    sample[key][self.horizon : self.horizon + 1], -1, 1
                ).astype(np.float32)
                / 255.0
            )
            goal_dict[key] = self.normalizer["obs"][key].normalize(goal_dict[key])

        for key in self.lowdim_keys:
            goal_dict[key] = sample[key][self.horizon : self.horizon + 1].astype(
                np.float32
            )
            goal_dict[key] = self.normalizer["obs"][key].normalize(goal_dict[key])

        # Intermediate obs: uniformly sample a real frame strictly between
        # To_1 (index n_obs_steps - 1) and goal (index horizon), with a
        # margin m=2 away from both endpoints so neither segment collapses
        # to a near-zero embedding displacement.
        m = 2
        k_low = self.n_obs_steps + m
        k_high = self.horizon - m
        assert k_high > k_low, (
            f"inter_obs sampling range empty: n_obs_steps={self.n_obs_steps}, "
            f"horizon={self.horizon}, m={m}"
        )
        k = int(np.random.randint(k_low, k_high))
        inter_dict = {}
        for key in self.rgb_keys:
            inter_dict[key] = (
                np.moveaxis(sample[key][k : k + 1], -1, 1).astype(np.float32) / 255.0
            )
            inter_dict[key] = self.normalizer["obs"][key].normalize(inter_dict[key])

        for key in self.lowdim_keys:
            inter_dict[key] = sample[key][k : k + 1].astype(np.float32)
            inter_dict[key] = self.normalizer["obs"][key].normalize(inter_dict[key])

        # Action: first `horizon` frames
        action = sample["action"][: self.horizon].astype(np.float32)
        if self.delta_action_anchor == "current_obs":
            # Anchor at the last obs frame's eef pose (chunk-relative deltas).
            # The raw sample dict still has the un-normalized eef_pos / quat
            # because we read it via sample[key], not via the normalized
            # obs_dict above (which has been MinMax'd).
            anchor_pos = sample["robot0_eef_pos"][self.n_obs_steps - 1].astype(np.float32)
            anchor_quat = sample["robot0_eef_quat"][self.n_obs_steps - 1].astype(np.float32)
            from mip.franka_delta_transform import to_delta
            action = to_delta(action, anchor_pos, anchor_quat)
        action = self.normalizer["action"].normalize(action)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "goal_obs": dict_apply(goal_dict, torch.tensor),
            "inter_obs": dict_apply(inter_dict, torch.tensor),
            "action": torch.tensor(action),
            "optimality": torch.tensor(self.optimality_label, dtype=torch.long),
        }
        return torch_data


def _merge_normalizers(normalizers):
    """Merge multiple normalizers by taking global min/max across all.

    For MinMaxNormalizer keys, computes element-wise min/max across datasets.
    ImageNormalizer keys are passed through unchanged (stateless).
    """
    merged = defaultdict(dict)
    # Merge obs normalizers
    all_obs_keys = set()
    for norm in normalizers:
        all_obs_keys.update(norm["obs"].keys())
    for key in all_obs_keys:
        sub_norms = [n["obs"][key] for n in normalizers if key in n["obs"]]
        if all(isinstance(n, ImageNormalizer) for n in sub_norms):
            merged["obs"][key] = sub_norms[0]
        elif all(isinstance(n, MinMaxNormalizer) for n in sub_norms):
            global_min = np.minimum.reduce([n.min for n in sub_norms])
            global_max = np.maximum.reduce([n.max for n in sub_norms])
            dummy = np.stack([global_min, global_max])
            merged["obs"][key] = MinMaxNormalizer(dummy)
        else:
            raise ValueError(
                f"Inconsistent normalizer types for obs key '{key}': "
                f"{[type(n).__name__ for n in sub_norms]}"
            )
    # Merge action normalizer. For QuantileNormalizer (delta path) the .min/.max
    # anchors are the per-dim q01/q99; merging by union of [q01, q99] across
    # datasets is an approximation of the global quantiles (errs toward a wider,
    # less aggressive range — safe). MinMax merge is byte-for-byte unchanged, so
    # robomimic action merges are unaffected.
    act_norms = [n["action"] for n in normalizers]
    global_min = np.minimum.reduce([n.min for n in act_norms])
    global_max = np.maximum.reduce([n.max for n in act_norms])
    if all(isinstance(n, QuantileNormalizer) for n in act_norms):
        merged["action"] = QuantileNormalizer.from_bounds(global_min, global_max)
    else:
        dummy = np.stack([global_min, global_max])
        merged["action"] = MinMaxNormalizer(dummy)
    return merged


def make_idm_dataset(task_config, mode="train", normalizer=None):
    """Create IDM dataset(s) from one or more HDF5 files.

    Uses task_config.dataset_paths (list) or falls back to task_config.dataset_path.
    Returns a ConcatDataset if multiple paths, or a single dataset otherwise.

    Args:
        task_config: Task configuration.
        mode: "train" or "val".
        normalizer: If provided, use this normalizer for all datasets instead of
            computing a new one. Useful for ensuring downstream tasks (e.g., goal
            predictor) use the same normalizer as the pretrained IDM.
    """
    paths = task_config.dataset_paths
    if paths is None:
        # Fall back to single path
        if hasattr(task_config, "dataset_path") and task_config.dataset_path:
            paths = [os.path.expanduser(task_config.dataset_path)]
        else:
            raise ValueError(
                "Either dataset_paths or dataset_path must be provided for IDM training"
            )
    else:
        paths = [os.path.expanduser(p) for p in paths]

    logger.info(f"IDM dataset paths: {paths}")

    filter_success = getattr(task_config, "filter_success", False)

    common_kwargs = dict(
        shape_meta=task_config.shape_meta,
        n_obs_steps=task_config.obs_steps,
        horizon=task_config.horizon,
        pad_before=task_config.obs_steps - 1,
        pad_after=task_config.act_steps - 1,
        abs_action=task_config.abs_action,
        mode=mode,
        delta_action_anchor=getattr(task_config, "delta_action_anchor", None),
        delta_action_normalizer=getattr(
            task_config, "delta_action_normalizer", "quantile"
        ),
    )

    # Create primary (expert) dataset with val split applied. The primary
    # path is tagged optimality=0 (expert); all additional paths are tagged
    # optimality=1 (null/play) so CFG-aware agents see real labels per sample.
    datasets = []
    primary_ds = RobomimicImageIDMDataset(
        dataset_dir=paths[0],
        val_dataset_percentage=task_config.val_dataset_percentage,
        optimality_label=0,
        **common_kwargs,
    )
    datasets.append(primary_ds)
    logger.info(f"Primary IDM dataset (optimality=0/expert): {len(primary_ds)} samples")

    # Create additional (rollout) datasets — each with own normalizer first
    for path in paths[1:]:
        ds = RobomimicImageIDMDataset(
            dataset_dir=path,
            val_dataset_percentage=0.0,
            filter_success=filter_success,
            optimality_label=1,
            **common_kwargs,
        )
        datasets.append(ds)
        logger.info(f"Additional IDM dataset (optimality=1/play): {len(ds)} samples")

    # Apply normalizer: use provided one, or merge across all datasets
    if normalizer is not None:
        for ds in datasets:
            ds.normalizer = normalizer
        logger.info("Using provided normalizer for all IDM datasets")
    elif len(datasets) > 1:
        merged_normalizer = _merge_normalizers([ds.normalizer for ds in datasets])
        for ds in datasets:
            ds.normalizer = merged_normalizer
        logger.info("Merged normalizers across all IDM datasets")

    if len(datasets) == 1:
        return datasets[0]

    return torch.utils.data.ConcatDataset(datasets)


class RobomimicImageREPADataset(BaseDataset):
    """Image dataset for on-the-fly LAM inference.

    Returns raw LAM camera images (horizon+1 frames) alongside the standard
    obs dict so the training loop can run LAM inference on them.
    """

    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        camera_keys=None,
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode
        self.extra_camera_keys = camera_keys  # None = all rgb_keys

        self.replay_buffer = _convert_robomimic_to_replay(
            store=zarr.storage.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            abs_action=abs_action,
            rotation_transformer=self.rotation_transformer,
            val_dataset_percentage=val_dataset_percentage,
            mode=mode,
        )

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        # Determine which rgb_keys are used for extra_raw_images
        if self.extra_camera_keys is not None:
            extra_rgb_keys = [k for k in rgb_keys if k in self.extra_camera_keys]
        else:
            extra_rgb_keys = list(rgb_keys)
        non_extra_rgb_keys = [k for k in rgb_keys if k not in extra_rgb_keys]

        # BUG FIX: Only limit non-extra keys to n_obs_steps.
        # extra camera keys need ALL horizon+1 frames (not just n_obs_steps),
        # otherwise SequenceSampler fills frames beyond n_obs_steps with NaN.
        key_first_k = {}
        if n_obs_steps is not None:
            for key in non_extra_rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon + 1,  # need horizon+1 obs to form horizon action pairs
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.extra_rgb_keys = extra_rgb_keys
        self.abs_action = abs_action
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
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
        extra_raw_images = {}
        for key in self.rgb_keys:
            # For extra cameras: extract all horizon+1 raw frames as float32/255
            if key in self.extra_rgb_keys:
                extra_raw_images[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 255.0  # (horizon+1, H, W, C)

            # For obs: first n_obs_steps frames, channel-first, normalized
            obs_dict[key] = (
                np.moveaxis(sample[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        # action: slice to horizon (sampler returns horizon+1 length)
        action = sample["action"][: self.horizon].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
            "extra_raw_images": dict_apply(extra_raw_images, torch.tensor),
        }
        return torch_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    val_dataset_percentage=0.0,
    mode="train",
    filter_success=False,
):
    """Convert Robomimic dataset to ReplayBuffer.

    A ReplayBuffer is a `zarr.Group` or Dict[str, dict] that contains the following keys:
    - data: zarr.Group or Dict[str, dict]
        Contains the data. All data should be stored as numpy arrays with the same length.
    - meta: zarr.Group or Dict[str, dict]
        Contains key "episode_ends", which is a numpy array of shape (n_episodes,) that contains the
        end index of each episode in the data.

    Args:
    - store: zarr.Store
        zarr.MemoryStore()
    - shape_meta: dict
        Shape metadata of the dataset. Should contain keys 'obs', 'action'.
        For example:
        shape_meta = {
            "action": {"shape": [10, ]},
            "obs": {
                "agentview_image": {"shape": [84, 84, 3], "type": "rgb"},
                "robot0_eef_pos":  {"shape": [3, ],       "type": "low_dim"},
            }}
    - dataset_path: str
        Path to the Robomimic dataset
    - abs_action: bool
        Whether to use position or velocity control
    - rotation_transformer: RotationTransformer
        Rotation transformer to convert rotation representation
    """
    """ Dataset structure of Can-PH, as an example:
    - data
        - demo_0
            - actions  (118, 7)
            - dones     (118, )
            - next_obs
                - agentview_image  (118, 84, 84, 3)
                - object            (118, 14)
                - robot0_eef_pos   (118, 3)
                - robot0_eef_quat
                - robot0_eef_vel_ang
                - robot0_eef_vel_lin
                - robot0_eye_in_hand_image
                - robot0_gripper_qpos
                - robot0_gripper_qvel
                - robot0_joint_pos
                - robot0_joint_pos_cos
                - robot0_joint_pos_sin
                - robot0_joint_vel
            - obs
                ...
            - rewards   (118, )
            - states    (118, 71)
        - demo_1
        ...(x200 demos)
    - mask
        - 20_percent
        - 20_percent_train
        - 20_percent_valid
        - 50_percent
        - 50_percent_train
        - 50_percent_valid
        - train (180,)
        - valid (20,)

    Suppose that the `shape_meta` is:
    shape_meta = {
    "action": {"shape": [10, ]},
    "obs": {
        "agentview_image": {
            "shape": [3, 84, 84], "type": "rgb", },
        "robot0_eye_in_hand_image": {
            "shape": [3, 84, 84], "type": "rgb", },
        "robot0_eef_pos": {
            "shape": [3, ], "type": "low_dim", },
        "robot0_eef_quat": {
            "shape": [4, ], "type": "low_dim", },
        "robot0_gripper_qpos": {
            "shape": [2, ], "type": "low_dim", }, }}
    """

    import multiprocessing

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = []
    lowdim_keys = []
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)
    # rgb_keys = ['agentview_image', 'robot0_eye_in_hand_image']
    # lowdim_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

    # create zarr group
    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file["data"]
        total_demos = len(demos)

        # Calculate split indices
        if val_dataset_percentage > 0.0:
            val_count = int(total_demos * val_dataset_percentage)
            train_count = total_demos - val_count

            # Use deterministic split based on indices
            if mode == "train":
                demo_indices = list(range(train_count))
            elif mode == "val":
                demo_indices = list(range(train_count, total_demos))
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
        else:
            # Use all data for training when no validation split
            demo_indices = list(range(total_demos))

        # Filter demos by binary reward (keep only successful episodes)
        if filter_success:
            filtered = []
            for i in demo_indices:
                demo = demos[f"demo_{i}"]
                if "rewards" in demo:
                    if np.sum(demo["rewards"][:]) > 0:
                        filtered.append(i)
                else:
                    logger.warning(f"demo_{i} has no rewards key, keeping it")
                    filtered.append(i)
            logger.info(
                f"Reward filter: kept {len(filtered)}/{len(demo_indices)} successful demos"
            )
            if len(filtered) == 0:
                raise ValueError(
                    f"All {len(demo_indices)} demos filtered out by reward filter in {dataset_path}"
                )
            demo_indices = filtered

        episode_ends = []
        prev_end = 0
        for i in demo_indices:
            demo = demos[f"demo_{i}"]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1] if episode_ends else 0
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.create_array(
            name="episode_ends",
            data=np.array(episode_ends, dtype=np.int64),
            compressor=None,
            overwrite=True,
        )

        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc=f"Loading {mode} lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            this_data = []
            for i in demo_indices:
                demo = demos[f"demo_{i}"]
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0) if this_data else np.array([])
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["action"]["shape"]
                )
            else:
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["obs"][key]["shape"]
                )
            _ = data_group.create_array(
                name=key,
                data=this_data,
                chunks=this_data.shape,
                compressor=None,
                overwrite=True,
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys),
            desc=f"Loading {mode} image data",
            mininterval=1.0,
        ) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    # Use None compressor for zarr v3 compatibility in tests
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=None,
                        dtype=np.uint8,
                    )
                    for demo_list_idx, episode_idx in enumerate(demo_indices):
                        demo = demos[f"demo_{episode_idx}"]
                        hdf5_arr = demo["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[demo_list_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


class RobomimicImageLAMDataset(BaseDataset):
    """Image dataset that loads precomputed LAM latent actions from HDF5.

    Expects the HDF5 to contain precomputed latent actions at:
        demo_i/latent_actions/fs{k}/{camera_key}          (T, 32)
        demo_i/latent_actions_prebn/fs{k}/{camera_key}    (T, 1024)  [optional]
    """

    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        lam_frame_skips=None,
        camera_keys=None,
        lam_latent_type="prebn",
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode
        self.lam_frame_skips = lam_frame_skips or []
        self.lam_camera_keys = camera_keys
        self.lam_latent_type = lam_latent_type

        self.replay_buffer = _convert_robomimic_lam_to_replay(
            store=zarr.storage.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            abs_action=abs_action,
            rotation_transformer=self.rotation_transformer,
            val_dataset_percentage=val_dataset_percentage,
            mode=mode,
            lam_frame_skips=self.lam_frame_skips + [horizon],
            lam_camera_keys=self.lam_camera_keys,
            lam_latent_type=self.lam_latent_type,
        )

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        # Build latent action keys explicitly from config
        effective_cam_keys = self.lam_camera_keys or rgb_keys
        prefix = "latent_action_prebn" if lam_latent_type == "prebn" else "latent_action"
        self.latent_action_keys = []
        assert len(self.lam_frame_skips) == 1 # TODO: haven't fully supported alignment with multiple frame skips yet
        for cam in effective_cam_keys:
            for fs in self.lam_frame_skips:
                key = f"{prefix}_fs{fs}_{cam}"
                if key in self.replay_buffer:
                    self.latent_action_keys.append(key)

        self.cls_keys = []
        for cam in effective_cam_keys:
            self.cls_keys.append(f"{prefix}_fs{horizon}_{cam}")

        key_first_k = {}
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
        self.abs_action = abs_action
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def __str__(self) -> str:
        return (
            f"Keys: {self.replay_buffer.keys()} "
            f"Steps: {self.replay_buffer.n_steps} "
            f"Episodes: {self.replay_buffer.n_episodes}"
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(sample[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        # Build latent action dict (single type selected by lam_latent_type)
        latent_actions = {}
        for key in self.latent_action_keys:
            latent_actions[key] = sample[key].astype(np.float32)

        cls_tokens = {}
        for key in self.cls_keys:
            cls_tokens[key] = sample[key][:1].astype(np.float32)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
            "latent_actions": dict_apply(latent_actions, torch.tensor),
            "cls_tokens": dict_apply(cls_tokens, torch.tensor),
        }
        return torch_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


def _convert_robomimic_lam_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    lam_frame_skips,
    lam_camera_keys=None,
    lam_latent_type="prebn",
    n_workers=None,
    max_inflight_tasks=None,
    val_dataset_percentage=0.0,
    mode="train",
):
    """Convert Robomimic dataset with precomputed LAM latent actions to ReplayBuffer.

    Extends _convert_robomimic_to_replay by also loading latent action data from
    demo_i/latent_actions/fs{k}/{camera_key} in the HDF5.
    """
    assert lam_latent_type in ("prebn", "bn"), f"Invalid lam_latent_type: {lam_latent_type}"
    import multiprocessing

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = []
    lowdim_keys = []
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    # create zarr group
    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file["data"]
        total_demos = len(demos)

        # Calculate split indices
        if val_dataset_percentage > 0.0:
            val_count = int(total_demos * val_dataset_percentage)
            train_count = total_demos - val_count
            if mode == "train":
                demo_indices = list(range(train_count))
            elif mode == "val":
                demo_indices = list(range(train_count, total_demos))
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
        else:
            demo_indices = list(range(total_demos))

        episode_ends = []
        prev_end = 0
        for i in demo_indices:
            demo = demos[f"demo_{i}"]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1] if episode_ends else 0
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.create_array(
            name="episode_ends",
            data=np.array(episode_ends, dtype=np.int64),
            compressor=None,
            overwrite=True,
        )

        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc=f"Loading {mode} lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            this_data = []
            for i in demo_indices:
                demo = demos[f"demo_{i}"]
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0) if this_data else np.array([])
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["action"]["shape"]
                )
            else:
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["obs"][key]["shape"]
                )
            _ = data_group.create_array(
                name=key,
                data=this_data,
                chunks=this_data.shape,
                compressor=None,
                overwrite=True,
            )

        # save image data
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys),
            desc=f"Loading {mode} image data",
            mininterval=1.0,
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=None,
                        dtype=np.uint8,
                    )
                    for demo_list_idx, episode_idx in enumerate(demo_indices):
                        demo = demos[f"demo_{episode_idx}"]
                        hdf5_arr = demo["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[demo_list_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

        # Auto-detect LAM camera keys if not provided
        if lam_camera_keys is None:
            # Discover from the HDF5 group matching the requested latent type
            first_demo = demos[f"demo_{demo_indices[0]}"]
            hdf5_group = "latent_actions_prebn" if lam_latent_type == "prebn" else "latent_actions"
            if hdf5_group in first_demo:
                # Get camera keys from first available frame_skip
                first_fs_key = list(first_demo[hdf5_group].keys())[0]
                lam_camera_keys = sorted(first_demo[hdf5_group][first_fs_key].keys())
            else:
                lam_camera_keys = []

        # Load precomputed latent action data
        for fs in lam_frame_skips:
            for cam_key in lam_camera_keys:
                if lam_latent_type == "prebn":
                    zarr_key = f"latent_action_prebn_fs{fs}_{cam_key}"
                    hdf5_path = f"latent_actions_prebn/fs{fs}/{cam_key}"
                else:
                    zarr_key = f"latent_action_fs{fs}_{cam_key}"
                    hdf5_path = f"latent_actions/fs{fs}/{cam_key}"

                la_data = []
                for i in demo_indices:
                    demo = demos[f"demo_{i}"]
                    if hdf5_path in demo:
                        la_data.append(demo[hdf5_path][:].astype(np.float32))
                    else:
                        raise KeyError(
                            f"Missing precomputed latent actions at data/demo_{i}/{hdf5_path}. "
                            f"Run mip/networks/lam/precompute_lam.py first."
                        )
                la_data = np.concatenate(la_data, axis=0)
                assert la_data.shape[0] == n_steps, (
                    f"Latent action steps mismatch: {la_data.shape[0]} vs {n_steps}"
                )
                _ = data_group.create_array(
                    name=zarr_key,
                    data=la_data,
                    chunks=la_data.shape,
                    compressor=None,
                    overwrite=True,
                )

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


class RobomimicImageDINODataset(BaseDataset):
    """Image dataset that loads precomputed DINO latent from HDF5.

    Expects the HDF5 to contain precomputed latent actions at:
        demo_i/dino_cls/{dino_model}/{camera_key}    (T, 384 or 768)
        demo_i/dino_patch_mean/{dino_model}/{camera_key}    (T, 384 or 768)
    """

    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        dino_model="vits16plus",
        dino_types=None,
        camera_keys=None,
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode
        if dino_types is None:
            dino_types = ["cls"]
        self.dino_types = dino_types
        self.dino_model = dino_model
        self.dino_camera_keys = camera_keys

        self.replay_buffer = _convert_robomimic_dino_to_replay(
            store=zarr.storage.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            abs_action=abs_action,
            rotation_transformer=self.rotation_transformer,
            val_dataset_percentage=val_dataset_percentage,
            mode=mode,
            dino_model=self.dino_model,
            dino_types=self.dino_types,
            dino_camera_keys=self.dino_camera_keys,
        )

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        # Build dino latents keys explicitly from config
        effective_cam_keys = self.dino_camera_keys or rgb_keys
        self.dino_latent_keys = []
        for cam in effective_cam_keys:
            for dino_type in self.dino_types:
                key = f"dino_{dino_type}_{self.dino_model}_{cam}"
                if key in self.replay_buffer:
                    self.dino_latent_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon + 1,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def __str__(self) -> str:
        return (
            f"Keys: {self.replay_buffer.keys()} "
            f"Steps: {self.replay_buffer.n_steps} "
            f"Episodes: {self.replay_buffer.n_episodes}"
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(sample[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        # Build dino latents dict (single mnodel selected by dino_model)
        dino_latents = {}
        for key in self.dino_latent_keys:
            dino_latents[key] = sample[key].astype(np.float32)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
            "dino_latents": dict_apply(dino_latents, torch.tensor),
        }
        return torch_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


def _convert_robomimic_dino_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    dino_model="vits16plus",
    dino_types=None,
    dino_camera_keys=None,
    n_workers=None,
    max_inflight_tasks=None,
    val_dataset_percentage=0.0,
    mode="train",
):
    """Convert Robomimic dataset with precomputed LAM latent actions to ReplayBuffer.

    Extends _convert_robomimic_to_replay by also loading latent action data from
    demo_i/latent_actions/fs{k}/{camera_key} in the HDF5.
    """
    assert dino_model in ("vits16plus", "vitb16")
    import multiprocessing

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = []
    lowdim_keys = []
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    # create zarr group
    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file["data"]
        total_demos = len(demos)

        # Calculate split indices
        if val_dataset_percentage > 0.0:
            val_count = int(total_demos * val_dataset_percentage)
            train_count = total_demos - val_count
            if mode == "train":
                demo_indices = list(range(train_count))
            elif mode == "val":
                demo_indices = list(range(train_count, total_demos))
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
        else:
            demo_indices = list(range(total_demos))

        episode_ends = []
        prev_end = 0
        for i in demo_indices:
            demo = demos[f"demo_{i}"]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1] if episode_ends else 0
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.create_array(
            name="episode_ends",
            data=np.array(episode_ends, dtype=np.int64),
            compressor=None,
            overwrite=True,
        )

        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc=f"Loading {mode} lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            this_data = []
            for i in demo_indices:
                demo = demos[f"demo_{i}"]
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0) if this_data else np.array([])
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["action"]["shape"]
                )
            else:
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["obs"][key]["shape"]
                )
            _ = data_group.create_array(
                name=key,
                data=this_data,
                chunks=this_data.shape,
                compressor=None,
                overwrite=True,
            )

        # save image data
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys),
            desc=f"Loading {mode} image data",
            mininterval=1.0,
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=None,
                        dtype=np.uint8,
                    )
                    for demo_list_idx, episode_idx in enumerate(demo_indices):
                        demo = demos[f"demo_{episode_idx}"]
                        hdf5_arr = demo["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[demo_list_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

        # Auto-detect dino latents camera keys if not provided
        if dino_camera_keys is None:
            # Discover from the HDF5 group matching the requested latent type
            first_demo = demos[f"demo_{demo_indices[0]}"]
            # Get camera keys from first available dino type
            hdf5_group = f"dino_{dino_types[0]}"
            if hdf5_group in first_demo:
                dino_camera_keys = sorted(first_demo[hdf5_group][dino_model].keys())
            else:
                dino_camera_keys = []
        # Load precomputed latent action data
        for dino_type in dino_types:
            for cam_key in dino_camera_keys:
                zarr_key = f"dino_{dino_type}_{dino_model}_{cam_key}"
                hdf5_path = f"dino_{dino_type}/{dino_model}/{cam_key}"

                dino_latents = []
                for i in demo_indices:
                    demo = demos[f"demo_{i}"]
                    if hdf5_path in demo:
                        dino_latents.append(demo[hdf5_path][:].astype(np.float32))
                    else:
                        raise KeyError(
                            f"Missing precomputed latent actions at data/demo_{i}/{hdf5_path}. "
                            f"Run mip/networks/dino/precompute_dino.py first."
                        )
                dino_latents = np.concatenate(dino_latents, axis=0)
                assert dino_latents.shape[0] == n_steps, (
                    f"DINO latent steps mismatch: {dino_latents.shape[0]} vs {n_steps}"
                )
                _ = data_group.create_array(
                    name=zarr_key,
                    data=dino_latents,
                    chunks=dino_latents.shape,
                    compressor=None,
                    overwrite=True,
                )

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
