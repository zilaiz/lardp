"""Rollout recorder for collecting evaluation trajectories as HDF5.

Records per-step observations, actions, rewards, and dones during policy
evaluation and appends them to an HDF5 file in the robomimic dataset format.
"""

from pathlib import Path

import h5py
import numpy as np
from loguru import logger


class RolloutRecorder:
    """Records per-step rollout data and appends to HDF5 in robomimic format."""

    def __init__(self, obs_keys, obs_type="state", shape_meta=None):
        self.obs_keys = obs_keys
        self.obs_type = obs_type
        self.shape_meta = shape_meta
        self.episodes = []
        self._current_episode = None
        self._recording = False

    def start_episode(self):
        """Initialize a new episode buffer."""
        self._current_episode = {
            "obs": {key: [] for key in self.obs_keys},
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        self._recording = True

    def record_initial_obs(self, obs_dict):
        """Record the observation from env.reset() (first obs of the episode)."""
        if not self._recording or self._current_episode is None:
            return
        for key in self.obs_keys:
            self._current_episode["obs"][key].append(
                np.array(obs_dict[key], copy=True)
            )

    def record_step(self, action, obs_after_dict, reward, done):
        """Record one inner environment step.

        Args:
            action: The action taken at this step.
            obs_after_dict: The observation dict AFTER taking the action.
            reward: Scalar reward.
            done: Whether the episode terminated.
        """
        if not self._recording or self._current_episode is None:
            return
        self._current_episode["actions"].append(np.array(action, copy=True))
        self._current_episode["rewards"].append(float(reward))
        self._current_episode["dones"].append(bool(done))
        for key in self.obs_keys:
            self._current_episode["obs"][key].append(
                np.array(obs_after_dict[key], copy=True)
            )

    def end_episode(self):
        """Finalize the current episode and store it."""
        if self._current_episode is None:
            return
        ep = self._current_episode
        self._current_episode = None
        self._recording = False

        # Skip empty episodes
        if len(ep["actions"]) == 0:
            return

        # Convert lists to arrays
        for key in self.obs_keys:
            arr = np.array(ep["obs"][key])
            # We have T+1 observations (initial + T post-action).
            # Drop the last to align obs[0:T] with actions[0:T].
            ep["obs"][key] = arr[:-1]
        ep["actions"] = np.array(ep["actions"])
        ep["rewards"] = np.array(ep["rewards"])
        ep["dones"] = np.array(ep["dones"])

        self.episodes.append(ep)

    def append_hdf5(self, path, max_demos=None):
        """Append collected episodes to an HDF5 file in robomimic format.

        Creates the file if it doesn't exist, otherwise appends new demos
        with incrementing indices.

        Args:
            path: Path to the HDF5 file.
            max_demos: If set, stop appending once total demos reaches this cap.
        """
        if len(self.episodes) == 0:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(path), "a") as f:
            if "data" not in f:
                data_grp = f.create_group("data")
                data_grp.attrs["num_demos"] = 0
                data_grp.attrs["total"] = 0
            else:
                data_grp = f["data"]

            existing_demos = int(data_grp.attrs["num_demos"])
            total_samples = int(data_grp.attrs["total"])

            # Cap the number of episodes to append
            episodes = self.episodes
            if max_demos is not None and existing_demos >= max_demos:
                logger.info(f"Rollout cap reached ({existing_demos}/{max_demos}), skipping append")
                return
            if max_demos is not None:
                remaining = max_demos - existing_demos
                episodes = episodes[:remaining]

            for i, ep in enumerate(episodes):
                demo_idx = existing_demos + i
                demo_grp = data_grp.create_group(f"demo_{demo_idx}")

                # obs group
                obs_grp = demo_grp.create_group("obs")
                for key in self.obs_keys:
                    arr = ep["obs"][key]
                    # Convert float image tensors to uint8 to match demo format.
                    # Policy outputs (T, C, H, W) float32 in [0, 1];
                    # demos store (T, H, W, C) uint8 in [0, 255].
                    if "image" in key and arr.dtype in (np.float32, np.float64):
                        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                        if arr.ndim == 4:
                            arr = arr.transpose(0, 2, 3, 1)
                    obs_grp.create_dataset(key, data=arr)

                # actions, rewards, dones
                demo_grp.create_dataset("actions", data=ep["actions"])
                demo_grp.create_dataset("rewards", data=ep["rewards"])
                demo_grp.create_dataset("dones", data=ep["dones"])

                T = ep["actions"].shape[0]
                demo_grp.attrs["num_samples"] = T
                total_samples += T

            data_grp.attrs["num_demos"] = existing_demos + len(episodes)
            data_grp.attrs["total"] = total_samples

        logger.info(
            f"Appended {len(episodes)} rollout episodes to {path} "
            f"(total demos: {existing_demos + len(episodes)}"
            f"{f', cap: {max_demos}' if max_demos else ''})"
        )

    def clear(self):
        """Reset for the next eval round."""
        self.episodes = []
        self._current_episode = None
        self._recording = False
