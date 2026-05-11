"""Eval-only entrypoint for the retrieval goal predictor.

There's nothing to train: the agent loads the frozen IDM and a precomputed
retrieval index, then runs the standard env-eval loop. One row per
(task, K, query_space, distance, aggregation) cell.
"""

import os
import time
from contextlib import contextmanager

import hydra
import loguru
import numpy as np
import torch

# Set MuJoCo rendering backend before importing any robomimic/mujoco modules.
os.environ["MUJOCO_GL"] = "egl"  # noqa: E402

from mip.agent_goal_predictor_retrieval import RetrievalGoalAgent  # noqa: E402
from mip.config import Config  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.logger import Logger  # noqa: E402
from mip.samplers import get_default_step_list  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, record_dict: dict):
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


class _NormalizerStub:
    """Replaces ``base_dataset`` in the eval loop. Holds just the normalizer
    and a rotation transformer for ``undo_transform_action`` — the two things
    the eval loop actually consumes from the dataset object. Avoids loading
    the HDF5 + zarr image cache (which can take ~90 s for tool_hang)."""

    def __init__(self, normalizer, rotation_rep: str = "rotation_6d"):
        from mip.datasets.robomimic_dataset import RotationTransformer

        self.normalizer = normalizer
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

    # Lifted verbatim from RobomimicReplayImageDataset.undo_transform_action.
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


def eval(config: Config, envs, base_dataset, agent, logger, num_steps: int = 1):
    """Mirror the regressor's eval loop, with the retrieval agent in place.

    ``base_dataset`` only needs ``.normalizer`` and ``.undo_transform_action``;
    a ``_NormalizerStub`` works in place of the real dataset.
    """
    episode_rewards = []
    episode_steps = []
    episode_success = []

    inference_times = {
        "normalize": [],
        "sample": [],
        "unnormalize": [],
        "env_step": [],
    }

    for i in range(config.log.eval_episodes // config.task.num_envs):
        ep_reward = [0.0] * config.task.num_envs
        obs, _ = envs.reset()
        t = 0

        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))

        while t < config.task.max_episode_steps:
            with timed("normalize", inference_times):
                if config.task.obs_type == "image":
                    obs_raw = obs
                    obs_dict = {}
                    for k in obs_raw:
                        obs_k = obs_raw[k].astype(np.float32)
                        obs_k = base_dataset.normalizer["obs"][k].normalize(obs_k)
                        obs_k = torch.tensor(
                            obs_k,
                            device=config.optimization.device,
                            dtype=torch.float32,
                        )
                        obs_dict[k] = obs_k
                    obs = obs_dict
                else:
                    raise NotImplementedError(
                        "Retrieval goal predictor currently only supports image obs"
                    )

                act_0 = torch.randn(
                    (config.task.num_envs, config.task.horizon, config.task.act_dim),
                    device=config.optimization.device,
                )

            with timed("sample", inference_times):
                act_normed = agent.sample(
                    act_0=act_0,
                    obs=obs,
                    num_steps=num_steps,
                    use_ema=False,
                )

            with timed("unnormalize", inference_times):
                act_normed = act_normed.detach().to("cpu").numpy()
                act = base_dataset.normalizer["action"].unnormalize(act_normed)

                start = config.task.obs_steps - 1
                end = start + config.task.act_steps
                act = act[:, start:end, :]

                if config.task.abs_action and config.task.env_name in [
                    "can",
                    "lift",
                    "square",
                    "tool_hang",
                    "transport",
                ]:
                    act = base_dataset.undo_transform_action(act)

            with timed("env_step", inference_times):
                obs, reward, terminated, truncated, info = envs.step(act)
                _ = terminated | truncated
                ep_reward += reward
                t += config.task.act_steps

        success = [1.0 if s > 0 else 0.0 for s in ep_reward]
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} "
        f"Mean reward: {np.nanmean(episode_rewards)} "
        f"Mean success: {np.nanmean(episode_success)}"
    )

    if inference_times["sample"]:
        loguru.logger.info(
            f"Inference perf - "
            f"Normalize: {np.mean(inference_times['normalize']) * 1000:.2f}ms, "
            f"Sample: {np.mean(inference_times['sample']) * 1000:.2f}ms, "
            f"Unnormalize: {np.mean(inference_times['unnormalize']) * 1000:.2f}ms, "
            f"Env step: {np.mean(inference_times['env_step']) * 1000:.2f}ms"
        )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }
    if inference_times["sample"]:
        metrics[f"perf/inference_normalize_ms_{num_steps}"] = (
            np.mean(inference_times["normalize"]) * 1000
        )
        metrics[f"perf/inference_sample_ms_{num_steps}"] = (
            np.mean(inference_times["sample"]) * 1000
        )
        metrics[f"perf/inference_unnormalize_ms_{num_steps}"] = (
            np.mean(inference_times["unnormalize"]) * 1000
        )
        metrics[f"perf/inference_env_step_ms_{num_steps}"] = (
            np.mean(inference_times["env_step"]) * 1000
        )
    return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Eval-only pipeline for retrieval goal predictor."""
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("warn")
        torch.set_float32_matmul_precision("high")

    set_seed(config.optimization.seed)
    limit_threads(1)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, info = envs.reset()
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        config.task.obs_dim = obs.shape[-1]
    loguru.logger.info("Finished setting up env")

    # Use the IDM's matching normalizer so observations and the index are in
    # the same space. We only need the normalizer + a rotation transformer for
    # this eval — the full HDF5 + zarr image cache is not needed, so wrap them
    # in a stub instead of calling make_idm_dataset (saves ~90s on tool_hang).
    import pickle

    normalizer_path = os.path.join(
        os.path.dirname(config.optimization.idm_checkpoint_path), "normalizer.pkl"
    )
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            f"IDM normalizer not found at {normalizer_path}. The retrieval eval "
            "requires the normalizer that was used to train the IDM (and to build "
            "the retrieval index); building a new one here would be inconsistent."
        )
    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)
    loguru.logger.info(f"Loaded IDM normalizer from {normalizer_path}")

    base_dataset = _NormalizerStub(normalizer)
    loguru.logger.info(
        "Skipped HDF5/zarr load — using normalizer + rotation transformer stub"
    )

    agent = RetrievalGoalAgent(config)
    agent.eval()

    num_steps_list = get_default_step_list(config.optimization.loss_type)
    # Fold every num_steps level into a single wandb log call — metric keys are
    # already suffixed ``_{num_steps}`` so they don't collide, and using a
    # single increasing step avoids wandb's "monotonically increasing" warning
    # (it would drop later rows since num_steps_list is descending: [9, 3, 1]).
    combined = {"step": 0}
    for num_steps in num_steps_list:
        num_steps = int(num_steps)  # JSON logger doesn't like numpy int64
        combined.update(eval(config, envs, base_dataset, agent, logger, num_steps))
    logger.log(combined, category="eval")
    all_metrics = combined

    for key, val in all_metrics.items():
        if "mean_success" in key:
            loguru.logger.info(f"{key} - {val}")


if __name__ == "__main__":
    main()
