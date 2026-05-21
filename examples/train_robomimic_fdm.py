"""BYOL-style forward-dynamics encoder pretraining on robomimic.

Trains an obs encoder by self-predicting the next-state embedding from
(z_obs, action_chunk). Uses ``FDMAgent`` (online dynamics-predictor MLP +
EMA target encoder, no projector, BYOL cosine loss).

Compared to ``train_robomimic_idm_fdm.py`` this pipeline:

  * Uses ``FDMAgent`` + the ``fdm`` network (no IDM flow-matching).
  * Does NOT wrap the encoder in ``GoalDropoutEncoder`` — BYOL-FDM doesn't
    use a learned uncond_emb, and ``LBMDiTJointDDTFrozenAgent`` accepts the
    plain encoder layout via its has_goal_dropout=False branch.
  * Passes ``obs`` (To_obs frames) and ``goal_obs`` (1 frame) separately to
    the agent instead of stacking them — the online side only encodes obs,
    the EMA target only encodes s'.
  * Skips eval-during-training. Cosine-sim and feature-std diagnostics
    are the in-loop health signals; downstream rollout eval is done after
    the encoder is plugged into ``LBMDiTJointDDTFrozenAgent``.
"""

import os
import time
from contextlib import contextmanager

import hydra
import loguru
import numpy as np
import torch
from tensordict import TensorDict
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set MuJoCo rendering backend before importing any robomimic/mujoco modules
os.environ["MUJOCO_GL"] = "egl"  # noqa: E402

from mip.agent_fdm import FDMAgent  # noqa: E402
from mip.config import Config  # noqa: E402
from mip.dataset_utils import loop_dataloader  # noqa: E402
from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.logger import Logger  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, record_dict: dict):
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


def train(config: Config, dataset, agent, logger, resume_state=None):
    """BYOL-FDM training loop.

    Splits the batch into (obs, goal_obs, act) and forwards them to the
    agent. The agent encodes obs with the online encoder and goal_obs with
    the EMA target encoder internally.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    loop_loader = loop_dataloader(dataloader)

    lr_scheduler = CosineAnnealingLR(
        agent.optimizer,
        T_max=config.optimization.gradient_steps,
    )

    start_step = 0
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        loguru.logger.info(f"Resuming training from step {start_step}")
        for _ in range(start_step):
            lr_scheduler.step()

    info_list = []
    start_time = time.time()
    perf_times = {
        "data_load": [],
        "preprocess": [],
        "update": [],
        "total_step": [],
    }

    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        with timed("total_step", perf_times):
            with timed("data_load", perf_times):
                batch = next(loop_loader)

            with timed("preprocess", perf_times):
                if config.task.obs_type != "image":
                    raise NotImplementedError(
                        "BYOL-FDM training currently only supports image observations"
                    )

                # Build obs (B, To_obs, ...) and goal_obs (B, 1, ...) per key.
                # No stacking — the agent runs them through separate encoder
                # branches (online for obs, EMA target for goal).
                obs_batch = batch["obs"]
                goal_batch = batch["goal_obs"]
                obs_dict = {}
                goal_dict = {}
                for k in obs_batch:
                    obs_dict[k] = obs_batch[k][:, : config.task.obs_steps].to(
                        config.optimization.device
                    )
                    goal_dict[k] = goal_batch[k].to(config.optimization.device)

                batch_size = next(iter(obs_dict.values())).shape[0]
                obs = TensorDict(obs_dict, batch_size=batch_size)
                goal_obs = TensorDict(goal_dict, batch_size=batch_size)

                act = batch["action"].to(config.optimization.device)
                act = act[:, : config.task.horizon, :]  # (B, horizon, act_dim)

            with timed("update", perf_times):
                info = agent.update(act, obs, goal_obs)
                lr_scheduler.step()

            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    info[k] = v.item()
            info_list.append(info)

        # log metrics
        if ((n_gradient_step + 1) % config.log.log_freq) == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except Exception as e:
                    loguru.logger.error(f"Error calculating {key}: {e}")
                    metrics[key] = np.nan

            if perf_times["total_step"]:
                window = config.log.log_freq
                metrics["perf/data_load_ms"] = (
                    np.mean(perf_times["data_load"][-window:]) * 1000
                )
                metrics["perf/preprocess_ms"] = (
                    np.mean(perf_times["preprocess"][-window:]) * 1000
                )
                metrics["perf/update_ms"] = (
                    np.mean(perf_times["update"][-window:]) * 1000
                )
                metrics["perf/total_step_ms"] = (
                    np.mean(perf_times["total_step"][-window:]) * 1000
                )
                metrics["perf/steps_per_sec"] = 1.0 / np.mean(
                    perf_times["total_step"][-window:]
                )

            logger.log(metrics, category="train")
            info_list = []

        if ((n_gradient_step + 1) % config.log.save_freq) == 0:
            loguru.logger.info("Save model...")
            logger.save_agent(agent=agent, identifier="latest")
            logger.save_agent(agent=agent, identifier=f"step_{n_gradient_step + 1}")


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline for BYOL-FDM encoder pretraining."""
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("warn")
        torch.set_float32_matmul_precision("high")

    set_seed(config.optimization.seed)
    limit_threads(1)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    # env setup is only needed to populate task.obs_dim for image tasks.
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, info = envs.reset()
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        config.task.obs_dim = obs.shape[-1]
    envs.close()
    loguru.logger.info("Finished setting up env (closed; eval is not run inline)")

    # dataset setup — uses IDM dataset with goal obs and multi-HDF5 support
    dataset = make_idm_dataset(config.task)
    loguru.logger.info("Finished setting up IDM dataset")

    # Save normalizer so downstream tasks (e.g., joint_ddt_frozen, compute_goal_stats)
    # use the same one.
    import pickle

    base_dataset = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )
    normalizer_path = os.path.join(logger._model_dir, "normalizer.pkl")
    with open(normalizer_path, "wb") as f:
        pickle.dump(base_dataset.normalizer, f)
    loguru.logger.info(f"Saved dataset normalizer to {normalizer_path}")

    agent = FDMAgent(config)
    loguru.logger.info(
        f"Created FDMAgent (ema_rate={config.optimization.ema_rate})"
    )

    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.optimization.auto_resume:
        checkpoint_base_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}"
        )
        checkpoint_path = logger.find_latest_checkpoint(checkpoint_base_name)
        if checkpoint_path:
            loguru.logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
            resume_state = agent.load(str(checkpoint_path), load_optimizer=True)
        else:
            loguru.logger.info("No checkpoint found, starting training from scratch")

    if config.mode == "train":
        train(config, dataset, agent, logger, resume_state=resume_state)
    else:
        raise ValueError(
            f"BYOL-FDM pipeline only supports mode=train; got mode={config.mode}"
        )


if __name__ == "__main__":
    main()
