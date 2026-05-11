"""Training pipeline for the deterministic goal-predictor regressor.

Mirrors ``train_robomimic_goal_predictor_dit.py`` exactly — same dataset,
same eval loop, same inputs to ``agent.update`` — but instantiates a
``GoalPredictorRegressorAgent`` (deterministic head) instead of the DiT
flow-matching agent. The frozen IDM, normalizer and goal stats are loaded
the same way.
"""

import os
import time
from contextlib import contextmanager

import hydra
import loguru
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set MuJoCo rendering backend before importing any robomimic/mujoco modules
os.environ["MUJOCO_GL"] = "egl"  # noqa: E402

from mip.agent_goal_predictor_regressor import GoalPredictorRegressorAgent  # noqa: E402
from mip.config import Config  # noqa: E402
from mip.dataset_utils import loop_dataloader  # noqa: E402
from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.logger import (  # noqa: E402
    Logger,
    compute_average_metrics,
    update_best_metrics,
)
from mip.samplers import get_default_step_list  # noqa: E402
from mip.scheduler import WarmupAnnealingScheduler  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, record_dict: dict):
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


def train(config: Config, envs, dataset, agent, logger, resume_state=None):
    """Goal predictor regressor training loop."""
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

    # Kept for interface parity with the DiT pipeline; the regressor itself
    # has no time variable, so delta_t is not used inside the agent.
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        loguru.logger.info(f"Resuming training from step {start_step}")
        loguru.logger.info(f"Restored best metrics: {best_metrics}")
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
                from tensordict import TensorDict

                if config.task.obs_type == "image":
                    obs_batch = batch["obs"]
                    obs_dict = {}
                    for k in obs_batch:
                        obs_dict[k] = obs_batch[k][:, : config.task.obs_steps].to(
                            config.optimization.device
                        )

                    goal_batch = batch["goal_obs"]
                    goal_dict = {}
                    for k in goal_batch:
                        goal_dict[k] = goal_batch[k].to(config.optimization.device)

                    batch_size = next(iter(obs_dict.values())).shape[0]
                    obs = TensorDict(obs_dict, batch_size=batch_size)
                    goal_obs = TensorDict(goal_dict, batch_size=batch_size)
                else:
                    raise NotImplementedError(
                        "Goal predictor regressor training currently only supports "
                        "image observations"
                    )

                act = batch["action"].to(config.optimization.device)
                act = act[:, : config.task.horizon, :]

            with timed("update", perf_times):
                delta_t_scalar = warmup_scheduler(n_gradient_step)
                batch_size = act.shape[0]
                delta_t = torch.full(
                    (batch_size,),
                    delta_t_scalar,
                    device=config.optimization.device,
                )
                info = agent.update(act, obs, goal_obs, delta_t)
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
                "delta_t": delta_t_scalar,
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

        if ((n_gradient_step + 1) % config.log.eval_freq) == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics.update(
                    eval(config, envs, dataset, agent, logger, num_steps)
                )

            old_best_metrics = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)

            primary_metric_key = f"mean_success_{num_steps_list[0]}"
            if primary_metric_key in metrics:
                is_new_best = (
                    primary_metric_key not in old_best_metrics
                    or metrics[primary_metric_key]
                    > old_best_metrics[primary_metric_key]
                )
                if is_new_best:
                    success_rate = metrics[primary_metric_key]
                    loguru.logger.info(
                        f"New best model! {primary_metric_key} = {success_rate:.4f}"
                    )
                    logger.save_agent(agent=agent, identifier="best")

                    checkpoint_base_name = (
                        f"{config.task.env_name}_{config.task.env_type}_"
                        f"{config.task.obs_type}_{config.optimization.loss_type}_"
                        f"{config.network.network_type}_{config.network.emb_dim}_"
                        f"seed{config.optimization.seed}"
                    )
                    training_state = {
                        "n_gradient_step": n_gradient_step,
                        "best_metrics": best_metrics,
                        "eval_history": eval_history,
                    }
                    logger.save_global_checkpoint(
                        agent,
                        checkpoint_base_name,
                        success_rate,
                        training_state=training_state,
                    )

            for key, value in best_metrics.items():
                metrics[f"best_{key}"] = value
            for key, value in avg_metrics.items():
                metrics[key] = value

            loguru.logger.info("Best metrics so far:")
            for key, value in best_metrics.items():
                loguru.logger.info(f"  {key}: {value:.4f}")
            if avg_metrics:
                loguru.logger.info("Average metrics (last 5 evals):")
                for key, value in avg_metrics.items():
                    loguru.logger.info(f"  {key}: {value:.4f}")

            logger.log(metrics, category="eval")
            agent.train()


def eval(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Evaluate the regressor + frozen IDM as a coupled policy."""
    episode_rewards = []
    episode_steps = []
    episode_success = []

    inference_times = {
        "normalize": [],
        "sample": [],
        "unnormalize": [],
        "env_step": [],
    }

    base_dataset = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )

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
                        "Goal predictor regressor eval currently only supports image obs"
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
                    use_ema=True,
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
    """Main pipeline for goal-predictor regressor training."""
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

    # Load normalizer from pretrained IDM to ensure consistent normalization.
    import pickle

    idm_normalizer = None
    if config.optimization.idm_checkpoint_path:
        normalizer_path = os.path.join(
            os.path.dirname(config.optimization.idm_checkpoint_path),
            "normalizer.pkl",
        )
        if os.path.exists(normalizer_path):
            with open(normalizer_path, "rb") as f:
                idm_normalizer = pickle.load(f)
            loguru.logger.info(f"Loaded IDM normalizer from {normalizer_path}")
        else:
            loguru.logger.warning(
                f"IDM normalizer not found at {normalizer_path}, "
                "will compute new one"
            )

    dataset = make_idm_dataset(config.task, normalizer=idm_normalizer)
    loguru.logger.info("Finished setting up IDM dataset")

    agent = GoalPredictorRegressorAgent(config)
    resume_state = None

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading goal-predictor regressor from {config.optimization.model_path}"
        )
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
    elif config.mode == "eval":
        agent.eval()
        num_steps_list = get_default_step_list(config.optimization.loss_type)
        for num_steps in num_steps_list:
            metrics = {"step": num_steps}
            metrics.update(eval(config, envs, dataset, agent, logger, num_steps))
            logger.log(metrics, category="eval")

        for key, val in metrics.items():
            if "mean_success" in key:
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
