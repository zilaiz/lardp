"""Training pipeline for the LBMDiTJointPTAgent on PushT.

Single-trunk PT joint state+action flow-matching policy on PushT image obs.
Self-contained (does NOT reuse the robomimic e2e train loop): the training
loop mirrors the joint pipeline (decoupled state/action flow times, optimality
+ CFG, target LN, encoder EMA, the joint_* knobs), but evaluation uses PushT's
coverage metric rather than robomimic's sparse-reward success.

Data: make_pusht_goal_dataset (expert zarr, optimality=0, with the train/val
demo split + optional rollout HDF5 mixed in as optimality=1). Env: PushT
make_vec_env. Agent: LBMDiTJointPTAgent (task-agnostic; reused unchanged).

Author: Zilai Zeng
"""

import os
import time

import hydra
import loguru
import numpy as np
import torch
from tensordict import TensorDict
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ.setdefault("MUJOCO_GL", "egl")  # noqa: E402 (harmless on PushT; pymunk render is CPU)

from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent  # noqa: E402
from mip.config import Config  # noqa: E402
from mip.dataset_utils import (  # noqa: E402
    loop_dataloader,
    make_expert_weighted_sampler,
)
from mip.datasets.pusht_dataset import make_pusht_goal_dataset  # noqa: E402
from mip.envs.pusht import make_vec_env  # noqa: E402
from mip.logger import (  # noqa: E402
    Logger,
    compute_average_metrics,
    update_best_metrics,
)
from mip.samplers import get_default_step_list  # noqa: E402
from mip.scheduler import WarmupAnnealingScheduler  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

torch.set_float32_matmul_precision("high")


def _read_optimality(batch, device):
    """Per-sample optimality labels {0=expert, 1=play}, or None when absent."""
    if "optimality" in batch:
        return batch["optimality"].to(device=device, dtype=torch.long)
    return None


def train(config: Config, envs, dataset, agent, logger, resume_state=None):
    """Joint state+action training loop with PushT coverage eval."""
    sampler = make_expert_weighted_sampler(
        dataset, config.optimization.expert_sample_fraction,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=8,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    loop_loader = loop_dataloader(dataloader)

    lr_scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=config.optimization.gradient_steps
    )
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
        for _ in range(start_step):
            lr_scheduler.step()

    info_list = []
    start_time = time.time()
    device = config.optimization.device

    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        batch = next(loop_loader)

        if config.task.obs_type != "image":
            raise NotImplementedError(
                "PushT joint pipeline currently only supports image obs"
            )
        obs_batch = batch["obs"]
        obs_dict = {
            k: obs_batch[k][:, : config.task.obs_steps].to(device)
            for k in obs_batch
        }
        goal_batch = batch["goal_obs"]
        goal_dict = {k: goal_batch[k].to(device) for k in goal_batch}
        batch_size = next(iter(obs_dict.values())).shape[0]
        obs = TensorDict(obs_dict, batch_size=batch_size)
        goal_obs = TensorDict(goal_dict, batch_size=batch_size)

        act = batch["action"][:, : config.task.horizon, :].to(device)
        optimality = _read_optimality(batch, device)

        delta_t_scalar = warmup_scheduler(n_gradient_step)
        delta_t = torch.full((batch_size,), delta_t_scalar, device=device)
        info = agent.update(act, obs, goal_obs, delta_t, optimality)
        lr_scheduler.step()

        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.item()
        info_list.append(info)

        if ((n_gradient_step + 1) % config.log.log_freq) == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([d[key] for d in info_list])
                except (KeyError, TypeError, ValueError) as e:
                    loguru.logger.error(f"Error calculating {key}: {e}")
                    metrics[key] = np.nan
            logger.log(metrics, category="train")
            info_list = []

        if ((n_gradient_step + 1) % config.log.save_freq) == 0:
            loguru.logger.info("Save model...")
            logger.save_agent(agent=agent, identifier="latest")

        if ((n_gradient_step + 1) % config.log.eval_freq) == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = (
                config.optimization.eval_num_steps
                or get_default_step_list(config.optimization.loss_type)
            )
            for num_steps in num_steps_list:
                metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps))

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
                        f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_seed{config.optimization.seed}"
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

            logger.log(metrics, category="eval")
            agent.train()


def evaluate(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Closed-loop PushT eval of the joint trunk (coverage metric)."""
    episode_rewards = []
    episode_steps = []
    episode_success = []

    # PushTMixedDataset exposes the (shared) normalizer; a single goal dataset
    # has it directly.
    normalizer = dataset.normalizer
    device = config.optimization.device

    for i in range(config.log.eval_episodes // config.task.num_envs):
        step_reward = []
        ep_reward = [0.0] * config.task.num_envs
        # Re-seed each env per round (the vec env's own reset seeding is stale).
        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * config.task.num_envs + j)
        obs, _ = envs.reset()
        t = 0

        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))

        while t < config.task.max_episode_steps:
            if config.task.obs_type != "image":
                raise NotImplementedError(
                    "PushT joint eval currently only supports image obs"
                )
            obs_raw = obs
            obs = {}
            for k in obs_raw:
                obs_k = obs_raw[k].astype(np.float32)
                obs_k = normalizer["obs"][k].normalize(obs_k)
                obs[k] = torch.tensor(obs_k, device=device, dtype=torch.float32)

            act_0 = torch.randn(
                (config.task.num_envs, config.task.horizon, config.task.act_dim),
                device=device,
            )
            act_normed = agent.sample(
                act_0=act_0, obs=obs, num_steps=num_steps, use_ema=True,
            )
            act_normed = act_normed.detach().to("cpu").numpy()
            act = normalizer["action"].unnormalize(act_normed)

            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            obs, reward, terminated, truncated, _ = envs.step(act)
            _ = terminated | truncated
            ep_reward += reward
            step_reward.append(reward)
            t += config.task.act_steps

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} "
        f"Mean reward: {np.nanmean(episode_rewards)} "
        f"Mean success: {np.nanmean(episode_success)}"
    )

    return {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("warn")
        torch.set_float32_matmul_precision("high")

    set_seed(config.optimization.seed)
    limit_threads(1)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        config.task.obs_dim = obs.shape[-1]
    loguru.logger.info("Finished setting up env")

    dataset = make_pusht_goal_dataset(config.task)
    loguru.logger.info("Finished setting up PushT goal dataset")

    agent = LBMDiTJointPTAgent(config)
    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading LBMDiTJointPT from {config.optimization.model_path}"
        )
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
    elif config.mode == "eval":
        agent.eval()
        num_steps_list = (
            config.optimization.eval_num_steps
            or get_default_step_list(config.optimization.loss_type)
        )
        for num_steps in num_steps_list:
            metrics = {"step": num_steps}
            metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps))
            logger.log(metrics, category="eval")
        for key, val in metrics.items():
            if "mean_success" in key:
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
