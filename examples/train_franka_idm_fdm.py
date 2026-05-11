"""Franka IDM + FDM training pipeline (real-robot, no sim env).

Mirrors examples/train_robomimic_idm_fdm.py but adapted for the Franka
real-robot setup:

- Loads the IDM dataset from one or more HDF5 files via ``make_idm_dataset``
  (the Franka conversion writes a robomimic-style HDF5, so the IDM loader
  works on it directly with ``abs_action=false`` — actions are already
  pre-expanded to 10-dim).
- No simulator: skips ``make_vec_env`` and rollout eval.
- Replaces inline rollout eval with periodic val-loss evaluation (IDM flow
  loss + FDM aux) on the held-out primary-HDF5 val split, mirroring
  train_franka.py's val tick.
- Saves the merged normalizer alongside checkpoints so downstream
  ``compute_goal_stats.py`` and goal-predictor training reuse the same scale.
"""

import os
import pickle
import time
from contextlib import contextmanager

import hydra
import loguru
import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.optim.lr_scheduler import CosineAnnealingLR

from mip.agent_idm_fdm import IDMFDMAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset, make_idm_dataset
from mip.logger import Logger
from mip.losses import get_norm
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import limit_threads, set_seed

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, perf: dict):
    t0 = time.perf_counter()
    yield
    perf[section].append(time.perf_counter() - t0)


def _stack_obs_goal(batch, config: Config) -> TensorDict:
    """Stack obs (To frames) + goal (1 frame) per key into (B, To+1, ...) on device."""
    device = config.optimization.device
    obs_batch = batch["obs"]
    goal_batch = batch["goal_obs"]
    obs_dict: dict = {}
    for k in obs_batch:
        obs = obs_batch[k][:, : config.task.obs_steps].to(device, non_blocking=True)
        goal = goal_batch[k].to(device, non_blocking=True)
        obs_dict[k] = torch.cat([obs, goal], dim=1)
    bs = next(iter(obs_dict.values())).shape[0]
    return TensorDict(obs_dict, batch_size=bs)


@torch.no_grad()
def compute_val_loss(
    agent: IDMFDMAgent,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    max_batches: int = 50,
) -> dict[str, float]:
    """IDM flow loss + FDM auxiliary loss on the val set (EMA modules)."""
    agent.eval()
    cfg = config.optimization
    encoder = agent.encoder_ema if cfg.ema_rate < 1 else agent.encoder
    flow_map = agent.flow_map_ema if cfg.ema_rate < 1 else agent.flow_map

    idm_losses: list[float] = []
    fdm_losses: list[float] = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        obs = _stack_obs_goal(batch, config)
        act = batch["action"][:, : config.task.horizon, :].to(cfg.device, non_blocking=True)

        encoded = encoder(obs, None)
        t = torch.empty(act.shape[0], device=cfg.device).uniform_(0, 1)
        act_0 = torch.randn_like(act)
        act_t = agent.interpolant.calc_It(t, act_0, act)
        act_t_dot = agent.interpolant.calc_It_dot(t, act_0, act)
        b_t = flow_map.get_velocity(t, act_t, encoded)
        idm_losses.append(float(torch.mean(get_norm(b_t - act_t_dot, cfg.norm_type)).item()))

        target_goal = encoded[:, -1].detach()
        pred_goal = flow_map.net.forward_predict(encoded, act)
        fdm_losses.append(float(F.mse_loss(pred_goal, target_goal).item()))

    agent.train()
    return {
        "val_idm_loss": float(np.mean(idm_losses)) if idm_losses else float("nan"),
        "val_fdm_loss": float(np.mean(fdm_losses)) if fdm_losses else float("nan"),
    }


def train(
    config: Config,
    dataset,
    val_dataset,
    agent: IDMFDMAgent,
    logger: Logger,
    resume_state: dict | None = None,
):
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    loop_train = loop_dataloader(train_loader)

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.optimization.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

    lr_scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=config.optimization.gradient_steps,
    )
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    start_step = 0
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        loguru.logger.info(f"Resuming training from step {start_step}")
        for _ in range(start_step):
            lr_scheduler.step()

    info_list: list[dict] = []
    start_time = time.time()
    perf = {"data_load": [], "preprocess": [], "update": [], "total_step": []}

    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        with timed("total_step", perf):
            with timed("data_load", perf):
                batch = next(loop_train)

            with timed("preprocess", perf):
                obs = _stack_obs_goal(batch, config)
                act = batch["action"][:, : config.task.horizon, :].to(
                    config.optimization.device, non_blocking=True,
                )

            with timed("update", perf):
                delta_t_scalar = warmup_scheduler(n_gradient_step)
                delta_t = torch.full(
                    (act.shape[0],), delta_t_scalar, device=config.optimization.device,
                )
                info = agent.update(act, obs, delta_t)
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
            for k in info:
                try:
                    metrics[k] = np.nanmean([i[k] for i in info_list])
                except Exception as e:
                    loguru.logger.error(f"Error averaging {k}: {e}")
                    metrics[k] = np.nan
            window = config.log.log_freq
            if perf["total_step"]:
                metrics["perf/data_load_ms"]  = np.mean(perf["data_load"][-window:])  * 1000
                metrics["perf/preprocess_ms"] = np.mean(perf["preprocess"][-window:]) * 1000
                metrics["perf/update_ms"]     = np.mean(perf["update"][-window:])     * 1000
                metrics["perf/total_step_ms"] = np.mean(perf["total_step"][-window:]) * 1000
                metrics["perf/steps_per_sec"] = 1.0 / np.mean(perf["total_step"][-window:])
            logger.log(metrics, category="train")
            info_list = []

        if ((n_gradient_step + 1) % config.log.eval_freq) == 0:
            eval_metrics: dict = {"step": n_gradient_step}
            if val_loader is not None:
                loguru.logger.info("Computing val IDM/FDM loss...")
                eval_metrics.update(compute_val_loss(agent, val_loader, config))
                for k, v in eval_metrics.items():
                    if k != "step":
                        loguru.logger.info(f"  {k} = {v:.4f}")
            logger.log(eval_metrics, category="eval")

        if ((n_gradient_step + 1) % config.log.save_freq) == 0:
            loguru.logger.info("Saving checkpoints (latest + step snapshot)...")
            logger.save_agent(agent=agent, identifier="latest")
            logger.save_agent(agent=agent, identifier=f"step_{n_gradient_step + 1}")


@hydra.main(version_base=None, config_path="configs/", config_name="main_franka")
def main(config: Config):
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("warn")
        torch.set_float32_matmul_precision("high")

    set_seed(config.optimization.seed)
    limit_threads(1)
    logger = Logger(config)
    loguru.logger.info("Logger ready")

    # Image obs_dim is set to network.emb_dim (the encoder output dim) — this
    # mirrors what train_robomimic_idm_fdm does when obs_type == "image".
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        raise NotImplementedError("Franka IDM+FDM training only supports image obs")

    # --- datasets (multi-HDF5 via make_idm_dataset) ---
    dataset = make_idm_dataset(config.task, mode="train")
    loguru.logger.info(f"Train IDM dataset: {len(dataset)} samples")

    base_dataset = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )

    # Val dataset: use only the *primary* HDF5's val split (the play datasets
    # have no held-out split). Share the merged normalizer from the train side
    # so val loss is on the same scale.
    val_dataset = None
    if config.task.val_dataset_percentage > 0.0:
        primary_path = (
            config.task.dataset_paths[0]
            if config.task.dataset_paths
            else config.task.dataset_path
        )
        val_dataset = RobomimicImageIDMDataset(
            dataset_dir=os.path.expanduser(primary_path),
            shape_meta=config.task.shape_meta,
            n_obs_steps=config.task.obs_steps,
            horizon=config.task.horizon,
            pad_before=config.task.obs_steps - 1,
            pad_after=config.task.act_steps - 1,
            abs_action=config.task.abs_action,
            val_dataset_percentage=config.task.val_dataset_percentage,
            mode="val",
            normalizer=base_dataset.normalizer,
        )
        loguru.logger.info(f"Val IDM dataset: {len(val_dataset)} samples")

    # Save the merged normalizer alongside checkpoints — downstream goal stats
    # and goal-predictor training will pick this up by sibling-of-checkpoint
    # convention.
    normalizer_path = os.path.join(logger._model_dir, "normalizer.pkl")
    with open(normalizer_path, "wb") as f:
        pickle.dump(base_dataset.normalizer, f)
    loguru.logger.info(f"Saved merged normalizer to {normalizer_path}")

    # --- agent ---
    agent = IDMFDMAgent(config)
    loguru.logger.info(
        f"Created IDMFDMAgent with fdm_loss_scale={config.optimization.fdm_loss_scale}"
    )

    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    else:
        loguru.logger.info(
            "Starting fresh (auto-resume not supported here; pass "
            "optimization.model_path to resume from latest.pt)"
        )

    if config.mode == "train":
        train(config, dataset, val_dataset, agent, logger, resume_state=resume_state)
    else:
        raise ValueError(
            f"Franka IDM+FDM pipeline only supports mode='train' (got {config.mode!r})"
        )


if __name__ == "__main__":
    main()
