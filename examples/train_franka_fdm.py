"""Franka BYOL-FDM encoder pretraining (real-robot, no sim env).

Mirrors ``examples/train_robomimic_fdm.py`` but adapted for the Franka
real-robot setup, paralleling how ``train_franka_idm_fdm.py`` adapts the
robomimic IDM+FDM pipeline:

- Loads the IDM dataset from one or more HDF5 files via ``make_idm_dataset``
  (the Franka conversion writes a robomimic-style HDF5).
- No simulator: skips ``make_vec_env`` and rollout eval.
- Replaces inline rollout eval with periodic val-loss evaluation (BYOL
  cos-sim) on the held-out primary-HDF5 val split, mirroring
  ``train_franka_idm_fdm.py``'s val tick.
- Saves the merged normalizer alongside checkpoints so downstream
  ``compute_goal_stats.py`` and goal-predictor / joint-DDT-frozen training
  reuse the same scale.
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

from mip.agent_fdm import FDMAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset, make_idm_dataset
from mip.logger import Logger
from mip.torch_utils import limit_threads, set_seed

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, perf: dict):
    t0 = time.perf_counter()
    yield
    perf[section].append(time.perf_counter() - t0)


def _split_obs_goal(batch, config: Config) -> tuple[TensorDict, TensorDict]:
    """Build obs (B, To_obs, ...) and goal_obs (B, 1, ...) per key on device.

    Unlike the IDM-FDM helper that *stacks* obs+goal, BYOL-FDM keeps them
    separate — the online encoder sees obs, the EMA target encoder sees
    goal_obs.
    """
    device = config.optimization.device
    obs_batch = batch["obs"]
    goal_batch = batch["goal_obs"]
    obs_dict: dict = {}
    goal_dict: dict = {}
    for k in obs_batch:
        obs_dict[k] = obs_batch[k][:, : config.task.obs_steps].to(
            device, non_blocking=True,
        )
        goal_dict[k] = goal_batch[k].to(device, non_blocking=True)
    bs = next(iter(obs_dict.values())).shape[0]
    return (
        TensorDict(obs_dict, batch_size=bs),
        TensorDict(goal_dict, batch_size=bs),
    )


@torch.no_grad()
def compute_val_loss(
    agent: FDMAgent,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    max_batches: int = 50,
) -> dict[str, float]:
    """BYOL cos-sim loss on the val set using the EMA modules.

    Uses encoder_ema for both obs and goal_obs (and flow_map_ema's dynamics
    predictor) so val measures the EMA model's quality on held-out data —
    that's what downstream consumes from the checkpoint.
    """
    agent.eval()
    cfg = config.optimization
    encoder = agent.encoder_ema if cfg.ema_rate < 1 else agent.encoder
    flow_map = agent.flow_map_ema if cfg.ema_rate < 1 else agent.flow_map

    cos_sims: list[float] = []
    pred_stds: list[float] = []
    target_stds: list[float] = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        obs, goal_obs = _split_obs_goal(batch, config)
        act = batch["action"][:, : config.task.horizon, :].to(
            cfg.device, non_blocking=True,
        )

        z_obs = encoder(obs, None)
        z_pred = flow_map.net.forward_online(z_obs, act)
        z_goal = encoder(goal_obs, None)[:, 0]

        pred_n = F.normalize(z_pred, dim=-1)
        target_n = F.normalize(z_goal, dim=-1)
        cos_sims.append(float((pred_n * target_n).sum(-1).mean().item()))
        pred_stds.append(float(z_pred.std(dim=0).mean().item()))
        target_stds.append(float(z_goal.std(dim=0).mean().item()))

    agent.train()
    return {
        "val_cos_sim": float(np.mean(cos_sims)) if cos_sims else float("nan"),
        "val_loss": (2.0 - 2.0 * float(np.mean(cos_sims))) if cos_sims else float("nan"),
        "val_pred_std": float(np.mean(pred_stds)) if pred_stds else float("nan"),
        "val_target_std": float(np.mean(target_stds)) if target_stds else float("nan"),
    }


def train(
    config: Config,
    dataset,
    val_dataset,
    agent: FDMAgent,
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
                obs, goal_obs = _split_obs_goal(batch, config)
                act = batch["action"][:, : config.task.horizon, :].to(
                    config.optimization.device, non_blocking=True,
                )

            with timed("update", perf):
                info = agent.update(act, obs, goal_obs)
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
                loguru.logger.info("Computing val BYOL cos-sim loss...")
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
    # mirrors what train_robomimic_fdm does when obs_type == "image".
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        raise NotImplementedError("Franka FDM training only supports image obs")

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
    agent = FDMAgent(config)
    loguru.logger.info(
        f"Created FDMAgent (ema_rate={config.optimization.ema_rate})"
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
            f"Franka FDM pipeline only supports mode='train' (got {config.mode!r})"
        )


if __name__ == "__main__":
    main()
