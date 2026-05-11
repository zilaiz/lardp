"""Franka goal-predictor DiT training pipeline (real-robot, no sim env).

Mirrors examples/train_robomimic_goal_predictor_dit.py but adapted for
the Franka real-robot setup:

- Loads the IDM dataset (with goal_obs) from one or more HDF5 files via
  ``make_idm_dataset``.
- Loads a frozen IDM checkpoint trained by ``train_franka_idm_fdm.py``.
  The IDM normalizer (saved alongside the IDM checkpoint as
  ``normalizer.pkl``) is reused so the goal predictor sees the same scale
  the IDM was trained on.
- No simulator, no rollout eval. Inline eval is replaced with periodic val
  loss (state flow loss + action reg loss) on the held-out primary HDF5's
  val split, mirroring train_franka.py's val tick.
"""

import os
import pickle
import time
from contextlib import contextmanager

import hydra
import loguru
import numpy as np
import torch
from tensordict import TensorDict
from torch.optim.lr_scheduler import CosineAnnealingLR

from mip.agent_goal_predictor_dit import GoalPredictorDiTAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset, make_idm_dataset
from mip.logger import Logger
from mip.losses import get_norm
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import at_least_ndim, limit_threads, set_seed

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, perf: dict):
    t0 = time.perf_counter()
    yield
    perf[section].append(time.perf_counter() - t0)


def _to_device_obs(batch, key, config: Config) -> TensorDict:
    """Move a per-key obs dict subset to device as a TensorDict."""
    device = config.optimization.device
    obs_batch = batch[key]
    obs_dict: dict = {}
    if key == "obs":
        for k in obs_batch:
            obs_dict[k] = obs_batch[k][:, : config.task.obs_steps].to(device, non_blocking=True)
    else:
        for k in obs_batch:
            obs_dict[k] = obs_batch[k].to(device, non_blocking=True)
    bs = next(iter(obs_dict.values())).shape[0]
    return TensorDict(obs_dict, batch_size=bs)


@torch.no_grad()
def compute_val_loss(
    agent: GoalPredictorDiTAgent,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    max_batches: int = 50,
) -> dict[str, float]:
    """State flow loss + action reg loss on the val set, using EMA goal-DiT.

    Action loss path mirrors training: one-step Euler denoising
    `x_pred_clean = x_t + (1 - t_flow) * v_pred`, denormalize, feed to IDM.
    """
    agent.eval()
    cfg = config.optimization

    state_losses: list[float] = []
    action_losses: list[float] = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        obs = _to_device_obs(batch, "obs", config)
        goal_obs = _to_device_obs(batch, "goal_obs", config)
        act = batch["action"][:, : config.task.horizon, :].to(cfg.device, non_blocking=True)

        z_t = agent._inner_encoder(obs, None)
        z_goal_raw = agent._inner_encoder(goal_obs, None)
        z_goal = agent._normalize(z_goal_raw)

        B = z_goal.shape[0]
        t_flow = torch.empty(B, device=z_goal.device).uniform_(0, 1)
        x0 = torch.randn_like(z_goal)
        x1 = z_goal
        x_t = agent.goal_interpolant.calc_It(t_flow, x0, x1)
        x_t_dot = agent.goal_interpolant.calc_It_dot(t_flow, x0, x1)
        v_pred, _, _ = agent.goal_dit_ema(
            x_t, t_flow, t_flow, z_t, align_depth=None,
        )
        state_losses.append(
            float(torch.mean(get_norm(v_pred - x_t_dot, cfg.norm_type)).item())
        )

        # Always compute action flow matching loss for monitoring,
        # regardless of action_reg_weight.
        t_flow_expanded = at_least_ndim(t_flow, x_t.dim())
        x_pred_clean = x_t + (1.0 - t_flow_expanded) * v_pred
        g_hat = agent._denormalize(x_pred_clean)
        obs_emb = torch.cat([z_t, g_hat], dim=1)
        t_act = torch.empty(B, device=z_goal.device).uniform_(0, 1)
        act_0 = torch.randn_like(act)
        act_t = agent.interpolant.calc_It(t_act, act_0, act)
        act_t_dot = agent.interpolant.calc_It_dot(t_act, act_0, act)
        b_t = agent.flow_map.get_velocity(t_act, act_t, obs_emb)
        action_losses.append(
            float(torch.mean(get_norm(b_t - act_t_dot, cfg.norm_type)).item())
        )

    agent.train()
    out = {"val_state_flow_loss": float(np.mean(state_losses)) if state_losses else float("nan")}
    if action_losses:
        out["val_action_reg_loss_unscaled"] = float(np.mean(action_losses))
    return out


def train(
    config: Config,
    dataset,
    val_dataset,
    agent: GoalPredictorDiTAgent,
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
                obs = _to_device_obs(batch, "obs", config)
                goal_obs = _to_device_obs(batch, "goal_obs", config)
                act = batch["action"][:, : config.task.horizon, :].to(
                    config.optimization.device, non_blocking=True,
                )

            with timed("update", perf):
                delta_t_scalar = warmup_scheduler(n_gradient_step)
                delta_t = torch.full(
                    (act.shape[0],), delta_t_scalar, device=config.optimization.device,
                )
                info = agent.update(act, obs, goal_obs, delta_t)
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
                loguru.logger.info("Computing val state-flow / action-reg loss...")
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

    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        raise NotImplementedError(
            "Franka goal predictor DiT training only supports image obs"
        )

    # --- Reuse the IDM's normalizer so train/eval are on the same scale the
    #     IDM was trained on. The IDM training script saves it as a sibling
    #     of the checkpoint.
    idm_path = config.optimization.idm_checkpoint_path
    if idm_path is None:
        raise ValueError(
            "optimization.idm_checkpoint_path must be set for goal predictor training"
        )
    idm_normalizer = None
    normalizer_path = os.path.join(os.path.dirname(idm_path), "normalizer.pkl")
    if os.path.exists(normalizer_path):
        with open(normalizer_path, "rb") as f:
            idm_normalizer = pickle.load(f)
        loguru.logger.info(f"Loaded IDM normalizer from {normalizer_path}")
    else:
        loguru.logger.warning(
            f"IDM normalizer not found at {normalizer_path}; computing a fresh one"
        )

    # --- datasets (multi-HDF5 via make_idm_dataset, train split) ---
    dataset = make_idm_dataset(config.task, mode="train", normalizer=idm_normalizer)
    loguru.logger.info(f"Train IDM dataset: {len(dataset)} samples")

    base_dataset = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )

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

    # --- agent (loads frozen IDM internally) ---
    agent = GoalPredictorDiTAgent(config)
    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading goal predictor DiT from {config.optimization.model_path}"
        )
        resume_state = agent.load(
            config.optimization.model_path, load_optimizer=True,
        )

    if config.mode == "train":
        train(config, dataset, val_dataset, agent, logger, resume_state=resume_state)
    else:
        raise ValueError(
            f"Franka goal predictor pipeline only supports mode='train' (got {config.mode!r})"
        )


if __name__ == "__main__":
    main()
