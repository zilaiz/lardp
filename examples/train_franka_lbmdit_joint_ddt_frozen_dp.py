"""Franka LBMDiTJointDDTFrozenDP training pipeline (real-robot, no sim env).

Same as ``train_franka_lbmdit_joint_ddt_frozen.py`` but the obs encoder is
loaded from a pretrained LBMDiT (DP) checkpoint instead of an IDM checkpoint.
The data normalizer is still expected as ``normalizer.pkl`` next to the DP
checkpoint (override with ``optimization.dp_normalizer_path`` if elsewhere).

Required config keys:
  - optimization.dp_checkpoint_path: path to pretrained LBMDiT checkpoint
  - optimization.dp_use_encoder_ema: True (default) loads encoder_ema,
    False loads encoder.
  - optimization.goal_stats_path: path to .pt with {mean, var} computed via
    ``scripts/compute_goal_stats_dp.py``.
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

from mip.agent_lbmdit_joint_ddt_frozen_dp import LBMDiTJointDDTFrozenDPAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader, make_expert_weighted_sampler
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset, make_idm_dataset
from mip.logger import Logger
from mip.losses import get_norm
from mip.networks.lbmdit_joint import LBMDiTJoint
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import limit_threads, set_seed

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, perf: dict):
    t0 = time.perf_counter()
    yield
    perf[section].append(time.perf_counter() - t0)


def _to_device_obs(batch, key, config: Config) -> TensorDict:
    device = config.optimization.device
    obs_batch = batch[key]
    obs_dict: dict = {}
    if key == "obs":
        for k in obs_batch:
            obs_dict[k] = obs_batch[k][:, : config.task.obs_steps].to(
                device, non_blocking=True,
            )
    else:
        for k in obs_batch:
            obs_dict[k] = obs_batch[k].to(device, non_blocking=True)
    bs = next(iter(obs_dict.values())).shape[0]
    return TensorDict(obs_dict, batch_size=bs)


def _read_optimality(batch: dict, batch_size: int, device) -> torch.Tensor:
    if "optimality" in batch:
        return batch["optimality"].to(device=device, dtype=torch.long)
    return torch.full(
        (batch_size,), LBMDiTJoint.NULL_IDX, dtype=torch.long, device=device,
    )


@torch.no_grad()
def compute_val_loss(
    agent: LBMDiTJointDDTFrozenDPAgent,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    max_batches: int = 50,
) -> dict[str, float]:
    """State + action FM loss on the val set, against the EMA trunk."""
    agent.eval()
    cfg = config.optimization
    device = cfg.device
    net = agent.net_ema

    state_losses: list[float] = []
    action_losses: list[float] = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        obs = _to_device_obs(batch, "obs", config)
        goal_obs = _to_device_obs(batch, "goal_obs", config)
        act = batch["action"][:, : config.task.horizon, :].to(device, non_blocking=True)
        B = act.shape[0]
        optimality = _read_optimality(batch, B, device)

        z_t = agent.encoder(obs, None)
        z_goal_raw = agent.encoder(goal_obs, None)
        target = agent._normalize_goal(z_goal_raw)

        eps = agent._t_eps
        lo, hi = eps, 1.0 - eps
        if agent._decouple_t:
            t_s_base = agent._sample_base_t((B,), device, lo, hi)
            t_a_base = agent._sample_base_t((B,), device, lo, hi)
            t_state = agent._apply_t_shift(t_s_base, agent._shift_state)
            t_action = agent._apply_t_shift(t_a_base, agent._shift_action)
        else:
            shared_base = agent._sample_base_t((B,), device, lo, hi)
            shared = agent._apply_t_shift(shared_base, agent._shift_state)
            t_state = shared
            t_action = shared

        s_noise = torch.randn_like(target)
        a_noise = torch.randn_like(act)
        s_t = agent.interpolant.calc_It(t_state, s_noise, target)
        s_t_dot = agent.interpolant.calc_It_dot(t_state, s_noise, target)
        a_t = agent.interpolant.calc_It(t_action, a_noise, act)
        a_t_dot = agent.interpolant.calc_It_dot(t_action, a_noise, act)

        v_state, v_action, _ = net(
            x_state=s_t, x_action=a_t,
            s=t_state, t=t_action,
            condition=z_t, optimality_idx=optimality,
        )
        state_losses.append(float(
            (torch.mean(get_norm(v_state - s_t_dot, cfg.norm_type))
             / float(net.obs_dim)).item()
        ))
        action_losses.append(float(
            (torch.mean(get_norm(v_action - a_t_dot, cfg.norm_type))
             / float(net.act_dim)).item()
        ))

    agent.train()
    return {
        "val_state_loss": float(np.mean(state_losses)) if state_losses else float("nan"),
        "val_action_loss": float(np.mean(action_losses)) if action_losses else float("nan"),
    }


def train(
    config: Config,
    dataset,
    val_dataset,
    agent: LBMDiTJointDDTFrozenDPAgent,
    logger: Logger,
    resume_state: dict | None = None,
):
    sampler = make_expert_weighted_sampler(
        dataset, config.optimization.expert_sample_fraction,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4,
        shuffle=(sampler is None),
        sampler=sampler,
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
                B = act.shape[0]
                optimality = _read_optimality(batch, B, config.optimization.device)

            with timed("update", perf):
                delta_t_scalar = warmup_scheduler(n_gradient_step)
                delta_t = torch.full(
                    (B,), delta_t_scalar, device=config.optimization.device,
                )
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
                loguru.logger.info("Computing val state/action FM loss...")
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
            "Franka LBMDiTJointDDTFrozenDP training only supports image obs"
        )

    dp_path = config.optimization.dp_checkpoint_path
    if dp_path is None:
        raise ValueError(
            "optimization.dp_checkpoint_path must be set for "
            "joint DDT DP-frozen training"
        )

    # Try to reuse the data normalizer saved alongside the DP checkpoint —
    # mirrors the IDM-frozen pipeline so the encoder sees the scale it was
    # trained on. Falls back to a fresh one if the LBMDiT trainer didn't
    # save one (base LBMDiT training scripts do not save normalizer.pkl).
    dp_normalizer = None
    normalizer_path = os.path.join(os.path.dirname(dp_path), "normalizer.pkl")
    if os.path.exists(normalizer_path):
        with open(normalizer_path, "rb") as f:
            dp_normalizer = pickle.load(f)
        loguru.logger.info(f"Loaded DP normalizer from {normalizer_path}")
    else:
        loguru.logger.warning(
            f"DP normalizer not found at {normalizer_path}; computing a "
            "fresh one. The encoder will see a slightly different obs scale "
            "than at LBMDiT training time — this is usually fine for image "
            "obs but worth flagging."
        )

    dataset = make_idm_dataset(config.task, mode="train", normalizer=dp_normalizer)
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

    agent = LBMDiTJointDDTFrozenDPAgent(config)
    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading LBMDiTJointDDTFrozenDP from {config.optimization.model_path}"
        )
        resume_state = agent.load(
            config.optimization.model_path, load_optimizer=True,
        )

    if config.mode == "train":
        train(config, dataset, val_dataset, agent, logger, resume_state=resume_state)
    else:
        raise ValueError(
            f"Franka joint DDT DP-frozen pipeline only supports mode='train' "
            f"(got {config.mode!r})"
        )


if __name__ == "__main__":
    main()
