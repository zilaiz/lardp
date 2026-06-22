"""Training pipeline for Franka real-robot image datasets.

Stripped counterpart to `examples/train_robomimic.py`:
- No simulator / vectorized env / rollout-based eval.
- Val loss on held-out demos replaces sim eval.
- Image-only observation branch (no DINO precomputed features, no state branch).
- One periodic tick (`log.eval_freq`) drives val-loss evaluation and both
  `latest.pt` (for auto-resume) and `step_{N}.pt` (accumulating snapshots
  you can evaluate on the real robot later).

Assumes the dataset HDF5 was produced by
`examples/process_dataset/convert_franka_coffee_pod.py`.
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

# Import mip modules (no MUJOCO env, no sim dependencies)
from mip.agent import TrainingAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.franka_dataset import make_franka_dataset
from mip.logger import Logger
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import limit_threads, set_seed

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, record_dict: dict):
    """Context manager for timing code sections."""
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


def _preprocess_batch(batch, config: Config) -> tuple[TensorDict, torch.Tensor]:
    """Move a dataloader batch to the training device and slice to the horizon.

    Image branch only: every obs key under `batch["obs"]` is treated as a raw
    tensor (RGB or low-dim) and truncated to `obs_steps`. Action is truncated
    to `horizon`.
    """
    device = config.optimization.device
    obs_dict: dict[str, torch.Tensor] = {}
    for k, v in batch["obs"].items():
        obs_dict[k] = v[:, : config.task.obs_steps, ...].to(device, non_blocking=True)
    batch_size = next(iter(obs_dict.values())).shape[0]
    obs = TensorDict(obs_dict, batch_size=batch_size)
    act = batch["action"][:, : config.task.horizon, :].to(device, non_blocking=True)
    return obs, act


@torch.no_grad()
def compute_val_loss(
    agent: TrainingAgent,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    max_batches: int = 50,
) -> float:
    """Flow-matching loss on the val set, using EMA weights.

    Mirrors the training loss (same interpolant, same sampled delta_t) but
    without backprop. Uses EMA model for consistency with inference-time
    behavior.
    """
    agent.eval()
    ema_flow_map = agent.flow_map_ema if config.optimization.ema_rate < 1 else agent.flow_map
    ema_encoder = agent.encoder_ema if config.optimization.ema_rate < 1 else agent.encoder

    losses: list[float] = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        obs, act = _preprocess_batch(batch, config)
        # delta_t for val: use a neutral value (no warmup schedule dependence).
        delta_t = torch.full(
            (act.shape[0],),
            config.optimization.min_value,
            device=config.optimization.device,
        )
        loss, _ = agent.loss_fn(
            config.optimization, ema_flow_map, ema_encoder, agent.interpolant,
            act, obs, delta_t,
        )
        losses.append(float(loss.item()))

    agent.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(
    config: Config,
    dataset,
    val_dataset,
    agent: TrainingAgent,
    logger: Logger,
    resume_state: dict | None = None,
):
    # --- dataloaders ---
    # Multi-worker loading: the zarr MemoryStore is pickle/fork-safe (all arrays
    # are in memory, no open HDF5 handles remain after __init__). With 6 CPUs
    # per slurm job, 4 workers for train leaves CPU headroom for the main loop;
    # persistent_workers=True amortizes the fork cost over training.
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        # IMPORTANT: drop_last=True required for CUDA graphs (static shapes)
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

    # --- schedulers ---
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

    # --- resume ---
    start_step = 0
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        loguru.logger.info(f"Resuming training from step {start_step}")
        for _ in range(start_step):
            lr_scheduler.step()

    # --- loop ---
    info_list: list[dict] = []
    start_time = time.time()
    perf = {"data_load": [], "preprocess": [], "update": [], "total_step": []}

    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        with timed("total_step", perf):
            with timed("data_load", perf):
                batch = next(loop_train)

            with timed("preprocess", perf):
                obs, act = _preprocess_batch(batch, config)
                # Optional optimality (expert/play) labels — present only for a
                # mixed franka dataset; consumed only when network.use_optimality.
                optimality = batch.get("optimality")
                if optimality is not None:
                    optimality = optimality.to(config.optimization.device)

            with timed("update", perf):
                delta_t_scalar = warmup_scheduler(n_gradient_step)
                delta_t = torch.full(
                    (act.shape[0],), delta_t_scalar, device=config.optimization.device
                )
                info = agent.update(act, obs, delta_t, optimality=optimality)
                lr_scheduler.step()

            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    info[k] = v.item()
            info_list.append(info)

        # --- log ---
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
            # Perf counters
            window = config.log.log_freq
            if perf["total_step"]:
                metrics["perf/data_load_ms"]  = np.mean(perf["data_load"][-window:])  * 1000
                metrics["perf/preprocess_ms"] = np.mean(perf["preprocess"][-window:]) * 1000
                metrics["perf/update_ms"]     = np.mean(perf["update"][-window:])     * 1000
                metrics["perf/total_step_ms"] = np.mean(perf["total_step"][-window:]) * 1000
                metrics["perf/steps_per_sec"] = 1.0 / np.mean(perf["total_step"][-window:])
            logger.log(metrics, category="train")
            info_list = []

        # --- eval tick: val loss + latest.pt + step_{N}.pt ---
        if ((n_gradient_step + 1) % config.log.eval_freq) == 0:
            eval_metrics: dict[str, float] = {"step": n_gradient_step}
            if val_loader is not None:
                loguru.logger.info("Computing val loss...")
                val_loss = compute_val_loss(agent, val_loader, config)
                eval_metrics["val_loss"] = val_loss
                loguru.logger.info(f"  val_loss = {val_loss:.4f}")
            logger.log(eval_metrics, category="eval")

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

    # --- datasets ---
    dataset = make_franka_dataset(config.task, mode="train")
    loguru.logger.info(f"Train dataset: {len(dataset)} samples")
    val_dataset = None
    if config.task.val_dataset_percentage > 0.0:
        val_dataset = make_franka_dataset(config.task, mode="val")
        # share the train normalizer so val loss is computed on the same scale
        val_dataset.normalizer = dataset.normalizer
        loguru.logger.info(f"Val dataset:   {len(val_dataset)} samples")

    # --- agent + auto-resume ---
    # (no obs_dim override needed: image encoder reads shape_meta, not obs_dim)
    agent = TrainingAgent(config)
    resume_state = None

    pretrained_encoder_path = getattr(config.optimization, "pretrained_encoder_path", None)
    if pretrained_encoder_path and pretrained_encoder_path != "None":
        agent.load_pretrained_encoder(
            pretrained_encoder_path,
            freeze=getattr(config.optimization, "freeze_encoder", False),
        )

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    else:
        # No auto-resume — Franka pipeline does not write a global checkpoint.
        # To resume a crashed run, re-launch with
        #   optimization.model_path=logs/<exp>/<ts>/models/model_latest.pt
        loguru.logger.info("Starting fresh (auto-resume not supported here; pass "
                           "optimization.model_path to resume from latest.pt)")

    if config.mode == "train":
        train(config, dataset, val_dataset, agent, logger, resume_state=resume_state)
    else:
        raise ValueError(
            f"Franka training pipeline supports mode='train' only (got {config.mode!r}). "
            "Real-robot eval is handled by a separate driver script."
        )


if __name__ == "__main__":
    main()
