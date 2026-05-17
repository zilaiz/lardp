"""Compute per-element mean/variance of encoded delta features.

Mirrors ``compute_goal_stats.py`` but accumulates the goal *delta*

    delta = z_goal − z_last_obs

over the training dataset. Output stats are intended for
``DeltaPredictorDDTNSAgent.delta_stats_path``.

Usage:
    python scripts/compute_delta_stats.py \
        --idm_checkpoint /path/to/models/model_best.pt \
        --dataset_path /path/to/data.hdf5 \
        --output_path /path/to/delta_stats.pt \
        --config_dir examples/configs \
        --task_config tool_hang_ph_image_gp \
        --network_config lbmidm_v2_delta
"""

from __future__ import annotations

import argparse
import os
import sys

import loguru
import torch
from tqdm import tqdm

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402


def compute_stats(
    encoder: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    obs_steps: int,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Encode ``goal_obs`` and the last frame of ``obs`` per sample, compute
    delta = z_goal − z_last_obs, then accumulate mean / var per element.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    total_sum = None
    total_sq_sum = None
    n = 0

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing delta stats"):
            obs_batch = batch["obs"]
            goal_batch = batch["goal_obs"]

            from tensordict import TensorDict

            if isinstance(obs_batch, dict):
                obs_dict = {
                    k: v[:, :obs_steps].to(device) for k, v in obs_batch.items()
                }
                B_local = next(iter(obs_dict.values())).shape[0]
                obs_td = TensorDict(obs_dict, batch_size=B_local)
            else:
                obs_td = obs_batch[:, :obs_steps].to(device)

            if isinstance(goal_batch, dict):
                goal_dict = {k: v.to(device) for k, v in goal_batch.items()}
                B_local = next(iter(goal_dict.values())).shape[0]
                goal_td = TensorDict(goal_dict, batch_size=B_local)
            else:
                goal_td = goal_batch.to(device)

            z_obs = encoder(obs_td, None)        # (B, To, emb_dim)
            z_goal = encoder(goal_td, None)      # (B, 1,  emb_dim)
            delta = (z_goal[:, 0] - z_obs[:, -1]).double()   # (B, emb_dim)

            if total_sum is None:
                emb_dim = delta.shape[-1]
                total_sum = torch.zeros(emb_dim, dtype=torch.float64, device=device)
                total_sq_sum = torch.zeros(emb_dim, dtype=torch.float64, device=device)

            total_sum += delta.sum(dim=0)
            total_sq_sum += (delta ** 2).sum(dim=0)
            n += delta.shape[0]

    mean = total_sum / n
    var = (total_sq_sum / n - mean ** 2).clamp(min=0)
    return {"mean": mean.float().cpu(), "var": var.float().cpu()}


def main():
    parser = argparse.ArgumentParser(description="Compute delta embedding stats")
    parser.add_argument("--idm_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--normalizer_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--network_config", type=str, default="lbmidm_v2_delta")
    args = parser.parse_args()

    device = args.device

    import hydra
    from omegaconf import OmegaConf

    config_abs = os.path.abspath(args.config_dir)
    overrides = [f"task={args.task_config}", f"network={args.network_config}"]
    with hydra.initialize_config_dir(config_dir=config_abs):
        cfg = hydra.compose(config_name="main", overrides=overrides)

    network_cfg = cfg.network
    task_cfg = cfg.task

    loguru.logger.info(f"Loading IDM checkpoint from {args.idm_checkpoint}")
    checkpoint = torch.load(args.idm_checkpoint, map_location=device,
                             weights_only=False)

    from mip.encoders import GoalDropoutEncoder
    from mip.network_utils import get_encoder

    encoder = get_encoder(network_cfg, task_cfg).to(device)
    encoder_sd = checkpoint["encoder"]
    has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
    if has_goal_dropout:
        enc_out_dim = network_cfg.get("encoder_out_dim") or network_cfg.emb_dim
        encoder = GoalDropoutEncoder(
            encoder, enc_out_dim, task_cfg.obs_steps,
        ).to(device)
        loguru.logger.info("IDM checkpoint has GoalDropoutEncoder; wrapping")
    encoder.load_state_dict(encoder_sd)
    encoder.requires_grad_(False)
    encoder.eval()

    inner_encoder = (
        encoder.encoder if isinstance(encoder, GoalDropoutEncoder) else encoder
    )

    import pickle

    normalizer_path = args.normalizer_path
    if normalizer_path is None:
        normalizer_path = os.path.join(
            os.path.dirname(args.idm_checkpoint), "normalizer.pkl",
        )
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            f"Normalizer not found at {normalizer_path}. "
            "Provide --normalizer_path or ensure normalizer.pkl is sibling "
            "to the IDM checkpoint."
        )
    loguru.logger.info(f"Loading normalizer from {normalizer_path}")
    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    OmegaConf.update(task_cfg, "dataset_paths", [args.dataset_path])
    dataset = make_idm_dataset(task_cfg, normalizer=normalizer)
    loguru.logger.info(f"Dataset size: {len(dataset)}")

    stats = compute_stats(
        inner_encoder, dataset, obs_steps=task_cfg.obs_steps,
        device=device, batch_size=args.batch_size,
    )

    loguru.logger.info(f"Delta mean norm: {stats['mean'].norm():.4f}")
    loguru.logger.info(
        f"Delta var: mean={stats['var'].mean():.4f} "
        f"min={stats['var'].min():.4f} max={stats['var'].max():.4f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(stats, args.output_path)
    loguru.logger.info(f"Saved delta stats to {args.output_path}")


if __name__ == "__main__":
    main()
