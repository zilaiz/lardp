"""Compute per-element mean/variance of encoded goal features for z-score normalization.

Runs the frozen encoder from a pretrained IDM checkpoint over the training dataset,
encodes all goal observations, and saves (mean, var) statistics to a .pt file.
Uses the normalizer saved during IDM training to ensure consistent input normalization.

The normalizer is automatically located as a sibling of the IDM checkpoint
({checkpoint_dir}/normalizer.pkl). Use --normalizer_path to override.

Usage:
    python scripts/compute_goal_stats.py \
        --idm_checkpoint /path/to/models/model_best.pt \
        --dataset_path /path/to/data.hdf5 \
        --output_path /path/to/goal_stats.pt \
        --config_dir examples/configs \
        --task_config can_ph_image_gp \
        --network_config lbmidm
"""

from __future__ import annotations

import argparse
import os
import sys

import loguru
import torch
from tqdm import tqdm

# Set MuJoCo rendering backend before importing robomimic modules
os.environ["MUJOCO_GL"] = "egl"

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402


def compute_stats(
    encoder: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Compute mean and variance of encoded goal features via batch accumulation.

    Two-pass approach: accumulate sum and sum-of-squares per batch,
    then derive mean and variance at the end.

    Args:
        encoder: frozen encoder module
        dataset: IDM dataset that returns {"goal_obs": ..., ...}
        device: compute device
        batch_size: dataloader batch size

    Returns:
        {"mean": (emb_dim,), "var": (emb_dim,)}
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    # Accumulate in float64 to avoid precision loss over large datasets
    total_sum = None
    total_sq_sum = None
    n = 0

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing goal stats"):
            goal_batch = batch["goal_obs"]

            # Build goal obs tensor/dict
            if isinstance(goal_batch, dict):
                from tensordict import TensorDict

                goal_dict = {k: v.to(device) for k, v in goal_batch.items()}
                batch_size_actual = next(iter(goal_dict.values())).shape[0]
                goal_obs = TensorDict(goal_dict, batch_size=batch_size_actual)
            else:
                goal_obs = goal_batch.to(device)

            z_goal = encoder(goal_obs, None)  # (B, 1, emb_dim)
            z_goal = z_goal.squeeze(1).double()  # (B, emb_dim) in float64

            if total_sum is None:
                emb_dim = z_goal.shape[-1]
                total_sum = torch.zeros(emb_dim, dtype=torch.float64, device=device)
                total_sq_sum = torch.zeros(emb_dim, dtype=torch.float64, device=device)

            total_sum += z_goal.sum(dim=0)
            total_sq_sum += (z_goal**2).sum(dim=0)
            n += z_goal.shape[0]

    mean = total_sum / n
    var = (total_sq_sum / n - mean**2).clamp(min=0)  # clamp for numerical safety
    return {"mean": mean.float().cpu(), "var": var.float().cpu()}


def main():
    parser = argparse.ArgumentParser(description="Compute goal embedding normalization stats")
    parser.add_argument("--idm_checkpoint", type=str, required=True, help="Path to pretrained IDM checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for stats .pt file")
    parser.add_argument("--normalizer_path", type=str, default=None, help="Path to normalizer.pkl (default: inferred from idm_checkpoint dir)")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--config_dir", type=str, required=True, help="Hydra config directory (e.g., examples/configs)")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name (e.g., can_ph_image_gp)")
    parser.add_argument("--network_config", type=str, default="lbmidm", help="Network config matching the IDM checkpoint architecture (default: lbmidm)")

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

    # Load IDM checkpoint to get encoder
    loguru.logger.info(f"Loading IDM checkpoint from {args.idm_checkpoint}")
    checkpoint = torch.load(args.idm_checkpoint, map_location=device, weights_only=False)

    # We need to instantiate the encoder — use network_utils
    from mip.network_utils import get_encoder

    encoder = get_encoder(network_cfg, task_cfg).to(device)

    # Handle GoalDropoutEncoder wrapper
    from mip.encoders import GoalDropoutEncoder

    encoder_sd = checkpoint["encoder"]
    has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
    if has_goal_dropout:
        enc_out_dim = network_cfg.get("encoder_out_dim") or network_cfg.emb_dim
        encoder = GoalDropoutEncoder(
            encoder, enc_out_dim, task_cfg.obs_steps
        ).to(device)
        loguru.logger.info("IDM checkpoint has GoalDropoutEncoder, wrapping encoder")

    encoder.load_state_dict(encoder_sd)
    encoder.requires_grad_(False)
    encoder.eval()

    # Get inner encoder (bypass GoalDropoutEncoder if present)
    inner_encoder = encoder.encoder if isinstance(encoder, GoalDropoutEncoder) else encoder

    loguru.logger.info("Encoder loaded successfully")

    # Load normalizer saved during IDM training
    import pickle

    normalizer_path = args.normalizer_path
    if normalizer_path is None:
        # Infer from IDM checkpoint: {checkpoint_dir}/normalizer.pkl
        normalizer_path = os.path.join(os.path.dirname(args.idm_checkpoint), "normalizer.pkl")
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            f"Normalizer not found at {normalizer_path}. "
            "Provide --normalizer_path or ensure normalizer.pkl is in the IDM checkpoint directory."
        )
    loguru.logger.info(f"Loading normalizer from {normalizer_path}")
    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    # Create dataset with the IDM's normalizer
    OmegaConf.update(task_cfg, "dataset_paths", [args.dataset_path])
    dataset = make_idm_dataset(task_cfg, normalizer=normalizer)
    loguru.logger.info(f"Dataset size: {len(dataset)}")

    # Compute stats
    stats = compute_stats(inner_encoder, dataset, device=device, batch_size=args.batch_size)

    loguru.logger.info(f"Mean norm: {stats['mean'].norm():.4f}")
    loguru.logger.info(f"Var mean: {stats['var'].mean():.4f}, min: {stats['var'].min():.4f}, max: {stats['var'].max():.4f}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(stats, args.output_path)
    loguru.logger.info(f"Saved goal stats to {args.output_path}")


if __name__ == "__main__":
    main()
