"""Compute per-element mean/variance of (encoded goal - encoded last obs).

Counterpart to `compute_goal_stats.py` for a delta-target goal predictor.
Runs the frozen encoder from a pretrained IDM checkpoint over the dataset,
encodes the obs window and the goal frame, and accumulates statistics of
the delta `g - last_obs` (where `last_obs` is the last token of the encoded
obs window). Saves `(delta_mean, delta_var)` to a .pt file for z-score
normalization of the diffusion target.

Note: for the per-frame encoders in this repo (MultiImageObsEncoder,
PerStepMLPEncoder, PrecomputedDINOEncoder, PrecomputedLAMEncoder), the last
token of `encoder(obs)` equals `encoder(obs[:, -1:])`. The legacy MLPEncoder
fuses across `To`, in which case these stats reflect the fused last-token
embedding (which is what training would subtract anyway).

The normalizer is auto-located as a sibling of the IDM checkpoint
({checkpoint_dir}/normalizer.pkl). Use --normalizer_path to override.

Usage:
    python scripts/compute_goal_delta_stats.py \
        --idm_checkpoint /path/to/models/model_best.pt \
        --dataset_path /path/to/data.hdf5 \
        --output_path /path/to/goal_delta_stats.pt \
        --config_dir examples/configs \
        --task_config can_ph_image_gp \
        --network_config lbmidm_v2
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


def _to_device_obs(obs_batch, device):
    """Move obs (tensor or dict-of-tensors) to device, wrap dict as TensorDict."""
    if isinstance(obs_batch, dict):
        from tensordict import TensorDict

        d = {k: v.to(device) for k, v in obs_batch.items()}
        b = next(iter(d.values())).shape[0]
        return TensorDict(d, batch_size=b)
    return obs_batch.to(device)


def compute_delta_stats(
    encoder: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Compute mean/var of `g - last_obs` in latent space via batch accumulation.

    Args:
        encoder: frozen encoder module
        dataset: IDM dataset returning {"obs": ..., "goal_obs": ..., ...}
        device: compute device
        batch_size: dataloader batch size

    Returns:
        {"delta_mean": (emb_dim,), "delta_var": (emb_dim,)}
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    # Accumulate in float64 to avoid precision loss over large datasets
    delta_sum = None
    delta_sq_sum = None
    n = 0

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing goal-delta stats"):
            goal_obs = _to_device_obs(batch["goal_obs"], device)
            obs = _to_device_obs(batch["obs"], device)

            z_goal = encoder(goal_obs, None)  # (B, 1, emb_dim)
            z_obs = encoder(obs, None)  # (B, To, emb_dim)

            z_goal = z_goal.squeeze(1).double()  # (B, emb_dim)
            z_last = z_obs[:, -1, :].double()  # (B, emb_dim)
            z_delta = z_goal - z_last

            if delta_sum is None:
                emb_dim = z_delta.shape[-1]
                delta_sum = torch.zeros(emb_dim, dtype=torch.float64, device=device)
                delta_sq_sum = torch.zeros(emb_dim, dtype=torch.float64, device=device)

            delta_sum += z_delta.sum(dim=0)
            delta_sq_sum += (z_delta**2).sum(dim=0)
            n += z_delta.shape[0]

    delta_mean = delta_sum / n
    delta_var = (delta_sq_sum / n - delta_mean**2).clamp(min=0)
    return {
        "delta_mean": delta_mean.float().cpu(),
        "delta_var": delta_var.float().cpu(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute (goal - last_obs) embedding normalization stats"
    )
    parser.add_argument("--idm_checkpoint", type=str, required=True, help="Path to pretrained IDM checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for delta stats .pt file")
    parser.add_argument("--normalizer_path", type=str, default=None, help="Path to normalizer.pkl (default: inferred from idm_checkpoint dir)")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--config_dir", type=str, required=True, help="Hydra config directory (e.g., examples/configs)")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name (e.g., can_ph_image_gp)")
    parser.add_argument("--network_config", type=str, default="lbmidm_v2", help="Network config matching the IDM checkpoint architecture (default: lbmidm_v2)")
    parser.add_argument(
        "--val_pct", type=float, default=None,
        help="Override task.val_dataset_percentage (i.e., compute stats only on the train subset). "
             "Default: keep whatever the task config says (typically 0.0 → all demos).",
    )

    args = parser.parse_args()

    device = args.device

    import hydra
    from omegaconf import OmegaConf

    config_abs = os.path.abspath(args.config_dir)
    overrides = [f"task={args.task_config}", f"network={args.network_config}"]
    if args.val_pct is not None:
        overrides.append(f"task.val_dataset_percentage={args.val_pct}")
    with hydra.initialize_config_dir(config_dir=config_abs):
        cfg = hydra.compose(config_name="main", overrides=overrides)

    network_cfg = cfg.network
    task_cfg = cfg.task

    # Load IDM checkpoint to get encoder
    loguru.logger.info(f"Loading IDM checkpoint from {args.idm_checkpoint}")
    checkpoint = torch.load(args.idm_checkpoint, map_location=device, weights_only=False)

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

    # Use inner encoder so output is the raw (B, T, emb_dim), bypassing
    # uncond_emb padding / goal-dropout substitution.
    inner_encoder = encoder.encoder if isinstance(encoder, GoalDropoutEncoder) else encoder

    loguru.logger.info("Encoder loaded successfully")

    # Load normalizer saved during IDM training
    import pickle

    normalizer_path = args.normalizer_path
    if normalizer_path is None:
        normalizer_path = os.path.join(os.path.dirname(args.idm_checkpoint), "normalizer.pkl")
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            f"Normalizer not found at {normalizer_path}. "
            "Provide --normalizer_path or ensure normalizer.pkl is in the IDM checkpoint directory."
        )
    loguru.logger.info(f"Loading normalizer from {normalizer_path}")
    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    OmegaConf.update(task_cfg, "dataset_paths", [args.dataset_path])
    dataset = make_idm_dataset(task_cfg, normalizer=normalizer)
    loguru.logger.info(f"Dataset size: {len(dataset)}")

    stats = compute_delta_stats(inner_encoder, dataset, device=device, batch_size=args.batch_size)

    dm, dv = stats["delta_mean"], stats["delta_var"]
    loguru.logger.info(f"Delta mean norm: {dm.norm():.4f}")
    loguru.logger.info(
        f"Delta var  mean: {dv.mean():.4f}, min: {dv.min():.4f}, max: {dv.max():.4f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(stats, args.output_path)
    loguru.logger.info(f"Saved goal-delta stats to {args.output_path}")


if __name__ == "__main__":
    main()
