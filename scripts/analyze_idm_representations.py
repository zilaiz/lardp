"""Analyze IDM encoder representation space for checkpoint selection insights.

Computes per-trajectory metrics over the expert dataset for a given IDM checkpoint:

1. Temporal smoothness:
   - Cosine similarity between consecutive encoder embeddings along trajectories
   - Local Lipschitz estimate: ||z_{t+1} - z_t|| / ||o_{t+1} - o_t||  (lowdim obs)

2. IDM unconditional vs conditional action prediction quality:
   - Conditional MSE:   E[ || v(t, a_t; [z_obs, z_goal]) - a_dot_t ||^2 ]
   - Unconditional MSE: E[ || v(t, a_t; [z_obs, uncond])  - a_dot_t ||^2 ]
   - CFG headroom = unconditional_MSE - conditional_MSE

All metrics are printed and optionally saved to a JSON file.

Usage:
    python scripts/analyze_idm_representations.py \
        --idm_checkpoint logs/.../models/model_step_90000.pt \
        --dataset_path data/robomimic/transport/ph/image_v15.hdf5 \
        --config_dir examples/configs \
        --task_config transport_ph_image_gp \
        --network_config lbmidm \
        --output_path analysis_transport_90k.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import loguru
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.flow_map import FlowMap  # noqa: E402
from mip.interpolant import Interpolant  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_idm(args, device):
    """Load frozen IDM encoder + flow_map from checkpoint.

    Returns (inner_encoder, encoder_with_wrapper, flow_map, normalizer, uncond_emb_or_None).
    """
    import pickle

    import hydra
    from omegaconf import OmegaConf

    config_abs = os.path.abspath(args.config_dir)
    overrides = [f"task={args.task_config}", f"network={args.network_config}"]
    with hydra.initialize_config_dir(config_dir=config_abs):
        cfg = hydra.compose(config_name="main", overrides=overrides)

    network_cfg = cfg.network
    task_cfg = cfg.task

    checkpoint = torch.load(args.idm_checkpoint, map_location=device, weights_only=False)

    from mip.network_utils import get_encoder, get_network

    # Encoder
    encoder = get_encoder(network_cfg, task_cfg).to(device)
    encoder_sd = checkpoint["encoder"]

    from mip.encoders import GoalDropoutEncoder

    has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
    enc_out_dim = network_cfg.get("encoder_out_dim", None) or network_cfg.emb_dim
    if has_goal_dropout:
        encoder = GoalDropoutEncoder(encoder, enc_out_dim, task_cfg.obs_steps).to(device)
        loguru.logger.info("Wrapping with GoalDropoutEncoder")
    encoder.load_state_dict(encoder_sd)
    encoder.requires_grad_(False)
    encoder.eval()

    inner_encoder = encoder.encoder if isinstance(encoder, GoalDropoutEncoder) else encoder
    uncond_emb = encoder.uncond_emb if isinstance(encoder, GoalDropoutEncoder) else None

    # FlowMap
    net = get_network(network_cfg, task_cfg)
    flow_map = FlowMap(net).to(device)
    flow_map.load_state_dict(checkpoint["flow_map"])
    flow_map.requires_grad_(False)
    flow_map.eval()

    # Normalizer
    normalizer_path = args.normalizer_path
    if normalizer_path is None:
        normalizer_path = os.path.join(os.path.dirname(args.idm_checkpoint), "normalizer.pkl")
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(f"Normalizer not found at {normalizer_path}")
    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    return inner_encoder, encoder, flow_map, normalizer, uncond_emb, task_cfg, network_cfg


# ---------------------------------------------------------------------------
# Analysis 1: Temporal smoothness  (cos-sim + local Lipschitz)
# ---------------------------------------------------------------------------

def analyze_temporal_smoothness(
    inner_encoder: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str,
    batch_size: int = 128,
) -> dict:
    """Encode every sample's current obs (To frames) and goal obs (1 frame).

    Since the dataset samples overlapping windows, we compute per-sample metrics:
      - cos_sim(z_obs_last, z_goal): cosine sim between the last obs frame embedding
        and the goal frame embedding (which is horizon steps ahead)
      - ||z_goal - z_obs_last||: L2 distance to goal in embedding space
      - lowdim Lipschitz: ||z_goal - z_obs_last|| / ||o_goal_lowdim - o_obs_lowdim_last||
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False,
    )

    cos_sims = []
    l2_dists = []
    lipschitz_ratios = []
    obs_l2_dists = []

    inner_encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Temporal smoothness"):
            obs_batch = batch["obs"]
            goal_batch = batch["goal_obs"]

            # Build obs and goal TensorDicts
            from tensordict import TensorDict

            obs_dict = {k: v.to(device) for k, v in obs_batch.items()}
            goal_dict = {k: v.to(device) for k, v in goal_batch.items()}
            B = next(iter(obs_dict.values())).shape[0]
            obs_td = TensorDict(obs_dict, batch_size=B)
            goal_td = TensorDict(goal_dict, batch_size=B)

            z_obs = inner_encoder(obs_td, None)   # (B, To, emb_dim)
            z_goal = inner_encoder(goal_td, None)  # (B, 1, emb_dim)

            z_last = z_obs[:, -1, :]    # (B, emb_dim)
            z_g = z_goal[:, 0, :]       # (B, emb_dim)

            # Cosine similarity
            cs = F.cosine_similarity(z_last, z_g, dim=-1)  # (B,)
            cos_sims.append(cs.cpu())

            # L2 distance in embedding space
            l2 = (z_last - z_g).norm(dim=-1)  # (B,)
            l2_dists.append(l2.cpu())

            # Lowdim Lipschitz: find lowdim keys
            lowdim_keys = [k for k in obs_batch if "image" not in k.lower()]
            if lowdim_keys:
                # last obs frame lowdim vs goal lowdim
                o_last_parts = []
                o_goal_parts = []
                for k in lowdim_keys:
                    o_last_parts.append(obs_batch[k][:, -1, :])   # (B, feat)
                    o_goal_parts.append(goal_batch[k][:, 0, :])   # (B, feat)
                o_last = torch.cat(o_last_parts, dim=-1)  # (B, total_lowdim)
                o_goal = torch.cat(o_goal_parts, dim=-1)

                o_l2 = (o_last - o_goal).norm(dim=-1).clamp(min=1e-8)  # (B,)
                lip = l2.cpu() / o_l2
                lipschitz_ratios.append(lip)
                obs_l2_dists.append(o_l2)

    cos_sims = torch.cat(cos_sims)
    l2_dists = torch.cat(l2_dists)

    results = {
        "cos_sim_mean": cos_sims.mean().item(),
        "cos_sim_std": cos_sims.std().item(),
        "cos_sim_median": cos_sims.median().item(),
        "cos_sim_q10": cos_sims.quantile(0.1).item(),
        "cos_sim_q90": cos_sims.quantile(0.9).item(),
        "embedding_l2_mean": l2_dists.mean().item(),
        "embedding_l2_std": l2_dists.std().item(),
        "embedding_l2_median": l2_dists.median().item(),
    }

    if lipschitz_ratios:
        lip = torch.cat(lipschitz_ratios)
        obs_l2 = torch.cat(obs_l2_dists)
        results.update({
            "lipschitz_mean": lip.mean().item(),
            "lipschitz_std": lip.std().item(),
            "lipschitz_median": lip.median().item(),
            "lipschitz_q90": lip.quantile(0.9).item(),
            "lipschitz_q95": lip.quantile(0.95).item(),
            "obs_lowdim_l2_mean": obs_l2.mean().item(),
        })

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Unconditional vs conditional IDM action prediction
# ---------------------------------------------------------------------------

def analyze_idm_action_quality(
    inner_encoder: torch.nn.Module,
    flow_map: FlowMap,
    uncond_emb: torch.Tensor | None,
    dataset: torch.utils.data.Dataset,
    device: str,
    obs_steps: int,
    batch_size: int = 128,
    n_time_samples: int = 8,
) -> dict:
    """Measure IDM velocity matching error with and without goal conditioning.

    For each batch, sample `n_time_samples` random time points and compute:
      - conditional_mse:    ||v(t, a_t; [z_obs, z_goal]) - a_dot_t||^2
      - unconditional_mse:  ||v(t, a_t; [z_obs, uncond]) - a_dot_t||^2

    Uses linear interpolant: a_t = (1-t)*a_0 + t*a_1, a_dot_t = a_1 - a_0.
    """
    if uncond_emb is None:
        loguru.logger.warning("No uncond_emb found — skipping unconditional analysis")
        return {}

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False,
    )

    interpolant = Interpolant("linear")

    cond_mses = []
    uncond_mses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="IDM action quality"):
            obs_batch = batch["obs"]
            goal_batch = batch["goal_obs"]
            act = batch["action"].to(device)

            from tensordict import TensorDict

            obs_dict = {k: v.to(device) for k, v in obs_batch.items()}
            goal_dict = {k: v.to(device) for k, v in goal_batch.items()}
            B = next(iter(obs_dict.values())).shape[0]
            obs_td = TensorDict(obs_dict, batch_size=B)
            goal_td = TensorDict(goal_dict, batch_size=B)

            z_obs = inner_encoder(obs_td, None)   # (B, To, emb_dim)
            z_goal = inner_encoder(goal_td, None)  # (B, 1, emb_dim)

            # Conditional embedding: [z_obs, z_goal]
            cond_emb = torch.cat([z_obs, z_goal], dim=1)  # (B, To+1, emb_dim)

            # Unconditional embedding: [z_obs, uncond]
            uncond = uncond_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            uncond_emb_full = torch.cat([z_obs, uncond], dim=1)  # (B, To+1, emb_dim)

            for _ in range(n_time_samples):
                t = torch.rand(B, device=device)
                a_0 = torch.randn_like(act)
                a_1 = act

                a_t = interpolant.calc_It(t, a_0, a_1)
                a_dot = interpolant.calc_It_dot(t, a_0, a_1)

                v_cond = flow_map.get_velocity(t, a_t, cond_emb)
                v_uncond = flow_map.get_velocity(t, a_t, uncond_emb_full)

                cond_err = ((v_cond - a_dot) ** 2).mean(dim=(1, 2))    # (B,)
                uncond_err = ((v_uncond - a_dot) ** 2).mean(dim=(1, 2))  # (B,)

                cond_mses.append(cond_err.cpu())
                uncond_mses.append(uncond_err.cpu())

    cond_mses = torch.cat(cond_mses)
    uncond_mses = torch.cat(uncond_mses)

    return {
        "conditional_mse_mean": cond_mses.mean().item(),
        "conditional_mse_std": cond_mses.std().item(),
        "unconditional_mse_mean": uncond_mses.mean().item(),
        "unconditional_mse_std": uncond_mses.std().item(),
        "cfg_headroom": (uncond_mses.mean() - cond_mses.mean()).item(),
        "cfg_headroom_ratio": (uncond_mses.mean() / cond_mses.mean().clamp(min=1e-8)).item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze IDM representation space")
    parser.add_argument("--idm_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config_dir", type=str, required=True, help="e.g., examples/configs")
    parser.add_argument("--task_config", type=str, required=True, help="e.g., transport_ph_image_gp")
    parser.add_argument("--network_config", type=str, default="lbmidm")
    parser.add_argument("--normalizer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_time_samples", type=int, default=8, help="Random time samples for IDM action analysis")
    parser.add_argument("--val_dataset_percentage", type=float, default=0.0, help="Fraction of demos held out as val (uses first (1-pct)*N demos)")
    parser.add_argument("--max_samples", type=int, default=None, help="Subsample dataset to at most N samples after loading")
    args = parser.parse_args()

    device = args.device
    loguru.logger.info(f"Analyzing IDM checkpoint: {args.idm_checkpoint}")

    inner_encoder, encoder, flow_map, normalizer, uncond_emb, task_cfg, network_cfg = load_idm(args, device)

    from omegaconf import OmegaConf

    OmegaConf.update(task_cfg, "dataset_paths", [args.dataset_path])
    if args.val_dataset_percentage > 0.0:
        OmegaConf.update(task_cfg, "val_dataset_percentage", args.val_dataset_percentage)
        loguru.logger.info(f"Using val_dataset_percentage={args.val_dataset_percentage}")

    dataset = make_idm_dataset(task_cfg, normalizer=normalizer)
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        dataset = dataset.datasets[0]
    loguru.logger.info(f"Dataset size: {len(dataset)}")

    if args.max_samples is not None and args.max_samples < len(dataset):
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=g)[: args.max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        loguru.logger.info(f"Subsampled to {len(dataset)} samples")

    results = {"checkpoint": args.idm_checkpoint, "dataset": args.dataset_path}

    # --- Analysis 1: Temporal smoothness (obs-to-goal) ---
    loguru.logger.info("=" * 60)
    loguru.logger.info("Analysis 1: Temporal smoothness (last obs frame -> goal)")
    loguru.logger.info("=" * 60)
    smoothness = analyze_temporal_smoothness(inner_encoder, dataset, device, args.batch_size)
    results["temporal_smoothness"] = smoothness
    for k, v in smoothness.items():
        loguru.logger.info(f"  {k}: {v:.6f}")

    # --- Analysis 2: IDM conditional vs unconditional ---
    loguru.logger.info("=" * 60)
    loguru.logger.info("Analysis 2: IDM action quality (conditional vs unconditional)")
    loguru.logger.info("=" * 60)
    action_quality = analyze_idm_action_quality(
        inner_encoder, flow_map, uncond_emb, dataset, device,
        obs_steps=task_cfg.obs_steps, batch_size=args.batch_size,
        n_time_samples=args.n_time_samples,
    )
    results["action_quality"] = action_quality
    for k, v in action_quality.items():
        loguru.logger.info(f"  {k}: {v:.6f}")

    # --- Summary ---
    loguru.logger.info("=" * 60)
    loguru.logger.info("Summary")
    loguru.logger.info("=" * 60)
    if smoothness:
        loguru.logger.info(f"  Obs->Goal cos-sim:    {smoothness['cos_sim_mean']:.4f} +/- {smoothness['cos_sim_std']:.4f}")
        loguru.logger.info(f"  Embedding L2 dist:    {smoothness['embedding_l2_mean']:.4f} +/- {smoothness['embedding_l2_std']:.4f}")
        if "lipschitz_mean" in smoothness:
            loguru.logger.info(f"  Local Lipschitz:      {smoothness['lipschitz_mean']:.4f} (median {smoothness['lipschitz_median']:.4f})")
    if action_quality:
        loguru.logger.info(f"  Cond. action MSE:     {action_quality['conditional_mse_mean']:.4f}")
        loguru.logger.info(f"  Uncond. action MSE:   {action_quality['unconditional_mse_mean']:.4f}")
        loguru.logger.info(f"  CFG headroom:         {action_quality['cfg_headroom']:.4f} (ratio {action_quality['cfg_headroom_ratio']:.2f}x)")

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        loguru.logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
