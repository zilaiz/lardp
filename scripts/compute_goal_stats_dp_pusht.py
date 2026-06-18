"""Compute per-dim mean/var of frozen DP-encoder goal features on PushT.

PushT sibling of ``scripts/compute_goal_stats_dp.py``. The robomimic version
uses ``make_idm_dataset``; PushT needs ``make_pusht_goal_dataset``, which
carries its own MinMax / Image normalizer (matching how the DP encoder was
trained), so no external ``--normalizer_path`` is needed.

Loads ``encoder_ema`` (default) from a pretrained LBMDiT (DP) checkpoint,
encodes ``goal_obs`` over the dataset, and saves ``{"mean", "var"}`` for the
z-score normalization of the FM state target in
``LBMDiTJointPTFrozenTargetAgent`` (``goal_stats_path``).

Usage:
    python scripts/compute_goal_stats_dp_pusht.py \\
        --dp_checkpoint logs/pusht_ph_image_flow_None_lbmdit_256_seed0_horizon16_ed256_0.8/<ts>/models/model_best.pt \\
        --output_path   goal_stats/pusht_dp_seed0.pt \\
        --config_dir    examples/configs \\
        --task_config   pusht_image \\
        --network_config lbmdit
"""

from __future__ import annotations

import argparse
import os
import sys

import loguru
import torch

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mip.datasets.pusht_dataset import make_pusht_goal_dataset  # noqa: E402
from scripts.compute_goal_stats_dp import compute_stats  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Compute PushT goal-embedding z-score stats from a "
                    "pretrained LBMDiT (DP) checkpoint."
    )
    parser.add_argument(
        "--dp_checkpoint", type=str, required=True,
        help="Path to pretrained LBMDiT (DP) checkpoint (top-level keys: "
             "encoder, encoder_ema, flow_map, flow_map_ema, optimizer).",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--config_dir", type=str, required=True,
        help="Hydra config directory (e.g., examples/configs).",
    )
    parser.add_argument(
        "--task_config", type=str, required=True,
        help="PushT task config name (must match the DP checkpoint's obs "
             "setup, e.g. pusht_image).",
    )
    parser.add_argument(
        "--network_config", type=str, default="lbmdit",
        help="Network config matching the DP checkpoint's encoder "
             "architecture (default: lbmdit).",
    )
    parser.add_argument(
        "--use_encoder_ema", action=argparse.BooleanOptionalAction, default=True,
        help="Load weights from 'encoder_ema' (default) or 'encoder'.",
    )
    args = parser.parse_args()
    device = args.device

    import hydra
    from omegaconf import OmegaConf  # noqa: F401  (kept for parity / future use)

    config_abs = os.path.abspath(args.config_dir)
    overrides = [f"task={args.task_config}", f"network={args.network_config}"]
    with hydra.initialize_config_dir(config_dir=config_abs):
        cfg = hydra.compose(config_name="main", overrides=overrides)
    network_cfg = cfg.network
    task_cfg = cfg.task

    # --- Load DP checkpoint, pick encoder vs encoder_ema (same checks as the
    # robomimic exporter / the frozen-target agent loader) ---
    loguru.logger.info(f"Loading DP checkpoint from {args.dp_checkpoint}")
    checkpoint = torch.load(
        args.dp_checkpoint, map_location=device, weights_only=False,
    )
    encoder_key = "encoder_ema" if args.use_encoder_ema else "encoder"
    if encoder_key not in checkpoint:
        available = sorted(k for k in checkpoint if "encoder" in k)
        raise KeyError(
            f"DP checkpoint has no '{encoder_key}' key. Available: {available}."
        )
    encoder_sd = checkpoint[encoder_key]
    if "uncond_emb" in encoder_sd or any(
        k.startswith("encoder.") for k in encoder_sd
    ):
        raise RuntimeError(
            f"State dict at '{encoder_key}' looks like a GoalDropoutEncoder "
            f"(has 'encoder.' prefixed keys or 'uncond_emb'). Use an LBMDiT "
            f"(DP) checkpoint."
        )

    from mip.network_utils import get_encoder

    encoder = get_encoder(network_cfg, task_cfg).to(device)
    encoder.load_state_dict(encoder_sd)
    encoder.requires_grad_(False)
    encoder.eval()
    loguru.logger.info(
        f"Loaded {encoder_key} from DP checkpoint "
        f"({sum(v.numel() for v in encoder_sd.values()):,} params)"
    )

    # --- PushT dataset (own normalizer; no external normalizer needed) ---
    dataset = make_pusht_goal_dataset(task_cfg)
    loguru.logger.info(f"Dataset size: {len(dataset)}")

    # --- Encode + accumulate stats (reuse the robomimic accumulator) ---
    stats = compute_stats(
        encoder, dataset, device=device, batch_size=args.batch_size,
    )

    loguru.logger.info(f"Mean norm: {stats['mean'].norm():.4f}")
    loguru.logger.info(
        f"Var mean: {stats['var'].mean():.4f}, "
        f"min: {stats['var'].min():.4f}, max: {stats['var'].max():.4f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(stats, args.output_path)
    loguru.logger.info(f"Saved goal stats to {args.output_path}")


if __name__ == "__main__":
    main()
