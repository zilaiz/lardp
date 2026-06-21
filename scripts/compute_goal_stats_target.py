r"""Compute per-dim z-score stats (mean/var) of a pluggable TargetEncoder's goal
features — unified across all ``target_encoder_type``s (dp / lewm / dinov2 /
siglip). The frozen-target agent (``LBMDiTJointPTFrozenTargetAgent``) z-scores
its FM state target by these stats (``goal_stats_path``).

This supersedes ``compute_goal_stats_dp.py`` / ``_pusht.py`` for the foreign
encoders: it builds the encoder via ``mip.target_encoders.get_target_encoder``
(so each owns its own loading + preprocessing) and reuses the same two-pass
accumulator (``compute_stats``), which calls ``TargetEncoder.embed``.

Dataset: robomimic uses ``make_idm_dataset`` (the image normalizer is the fixed
``x*2-1``, which the foreign encoders invert — so a fresh normalizer is fine);
pusht uses ``make_pusht_goal_dataset``. For multi-camera robomimic you MUST set
``--target_encoder_image_key`` (e.g. agentview_image); pusht auto-resolves its
single rgb key.

Usage (pusht, DINOv2-S):
    python scripts/compute_goal_stats_target.py \\
        --dataset pusht --task_config pusht_image --network_config lbmdit \\
        --target_encoder_type dinov2 \\
        --target_encoder_path ~/data/zzeng28/ckpts/dinov2_small \\
        --output_path goal_stats/pusht_dinov2_small.pt

Usage (robomimic, SigLIP, pick a camera):
    python scripts/compute_goal_stats_target.py \\
        --dataset robomimic --task_config can_ph_image_gp --network_config lbmdit \\
        --dataset_path data/robomimic/can/ph/image_v15.hdf5 \\
        --target_encoder_type siglip \\
        --target_encoder_path ~/data/zzeng28/ckpts/siglip_base_patch16_224 \\
        --target_encoder_image_key agentview_image \\
        --output_path goal_stats/can_siglip.pt

DP also works (parity with compute_goal_stats_dp.py):
    ... --target_encoder_type dp --dp_checkpoint <ckpt> [--no-use_encoder_ema]
"""

from __future__ import annotations

import argparse
import os
import sys

import loguru
import torch

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.compute_goal_stats_dp import compute_stats  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="Compute goal z-score stats for any pluggable TargetEncoder."
    )
    p.add_argument(
        "--target_encoder_type", required=True,
        choices=["dp", "lewm", "dinov2", "siglip"],
    )
    p.add_argument(
        "--target_encoder_path", default=None,
        help="Checkpoint dir/file for foreign encoders (lewm/dinov2/siglip).",
    )
    p.add_argument(
        "--target_encoder_image_key", default=None,
        help="Rgb obs key the image-only encoder reads. Required for "
             "multi-camera tasks; auto for single-rgb (pusht).",
    )
    p.add_argument(
        "--dp_checkpoint", default=None,
        help="For --target_encoder_type dp: the LBMDiT (DP) checkpoint.",
    )
    p.add_argument(
        "--use_encoder_ema", action=argparse.BooleanOptionalAction, default=True,
        help="DP only: load 'encoder_ema' (default) or 'encoder'.",
    )
    p.add_argument("--dataset", required=True, choices=["robomimic", "pusht"])
    p.add_argument("--config_dir", default="examples/configs")
    p.add_argument("--task_config", required=True)
    p.add_argument("--network_config", default="lbmdit")
    p.add_argument(
        "--dataset_path", nargs="*", default=None,
        help="robomimic only: override task.dataset_paths (else use the "
             "task config's own list).",
    )
    p.add_argument(
        "--normalizer_path", default=None,
        help="robomimic only, optional. Foreign image encoders only use the "
             "fixed x*2-1 image normalizer, so this rarely matters for them.",
    )
    p.add_argument("--output_path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()
    device = args.device

    import hydra
    from omegaconf import OmegaConf

    config_abs = os.path.abspath(args.config_dir)
    overrides = [f"task={args.task_config}", f"network={args.network_config}"]
    with hydra.initialize_config_dir(config_dir=config_abs):
        cfg = hydra.compose(config_name="main", overrides=overrides)

    # Inject the target-encoder selection into the config the factory reads.
    OmegaConf.update(cfg, "optimization.device", device)
    OmegaConf.update(cfg, "optimization.target_encoder_type", args.target_encoder_type)
    if args.target_encoder_path is not None:
        OmegaConf.update(
            cfg, "optimization.target_encoder_path", args.target_encoder_path
        )
    if args.target_encoder_image_key is not None:
        OmegaConf.update(
            cfg, "optimization.target_encoder_image_key",
            args.target_encoder_image_key,
        )
    if args.target_encoder_type == "dp":
        if args.dp_checkpoint is None:
            raise ValueError("--dp_checkpoint required for --target_encoder_type dp")
        OmegaConf.update(cfg, "optimization.dp_checkpoint_path", args.dp_checkpoint)
        OmegaConf.update(cfg, "optimization.dp_use_encoder_ema", args.use_encoder_ema)

    from mip.target_encoders import get_target_encoder

    target_encoder = get_target_encoder(cfg)  # frozen, eval, on device

    # --- Dataset ---
    task_cfg = cfg.task
    if args.dataset == "pusht":
        from mip.datasets.pusht_dataset import make_pusht_goal_dataset

        dataset = make_pusht_goal_dataset(task_cfg)
    else:
        import pickle

        from mip.datasets.robomimic_dataset import make_idm_dataset

        normalizer = None
        if args.normalizer_path and os.path.exists(args.normalizer_path):
            loguru.logger.info(f"Loading normalizer from {args.normalizer_path}")
            with open(args.normalizer_path, "rb") as f:
                normalizer = pickle.load(f)
        if args.dataset_path:
            OmegaConf.update(task_cfg, "dataset_paths", list(args.dataset_path))
        loguru.logger.info(f"Dataset paths: {list(task_cfg.dataset_paths)}")
        dataset = make_idm_dataset(task_cfg, normalizer=normalizer)
    loguru.logger.info(f"Dataset size: {len(dataset)}")

    # --- Encode + accumulate (compute_stats calls target_encoder.embed) ---
    stats = compute_stats(
        target_encoder, dataset, device=device, batch_size=args.batch_size,
    )
    loguru.logger.info(
        f"D={stats['mean'].numel()} | mean norm {stats['mean'].norm():.4f} | "
        f"var mean {stats['var'].mean():.4f} "
        f"min {stats['var'].min():.4f} max {stats['var'].max():.4f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(stats, args.output_path)
    loguru.logger.info(f"Saved goal stats to {args.output_path}")


if __name__ == "__main__":
    main()
