"""Compute per-element mean/variance of encoded goal features for z-score
normalization, using the encoder from a pretrained **LBMDiT (DP)** checkpoint.

Sister of ``scripts/compute_goal_stats.py`` (IDM checkpoint). Differences:
  - LBMDiT saves ``encoder`` and ``encoder_ema`` as separate top-level keys
    (see ``mip/agent.py:save``), neither wrapped in ``GoalDropoutEncoder``.
    We load whichever the user picked via ``--use_encoder_ema`` (default
    True; matches the agent's default for downstream use).
  - The data normalizer is *optional* — LBMDiT training scripts do not
    save ``normalizer.pkl`` next to the checkpoint. If a normalizer is
    available (e.g. you saved one alongside the LBMDiT run, or you reuse
    one from a sibling IDM run), pass it via ``--normalizer_path``; the
    script falls back to a fresh dataset-computed normalizer otherwise.

Usage:
    python scripts/compute_goal_stats_dp.py \\
        --dp_checkpoint /path/to/lbmdit/model_best.pt \\
        --dataset_path /path/to/data.hdf5 \\
        --output_path /path/to/goal_stats.pt \\
        --config_dir examples/configs \\
        --task_config can_ph_image_gp \\
        --network_config lbmdit
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
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Two-pass accumulation: sum, sum-of-squares -> mean, var.

    Returns ``{"mean": (emb_dim,), "var": (emb_dim,)}`` on CPU in float32.
    """
    # num_workers>0: __getitem__ decodes images off a (networked) zarr store, so
    # single-threaded loading dominates (~40s/batch). Parallel workers + pinned
    # memory overlap that IO with the GPU encode and cut wall-clock by ~10x.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    total_sum = None
    total_sq_sum = None
    n = 0

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing goal stats"):
            goal_batch = batch["goal_obs"]

            if isinstance(goal_batch, dict):
                from tensordict import TensorDict

                goal_dict = {k: v.to(device) for k, v in goal_batch.items()}
                B = next(iter(goal_dict.values())).shape[0]
                goal_obs = TensorDict(goal_dict, batch_size=B)
            else:
                goal_obs = goal_batch.to(device)

            # A pluggable TargetEncoder exposes .embed(goal)->(B,1,D); a raw
            # encoder is called as encoder(goal, None). Support both.
            z_goal = (
                encoder.embed(goal_obs)
                if hasattr(encoder, "embed")
                else encoder(goal_obs, None)
            )                                        # (B, 1, emb_dim)
            z_goal = z_goal.squeeze(1).double()      # (B, emb_dim) f64

            if total_sum is None:
                emb_dim = z_goal.shape[-1]
                total_sum = torch.zeros(emb_dim, dtype=torch.float64, device=device)
                total_sq_sum = torch.zeros(emb_dim, dtype=torch.float64, device=device)

            total_sum += z_goal.sum(dim=0)
            total_sq_sum += (z_goal ** 2).sum(dim=0)
            n += z_goal.shape[0]

    if n == 0:
        raise RuntimeError("Empty dataset — no goal observations to encode.")

    mean = total_sum / n
    var = (total_sq_sum / n - mean ** 2).clamp(min=0)  # numerical safety
    return {"mean": mean.float().cpu(), "var": var.float().cpu()}


def main():
    parser = argparse.ArgumentParser(
        description="Compute goal embedding normalization stats from a "
                    "pretrained LBMDiT (DP) checkpoint."
    )
    parser.add_argument(
        "--dp_checkpoint", type=str, required=True,
        help="Path to pretrained LBMDiT (DP) checkpoint (top-level keys: "
             "encoder, encoder_ema, flow_map, flow_map_ema, optimizer).",
    )
    parser.add_argument(
        "--dataset_path", type=str, nargs="*", default=None,
        help="Optional. One or more HDF5 paths overriding task.dataset_paths. "
             "Omit to use the task config's own dataset_paths — e.g. pass a "
             "'_mixed' task config (expert + rollouts) to compute stats over "
             "the combined (play-inclusive) goal distribution.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--normalizer_path", type=str, default=None,
        help="Optional. Path to data normalizer.pkl. If omitted and one "
             "exists next to the DP checkpoint we use that; otherwise a "
             "fresh normalizer is computed from the dataset.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--config_dir", type=str, required=True,
        help="Hydra config directory (e.g., examples/configs).",
    )
    parser.add_argument(
        "--task_config", type=str, required=True,
        help="Task config name (e.g., can_ph_image_gp).",
    )
    parser.add_argument(
        "--network_config", type=str, default="lbmdit",
        help="Network config matching the LBMDiT checkpoint's encoder "
             "architecture (default: lbmdit).",
    )
    parser.add_argument(
        "--use_encoder_ema", action=argparse.BooleanOptionalAction, default=True,
        help="Load weights from 'encoder_ema' (default) or 'encoder'.",
    )

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

    # --- Load DP checkpoint, pick encoder vs encoder_ema ---
    loguru.logger.info(f"Loading DP checkpoint from {args.dp_checkpoint}")
    checkpoint = torch.load(
        args.dp_checkpoint, map_location=device, weights_only=False,
    )

    encoder_key = "encoder_ema" if args.use_encoder_ema else "encoder"
    if encoder_key not in checkpoint:
        available = sorted(k for k in checkpoint.keys() if "encoder" in k)
        raise KeyError(
            f"DP checkpoint has no '{encoder_key}' key. Available: {available}."
        )
    encoder_sd = checkpoint[encoder_key]

    if "uncond_emb" in encoder_sd or any(
        k.startswith("encoder.") for k in encoder_sd
    ):
        raise RuntimeError(
            f"State dict at '{encoder_key}' looks like a GoalDropoutEncoder "
            f"(has 'encoder.' prefixed keys or 'uncond_emb'). LBMDiT does "
            f"not wrap; use scripts/compute_goal_stats.py for IDM checkpoints."
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

    # --- Optional data normalizer ---
    import pickle

    normalizer = None
    nz_path = args.normalizer_path
    if nz_path is None:
        candidate = os.path.join(
            os.path.dirname(args.dp_checkpoint), "normalizer.pkl",
        )
        if os.path.exists(candidate):
            nz_path = candidate
    if nz_path is not None and os.path.exists(nz_path):
        loguru.logger.info(f"Loading normalizer from {nz_path}")
        with open(nz_path, "rb") as f:
            normalizer = pickle.load(f)
    else:
        loguru.logger.warning(
            "No data normalizer provided / found — dataset will compute "
            "a fresh one. Encoder will see a slightly different scale than "
            "at LBMDiT training time; usually OK for image obs."
        )

    # --- Dataset ---
    # Override task.dataset_paths only when --dataset_path is given; otherwise
    # use the task config's own list (e.g. a _mixed config = expert + rollouts,
    # so the stats cover the play-inclusive goal distribution).
    if args.dataset_path:
        OmegaConf.update(task_cfg, "dataset_paths", list(args.dataset_path))
    loguru.logger.info(f"Dataset paths: {list(task_cfg.dataset_paths)}")
    dataset = make_idm_dataset(task_cfg, normalizer=normalizer)
    loguru.logger.info(f"Dataset size: {len(dataset)}")

    # --- Encode + accumulate stats ---
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
