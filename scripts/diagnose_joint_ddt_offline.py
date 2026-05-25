"""Offline diagnostic for an LBMDiTJointDDTAgent checkpoint.

Purpose: when an agent shows OOD behavior on the robot, before re-training
or chasing deployment-side bugs, verify the checkpoint itself is sane by
feeding it real training-set observations and comparing its predictions to
the dataset's ground-truth actions. If the model can't reproduce its own
training data, the checkpoint is the issue. If it can, the on-robot failure
is downstream (compounding error / OOD obs / preprocessing drift).

Checks performed:
  1. learnable_state_token health (norm/std/abs_max for net + net_ema).
  2. Per-channel stats of target_ln(encoder(obs)) on N training samples
     (mean, std, effective rank) — both live and EMA encoder stacks.
  3. Per-dimension MSE between predicted and ground-truth NORMALIZED actions
     on N training samples, at a few num_steps values.
  4. Per-dimension predicted-action distribution (min/max/mean/std vs the
     training action range [-1, 1]) — flags OOD predictions on in-dist obs.

Usage:
    python scripts/diagnose_joint_ddt_offline.py \\
        --ckpt_path  logs/<exp>/<ts>/models/model_step_100000.pt \\
        --config_path outputs/<date>/<time>/.hydra/config.yaml \\
        --num_samples 32 \\
        --num_steps_list 5 25
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

LARDP_PATH = Path(__file__).resolve().parents[1]
if str(LARDP_PATH) not in sys.path:
    sys.path.append(str(LARDP_PATH))

from mip.agent_lbmdit_joint_ddt import LBMDiTJointDDTAgent
from mip.dataset_utils import dict_apply
from mip.datasets.robomimic_dataset import make_idm_dataset


def _fmt_tensor_stats(t: torch.Tensor) -> str:
    return (
        f"shape={tuple(t.shape)} "
        f"norm={t.norm().item():.4e} "
        f"std={t.std().item():.4e} "
        f"abs_max={t.abs().max().item():.4e}"
    )


def _effective_rank(x: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank = exp(-sum(p_i log p_i)) where p_i = sigma_i / sum(sigma).
    Reference: Roy & Vetterli, 2007. Used by the user's memory as the right
    encoder health signal (target_std is an LN artifact).
    """
    x_flat = x.reshape(-1, x.shape[-1]).float()
    x_flat = x_flat - x_flat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x_flat)
    p = s / (s.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))


def diagnose(args):
    cfg = OmegaConf.load(args.config_path)
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)
    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)

    device = torch.device(cfg.optimization.device)
    print(f"\n== device: {device} ==")
    print(f"   ckpt:   {args.ckpt_path}")
    print(f"   config: {args.config_path}")

    print("\n[1/4] Building agent + loading checkpoint...")
    agent = LBMDiTJointDDTAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()

    # -----------------------------------------------------------------------
    # 1. learnable_state_token health
    # -----------------------------------------------------------------------
    print("\n== (1) learnable_state_token health ==")
    for tag, net in [("net (live)", agent.net), ("net_ema", agent.net_ema)]:
        if hasattr(net, "replace_x_state") and net.replace_x_state:
            t = net.learnable_state_token.detach()
            print(f"  {tag:14s}: {_fmt_tensor_stats(t)} "
                  f"all_zero={bool((t == 0).all())}")
        else:
            print(f"  {tag:14s}: replace_x_state=False (no token)")

    # -----------------------------------------------------------------------
    # 2. Load training dataset and build a small batch
    # -----------------------------------------------------------------------
    # Optional: override the dataset path (e.g. point at image_heldout.hdf5).
    # When overridden, you almost always want --normalizer_path too — the
    # heldout file alone would compute a DIFFERENT normalizer and the action
    # scale would be wrong relative to the trained model.
    if args.dataset_path is not None:
        OmegaConf.update(cfg, "task.dataset_paths", [args.dataset_path], merge=False)
        OmegaConf.update(cfg, "task.dataset_path", None, merge=False)
        print(f"\n[2/4] Loading dataset override: {args.dataset_path}")
        if args.normalizer_path is None:
            print("   WARNING: no --normalizer_path; the heldout file will "
                  "compute its own normalizer (action scale will NOT match "
                  "the trained model). Pass --normalizer_path for a valid test.")
    else:
        print("\n[2/4] Loading training dataset for in-distribution samples...")

    norm_override = None
    if args.normalizer_path is not None:
        with open(args.normalizer_path, "rb") as f:
            norm_override = pickle.load(f)
        print(f"   using normalizer from: {args.normalizer_path}")
        print(f"   normalizer obs keys: {sorted(norm_override['obs'].keys())}")

    # Heldout HDF5 typically has no val/train split; set val_dataset_percentage=0
    # so all of it is available in train mode. (When loading the original
    # training file we keep cfg.task.val_dataset_percentage unchanged.)
    if args.dataset_path is not None:
        OmegaConf.update(cfg, "task.val_dataset_percentage", 0.0, merge=False)

    dataset = make_idm_dataset(cfg.task, mode="train", normalizer=norm_override)
    n = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    print(f"   dataset size={len(dataset)}, sampling {n} indices")

    sample_keys = list(cfg.task.shape_meta["obs"].keys())
    obs_batch = {k: [] for k in sample_keys}
    act_batch_normed = []
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
        # ground-truth horizon-long action chunk, normalized
        act_batch_normed.append(s["action"][: cfg.task.horizon])

    obs_torch = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in obs_batch.items()
    }
    # shapes: rgb (n, To, 3, H, W), low_dim (n, To, D)
    act_true_normed = torch.from_numpy(
        np.stack([t.numpy() for t in act_batch_normed])
    ).to(device)
    print(f"   obs shapes: {[(k, tuple(v.shape)) for k, v in obs_torch.items()]}")
    print(f"   act_true_normed: {tuple(act_true_normed.shape)}")

    # -----------------------------------------------------------------------
    # 3. Encoder output health on training data
    # -----------------------------------------------------------------------
    print("\n== (2) encoder output stats on training obs ==")
    with torch.no_grad():
        for tag, enc, ln in [
            ("live", agent.encoder, agent.target_ln),
            ("EMA",  agent.encoder_ema, agent.target_ln_ema),
        ]:
            z_raw = enc(obs_torch, None)
            z_t = ln(z_raw)
            print(f"  [{tag}] raw  enc(obs): {_fmt_tensor_stats(z_raw)} "
                  f"eff_rank={_effective_rank(z_raw):.2f}")
            print(f"  [{tag}] target_ln(.): {_fmt_tensor_stats(z_t)} "
                  f"eff_rank={_effective_rank(z_t):.2f}")

    # -----------------------------------------------------------------------
    # 4. Per-sample action prediction error vs ground truth (NORMALIZED)
    # -----------------------------------------------------------------------
    print("\n== (3) predicted vs true (normalized) action on training samples ==")
    dim_names = ["pos_x", "pos_y", "pos_z",
                 "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
                 "gripper"]

    for steps in args.num_steps_list:
        print(f"\n  num_steps = {steps}")
        with torch.no_grad():
            act_0 = torch.randn(
                (n, cfg.task.horizon, cfg.task.act_dim), device=device,
            )
            act_pred = agent.sample(
                act_0=act_0, obs=obs_torch, num_steps=steps, use_ema=True,
            )
        err = act_pred - act_true_normed         # (n, H, 10)
        mse_per_dim = (err ** 2).mean(dim=(0, 1))  # (10,)
        mae_per_dim = err.abs().mean(dim=(0, 1))   # (10,)
        pred_min = act_pred.amin(dim=(0, 1))
        pred_max = act_pred.amax(dim=(0, 1))
        pred_std = act_pred.std(dim=(0, 1))
        true_min = act_true_normed.amin(dim=(0, 1))
        true_max = act_true_normed.amax(dim=(0, 1))
        true_std = act_true_normed.std(dim=(0, 1))

        print(f"    overall MSE={mse_per_dim.mean().item():.4f}   "
              f"MAE={mae_per_dim.mean().item():.4f}   "
              f"||pred - true||_2/sqrt(n*H*D) "
              f"={err.pow(2).mean().sqrt().item():.4f}")
        print(f"    {'dim':<8s}{'MSE':>10s}{'MAE':>10s}"
              f"{'pred[min,max]':>22s}{'true[min,max]':>22s}"
              f"{'pred_std':>10s}{'true_std':>10s}")
        for i, name in enumerate(dim_names):
            ood_flag = ""
            if pred_min[i].item() < true_min[i].item() - 0.05:
                ood_flag += " ↓"
            if pred_max[i].item() > true_max[i].item() + 0.05:
                ood_flag += " ↑"
            print(f"    {name:<8s}{mse_per_dim[i].item():>10.4f}"
                  f"{mae_per_dim[i].item():>10.4f}"
                  f"   [{pred_min[i].item():+.3f},{pred_max[i].item():+.3f}]"
                  f"   [{true_min[i].item():+.3f},{true_max[i].item():+.3f}]"
                  f"   {pred_std[i].item():>7.3f}   {true_std[i].item():>7.3f}"
                  f"{ood_flag}")

    # -----------------------------------------------------------------------
    # 5. Stability across noise draws (sanity for stochasticity in sampling)
    # -----------------------------------------------------------------------
    print("\n== (4) action variance across noise draws (first 4 samples) ==")
    with torch.no_grad():
        K = 8
        preds = []
        for k in range(K):
            torch.manual_seed(args.seed + k)
            act_0 = torch.randn(
                (n, cfg.task.horizon, cfg.task.act_dim), device=device,
            )
            preds.append(agent.sample(
                act_0=act_0, obs=obs_torch,
                num_steps=args.num_steps_list[0], use_ema=True,
            ))
        preds = torch.stack(preds, dim=0)        # (K, n, H, 10)
        # variance per (sample, step, dim) across the K noise draws, averaged
        # then per dim
        var_per_dim = preds.var(dim=0).mean(dim=(0, 1))   # (10,)
        for i, name in enumerate(dim_names):
            print(f"    {name:<8s}  std_across_noise={var_per_dim[i].sqrt().item():.4f}")
        print("    (low values mean the policy is nearly deterministic on this "
              "obs across different action-noise inits; high values mean it's "
              "still treating the obs as multi-modal)")

    print("\n== done ==")
    print("Reading the output:")
    print("  - learnable_state_token near zero (norm ~1e-3): expected for this "
          "config; trunk is conditioning only on obs.")
    print("  - encoder effective rank low (<<emb_dim): channel collapse, the "
          "encoder isn't using its capacity.")
    print("  - overall MSE > ~0.05 on in-distribution training data: the "
          "checkpoint cannot reproduce its own data; re-train.")
    print("  - OOD flags (↑/↓) on in-distribution obs: the model is "
          "extrapolating outside the training action range even on data it "
          "was trained on — definitive failure mode.")
    print("  - low std_across_noise: deterministic policy (fine).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Override task.dataset_paths with a single file "
                             "(e.g. image_heldout.hdf5). Pair with "
                             "--normalizer_path for a meaningful test.")
    parser.add_argument("--normalizer_path", type=str, default=None,
                        help="Path to normalizer.pkl from the run the ckpt "
                             "was trained on. Required when --dataset_path "
                             "is overridden, otherwise the heldout file would "
                             "compute its own normalizer with wrong action "
                             "scale.")
    parser.add_argument("--num_samples", type=int, default=32,
                        help="How many dataset indices to draw")
    parser.add_argument("--num_steps_list", type=int, nargs="+", default=[5, 25],
                        help="Sampler step counts to evaluate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose(args)
