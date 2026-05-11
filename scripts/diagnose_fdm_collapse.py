"""Offline diagnostic: did the IDM+FDM training collapse the encoder?

Loads a trained ``IDMFDMAgent`` checkpoint and probes the encoder + FDM head
on the val split. Reports the metrics needed to distinguish:

  (A) Encoder collapse — Var(z_goal) shrunk; FDM MSE small only because the
      target is near-constant.
  (B) Trivial obs≈goal — FDM head learned a near-identity mapping that
      ignores the action.
  (C) Genuine forward-dynamics learning — FDM uses the action to bridge
      a non-trivial gap between obs and goal embeddings.

Reported on a held-out val batch (raw encoder space, no normalization):

  Encoder shape:
    var_per_dim_mean(z_goal), var_per_dim_mean(z_obs)
    ||z_goal||_2 mean, ||z_obs||_2 mean
    effective rank of z_goal (participation ratio of singular values)
    fraction of dims with std < 1e-3   -- "dead dims"
    mean cosine(z_obs_last, z_goal)    -- temporal correlation

  FDM head:
    mse(predicted_goal, z_goal)
    R^2 = 1 - mse / Var(z_goal)
    trivial baseline mse = mean per-dim Var(z_goal)
    fdm_loss / trivial_baseline                  -- < 1 means beats mean
    fdm_loss with act=0                          -- if ~unchanged, action ignored
    fdm_loss with act=shuffled_across_batch      -- ditto
    fdm_loss with act=Gaussian noise

Usage:

    python scripts/diagnose_fdm_collapse.py \
        --config_path outputs/2026-04-27/00-41-40/.hydra/config.yaml \
        --ckpt_path  logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_fdm_aux/2026_04_27_01_03_55/models/model_step_200000.pt \
        --num_batches 8

To trace evolution over training, pass ``--ckpt_glob`` instead of ``--ckpt_path``:

    python scripts/diagnose_fdm_collapse.py \
        --config_path outputs/2026-04-27/00-41-40/.hydra/config.yaml \
        --ckpt_glob 'logs/.../models/model_step_*.pt' \
        --num_batches 4
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys

import loguru
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mip.agent_idm_fdm import IDMFDMAgent  # noqa: E402
from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.torch_utils import set_seed  # noqa: E402


def _to_device(td_or_dict, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in td_or_dict.items()
    }


def _stack_obs_goal(obs_dict, goal_dict, obs_steps, device):
    """Replicate the train-loop preprocessing: stack To obs frames + 1 goal
    frame along dim=1 to get (B, To+1, ...) per key."""
    out = {}
    for k in obs_dict:
        o = obs_dict[k][:, :obs_steps].to(device)       # (B, To, ...)
        g = goal_dict[k].to(device)                      # (B, 1, ...)
        out[k] = torch.cat([o, g], dim=1)                # (B, To+1, ...)
    return out


def _effective_rank(M: torch.Tensor) -> float:
    """Participation ratio of singular values. For a (N, D) matrix:
       (sum sv)^2 / sum sv^2. Equals D for white matrix, 1 for rank-1."""
    M = M - M.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(M)
    return float((sv.sum() ** 2 / (sv.pow(2).sum() + 1e-12)).item())


def _fdm_mse(net, condition, act):
    pred = net.forward_predict(condition, act)
    target = condition[:, -1]
    return F.mse_loss(pred, target).item(), pred, target


def _summarize_split(name, z, var_per_dim):
    """name is a label; z is a (N, D) tensor; var_per_dim is (D,)."""
    norm = z.norm(dim=-1).mean().item()
    var_mean = var_per_dim.mean().item()
    std_per_dim = var_per_dim.clamp_min(0).sqrt()
    dead = (std_per_dim < 1e-3).float().mean().item()
    eff_rank = _effective_rank(z)
    return {
        "label": name,
        "n": int(z.shape[0]),
        "d": int(z.shape[1]),
        "var_per_dim_mean": var_mean,
        "norm_l2_mean": norm,
        "frac_dead_dims": dead,
        "effective_rank": eff_rank,
    }


def diagnose_one_ckpt(
    ckpt_path: str,
    cfg,
    device: str,
    dataloader,
    obs_steps: int,
    horizon: int,
    num_batches: int,
):
    """Run all diagnostics on one checkpoint, return a flat dict."""
    agent = IDMFDMAgent(cfg)
    agent.load(ckpt_path, load_optimizer=False)
    encoder = agent.encoder.eval()
    fdm_net = agent.flow_map.net.eval()

    z_obs_chunks = []     # last obs frame
    z_goal_chunks = []    # goal frame
    cos_obs_goal = []
    fdm_mse_real = []
    fdm_mse_zero = []
    fdm_mse_shuf = []
    fdm_mse_rand = []
    # Direction-only metrics: do predicted_goal and target_z_goal point the
    # same way regardless of magnitude?  If FDM is just shrinking magnitudes,
    # cos_pred_target stays low while MSE drops.
    cos_pred_target = []
    pred_norm_chunks = []
    target_norm_chunks = []
    norm_only_mse = []   # MSE between L2-normalized vectors (direction error)

    n_done = 0
    with torch.no_grad():
        for batch in dataloader:
            if n_done >= num_batches:
                break

            obs = _to_device(batch["obs"], device)
            goal = _to_device(batch["goal_obs"], device)
            stacked = _stack_obs_goal(obs, goal, obs_steps, device)

            from tensordict import TensorDict
            B = next(iter(stacked.values())).shape[0]
            obs_td = TensorDict(stacked, batch_size=B)

            encoded = encoder(obs_td, None)              # (B, To+1, D)
            z_obs_last = encoded[:, obs_steps - 1]        # last obs frame
            z_goal = encoded[:, -1]
            z_obs_chunks.append(z_obs_last.cpu())
            z_goal_chunks.append(z_goal.cpu())

            cos = F.cosine_similarity(z_obs_last, z_goal, dim=-1)
            cos_obs_goal.append(cos.cpu())

            act = batch["action"].to(device)[:, :horizon, :]

            # (1) Real action — also pull predicted_goal for direction metrics
            pred = fdm_net.forward_predict(encoded, act)   # (B, D)
            target = encoded[:, -1]                         # (B, D)
            fdm_mse_real.append(F.mse_loss(pred, target).item())

            cos_pt = F.cosine_similarity(pred, target, dim=-1)
            cos_pred_target.append(cos_pt.cpu())
            pred_norm_chunks.append(pred.norm(dim=-1).cpu())
            target_norm_chunks.append(target.norm(dim=-1).cpu())

            # Direction-only MSE: normalize both to unit length, then MSE.
            # = 2 * (1 - cos) per sample (Eucl-distance² of unit vectors).
            pred_n = F.normalize(pred, dim=-1)
            target_n = F.normalize(target, dim=-1)
            norm_only_mse.append(
                F.mse_loss(pred_n, target_n).item()
            )

            # (2) Zero action
            mse, _, _ = _fdm_mse(fdm_net, encoded, torch.zeros_like(act))
            fdm_mse_zero.append(mse)

            # (3) Shuffled action — break (s, a) correspondence
            perm = torch.randperm(B, device=device)
            mse, _, _ = _fdm_mse(fdm_net, encoded, act[perm])
            fdm_mse_shuf.append(mse)

            # (4) Random Gaussian action (matched scale)
            mse, _, _ = _fdm_mse(
                fdm_net, encoded, torch.randn_like(act) * act.std(),
            )
            fdm_mse_rand.append(mse)

            n_done += 1

    z_obs = torch.cat(z_obs_chunks, dim=0)        # (N, D)
    z_goal = torch.cat(z_goal_chunks, dim=0)
    cos_obs_goal = torch.cat(cos_obs_goal, dim=0)

    var_obs = z_obs.var(dim=0, unbiased=False)
    var_goal = z_goal.var(dim=0, unbiased=False)

    obs_summary = _summarize_split("z_obs_last", z_obs, var_obs)
    goal_summary = _summarize_split("z_goal", z_goal, var_goal)

    fdm_real = float(np.mean(fdm_mse_real))
    fdm_zero = float(np.mean(fdm_mse_zero))
    fdm_shuf = float(np.mean(fdm_mse_shuf))
    fdm_rand = float(np.mean(fdm_mse_rand))

    trivial_mse = float(var_goal.mean().item())   # MSE of predicting per-dim mean
    r2 = 1.0 - fdm_real / max(trivial_mse, 1e-12)

    # Trivial-identity baseline: predict z_obs_last directly. If FDM beats this
    # only by a hair, it's basically the obs encoder routed through.
    identity_mse = float(((z_obs - z_goal).pow(2)).mean().item())

    cos_pt_t = torch.cat(cos_pred_target, dim=0)
    pred_norms = torch.cat(pred_norm_chunks, dim=0)
    target_norms = torch.cat(target_norm_chunks, dim=0)

    return {
        "ckpt": ckpt_path,
        "z_obs_last": obs_summary,
        "z_goal": goal_summary,
        "cos_obs_goal_mean": float(cos_obs_goal.mean().item()),
        "cos_obs_goal_std": float(cos_obs_goal.std().item()),
        "fdm_mse_real_action": fdm_real,
        "fdm_mse_zero_action": fdm_zero,
        "fdm_mse_shuffled_action": fdm_shuf,
        "fdm_mse_random_action": fdm_rand,
        "fdm_r2_vs_target_var": r2,
        "trivial_baseline_mse": trivial_mse,
        "identity_baseline_mse": identity_mse,  # MSE of "predict z_obs_last as z_goal"
        "fdm_loss_vs_trivial": fdm_real / max(trivial_mse, 1e-12),
        "fdm_loss_vs_identity": fdm_real / max(identity_mse, 1e-12),
        "action_sensitivity": fdm_zero / max(fdm_real, 1e-12),  # >>1 ⇒ uses act
        # --- Direction-only metrics (scale-invariant) ---
        # If FDM is "real alignment", cos_pred_target is high and
        # direction_mse is small. If FDM is "just magnitude shrinkage",
        # cos_pred_target stays low but raw MSE looks small.
        "cos_pred_target_mean": float(cos_pt_t.mean().item()),
        "cos_pred_target_std": float(cos_pt_t.std().item()),
        "pred_norm_mean": float(pred_norms.mean().item()),
        "target_norm_mean": float(target_norms.mean().item()),
        "norm_ratio_pred_over_target": float(
            (pred_norms / target_norms.clamp_min(1e-8)).mean().item()
        ),
        "direction_only_mse": float(np.mean(norm_only_mse)),
    }


def _step_from_path(path: str) -> int:
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _print_report(rows):
    print("\n" + "=" * 86)
    print("FDM-collapse diagnostic")
    print("=" * 86)
    for r in rows:
        print(f"\n--- ckpt: {r['ckpt']} ---")
        zg = r["z_goal"]
        zo = r["z_obs_last"]
        print(f"  z_goal:       N={zg['n']:>5}  D={zg['d']:>4}  "
              f"var/dim={zg['var_per_dim_mean']:.4e}  ||z||={zg['norm_l2_mean']:.3f}  "
              f"eff_rank={zg['effective_rank']:.2f}  dead_dims={zg['frac_dead_dims']:.2%}")
        print(f"  z_obs_last:   N={zo['n']:>5}  D={zo['d']:>4}  "
              f"var/dim={zo['var_per_dim_mean']:.4e}  ||z||={zo['norm_l2_mean']:.3f}  "
              f"eff_rank={zo['effective_rank']:.2f}  dead_dims={zo['frac_dead_dims']:.2%}")
        print(f"  cos(z_obs_last, z_goal) = "
              f"{r['cos_obs_goal_mean']:.4f} ± {r['cos_obs_goal_std']:.4f}")
        print(f"  trivial-mean baseline MSE  = {r['trivial_baseline_mse']:.4e}")
        print(f"  identity baseline MSE      = {r['identity_baseline_mse']:.4e}  "
              f"(=MSE if FDM just outputs z_obs_last)")
        print(f"  FDM MSE  real action       = {r['fdm_mse_real_action']:.4e}  "
              f"(R^2 vs Var(z_goal) = {r['fdm_r2_vs_target_var']:+.3f})")
        print(f"  FDM MSE  zero action       = {r['fdm_mse_zero_action']:.4e}")
        print(f"  FDM MSE  shuffled action   = {r['fdm_mse_shuffled_action']:.4e}")
        print(f"  FDM MSE  random Gaussian   = {r['fdm_mse_random_action']:.4e}")
        print(f"  fdm_loss / trivial-mean    = {r['fdm_loss_vs_trivial']:.4f}")
        print(f"  fdm_loss / identity        = {r['fdm_loss_vs_identity']:.4f}")
        print(f"  action sensitivity (zero/real) = {r['action_sensitivity']:.3f}  "
              f"(>>1 ⇒ head uses action; ≈1 ⇒ ignores it)")
        print(f"  --- direction (scale-invariant) ---")
        print(f"  cos(predicted, target)         = "
              f"{r['cos_pred_target_mean']:.4f} ± {r['cos_pred_target_std']:.4f}")
        print(f"  ||predicted||  / ||target||    = {r['norm_ratio_pred_over_target']:.4f}  "
              f"(predicted={r['pred_norm_mean']:.3f}, target={r['target_norm_mean']:.3f})")
        print(f"  direction-only MSE             = {r['direction_only_mse']:.4e}  "
              f"(MSE on L2-normalized vectors; chance ≈ 2/D = "
              f"{2.0/r['z_goal']['d']:.4e})")

    print("\n" + "-" * 86)
    print("Quick interpretation guide:")
    print("  • z_goal var/dim → 0 or eff_rank → 1   ⇒ encoder collapse (A).")
    print("  • action_sensitivity ≈ 1               ⇒ FDM head ignores the action (B).")
    print("  • fdm_loss / identity << 1             ⇒ FDM does more than copy obs (good).")
    print("  • cos(predicted, target) high AND      ⇒ FDM is genuinely aligning,")
    print("    direction-only MSE small                  not just shrinking magnitudes.")
    print("  • cos(predicted, target) low AND       ⇒ FDM is mostly magnitude shrinkage,")
    print("    raw MSE small                             not direction alignment.")
    print("-" * 86)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_glob", type=str, default=None,
                        help="Glob pattern over multiple checkpoints — sweeps "
                             "by training step (sorted ascending).")
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalizer_path", type=str, default=None)
    parser.add_argument(
        "--cfg_override", type=str, nargs="*", default=[],
        help="OmegaConf dotlist overrides, e.g. task.act_steps=6",
    )
    args = parser.parse_args()

    if (args.ckpt_path is None) == (args.ckpt_glob is None):
        raise ValueError("Provide exactly one of --ckpt_path or --ckpt_glob")

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(args.config_path)
    if args.cfg_override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.cfg_override))
    cfg.optimization.device = device
    cfg.optimization.use_compile = False
    cfg.optimization.use_cudagraphs = False
    cfg.optimization.auto_resume = False
    cfg.optimization.model_path = None
    if cfg.task.obs_type == "image":
        cfg.task.obs_dim = cfg.network.emb_dim

    # Normalizer (so dataset matches the IDM's training stats)
    if args.normalizer_path is not None:
        norm_path = args.normalizer_path
    else:
        # Default: sit next to the checkpoint(s)
        any_ckpt = args.ckpt_path or sorted(glob.glob(args.ckpt_glob))[0]
        norm_path = os.path.join(os.path.dirname(any_ckpt), "normalizer.pkl")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"normalizer.pkl missing at {norm_path}")
    with open(norm_path, "rb") as f:
        normalizer = pickle.load(f)
    loguru.logger.info(f"Loaded normalizer from {norm_path}")

    dataset = make_idm_dataset(cfg.task, mode=args.mode, normalizer=normalizer)
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        dataset = dataset.datasets[0]
    loguru.logger.info(f"Dataset (mode={args.mode}): {len(dataset)} samples")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    obs_steps = cfg.task.obs_steps
    horizon = cfg.task.horizon

    if args.ckpt_path:
        ckpt_paths = [args.ckpt_path]
    else:
        ckpt_paths = sorted(glob.glob(args.ckpt_glob), key=_step_from_path)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints matched: {args.ckpt_glob}")
        loguru.logger.info(f"Sweeping {len(ckpt_paths)} checkpoints")

    rows = []
    for ckpt in ckpt_paths:
        loguru.logger.info(f"Probing {ckpt}")
        row = diagnose_one_ckpt(
            ckpt, cfg, device, dataloader, obs_steps, horizon, args.num_batches,
        )
        rows.append(row)

    _print_report(rows)


if __name__ == "__main__":
    main()
