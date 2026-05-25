"""Vanilla-DP sibling of ``diagnose_joint_pt_pickup.py``.

Loads ``TrainingAgent`` (the ``mip.agent.TrainingAgent`` used by
``interface_example.py``) and runs the same pre-/post-pickup smoothness +
reversal analysis on held-out franka demos. Use this to compare the joint
pipeline against a vanilla DP baseline trained on the same dataset.

Normalizer format here is the .npz produced by
``scripts/export_franka_normalizer.py`` (different from the joint pipeline's
pickle).

Usage
-----
    python scripts/diagnose_dp_pickup.py \\
        --ckpt_path       logs/<exp>/<ts>/models/model_step_90000.pt \\
        --config_path     outputs/<date>/<time>/.hydra/config.yaml \\
        --normalizer_path checkpoints/franka_coffee_pod_cog_lbmdit_normalizer.npz \\
        --heldout_hdf5    data/franka_coffee_pod_cog/image_heldout.hdf5 \\
        --out_dir         outputs/diagnose_dp_pickup
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

LARDP_PATH = Path(__file__).resolve().parents[1]
if str(LARDP_PATH) not in sys.path:
    sys.path.insert(0, str(LARDP_PATH))

from mip.agent import TrainingAgent  # noqa: E402
from mip.dataset_utils import ImageNormalizer, MinMaxNormalizer  # noqa: E402

# Reuse the joint-PT diagnostic helpers — they're agent-agnostic.
from scripts.diagnose_joint_pt_pickup import (  # noqa: E402
    GRIPPER_OPEN_THRESHOLD,
    _build_obs_window,
    _err_per_dim,
    _first_gripper_closure,
    _grip_excursion_type,
    _grip_flips,
    _pos_reversals,
    _smoothness,
)


def _populate_minmax(n: MinMaxNormalizer, mn, mx, rng) -> MinMaxNormalizer:
    n.min = mn.astype(np.float32)
    n.max = mx.astype(np.float32)
    n.range = rng.astype(np.float32)
    return n


def _load_normalizer_npz(path: str) -> dict:
    """Mirror interface_example._load_normalizer_from_npz."""
    data = np.load(path, allow_pickle=True)
    keys = [str(k) for k in data["keys"]]

    def _make(prefix: str) -> MinMaxNormalizer:
        n = MinMaxNormalizer(
            np.zeros((1, data[f"{prefix}_min"].shape[0]), dtype=np.float32)
        )
        return _populate_minmax(
            n, data[f"{prefix}_min"], data[f"{prefix}_max"],
            data[f"{prefix}_range"],
        )

    norm = {"obs": {}, "action": _make("action")}
    for key in keys:
        norm["obs"][key] = _make(f"obs__{key}")
    return norm


def diagnose(args):
    cfg = OmegaConf.load(args.config_path)
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)
    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)
    if args.num_steps is not None:
        OmegaConf.update(cfg, "optimization.num_steps",
                         int(args.num_steps), merge=False)

    device = torch.device(cfg.optimization.device)
    print(f"\n== ckpt: {args.ckpt_path}")
    print(f"== config: {args.config_path}")
    print(f"== normalizer: {args.normalizer_path}")
    print(f"== heldout:  {args.heldout_hdf5}")
    print(f"== device: {device}")
    print(f"== num_steps: {cfg.optimization.num_steps} | "
          f"loss_type: {cfg.optimization.loss_type} | "
          f"network: {cfg.network.network_type}")

    print("\n[1/3] Building agent + loading checkpoint...")
    agent = TrainingAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()

    print(f"\n[2/3] Loading normalizer + held-out demos...")
    normalizer = _load_normalizer_npz(args.normalizer_path)
    shape_meta_obs = cfg.task.shape_meta["obs"]
    # The npz holds only low_dim norms; images use the standard
    # ImageNormalizer (x*2 - 1), applied per the joint-pt _build_obs_window
    # contract that normalizer["obs"][key].normalize(...) handles it.
    for key, meta in shape_meta_obs.items():
        if meta.get("type", "low_dim") == "rgb" and key not in normalizer["obs"]:
            normalizer["obs"][key] = ImageNormalizer()
    obs_steps = int(cfg.task.obs_steps)
    horizon = int(cfg.task.horizon)
    act_steps = int(cfg.task.act_steps)
    act_dim = int(cfg.task.act_dim)
    print(f"   obs_steps={obs_steps} horizon={horizon} act_steps={act_steps} "
          f"act_dim={act_dim}")

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.heldout_hdf5, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if args.num_demos > 0:
            demo_keys = demo_keys[: args.num_demos]
        all_demos = {
            k: {"obs": {kk: f["data"][k]["obs"][kk][:]
                        for kk in f["data"][k]["obs"]},
                "actions": f["data"][k]["actions"][:]}
            for k in demo_keys
        }

    print(f"\n[3/3] Running predictions on {len(demo_keys)} demos "
          f"(num_steps={cfg.optimization.num_steps})...")

    summary_rows = []

    for demo_key in demo_keys:
        demo = all_demos[demo_key]
        T = demo["actions"].shape[0]
        grip_q = demo["obs"]["robot0_gripper_qpos"][:].squeeze()
        pickup_idx = _first_gripper_closure(grip_q)
        gt_actions = demo["actions"]
        gt_normed = normalizer["action"].normalize(gt_actions).astype(np.float32)

        pred_chunks_normed = []
        chunk_starts = []
        for t0 in range(0, T - horizon + 1, act_steps):
            obs = _build_obs_window(demo, t0, obs_steps, shape_meta_obs,
                                    normalizer, device)
            with torch.no_grad():
                act_0 = torch.randn(
                    (1, horizon, act_dim), device=device,
                )
                pred = agent.sample(
                    act_0=act_0,
                    obs=obs,
                    num_steps=int(cfg.optimization.num_steps),
                    use_ema=True,
                )
            pred_chunks_normed.append(pred[0].detach().cpu().numpy())
            chunk_starts.append(t0)

        pre_smooth, post_smooth = [], []
        pre_err, post_err = [], []
        pre_pred_grip, post_pred_grip = [], []
        pre_rev_pred, post_rev_pred = [], []
        pre_rev_gt, post_rev_gt = [], []
        pre_flips_pred, post_flips_pred = [], []
        pre_flips_gt, post_flips_gt = [], []
        pre_excur_pred, post_excur_pred = [], []
        for t0, pred_np in zip(chunk_starts, pred_chunks_normed):
            start = obs_steps - 1
            end = start + act_steps
            exec_pred = pred_np[start:end]
            exec_gt = gt_normed[t0:t0 + act_steps]
            s = _smoothness(exec_pred)
            e = _err_per_dim(exec_pred, exec_gt)
            r_pred = _pos_reversals(exec_pred)
            r_gt = _pos_reversals(exec_gt)
            f_pred = _grip_flips(exec_pred)
            f_gt = _grip_flips(exec_gt)
            excur = _grip_excursion_type(exec_pred)
            grip_pred = exec_pred[:, 9]
            bucket_post = pickup_idx is not None and t0 >= pickup_idx
            if bucket_post:
                post_smooth.append(s)
                post_err.append(e)
                post_rev_pred.append(r_pred)
                post_rev_gt.append(r_gt)
                post_pred_grip.extend(grip_pred.tolist())
                post_flips_pred.append(f_pred)
                post_flips_gt.append(f_gt)
                post_excur_pred.append(excur)
            else:
                pre_smooth.append(s)
                pre_err.append(e)
                pre_rev_pred.append(r_pred)
                pre_rev_gt.append(r_gt)
                pre_pred_grip.extend(grip_pred.tolist())
                pre_flips_pred.append(f_pred)
                pre_flips_gt.append(f_gt)
                pre_excur_pred.append(excur)

        def _agg(lst_dict, k):
            if not lst_dict:
                return float("nan")
            return float(np.mean([x[k] for x in lst_dict]))

        def _agg_err(lst):
            if not lst:
                return np.full(act_dim, np.nan)
            return np.mean(np.stack(lst), axis=0)

        def _agg_rev(lst, k):
            if not lst:
                return float("nan")
            return float(np.mean([r[k] for r in lst]))

        summary_rows.append({
            "demo": demo_key,
            "pickup_idx": pickup_idx,
            "T": T,
            "pre_smooth_pos": _agg(pre_smooth, "pos"),
            "pre_smooth_rot6d": _agg(pre_smooth, "rot6d"),
            "pre_smooth_grip": _agg(pre_smooth, "grip"),
            "post_smooth_pos": _agg(post_smooth, "pos"),
            "post_smooth_rot6d": _agg(post_smooth, "rot6d"),
            "post_smooth_grip": _agg(post_smooth, "grip"),
            "pre_err": _agg_err(pre_err),
            "post_err": _agg_err(post_err),
            "pre_grip_pred_mean": float(np.mean(pre_pred_grip)) if pre_pred_grip else float("nan"),
            "post_grip_pred_mean": float(np.mean(post_pred_grip)) if post_pred_grip else float("nan"),
            "pre_grip_pred_std": float(np.std(pre_pred_grip)) if pre_pred_grip else float("nan"),
            "post_grip_pred_std": float(np.std(post_pred_grip)) if post_pred_grip else float("nan"),
            "pre_rev_pred_total": _agg_rev(pre_rev_pred, "total"),
            "post_rev_pred_total": _agg_rev(post_rev_pred, "total"),
            "pre_rev_gt_total": _agg_rev(pre_rev_gt, "total"),
            "post_rev_gt_total": _agg_rev(post_rev_gt, "total"),
            "pre_grip_flips_pred_mean": float(np.mean(pre_flips_pred)) if pre_flips_pred else float("nan"),
            "post_grip_flips_pred_mean": float(np.mean(post_flips_pred)) if post_flips_pred else float("nan"),
            "pre_grip_flips_gt_mean": float(np.mean(pre_flips_gt)) if pre_flips_gt else float("nan"),
            "post_grip_flips_gt_mean": float(np.mean(post_flips_gt)) if post_flips_gt else float("nan"),
            "pre_flip_chunk_frac": float(np.mean([f > 0 for f in pre_flips_pred])) if pre_flips_pred else float("nan"),
            "post_flip_chunk_frac": float(np.mean([f > 0 for f in post_flips_pred])) if post_flips_pred else float("nan"),
            "pre_excur": pre_excur_pred,
            "post_excur": post_excur_pred,
        })

        # Per-demo plot
        exec_pred_concat = np.concatenate(
            [pc[obs_steps - 1: obs_steps - 1 + act_steps]
             for pc in pred_chunks_normed],
            axis=0,
        )
        t_exec = np.concatenate(
            [np.arange(t0, t0 + act_steps) for t0 in chunk_starts]
        )
        err_mag = np.linalg.norm(exec_pred_concat - gt_normed[t_exec], axis=1)

        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        for i, lbl in enumerate(["x", "y", "z"]):
            axes[0].plot(np.arange(T), gt_normed[:, i], "-", lw=0.8, alpha=0.5,
                         label=f"GT {lbl}")
            axes[0].plot(t_exec, exec_pred_concat[:, i], "--", lw=0.8,
                         label=f"pred {lbl}")
        axes[0].set_ylabel("pos (normed)")
        axes[0].legend(ncol=2, fontsize=7)
        axes[0].set_title(f"{demo_key}  T={T}  pickup_idx={pickup_idx}  (vanilla DP)")

        axes[1].plot(np.arange(T), gt_normed[:, 9], "-", lw=1.0, label="GT grip")
        axes[1].plot(t_exec, exec_pred_concat[:, 9], "--", lw=0.8, label="pred grip")
        axes[1].axhline(0.0, color="k", lw=0.3, alpha=0.4)
        axes[1].set_ylabel("grip (normed)")
        axes[1].legend(fontsize=7)

        axes[2].plot(np.arange(T), grip_q, "-", lw=1.0, color="tab:purple",
                     label="gripper_qpos")
        axes[2].axhline(GRIPPER_OPEN_THRESHOLD, color="k", lw=0.3, ls=":",
                        label="closure thresh")
        axes[2].set_ylabel("gripper_qpos (raw)")
        axes[2].legend(fontsize=7)

        axes[3].plot(t_exec, err_mag, "-", lw=0.8, color="tab:red")
        axes[3].set_ylabel("||pred - gt|| (normed)")
        axes[3].set_xlabel("step")

        if pickup_idx is not None:
            for ax in axes:
                ax.axvline(pickup_idx, color="gray", lw=0.8, ls=":")

        fig.tight_layout()
        out_png = os.path.join(args.out_dir, f"{demo_key}.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"   wrote {out_png}")

    print(f"\n{'=' * 100}")
    print("Per-demo summary (executed-slice metrics, normalized space)")
    print(f"{'=' * 100}")
    hdr = (
        f"{'demo':<10s} {'pickup':>7s} {'T':>4s} | "
        f"{'pre_pos_smooth':>14s} {'post_pos_smooth':>15s} | "
        f"{'pre_grip_smooth':>15s} {'post_grip_smooth':>16s} | "
        f"{'pre_grip_std':>12s} {'post_grip_std':>13s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary_rows:
        print(
            f"{r['demo']:<10s} {str(r['pickup_idx']):>7s} {r['T']:>4d} | "
            f"{r['pre_smooth_pos']:>14.4f} {r['post_smooth_pos']:>15.4f} | "
            f"{r['pre_smooth_grip']:>15.4f} {r['post_smooth_grip']:>16.4f} | "
            f"{r['pre_grip_pred_std']:>12.4f} {r['post_grip_pred_std']:>13.4f}"
        )

    def _mean_nan(name):
        vals = [r[name] for r in summary_rows if np.isfinite(r[name])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n{'=' * 60}")
    print("Aggregate (mean across demos, pre vs post first pickup)")
    print(f"{'=' * 60}")
    for nm in ["smooth_pos", "smooth_rot6d", "smooth_grip"]:
        print(f"  {nm:<14s}: pre={_mean_nan('pre_' + nm):.4f}  "
              f"post={_mean_nan('post_' + nm):.4f}  "
              f"ratio={_mean_nan('post_' + nm) / max(_mean_nan('pre_' + nm), 1e-9):.2f}x")
    print(f"  grip_pred_std : pre={_mean_nan('pre_grip_pred_std'):.4f}  "
          f"post={_mean_nan('post_grip_pred_std'):.4f}")
    print(f"  grip_pred_mean: pre={_mean_nan('pre_grip_pred_mean'):.4f}  "
          f"post={_mean_nan('post_grip_pred_mean'):.4f}")
    print(f"  pos_reversals (mean # per 8-step chunk, dead_band=1e-3 normed):")
    print(f"    PRED : pre={_mean_nan('pre_rev_pred_total'):.2f}  "
          f"post={_mean_nan('post_rev_pred_total'):.2f}")
    print(f"    GT   : pre={_mean_nan('pre_rev_gt_total'):.2f}  "
          f"post={_mean_nan('post_rev_gt_total'):.2f}")
    print(f"  grip within-chunk binary flips (mean # per 8-step chunk):")
    print(f"    PRED : pre={_mean_nan('pre_grip_flips_pred_mean'):.3f}  "
          f"post={_mean_nan('post_grip_flips_pred_mean'):.3f}")
    print(f"    GT   : pre={_mean_nan('pre_grip_flips_gt_mean'):.3f}  "
          f"post={_mean_nan('post_grip_flips_gt_mean'):.3f}")
    print(f"  frac of chunks with >=1 grip flip:")
    print(f"    PRED : pre={_mean_nan('pre_flip_chunk_frac'):.2%}  "
          f"post={_mean_nan('post_flip_chunk_frac'):.2%}")

    cat_order = ["stable_open", "stable_close", "open->close", "close->open",
                 "spurious_close", "spurious_open", "complex"]
    pre_all = sum((r["pre_excur"] for r in summary_rows), [])
    post_all = sum((r["post_excur"] for r in summary_rows), [])

    def _hist(items: list[str]) -> dict[str, float]:
        n = max(len(items), 1)
        return {c: items.count(c) / n for c in cat_order}

    pre_h = _hist(pre_all)
    post_h = _hist(post_all)
    print(f"\n{'=' * 60}")
    print("Excursion-type breakdown (% of chunks)")
    print(f"  spurious_close = starts OPEN, closes mid-chunk, reopens "
          f"(phantom grasp)")
    print(f"  spurious_open  = starts CLOSED, opens mid-chunk, recloses "
          f"(phantom release — drops object!)")
    print(f"{'=' * 60}")
    print(f"  {'category':<16s} {'pre':>10s} {'post':>10s}")
    for c in cat_order:
        print(f"  {c:<16s} {pre_h[c]:>9.2%} {post_h[c]:>9.2%}")

    pre_err_stack = np.stack(
        [r["pre_err"] for r in summary_rows if np.all(np.isfinite(r["pre_err"]))]
    )
    post_err_stack = np.stack(
        [r["post_err"] for r in summary_rows if np.all(np.isfinite(r["post_err"]))]
    )
    pre_e = pre_err_stack.mean(axis=0) if pre_err_stack.size else np.full(act_dim, np.nan)
    post_e = post_err_stack.mean(axis=0) if post_err_stack.size else np.full(act_dim, np.nan)
    names = ["pos_x", "pos_y", "pos_z",
             "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
             "grip"]
    print(f"\n{'=' * 60}")
    print("Per-dim abs error (mean over demos, normed space)")
    print(f"{'=' * 60}")
    for i, nm in enumerate(names):
        ratio = post_e[i] / max(pre_e[i], 1e-9)
        print(f"  {nm:<6s}: pre={pre_e[i]:.4f}  post={post_e[i]:.4f}  ratio={ratio:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--normalizer_path", type=str, required=True)
    parser.add_argument("--heldout_hdf5", type=str,
                        default="data/franka_coffee_pod_cog/image_heldout.hdf5")
    parser.add_argument("--out_dir", type=str,
                        default="outputs/diagnose_dp_pickup")
    parser.add_argument("--num_demos", type=int, default=6)
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override optimization.num_steps from cfg")
    args = parser.parse_args()
    diagnose(args)
