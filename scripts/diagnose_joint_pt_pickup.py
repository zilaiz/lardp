"""Offline diagnostic for the joint-PT franka checkpoint that misbehaves
after the gripper closes on the coffee pod.

Loads ``LBMDiTJointPTAgent`` + the chosen normalizer + the held-out HDF5
and, for each demo, runs receding-horizon prediction at every chunk
boundary. Compares the predicted action chunk to ground-truth, computes
within-chunk smoothness, and stratifies by pre- vs post- first gripper
closure. Writes one PNG per demo plus a printed summary.

Usage
-----
    python scripts/diagnose_joint_pt_pickup.py \\
        --ckpt_path       logs/<exp>/<ts>/models/model_step_100000.pt \\
        --config_path     outputs/<date>/<time>/.hydra/config.yaml \\
        --normalizer_path logs/<other-run>/models/normalizer.pkl \\
        --heldout_hdf5    data/franka_coffee_pod_cog/image_heldout.hdf5 \\
        --out_dir         outputs/diagnose_joint_pt_pickup \\
        --num_demos       6
"""
from __future__ import annotations

import argparse
import os
import pickle
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

from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent  # noqa: E402

GRIPPER_OPEN_THRESHOLD = 0.8  # below this = closed (range is ~[0.47, 1.0])


def _load_normalizer(path: str) -> dict:
    if str(path).endswith(".npz"):
        # Reuse the DP-pickup npz loader (single source of truth for the
        # .npz format produced by scripts/export_franka_normalizer.py).
        from scripts.diagnose_dp_pickup import _load_normalizer_npz
        return _load_normalizer_npz(str(path))
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_obs_window(
    demo: dict,
    t: int,
    obs_steps: int,
    shape_meta_obs: dict,
    normalizer: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Mirror interface_lbmdit_joint_pt._normalize_obs but for an HDF5 slice.

    Returns a dict of tensors with shape (1, obs_steps, ...) per key. For
    t < obs_steps - 1, pads with repeats of the first step (same as how the
    robot controller would feed the first inference call).
    """
    out = {}
    indices = [max(0, t - (obs_steps - 1 - k)) for k in range(obs_steps)]
    for key, meta in shape_meta_obs.items():
        meta_type = meta.get("type", "low_dim")
        # HDF5 is HWC uint8 for rgb, low_dim is float32 (N, D).
        arr = demo["obs"][key][indices]  # (To, ...)
        if meta_type == "rgb":
            # arr shape: (To, H, W, 3) uint8 -> (To, 3, H, W) float [-1, 1]
            arr = np.moveaxis(arr, -1, -3).astype(np.float32) / 255.0
            arr = normalizer["obs"][key].normalize(arr).astype(np.float32)
        else:
            arr = arr.astype(np.float32, copy=False)
            if arr.ndim == 1:
                arr = arr[:, None]
            arr = normalizer["obs"][key].normalize(arr).astype(np.float32)
        t_arr = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, To, ...)
        out[key] = t_arr
    return out


def _first_gripper_closure(grip_q: np.ndarray) -> int | None:
    """Index of first step where gripper drops below threshold from open."""
    grip = grip_q.squeeze()
    closed = grip < GRIPPER_OPEN_THRESHOLD
    if not closed.any():
        return None
    # First True after at least one False, otherwise first True.
    first = int(np.argmax(closed))
    return first


def _smoothness(chunk: np.ndarray) -> dict[str, float]:
    """Within-chunk consecutive-step L2 of (pos, rot6d, grip).

    chunk: (T, 10) -> dict with mean step-norms per group.
    """
    if chunk.shape[0] < 2:
        return {"pos": 0.0, "rot6d": 0.0, "grip": 0.0}
    d = np.diff(chunk, axis=0)
    return {
        "pos": float(np.linalg.norm(d[:, :3], axis=1).mean()),
        "rot6d": float(np.linalg.norm(d[:, 3:9], axis=1).mean()),
        "grip": float(np.abs(d[:, 9]).mean()),
    }


def _grip_flips(chunk: np.ndarray) -> int:
    """Number of binary gripper-command changes WITHIN a single chunk.

    The controller does ``a[:,9] = (a[:,9] > 0.5).astype(float32)`` on
    unnormalized chunks (raw {0, 1}); in normed space the equivalent
    threshold is 0 (MinMax maps 0->-1, 1->+1). A "flip" is open->close or
    close->open inside the 8-step executed slice — i.e., the gripper
    physically commanded to change state mid-chunk.
    """
    binarized = (chunk[:, 9] > 0.0).astype(np.int8)
    return int((np.diff(binarized) != 0).sum())


def _grip_excursion_type(chunk: np.ndarray) -> str:
    """Classify the within-chunk grip behavior into:
      - "stable_open"   : binarized command is all 0 across the chunk.
      - "stable_close"  : binarized command is all 1 across the chunk.
      - "open->close"   : starts at 0, transitions to 1, no return (legit close).
      - "close->open"   : starts at 1, transitions to 0, no return (legit release).
      - "spurious_close": starts at 0, has 1s, ends at 0 (open->close->open
        excursion — phantom grasp during a reach/transit).
      - "spurious_open" : starts at 1, has 0s, ends at 1 (close->open->close
        excursion — phantom release while holding object; drops the pod).
      - "complex"       : any other pattern (e.g., multiple round trips).

    Convention reminder for this dataset: action[:,9]==1 commands the
    gripper to CLOSE; action[:,9]==0 commands OPEN. In normed space
    (MinMax->[-1,+1]), the threshold is 0.
    """
    b = (chunk[:, 9] > 0.0).astype(np.int8)
    if b.size < 2:
        return "stable_open" if b[0] == 0 else "stable_close"
    n_flips = int((np.diff(b) != 0).sum())
    if n_flips == 0:
        return "stable_open" if b[0] == 0 else "stable_close"
    if n_flips == 1:
        return "open->close" if b[0] == 0 else "close->open"
    if n_flips == 2:
        if b[0] == 0 and b[-1] == 0:
            return "spurious_close"
        if b[0] == 1 and b[-1] == 1:
            return "spurious_open"
        return "complex"
    return "complex"


def _pos_reversals(chunk: np.ndarray, dead_band: float = 1e-3) -> dict[str, int]:
    """Count direction reversals along pos x/y/z inside a chunk.

    A reversal is a sign flip of consecutive diffs ignoring moves below
    ``dead_band`` (so tiny jitter doesn't count). Returns dict with the
    count per axis and a total. Physically smooth single-segment motion
    should have 0 reversals; >1 within an 8-step chunk is suspicious.
    """
    out = {"x": 0, "y": 0, "z": 0}
    if chunk.shape[0] < 3:
        return {**out, "total": 0}
    d = np.diff(chunk[:, :3], axis=0)  # (T-1, 3)
    sig = np.where(np.abs(d) > dead_band, np.sign(d), 0)  # (T-1, 3)
    for i, k in enumerate(("x", "y", "z")):
        s = sig[:, i]
        s = s[s != 0]
        if s.size >= 2:
            out[k] = int((np.diff(s) != 0).sum())
    out["total"] = out["x"] + out["y"] + out["z"]
    return out


def _err_per_dim(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-dim absolute error averaged over chunk steps. (T, 10) -> (10,)"""
    return np.abs(pred - gt).mean(axis=0)


def diagnose(args):
    cfg = OmegaConf.load(args.config_path)
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)
    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)
    if args.joint_num_steps is not None:
        OmegaConf.update(cfg, "optimization.joint_num_steps",
                         int(args.joint_num_steps), merge=False)
    if args.joint_t_schedule is not None:
        OmegaConf.update(cfg, "optimization.joint_t_schedule",
                         str(args.joint_t_schedule), merge=False)
    if args.joint_sample_mode is not None:
        OmegaConf.update(cfg, "optimization.joint_sample_mode",
                         str(args.joint_sample_mode), merge=False)
    if args.joint_cfg_scale is not None:
        OmegaConf.update(cfg, "optimization.joint_cfg_scale",
                         float(args.joint_cfg_scale), merge=False)

    device = torch.device(cfg.optimization.device)
    print(f"\n== ckpt: {args.ckpt_path}")
    print(f"== config: {args.config_path}")
    print(f"== normalizer: {args.normalizer_path}")
    print(f"== heldout:  {args.heldout_hdf5}")
    print(f"== device: {device}")
    print(f"== joint_num_steps: {cfg.optimization.joint_num_steps} | "
          f"joint_decouple_t: {cfg.optimization.joint_decouple_t} | "
          f"joint_t_schedule: {cfg.optimization.joint_t_schedule}")

    print("\n[1/3] Building agent + loading checkpoint...")
    agent = LBMDiTJointPTAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()

    print(f"\n[2/3] Loading normalizer + held-out demos...")
    normalizer = _load_normalizer(args.normalizer_path)
    shape_meta_obs = cfg.task.shape_meta["obs"]
    # npz normalizers only carry low_dim stats; pad rgb keys with the
    # standard ImageNormalizer (x*2 - 1) so _build_obs_window can call
    # normalizer["obs"][k].normalize(...) on images uniformly.
    from mip.dataset_utils import ImageNormalizer
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
        all_demos = {k: {"obs": {kk: f["data"][k]["obs"][kk][:] for kk in f["data"][k]["obs"]},
                          "actions": f["data"][k]["actions"][:]} for k in demo_keys}

    print(f"\n[3/3] Running predictions on {len(demo_keys)} demos...")

    summary_rows = []  # one per (demo, phase in {pre, post}) for printed summary

    for demo_key in demo_keys:
        demo = all_demos[demo_key]
        T = demo["actions"].shape[0]
        grip_q = demo["obs"]["robot0_gripper_qpos"][:].squeeze()
        pickup_idx = _first_gripper_closure(grip_q)
        gt_actions = demo["actions"]  # (T, 10) — already in TRAINING units

        # Normalize GT actions for apples-to-apples comparison with the
        # model's normalized output.
        gt_normed = normalizer["action"].normalize(gt_actions).astype(np.float32)

        # Walk chunk boundaries: t = 0, act_steps, 2*act_steps, ... while
        # t + horizon <= T. This mirrors receding-horizon execution.
        pred_chunks_normed = []  # list of (horizon, 10)
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
                    num_steps=int(cfg.optimization.joint_num_steps),
                    use_ema=True,
                )
            pred_np = pred[0].detach().cpu().numpy()  # (horizon, 10)
            pred_chunks_normed.append(pred_np)
            chunk_starts.append(t0)

        # Stratify smoothness + error by pre / post pickup.
        pre_smooth, post_smooth = [], []
        pre_err, post_err = [], []
        pre_pred_grip, post_pred_grip = [], []
        pre_rev_pred, post_rev_pred = [], []
        pre_rev_gt, post_rev_gt = [], []
        pre_flips_pred, post_flips_pred = [], []
        pre_flips_gt, post_flips_gt = [], []
        pre_excur_pred, post_excur_pred = [], []
        # Per-chunk executed-slice binarized grip end-values (one int per chunk).
        # Used to count INTER-CHUNK flips: how often consecutive executed
        # chunks disagree on the final commanded gripper state.
        per_chunk_first_bin = []   # binarized first step
        per_chunk_last_bin = []    # binarized last step
        per_chunk_t0 = []
        for t0, pred_np in zip(chunk_starts, pred_chunks_normed):
            # The robot executes pred[obs_steps-1 : obs_steps-1+act_steps];
            # use the same slice for comparing executed slice smoothness.
            start = obs_steps - 1
            end = start + act_steps
            exec_pred = pred_np[start:end]  # (act_steps, 10)
            exec_gt = gt_normed[t0:t0 + act_steps]  # (act_steps, 10)
            s = _smoothness(exec_pred)
            e = _err_per_dim(exec_pred, exec_gt)  # (10,)
            r_pred = _pos_reversals(exec_pred)
            r_gt = _pos_reversals(exec_gt)
            f_pred = _grip_flips(exec_pred)
            f_gt = _grip_flips(exec_gt)
            excur = _grip_excursion_type(exec_pred)
            grip_pred = exec_pred[:, 9]
            bin_pred = (grip_pred > 0.0).astype(np.int8)
            per_chunk_first_bin.append(int(bin_pred[0]))
            per_chunk_last_bin.append(int(bin_pred[-1]))
            per_chunk_t0.append(t0)
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

        # ---- summary row per demo ----
        def _agg(lst_dict, k):
            if not lst_dict:
                return float("nan")
            return float(np.mean([x[k] for x in lst_dict]))

        def _agg_err(lst):
            if not lst:
                return np.full(act_dim, np.nan)
            return np.mean(np.stack(lst), axis=0)

        def _agg_rev(lst, key):
            if not lst:
                return float("nan")
            return float(np.mean([r[key] for r in lst]))

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
            # Gripper within-chunk flips (open->close or close->open).
            "pre_grip_flips_pred_mean": float(np.mean(pre_flips_pred)) if pre_flips_pred else float("nan"),
            "post_grip_flips_pred_mean": float(np.mean(post_flips_pred)) if post_flips_pred else float("nan"),
            "pre_grip_flips_gt_mean": float(np.mean(pre_flips_gt)) if pre_flips_gt else float("nan"),
            "post_grip_flips_gt_mean": float(np.mean(post_flips_gt)) if post_flips_gt else float("nan"),
            # Fraction of chunks with >=1 flip.
            "pre_flip_chunk_frac": float(np.mean([f > 0 for f in pre_flips_pred])) if pre_flips_pred else float("nan"),
            "post_flip_chunk_frac": float(np.mean([f > 0 for f in post_flips_pred])) if post_flips_pred else float("nan"),
            "pre_excur": pre_excur_pred,
            "post_excur": post_excur_pred,
            # Inter-chunk boundary flips: count consecutive chunks where the
            # binarized command at end-of-N != start-of-N+1.
            "inter_chunk_pre_flips": int(sum(
                1 for i in range(1, len(per_chunk_t0))
                if per_chunk_t0[i] < (pickup_idx if pickup_idx is not None else T)
                and per_chunk_last_bin[i - 1] != per_chunk_first_bin[i]
            )),
            "inter_chunk_post_flips": int(sum(
                1 for i in range(1, len(per_chunk_t0))
                if pickup_idx is not None and per_chunk_t0[i] >= pickup_idx
                and per_chunk_last_bin[i - 1] != per_chunk_first_bin[i]
            )),
            "n_pre_chunks": int(sum(
                1 for t in per_chunk_t0
                if t < (pickup_idx if pickup_idx is not None else T)
            )),
            "n_post_chunks": int(sum(
                1 for t in per_chunk_t0
                if pickup_idx is not None and t >= pickup_idx
            )),
        })

        # ---- per-demo plot ----
        # Flatten predicted chunks into a continuous trace using the
        # executed slice (start:end) so the trace mirrors what would have
        # been sent to the robot under receding-horizon execution.
        exec_pred_concat = np.concatenate(
            [pc[obs_steps - 1: obs_steps - 1 + act_steps] for pc in pred_chunks_normed],
            axis=0,
        )
        t_exec = np.concatenate(
            [np.arange(t0, t0 + act_steps) for t0 in chunk_starts]
        )
        # Per-step error magnitude along the trace.
        err_mag = np.linalg.norm(exec_pred_concat - gt_normed[t_exec], axis=1)

        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        # 1. Pos (normalized): GT vs pred
        for i, lbl in enumerate(["x", "y", "z"]):
            axes[0].plot(np.arange(T), gt_normed[:, i], "-", lw=0.8, alpha=0.5,
                         label=f"GT {lbl}")
            axes[0].plot(t_exec, exec_pred_concat[:, i], "--", lw=0.8,
                         label=f"pred {lbl}")
        axes[0].set_ylabel("pos (normed)")
        axes[0].legend(ncol=2, fontsize=7)
        axes[0].set_title(f"{demo_key}  T={T}  pickup_idx={pickup_idx}")

        # 2. Gripper (normalized): GT vs pred
        axes[1].plot(np.arange(T), gt_normed[:, 9], "-", lw=1.0, label="GT grip")
        axes[1].plot(t_exec, exec_pred_concat[:, 9], "--", lw=0.8, label="pred grip")
        axes[1].axhline(0.0, color="k", lw=0.3, alpha=0.4)
        axes[1].set_ylabel("grip (normed)")
        axes[1].legend(fontsize=7)

        # 3. Raw gripper qpos (state side, not action)
        axes[2].plot(np.arange(T), grip_q, "-", lw=1.0, color="tab:purple",
                     label="gripper_qpos")
        axes[2].axhline(GRIPPER_OPEN_THRESHOLD, color="k", lw=0.3, ls=":",
                        label="closure thresh")
        axes[2].set_ylabel("gripper_qpos (raw)")
        axes[2].legend(fontsize=7)

        # 4. Per-step prediction error magnitude (action L2 in normed space)
        axes[3].plot(t_exec, err_mag, "-", lw=0.8, color="tab:red")
        axes[3].set_ylabel("||pred − gt|| (normed)")
        axes[3].set_xlabel("step")

        if pickup_idx is not None:
            for ax in axes:
                ax.axvline(pickup_idx, color="gray", lw=0.8, ls=":")

        fig.tight_layout()
        out_png = os.path.join(args.out_dir, f"{demo_key}.png")
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"   wrote {out_png}")

    # ----------------------- printed summary -----------------------
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

    # Aggregate across demos.
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
    # Inter-chunk flips: how often does the grip command flip across chunk
    # boundaries? This corresponds to the deployment-time pattern where
    # consecutive 8-step chunks disagree on the gripper state.
    tot_pre_inter = sum(r["inter_chunk_pre_flips"] for r in summary_rows)
    tot_post_inter = sum(r["inter_chunk_post_flips"] for r in summary_rows)
    tot_pre_n = sum(max(r["n_pre_chunks"] - 1, 0) for r in summary_rows)
    tot_post_n = sum(max(r["n_post_chunks"] - 1, 0) for r in summary_rows)
    print(f"  inter-chunk grip flips (boundary mismatch between consecutive "
          f"executed chunks):")
    print(f"    PRED : pre={tot_pre_inter}/{tot_pre_n} "
          f"({tot_pre_inter / max(tot_pre_n, 1):.2%})  "
          f"post={tot_post_inter}/{tot_post_n} "
          f"({tot_post_inter / max(tot_post_n, 1):.2%})")

    # ---- excursion type breakdown ----
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

    # Per-dim error aggregate (10-dim action).
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
                        default="outputs/diagnose_joint_pt_pickup")
    parser.add_argument("--num_demos", type=int, default=6,
                        help="0 = all held-out demos")
    parser.add_argument("--joint_num_steps", type=int, default=None,
                        help="Override optimization.joint_num_steps from cfg")
    parser.add_argument("--joint_t_schedule", type=str, default=None,
                        choices=["diagonal", "state_first", "pyramid",
                                 "action_only"])
    parser.add_argument("--joint_sample_mode", type=str, default=None,
                        choices=["stochastic", "zero"])
    parser.add_argument("--joint_cfg_scale", type=float, default=None)
    args = parser.parse_args()
    diagnose(args)
