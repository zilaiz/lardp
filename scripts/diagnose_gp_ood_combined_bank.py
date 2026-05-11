"""GP OOD trajectory using a *combined* expert bank (training + held-out).

The earlier diagnostic built the expert bank only from the held-out HDF5
(22 episodes, 6383 action steps). That's a thin sample of the goal-
embedding manifold and could overstate the predicted→nearest-expert
distance simply because the bank misses parts of the manifold.

This script enlarges the bank by adding goal embeddings from the *training*
expert HDF5 (`image.hdf5`, 41 episodes, 13496 action steps) — the goals
the GP-DiT was actually trained on. If the GP can't even land near goals
it has seen during training, OOD-ness is a true structural failure, not
a held-out-sample artifact.

Z-score reference: the agent's loaded goal_stats (the same stats the GP
was trained against). All distances are reported in those units.

For each GP checkpoint:
  - paired error  ‖z_pred − z_exp_paired‖  (held-out only)
  - nearest-expert in the *combined* bank
  - mean bias from the combined-bank mean

Reference baselines computed once:
  - random expert-pair distance in the combined bank
  - expert → nearest-other-expert in the combined bank (split by source)
"""

import argparse
import os
import pickle
import sys
import time
from glob import glob

import hydra
import loguru
import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, "/oscar/data/csun45/zzeng28/repo/lardp")

from mip.agent_goal_predictor_dit import GoalPredictorDiTAgent  # noqa: E402
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

CONFIG_DIR = "/oscar/data/csun45/zzeng28/repo/lardp/examples/configs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_cfg(idm_ckpt, goal_stats, batch_size, goal_dit_d_model=512):
    overrides = [
        "task=franka_coffee_pod_cog_image_gp",
        "network=goal_predictor_dit_v2",
        f"network.goal_dit_d_model={goal_dit_d_model}",
        "optimization.loss_type=goal_predictor_dit",
        f"optimization.batch_size={batch_size}",
        f"optimization.idm_checkpoint_path={idm_ckpt}",
        f"optimization.goal_stats_path={goal_stats}",
        "optimization.auto_resume=False",
        "log.wandb_mode=disabled",
    ]
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = hydra.compose(config_name="main_franka", overrides=overrides)
    return cfg


def to_dev(d, slice_steps=None):
    out = {}
    for k, v in d.items():
        if slice_steps is not None:
            v = v[:, :slice_steps]
        out[k] = v.to(DEV, non_blocking=True)
    bs = next(iter(out.values())).shape[0]
    return TensorDict(out, batch_size=bs)


@torch.no_grad()
def goal_dit_ode(goal_flow_map, z_t, num_steps, generator):
    B, _, emb_dim = z_t.shape
    g = torch.randn(
        (B, 1, emb_dim), device=z_t.device, dtype=z_t.dtype, generator=generator,
    )
    schedule = np.linspace(0, 1, num_steps + 1)
    for i in range(num_steps):
        s_val, t_val = schedule[i], schedule[i + 1]
        s = torch.full((B,), s_val, device=z_t.device)
        v = goal_flow_map.get_velocity(s, g, z_t)
        g = g + v * (t_val - s_val)
    return g


def discover_steps(ckpt_dir, steps_str=None):
    paths = sorted(glob(os.path.join(ckpt_dir, "model_step_*.pt")))
    items = []
    for p in paths:
        try:
            s = int(os.path.basename(p).split("_step_")[1].split(".")[0])
        except Exception:
            continue
        items.append((s, p))
    items.sort(key=lambda x: x[0])
    if steps_str:
        wanted = {int(s) for s in steps_str.split(",")}
        items = [it for it in items if it[0] in wanted]
    return items


@torch.no_grad()
def encode_goals_from_dataset(agent, dataset, batch_size, num_workers, max_samples,
                              stride=1, slice_steps_obs=None, label="goals"):
    """Iterate dataset and encode goal frames with the frozen IDM encoder.

    Returns z_g (M, D) and (optionally) z_t (M, To, D) for the same samples.
    Subsamples at the given stride and caps at max_samples for memory.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True, drop_last=False,
    )
    g_list, t_list = [], []
    n_done = 0
    for batch in loader:
        if n_done >= max_samples:
            break
        # Stride-subsample within the batch.
        idx = torch.arange(0, batch["action"].shape[0], stride)
        if len(idx) == 0:
            continue
        goal_obs = to_dev({k: v[idx] for k, v in batch["goal_obs"].items()})
        z_g = agent._inner_encoder(goal_obs, None)[:, 0, :]  # (b, D)
        g_list.append(z_g)
        if slice_steps_obs is not None:
            obs = to_dev({k: v[idx] for k, v in batch["obs"].items()},
                         slice_steps=slice_steps_obs)
            z_t = agent._inner_encoder(obs, None)               # (b, To, D)
            t_list.append(z_t)
        n_done += len(idx)
        if n_done % 2000 < batch_size:
            loguru.logger.info(f"  encoded {label}: {n_done}")
    if n_done > max_samples:
        # Trim if we slightly overshot.
        out_g = torch.cat(g_list, dim=0)[:max_samples]
        out_t = torch.cat(t_list, dim=0)[:max_samples] if t_list else None
    else:
        out_g = torch.cat(g_list, dim=0)
        out_t = torch.cat(t_list, dim=0) if t_list else None
    return out_g, out_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gp-ckpt-dir", required=True)
    p.add_argument("--idm-ckpt", required=True)
    p.add_argument("--goal-stats", required=True)
    p.add_argument("--heldout-path", required=True,
                   help="Held-out HDF5 — predictions are made from these obs, "
                        "and these expert goals are the paired ground truth.")
    p.add_argument("--train-path", required=True,
                   help="Training expert HDF5 (image.hdf5) — its goal frames "
                        "are added to the expert bank only.")
    p.add_argument("--steps", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-heldout-batches", type=int, default=4,
                   help="How many held-out batches' obs to use for predictions.")
    p.add_argument("--bank-stride", type=int, default=4,
                   help="Subsample stride for the training bank (~density).")
    p.add_argument("--bank-max-samples", type=int, default=20000,
                   help="Cap on combined bank size to keep nearest-neighbor cheap.")
    p.add_argument("--nfe-goal", type=int, default=5)
    p.add_argument("--goal-dit-d-model", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-npz", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    limit_threads(1)
    torch.set_float32_matmul_precision("high")

    ckpt_list = discover_steps(args.gp_ckpt_dir, args.steps)
    if not ckpt_list:
        raise FileNotFoundError(f"No GP checkpoints in {args.gp_ckpt_dir}")
    loguru.logger.info(f"Tracking {len(ckpt_list)} GP checkpoints")

    norm_path = os.path.join(os.path.dirname(args.idm_ckpt), "normalizer.pkl")
    with open(norm_path, "rb") as f:
        normalizer = pickle.load(f)

    cfg = build_cfg(args.idm_ckpt, args.goal_stats, args.batch_size, args.goal_dit_d_model)
    cfg.task.obs_dim = cfg.network.emb_dim
    obs_steps = cfg.task.obs_steps
    horizon = cfg.task.horizon
    act_steps = cfg.task.act_steps

    def make_dataset(path):
        return RobomimicImageIDMDataset(
            dataset_dir=os.path.expanduser(path),
            shape_meta=cfg.task.shape_meta,
            n_obs_steps=obs_steps,
            horizon=horizon,
            pad_before=obs_steps - 1,
            pad_after=act_steps - 1,
            abs_action=cfg.task.abs_action,
            val_dataset_percentage=0.0,
            mode="train",
            normalizer=normalizer,
        )

    held_ds = make_dataset(args.heldout_path)
    train_ds = make_dataset(args.train_path)
    loguru.logger.info(f"held-out: {len(held_ds)}  |  train (bank): {len(train_ds)}")

    agent = GoalPredictorDiTAgent(cfg)
    agent.eval()

    # Z-score reference: the SAME goal_stats the GP was trained against.
    # agent._goal_mean shape (D,), agent._goal_var shape (D,).
    goal_mean = agent._goal_mean        # (D,)
    goal_var  = agent._goal_var
    goal_sd   = torch.sqrt(goal_var + agent._norm_eps)

    def to_z(x):
        return (x - goal_mean) / goal_sd

    # 1) Encode held-out goals + obs (cap to a few batches; predictions made on these).
    held_z_g, held_z_t = encode_goals_from_dataset(
        agent, held_ds, args.batch_size, args.num_workers,
        max_samples=args.batch_size * args.num_heldout_batches,
        stride=1, slice_steps_obs=obs_steps, label="held-out",
    )
    # 2) Encode training goals (bank only; subsample with stride for diversity).
    train_z_g, _ = encode_goals_from_dataset(
        agent, train_ds, args.batch_size, args.num_workers,
        max_samples=args.bank_max_samples,
        stride=args.bank_stride, slice_steps_obs=None, label="train (bank)",
    )

    held_z_g_z = to_z(held_z_g)             # (M_held, D) z-scored
    train_z_g_z = to_z(train_z_g)           # (M_train, D) z-scored
    bank_z = torch.cat([held_z_g_z, train_z_g_z], dim=0)   # combined, z-scored
    bank_np = bank_z.cpu().numpy()
    M_held = held_z_g_z.shape[0]
    M_bank = bank_np.shape[0]
    D = bank_np.shape[1]
    loguru.logger.info(f"bank size combined: {M_bank}  (held={M_held}, train={M_bank - M_held})  D={D}")

    bank_sq = (bank_np ** 2).sum(axis=-1)

    def nearest_dist(query_np, exclude_self_offset=None):
        """L2 to nearest bank row. If exclude_self_offset is not None, mask
        self-distance for the held-portion of the bank (rows 0..M_held)."""
        Q = query_np.shape[0]
        # Compute in chunks to keep memory reasonable.
        out = np.empty(Q)
        chunk = 512
        for i in range(0, Q, chunk):
            qi = query_np[i:i + chunk]
            qsq = (qi ** 2).sum(axis=-1, keepdims=True)
            cross = qi @ bank_np.T
            d2 = qsq + bank_sq[None, :] - 2.0 * cross
            d2 = np.maximum(d2, 0.0)
            if exclude_self_offset is not None:
                # self-distance is at column index = (i + local_row) + exclude_self_offset
                rows_global = np.arange(i, i + qi.shape[0])
                self_cols = rows_global + exclude_self_offset
                d2[np.arange(qi.shape[0]), self_cols] = np.inf
            out[i:i + chunk] = np.sqrt(d2.min(axis=-1))
        return out

    # 3) Reference baselines.
    rng = np.random.default_rng(args.seed)
    n_pairs = 5000
    pi = rng.integers(0, M_bank, n_pairs)
    pj = rng.integers(0, M_bank, n_pairs)
    pair_d = np.linalg.norm(bank_np[pi] - bank_np[pj], axis=-1)
    # Held-only NN (excluding self) within the COMBINED bank.
    nn_held = nearest_dist(bank_np[:M_held], exclude_self_offset=0)
    # Train-only NN (excluding self).
    nn_train = nearest_dist(bank_np[M_held:], exclude_self_offset=M_held)

    print()
    print("=" * 110)
    print(f"COMBINED expert bank reference (N={M_bank}, D={D}, "
          f"held={M_held}, train={M_bank - M_held})")
    print(f"  random pair z-distance:   "
          f"med={float(np.median(pair_d)):.3f}  P5={float(np.percentile(pair_d,5)):.3f}  "
          f"P95={float(np.percentile(pair_d,95)):.3f}")
    print(f"  held → nearest in bank:   "
          f"med={float(np.median(nn_held)):.3f}  P95={float(np.percentile(nn_held,95)):.3f}")
    print(f"  train → nearest in bank:  "
          f"med={float(np.median(nn_train)):.3f}  P95={float(np.percentile(nn_train,95)):.3f}")
    print("=" * 110)

    # 4) Sweep checkpoints.
    g_torch = torch.Generator(device=DEV).manual_seed(args.seed)
    rows = []
    for label_step, path in ckpt_list:
        t0 = time.time()
        agent.load(path)
        agent.eval()
        g_torch.manual_seed(args.seed)
        with torch.no_grad():
            pred_norm = goal_dit_ode(
                agent.goal_flow_map_ema, held_z_t, args.nfe_goal, g_torch,
            )                                                  # (M_held, 1, D)
            pred_raw = agent._denormalize(pred_norm)[:, 0, :]   # (M_held, D)
        pred_z = to_z(pred_raw)
        pred_np = pred_z.cpu().numpy()

        paired_err = np.linalg.norm(pred_np - held_z_g_z.cpu().numpy(), axis=-1)
        nn_combined = nearest_dist(pred_np, exclude_self_offset=None)
        # Bias: difference of pred mean from bank mean (bank z-score has mu≈0
        # but not exactly because we used training-time goal_stats, not bank-empirical).
        mu_pred = pred_np.mean(axis=0)
        mu_bank = bank_np.mean(axis=0)
        bias_z = float(np.linalg.norm(mu_pred - mu_bank))
        radius_pred = float(np.linalg.norm(pred_np, axis=-1).mean())

        dt = time.time() - t0
        rows.append({
            "step": label_step,
            "paired_err_med":   float(np.median(paired_err)),
            "paired_err_mean":  float(paired_err.mean()),
            "nn_combined_med":  float(np.median(nn_combined)),
            "nn_combined_mean": float(nn_combined.mean()),
            "nn_combined_p5":   float(np.percentile(nn_combined, 5)),
            "nn_combined_p95":  float(np.percentile(nn_combined, 95)),
            "bias_z":           bias_z,
            "radius_pred":      radius_pred,
            "wall_s":           dt,
        })
        loguru.logger.info(
            f"step={label_step:>7d}  "
            f"NN-combined med={rows[-1]['nn_combined_med']:.3f}  "
            f"P5={rows[-1]['nn_combined_p5']:.3f}  "
            f"P95={rows[-1]['nn_combined_p95']:.3f}  "
            f"paired med={rows[-1]['paired_err_med']:.3f}  "
            f"bias_z={bias_z:.3f}  ({dt:.1f}s)"
        )

    print()
    print(
        f"{'step':>7s}  {'paired_med':>10s}  {'NN_med':>7s}  {'NN_P5':>7s}  "
        f"{'NN_P95':>7s}  {'bias_z':>8s}  {'radius':>7s}"
    )
    for r in rows:
        print(
            f"{r['step']:>7d}  {r['paired_err_med']:>10.3f}  "
            f"{r['nn_combined_med']:>7.3f}  {r['nn_combined_p5']:>7.3f}  "
            f"{r['nn_combined_p95']:>7.3f}  {r['bias_z']:>8.3f}  "
            f"{r['radius_pred']:>7.3f}"
        )

    # Verdict — does the larger bank meaningfully reduce nearest-expert distance?
    nn_first = rows[0]["nn_combined_med"]
    nn_last  = rows[-1]["nn_combined_med"]
    held_nn_ref = float(np.median(nn_held))
    print()
    print(f"reference: held-expert→nearest-bank med = {held_nn_ref:.3f}  "
          f"(this is the 'on-manifold' baseline; predictions ≫ this means OOD)")
    print(f"GP NN-combined median: {nn_first:.3f} → {nn_last:.3f}  "
          f"(over training)")
    print(f"GP NN / held-expert NN ratio = {nn_last / held_nn_ref:.2f}× "
          f"(>>1 → predictions are far more isolated than experts even with the larger bank)")

    if args.out_npz:
        out = {
            "steps": np.array([r["step"] for r in rows]),
            **{k: np.array([r[k] for r in rows]) for k in rows[0].keys() if k != "step"},
            "ref_pair_d": pair_d,
            "ref_nn_held": nn_held,
            "ref_nn_train": nn_train,
        }
        np.savez(args.out_npz, **out)
        print(f"\nSaved to {args.out_npz}")


if __name__ == "__main__":
    main()
