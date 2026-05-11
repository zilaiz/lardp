"""Track GP OOD-ness across training checkpoints.

For each GP checkpoint, run goal-DiT inference on a fixed batch of held-out
obs and measure how off-manifold the predicted goal embeddings are
(relative to the frozen IDM encoder's image of expert goals). Reports:

  - GP-paired error ‖pred − exp‖_z         : per-sample displacement
  - nearest-expert distance ‖pred − exp_NN‖_z : where pred lands in the
                                              expert manifold neighborhood
  - mean bias ‖μ_pred − μ_exp‖_z            : systematic offset
  - global radius ‖z_pred‖₂                 : compared to √D ≈ ‖z_exp‖₂
  - per-dim Var(pred)/Var(exp) median       : collapse signal

Expert embeddings are computed ONCE with the frozen IDM encoder; only
goal-DiT weights are swapped per checkpoint. This makes the sweep cheap
(~5s/ckpt).

If OOD-ness (nearest-expert distance, paired error) grows monotonically
with training, it's an overfitting symptom. If it's flat from step 10k
onwards, it's an architectural / loss-shape issue.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gp-ckpt-dir", required=True)
    p.add_argument("--idm-ckpt", required=True)
    p.add_argument("--goal-stats", required=True)
    p.add_argument("--heldout-path", required=True)
    p.add_argument("--steps", type=str, default=None,
                   help="Comma-separated step numbers; default = all available.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-batches", type=int, default=4,
                   help="How many held-out batches to aggregate per checkpoint.")
    p.add_argument("--num-workers", type=int, default=2)
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
        raise FileNotFoundError(f"No GP checkpoints found in {args.gp_ckpt_dir}")
    loguru.logger.info(f"Tracking {len(ckpt_list)} GP checkpoints")

    norm_path = os.path.join(os.path.dirname(args.idm_ckpt), "normalizer.pkl")
    with open(norm_path, "rb") as f:
        normalizer = pickle.load(f)

    cfg = build_cfg(args.idm_ckpt, args.goal_stats, args.batch_size, args.goal_dit_d_model)
    cfg.task.obs_dim = cfg.network.emb_dim
    obs_steps = cfg.task.obs_steps
    horizon = cfg.task.horizon
    act_steps = cfg.task.act_steps

    dataset = RobomimicImageIDMDataset(
        dataset_dir=os.path.expanduser(args.heldout_path),
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
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=False,
    )
    loguru.logger.info(f"Held-out: {len(dataset)} samples")

    # Build agent (loads frozen IDM once); will swap goal-DiT per ckpt.
    agent = GoalPredictorDiTAgent(cfg)
    agent.eval()

    # Encode obs + expert goals ONCE (frozen encoder).
    z_t_list, exp_list = [], []
    n_done = 0
    for batch in loader:
        if n_done >= args.num_batches:
            break
        n_done += 1
        obs = to_dev(dict(batch["obs"]), slice_steps=obs_steps)
        goal_obs = to_dev(dict(batch["goal_obs"]))
        with torch.no_grad():
            z_t = agent._inner_encoder(obs, None)              # (B, To, D)
            z_g = agent._inner_encoder(goal_obs, None)[:, 0, :]  # (B, D)
        z_t_list.append(z_t)
        exp_list.append(z_g)
    z_t_all = torch.cat(z_t_list, dim=0)
    exp_all = torch.cat(exp_list, dim=0)
    N, D = exp_all.shape
    loguru.logger.info(f"Bank: N={N}, D={D}")

    # Z-score against expert bank (matching prior diagnostics).
    mu = exp_all.mean(dim=0)
    sd = exp_all.std(dim=0).clamp_min(1e-9)
    z_exp = (exp_all - mu) / sd
    var_exp_dim = z_exp.var(dim=0)  # = 1 by construction; just for ratio sanity.

    # Brute-force pairwise distance bank (1024×1024 fits in memory at fp32).
    # We'll reuse this for nearest-expert lookup (excluding self).
    z_exp_cpu = z_exp.cpu().numpy()
    # Pre-compute for nearest-expert lookup — N rows of distances.
    # Use squared distance trick for speed.
    expert_sq = (z_exp_cpu ** 2).sum(axis=-1)

    def nearest_expert_dist(z_pred_np):
        # z_pred_np: (N, D). Returns (N,) min L2 distance to any expert in bank.
        pred_sq = (z_pred_np ** 2).sum(axis=-1, keepdims=True)
        cross = z_pred_np @ z_exp_cpu.T
        d2 = pred_sq + expert_sq[None, :] - 2.0 * cross
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(d2.min(axis=-1))

    g_torch = torch.Generator(device=DEV).manual_seed(args.seed)

    rows = []
    for label_step, path in ckpt_list:
        t0 = time.time()
        agent.load(path)
        agent.eval()
        # Re-seed generator per ckpt so goal-DiT noise is comparable.
        g_torch.manual_seed(args.seed)
        with torch.no_grad():
            pred_norm = goal_dit_ode(
                agent.goal_flow_map_ema, z_t_all, args.nfe_goal, g_torch,
            )                                                  # (N, 1, D)
            pred_raw = agent._denormalize(pred_norm)[:, 0, :]   # (N, D)

        # Z-score against the SAME expert bank as everything else.
        z_pred = (pred_raw - mu) / sd
        z_pred_np = z_pred.cpu().numpy()

        # Metrics
        paired_err = np.linalg.norm(z_pred_np - z_exp_cpu, axis=-1)
        nn_dist    = nearest_expert_dist(z_pred_np)
        mu_pred    = z_pred_np.mean(axis=0)
        bias_z     = float(np.linalg.norm(mu_pred))   # mu_exp z-norm = 0 by construction
        var_pred   = z_pred_np.var(axis=0)
        var_ratio  = var_pred / np.maximum(var_exp_dim.cpu().numpy(), 1e-12)
        radius     = np.linalg.norm(z_pred_np, axis=-1)
        # Centered cos_sim with paired expert.
        pred_c = z_pred_np - mu_pred
        exp_c = z_exp_cpu  # already mean-zero in z-space
        cos_centered = (pred_c * exp_c).sum(axis=-1) / np.maximum(
            np.linalg.norm(pred_c, axis=-1) * np.linalg.norm(exp_c, axis=-1), 1e-9,
        )

        dt = time.time() - t0
        rows.append({
            "step": label_step,
            "paired_err_mean": float(paired_err.mean()),
            "paired_err_med":  float(np.median(paired_err)),
            "nn_dist_mean":    float(nn_dist.mean()),
            "nn_dist_med":     float(np.median(nn_dist)),
            "mean_bias_z":     bias_z,
            "var_ratio_med":   float(np.median(var_ratio)),
            "var_ratio_mean":  float(var_ratio.mean()),
            "radius_mean":     float(radius.mean()),
            "centered_cos_med": float(np.median(cos_centered)),
            "wall_s":          dt,
        })
        loguru.logger.info(
            f"step={label_step:>7d}  "
            f"‖pred−exp‖_z med={rows[-1]['paired_err_med']:.3f}  "
            f"NN-exp med={rows[-1]['nn_dist_med']:.3f}  "
            f"‖μ_pred‖_z={bias_z:.3f}  "
            f"var_ratio_med={rows[-1]['var_ratio_med']:.3f}  "
            f"radius={rows[-1]['radius_mean']:.3f}  ({dt:.1f}s)"
        )

    # Reference baselines (computed once on the bank).
    rng = np.random.default_rng(0)
    pairs_i = rng.integers(0, N, 5000)
    pairs_j = rng.integers(0, N, 5000)
    expert_pair_d = np.linalg.norm(z_exp_cpu[pairs_i] - z_exp_cpu[pairs_j], axis=-1)
    # Nearest-other-expert (excluding self).
    nn_self = []
    for i in range(N):
        d2 = expert_sq + expert_sq[i] - 2 * z_exp_cpu @ z_exp_cpu[i]
        d2[i] = np.inf
        nn_self.append(np.sqrt(max(d2.min(), 0.0)))
    nn_self = np.array(nn_self)

    print()
    print("=" * 110)
    print(f"GP OOD trajectory across training (N={N}, D={D}, NFE_goal={args.nfe_goal})")
    print("Reference: random expert-pair z-dist  med={:.3f}, P95={:.3f}  |  expert→nearest-other-expert med={:.3f}, P95={:.3f}".format(
        float(np.median(expert_pair_d)), float(np.percentile(expert_pair_d, 95)),
        float(np.median(nn_self)),       float(np.percentile(nn_self, 95)),
    ))
    print("=" * 110)
    print(
        f"{'step':>7s}  {'‖pred−exp‖':>11s}  {'NN-exp':>7s}  {'‖μ_pred‖':>8s}  "
        f"{'var_R_med':>9s}  {'radius':>7s}  {'cent_cos':>8s}"
    )
    for r in rows:
        print(
            f"{r['step']:>7d}  "
            f"med={r['paired_err_med']:>5.2f}  "
            f"{r['nn_dist_med']:>5.2f}  "
            f"{r['mean_bias_z']:>8.3f}  "
            f"{r['var_ratio_med']:>9.3f}  "
            f"{r['radius_mean']:>7.3f}  "
            f"{r['centered_cos_med']:>8.4f}"
        )

    # Quick verdict.
    nn_first = rows[0]["nn_dist_med"]
    nn_last  = rows[-1]["nn_dist_med"]
    pe_first = rows[0]["paired_err_med"]
    pe_last  = rows[-1]["paired_err_med"]
    bias_first = rows[0]["mean_bias_z"]
    bias_last  = rows[-1]["mean_bias_z"]
    print()
    print(f"Δ(NN-exp)        : {nn_first:.3f} → {nn_last:.3f}  "
          f"({'+' if nn_last > nn_first else ''}{(nn_last-nn_first)/max(nn_first,1e-6)*100:+.1f}% over training)")
    print(f"Δ(‖pred−exp‖_z)  : {pe_first:.3f} → {pe_last:.3f}  "
          f"({(pe_last-pe_first)/max(pe_first,1e-6)*100:+.1f}%)")
    print(f"Δ(mean bias z)   : {bias_first:.3f} → {bias_last:.3f}  "
          f"({(bias_last-bias_first)/max(bias_first,1e-6)*100:+.1f}%)")

    if args.out_npz:
        out = {
            "steps": np.array([r["step"] for r in rows]),
            **{k: np.array([r[k] for r in rows]) for k in rows[0].keys() if k != "step"},
            "ref_expert_pair_d": expert_pair_d,
            "ref_nn_self": nn_self,
        }
        np.savez(args.out_npz, **out)
        print(f"\nSaved trajectory arrays to {args.out_npz}")


if __name__ == "__main__":
    main()
