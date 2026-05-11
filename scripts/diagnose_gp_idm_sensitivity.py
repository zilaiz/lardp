"""IDM-vs-GP bottleneck diagnostic on Franka coffee-pod held-out data.

Question: when goal-predictor + IDM produces ~32 mm eef-position error
on held-out data while expert-goal + IDM produces ~1.7 mm, which is the
bottleneck — IDM sensitivity to embedding error, or goal-prediction
accuracy?

This script disentangles the two by running three diagnostics on a single
batch from the held-out HDF5:

  (A) Goal-predictor accuracy in raw + z-scored embedding space
      - mean(pred_emb), mean(exp_emb)             : per-dim bias check
      - per-dim Var(pred)/Var(exp)                : mode-collapse check
      - ‖pred − exp‖_z (mean / median)            : actual error magnitude
      - cos_sim(pred − μ_exp, exp − μ_exp)        : direction match
                                                    after centering

  (B) IDM sensitivity curve (calibration)
      Take expert goal embedding, perturb in z-scored space:
          perturbed = expert + ε · n,    n ~ N(0, I_emb_dim)
      For ε ∈ {0, 0.5, 1, 2, 4, 8, 16}, denormalize, feed to IDM, sample
      action chunk, measure L2(action[-1, :3], expert_goal_eef). The
      ε=0 row reproduces the IDM ceiling; the rest map "noise σ" → "eef
      error mm". The mean per-dim ‖pred − exp‖_z corresponds to one
      specific ε on this curve — the *matched-noise point*.

  (C) Verdict
      Compare GP_actual_L2 against IDM_L2 at the matched-noise ε:
        - GP_L2 ≈ matched_noise_L2 → IDM-sensitivity-bound: the GP is no
          worse than isotropic noise at the same magnitude; reducing GP
          error directly reduces eef error.
        - GP_L2 ≫ matched_noise_L2 → GP-accuracy-bound *and* errors are
          systematic (e.g. mode collapse) — bigger GP improvements
          required.
        - GP_L2 ≪ matched_noise_L2 → GP errors lie along IDM's
          insensitive directions; rare; means the IDM has structured
          insensitivity that the GP exploits.

Usage
-----
    python scripts/diagnose_gp_idm_sensitivity.py \\
        --gp-ckpt logs/.../goal_predictor.../models/model_step_200000.pt \\
        --idm-ckpt logs/.../flow_lbmidm_v2.../models/model_step_130000.pt \\
        --goal-stats logs/.../goal_stats_130000.pt \\
        --heldout-path data/franka_coffee_pod_cog/image_heldout.hdf5
"""

import argparse
import os
import pickle
import sys

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def to_device_obs(d, slice_steps, device):
    out = {}
    for k, v in d.items():
        if slice_steps is not None:
            v = v[:, :slice_steps]
        out[k] = v.to(device, non_blocking=True)
    bs = next(iter(out.values())).shape[0]
    return TensorDict(out, batch_size=bs)


@torch.no_grad()
def goal_dit_ode(goal_flow_map, z_t, num_steps, generator):
    B, _, emb_dim = z_t.shape
    device = z_t.device
    g = torch.randn(
        (B, 1, emb_dim), device=device, dtype=z_t.dtype, generator=generator,
    )
    schedule = np.linspace(0, 1, num_steps + 1)
    for i in range(num_steps):
        s_val, t_val = schedule[i], schedule[i + 1]
        s = torch.full((B,), s_val, device=device)
        v = goal_flow_map.get_velocity(s, g, z_t)
        g = g + v * (t_val - s_val)
    return g


@torch.no_grad()
def idm_ode(flow_map, obs_emb, B, Ta, act_dim, num_steps, sample_mode):
    device = obs_emb.device
    if sample_mode == "stochastic":
        act_s = torch.randn((B, Ta, act_dim), device=device, dtype=obs_emb.dtype)
    else:
        act_s = torch.zeros((B, Ta, act_dim), device=device, dtype=obs_emb.dtype)
    schedule = np.linspace(0, 1, num_steps + 1)
    for i in range(num_steps):
        s_val, t_val = schedule[i], schedule[i + 1]
        s = torch.full((B,), s_val, device=device)
        b = flow_map.get_velocity(s, act_s, obs_emb)
        act_s = act_s + b * (t_val - s_val)
    return act_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gp-ckpt", required=True)
    p.add_argument("--idm-ckpt", required=True)
    p.add_argument("--goal-stats", required=True)
    p.add_argument("--heldout-path", required=True)
    p.add_argument("--normalizer-path", default=None)
    p.add_argument("--batch-size", type=int, default=512,
                   help="One batch is enough for the diagnostic; ~512 is plenty.")
    p.add_argument("--num-batches", type=int, default=4,
                   help="Aggregate over a few batches for stable means.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--nfe-idm", type=int, default=9)
    p.add_argument("--nfe-goal", type=int, default=5)
    p.add_argument("--sample-mode", default="zero", choices=["zero", "stochastic"])
    p.add_argument("--goal-dit-d-model", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilons", type=str,
                   default="0,0.5,1,2,4,8,16",
                   help="Comma-separated noise σ values to sweep, in z-scored "
                        "embedding space (per-dim). Includes 0 (the IDM ceiling).")
    p.add_argument("--out-npz", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    limit_threads(1)
    torch.set_float32_matmul_precision("high")

    norm_path = args.normalizer_path or os.path.join(
        os.path.dirname(args.idm_ckpt), "normalizer.pkl",
    )
    with open(norm_path, "rb") as f:
        normalizer = pickle.load(f)
    loguru.logger.info(f"Loaded normalizer from {norm_path}")

    cfg = build_cfg(
        args.idm_ckpt, args.goal_stats, args.batch_size,
        goal_dit_d_model=args.goal_dit_d_model,
    )
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

    loguru.logger.info("Instantiating GoalPredictorDiTAgent (loads frozen IDM)...")
    agent = GoalPredictorDiTAgent(cfg)
    agent.load(args.gp_ckpt)
    agent.eval()

    eef_norm = normalizer["obs"]["robot0_eef_pos"]
    act_norm = normalizer["action"]
    epsilons = [float(e) for e in args.epsilons.split(",")]

    pred_emb_all, exp_emb_all = [], []  # raw embeddings (B, emb_dim)
    pred_l2_all = []                    # GP-predicted-goal eef L2 (B,)
    exp_l2_all = []                     # IDM-with-expert-goal eef L2 (B,)
    eps_l2_acc = {e: [] for e in epsilons}  # eef L2 per ε (each (B,))
    eps_dircos_acc = {e: [] for e in epsilons}

    g_torch = torch.Generator(device=DEVICE).manual_seed(args.seed)
    n_batches_done = 0

    for bi, batch in enumerate(loader):
        if n_batches_done >= args.num_batches:
            break
        n_batches_done += 1
        obs = to_device_obs(
            {k: v for k, v in batch["obs"].items()}, slice_steps=obs_steps, device=DEVICE,
        )
        goal_obs = to_device_obs(
            {k: v for k, v in batch["goal_obs"].items()}, slice_steps=None, device=DEVICE,
        )
        B = next(iter(goal_obs.values())).shape[0]
        act_dim = batch["action"].shape[-1]

        # 1) Encode obs + expert goal with the frozen IDM encoder.
        z_t = agent._inner_encoder(obs, None)                 # (B, To, emb_dim)
        z_goal_raw = agent._inner_encoder(goal_obs, None)      # (B, 1, emb_dim)
        expert_goal = z_goal_raw[:, 0, :]                      # (B, emb_dim)

        # 2) GP-predicted goal (EMA) in raw space.
        pred_norm = goal_dit_ode(
            agent.goal_flow_map_ema, z_t, args.nfe_goal, generator=g_torch,
        )                                                      # (B, 1, emb_dim) z-scored
        pred_raw = agent._denormalize(pred_norm)[:, 0, :]      # (B, emb_dim)
        pred_emb_all.append(pred_raw.cpu().numpy())
        exp_emb_all.append(expert_goal.cpu().numpy())

        # 3) Goal eef in world frame.
        goal_eef_norm = batch["goal_obs"]["robot0_eef_pos"][:, 0, :].cpu().numpy()
        goal_eef = eef_norm.unnormalize(goal_eef_norm)         # (B, 3)
        cur_eef_norm = batch["obs"]["robot0_eef_pos"][:, obs_steps - 1, :].cpu().numpy()
        cur_eef = eef_norm.unnormalize(cur_eef_norm)            # (B, 3)
        delta_expert = goal_eef - cur_eef
        delta_expert_norm = np.linalg.norm(delta_expert, axis=-1)

        # 4) IDM with GP-predicted goal.
        cond_pred = torch.cat([z_t, pred_raw.unsqueeze(1)], dim=1)
        act_pred_norm = idm_ode(
            agent.flow_map, cond_pred, B, horizon, act_dim, args.nfe_idm, args.sample_mode,
        )
        act_pred = act_norm.unnormalize(act_pred_norm.cpu().numpy())
        pred_l2 = np.linalg.norm(act_pred[:, -1, :3] - goal_eef, axis=-1)
        pred_l2_all.append(pred_l2)

        # 5) IDM with expert goal (ceiling).
        cond_exp = torch.cat([z_t, expert_goal.unsqueeze(1)], dim=1)
        act_exp_norm = idm_ode(
            agent.flow_map, cond_exp, B, horizon, act_dim, args.nfe_idm, args.sample_mode,
        )
        act_exp = act_norm.unnormalize(act_exp_norm.cpu().numpy())
        exp_l2 = np.linalg.norm(act_exp[:, -1, :3] - goal_eef, axis=-1)
        exp_l2_all.append(exp_l2)

        # 6) Sensitivity sweep — perturb EXPERT goal in z-scored space, run IDM.
        # ε is the per-dim std of additive isotropic Gaussian noise in z-scored
        # space (so emb_dim ⋅ ε² is the expected squared norm of the perturbation).
        exp_norm_z = agent._normalize(z_goal_raw)[:, 0, :]    # (B, emb_dim)
        for eps in epsilons:
            n = torch.randn_like(exp_norm_z)
            perturbed_z = exp_norm_z + eps * n
            perturbed_raw = perturbed_z * torch.sqrt(
                agent._goal_var + agent._norm_eps,
            ) + agent._goal_mean
            cond_pert = torch.cat([z_t, perturbed_raw.unsqueeze(1)], dim=1)
            act_pert_norm = idm_ode(
                agent.flow_map, cond_pert, B, horizon, act_dim, args.nfe_idm, args.sample_mode,
            )
            act_pert = act_norm.unnormalize(act_pert_norm.cpu().numpy())
            l2 = np.linalg.norm(act_pert[:, -1, :3] - goal_eef, axis=-1)
            delta_p = act_pert[:, -1, :3] - cur_eef
            cos = np.einsum("bi,bi->b", delta_p, delta_expert) / (
                np.maximum(np.linalg.norm(delta_p, axis=-1), 1e-9)
                * np.maximum(delta_expert_norm, 1e-9)
            )
            eps_l2_acc[eps].append(l2)
            eps_dircos_acc[eps].append(cos)

        loguru.logger.info(
            f"batch {bi}: B={B}  pred_l2_mean={pred_l2.mean()*1000:.2f}mm  "
            f"exp_l2_mean={exp_l2.mean()*1000:.2f}mm"
        )

    pred_emb = np.concatenate(pred_emb_all, axis=0)
    exp_emb = np.concatenate(exp_emb_all, axis=0)
    pred_l2_all = np.concatenate(pred_l2_all, axis=0)
    exp_l2_all = np.concatenate(exp_l2_all, axis=0)

    # ===== (A) Embedding-space accuracy =====
    mu_exp = exp_emb.mean(axis=0)
    mu_pred = pred_emb.mean(axis=0)
    var_exp = exp_emb.var(axis=0)
    var_pred = pred_emb.var(axis=0)
    bias_l2 = float(np.linalg.norm(mu_pred - mu_exp))
    bias_z_per_dim = (mu_pred - mu_exp) / np.sqrt(var_exp + 1e-9)
    bias_z_l2 = float(np.linalg.norm(bias_z_per_dim))
    var_ratio = var_pred / np.maximum(var_exp, 1e-12)

    # ‖pred − exp‖_z per sample (z-scored using per-dim expert var).
    err_z_per_dim = (pred_emb - exp_emb) / np.sqrt(var_exp + 1e-9)
    err_z_l2 = np.linalg.norm(err_z_per_dim, axis=-1)
    mean_err_z = float(err_z_l2.mean())

    # Centered cosine similarity (direction match after removing per-dim bias).
    pred_c = pred_emb - mu_exp
    exp_c = exp_emb - mu_exp
    cos_centered = np.einsum("bi,bi->b", pred_c, exp_c) / (
        np.maximum(np.linalg.norm(pred_c, axis=-1), 1e-9)
        * np.maximum(np.linalg.norm(exp_c, axis=-1), 1e-9)
    )

    # ===== (B) Sensitivity curve =====
    eps_summary = []
    for eps in epsilons:
        l2 = np.concatenate(eps_l2_acc[eps], axis=0)
        cos = np.concatenate(eps_dircos_acc[eps], axis=0)
        eps_summary.append({
            "eps": eps,
            "exp_z_norm": float(eps * np.sqrt(exp_emb.shape[1])),  # E[‖ε⋅n‖₂] ≈ ε√D
            "eef_l2_mean_mm": float(l2.mean() * 1000),
            "eef_l2_med_mm":  float(np.median(l2) * 1000),
            "dir_cos_mean": float(cos.mean()),
        })

    # ===== (C) Match GP error magnitude to a sensitivity-curve ε ===
    gp_z_norm = mean_err_z
    # ε corresponding to gp_z_norm: gp_z_norm = ε √D  →  ε = gp_z_norm / √D
    D = exp_emb.shape[1]
    eps_match = gp_z_norm / np.sqrt(D)

    # Interpolate L2 at eps_match from the swept curve.
    eps_arr = np.array([e["eps"] for e in eps_summary])
    l2_arr = np.array([e["eef_l2_mean_mm"] for e in eps_summary])
    if eps_match <= eps_arr.min():
        l2_match = float(l2_arr.min())
    elif eps_match >= eps_arr.max():
        l2_match = float(l2_arr.max())
    else:
        l2_match = float(np.interp(eps_match, eps_arr, l2_arr))

    print()
    print("=" * 78)
    print("(A) GOAL-PREDICTOR EMBEDDING ACCURACY")
    print("=" * 78)
    print(f"  samples:                          {pred_emb.shape[0]}")
    print(f"  embedding dim:                    {D}")
    print(f"  mean(pred) - mean(exp) ‖·‖₂:      {bias_l2:.4f} (raw)")
    print(f"  mean(pred) - mean(exp) z-L2:      {bias_z_l2:.4f}  (in σ_per-dim units)")
    print(f"  per-dim Var(pred)/Var(exp):       "
          f"min={var_ratio.min():.4f}  med={np.median(var_ratio):.4f}  "
          f"mean={var_ratio.mean():.4f}  max={var_ratio.max():.4f}")
    print(f"      [if median ≪ 1 → GP partially collapses to mean of expert goals]")
    print(f"  per-sample err z-L2 (mean):       {mean_err_z:.4f}  "
          f"(≈ ε ⋅ √D with ε ≈ {eps_match:.4f})")
    print(f"  per-sample err z-L2 (median):     {float(np.median(err_z_l2)):.4f}")
    print(f"  centered cos_sim(pred-μ, exp-μ):  "
          f"mean={float(cos_centered.mean()):.4f}  "
          f"med={float(np.median(cos_centered)):.4f}  "
          f"frac>0={float((cos_centered > 0).mean()):.3f}")
    print(f"      [≈1 → GP captures per-sample structure; ≈0 → collapsed to mean]")

    print()
    print("=" * 78)
    print("(B) IDM SENSITIVITY CURVE (perturb expert goal in z-scored embedding space)")
    print("=" * 78)
    print(f"  {'ε (per-dim σ)':>13s}  {'E[‖ε·n‖_z]':>11s}  {'eef L2 mean':>13s}  {'med':>9s}  {'dir_cos':>8s}")
    for e in eps_summary:
        print(
            f"  {e['eps']:>13.3f}  {e['exp_z_norm']:>11.3f}  "
            f"{e['eef_l2_mean_mm']:>10.2f} mm  {e['eef_l2_med_mm']:>6.2f} mm  "
            f"{e['dir_cos_mean']:>8.4f}"
        )

    print()
    print("=" * 78)
    print("(C) VERDICT — sensitivity-bound vs accuracy-bound")
    print("=" * 78)
    gp_l2_mm = float(pred_l2_all.mean() * 1000)
    exp_l2_mm = float(exp_l2_all.mean() * 1000)
    print(f"  GP-actual eef L2 (mean):          {gp_l2_mm:.2f} mm")
    print(f"  IDM-with-expert-goal eef L2:      {exp_l2_mm:.2f} mm  (the ceiling)")
    print(f"  GP error magnitude ‖pred−exp‖_z:  {gp_z_norm:.3f}  "
          f"(matches ε ≈ {eps_match:.3f} on the sweep)")
    print(f"  IDM eef L2 at ε={eps_match:.3f}:     ~{l2_match:.2f} mm  (matched-noise)")
    ratio = gp_l2_mm / max(l2_match, 1e-6)
    print(f"  GP_actual / matched_noise:        {ratio:.2f}")
    if ratio < 0.7:
        verdict = (
            "GP errors are aligned with IDM-INSENSITIVE directions "
            "(better than random noise of equal magnitude). Sensitivity is "
            "not the bottleneck — accuracy could improve, but the IDM is "
            "already absorbing GP errors better than a noise baseline would."
        )
    elif ratio < 1.3:
        verdict = (
            "SENSITIVITY-BOUND: GP error behaves like random isotropic noise "
            "in z-scored space at this magnitude. The IDM is ~equally "
            "sensitive in all embedding directions; reducing GP error in "
            "z-scored magnitude should reduce eef error proportionally."
        )
    else:
        verdict = (
            "GP-ACCURACY-BOUND with SYSTEMATIC error: GP_actual ≫ matched-"
            "noise L2, meaning GP errors are biased / mode-collapsed and "
            "drive the IDM further off than random noise of equal magnitude. "
            "The GP needs structural fixes (training, conditioning, capacity) "
            "rather than just smaller error."
        )
    print(f"  verdict: {verdict}")

    if args.out_npz:
        np.savez(
            args.out_npz,
            pred_emb=pred_emb, exp_emb=exp_emb,
            pred_l2=pred_l2_all, exp_l2=exp_l2_all,
            err_z_l2=err_z_l2,
            var_ratio=var_ratio,
            eps_arr=eps_arr, eps_l2_mm=l2_arr,
        )
        print(f"\n  raw arrays saved to {args.out_npz}")


if __name__ == "__main__":
    main()
