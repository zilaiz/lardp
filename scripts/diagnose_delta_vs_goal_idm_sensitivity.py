"""Disentangle two failure modes when a delta-predictor pipeline lags goal:

  Hypothesis A — "Delta IDM is too sensitive": the IDM's action trunk is
    fragile to errors in its third AdaLN slot, so even modest predictor
    errors blow up action quality.
  Hypothesis B — "Delta predictor is the bottleneck": the IDM is fine,
    but learning the delta target is harder so the predictor produces
    worse `delta_hat` than the goal predictor produces `g_hat`.

To separate the two, we run a calibrated noise-injection sweep on the
*frozen* IDMs:

  goal-cond:    z_goal'  = z_goal + ε · σ_goal  · N(0, I)
  delta-cond:   delta'   = delta  + ε · σ_delta · N(0, I)

where σ_goal / σ_delta are the per-dim std of z_goal / delta on the same
expert windows. For each ε we run the IDM's action ODE and report L2 vs
ground-truth actions.

  * If ε=0 produces the same low L2 for both → both IDMs fit expert data.
  * If goal IDM is much more tolerant at moderate ε → IDM is the bottleneck
    in the delta pipeline (Hypothesis A).
  * If both IDMs degrade similarly at matched ε → IDM is fine; the gap is
    in the predictor (Hypothesis B). Then run a trained predictor against
    each (--predict mode) to confirm.

Two modes:
  --mode noise   (default) — pure noise injection, no trained predictor
  --mode predict — if you have trained predictor ckpts, plug them in
                   directly (NOT YET WIRED — falls back to noise).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import h5py
import loguru
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_idm(ckpt_path, network_name, task_config, config_dir, device):
    """Load a frozen IDM (encoder + flow_map) into a TrainingAgent-like setup."""
    import hydra
    from hydra import initialize_config_dir
    from mip.encoders import GoalDropoutEncoder
    from mip.flow_map import FlowMap
    from mip.network_utils import get_encoder, get_network

    config_abs = str((Path(__file__).resolve().parents[1] / config_dir).resolve())
    with initialize_config_dir(config_dir=config_abs, version_base=None):
        cfg = hydra.compose(config_name="main",
                            overrides=[f"task={task_config}",
                                       f"network={network_name}"])
    task_cfg, network_cfg = cfg.task, cfg.network
    cfg.task.obs_dim = network_cfg.emb_dim

    encoder = get_encoder(network_cfg, task_cfg).to(device)
    net = get_network(network_cfg, task_cfg).to(device)
    flow_map = FlowMap(net).to(device)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder_sd = ck["encoder_ema"] if "encoder_ema" in ck else ck["encoder"]
    flow_map_sd = ck["flow_map_ema"] if "flow_map_ema" in ck else ck["flow_map"]

    has_wrapper = any(k.startswith("encoder.") for k in encoder_sd)
    if has_wrapper:
        enc_out_dim = network_cfg.encoder_out_dim or network_cfg.emb_dim
        encoder = GoalDropoutEncoder(
            encoder, enc_out_dim, task_cfg.obs_steps,
        ).to(device)
    encoder.load_state_dict(encoder_sd)
    encoder.eval().requires_grad_(False)
    inner_enc = encoder.encoder if isinstance(encoder, GoalDropoutEncoder) else encoder

    flow_map.load_state_dict(flow_map_sd)
    flow_map.eval().requires_grad_(False)
    return inner_enc, flow_map, cfg


def _load_normalizer(ckpt_path):
    with open(Path(ckpt_path).parent / "normalizer.pkl", "rb") as f:
        return pickle.load(f)


def _build_windows(dataset_path, task_cfg, normalizer, n_demos, n_windows, seed):
    rng = np.random.default_rng(seed)
    img_keys = sorted(k for k, v in task_cfg.shape_meta.obs.items() if v.type == "rgb")
    low_keys = sorted(k for k, v in task_cfg.shape_meta.obs.items() if v.type == "low_dim")
    To, H = task_cfg.obs_steps, task_cfg.horizon

    obs_list = {k: [] for k in img_keys + low_keys}
    goal_list = {k: [] for k in img_keys + low_keys}
    act_list = []

    with h5py.File(dataset_path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        chosen = rng.choice(len(names), size=min(n_demos, len(names)), replace=False)
        for di in chosen:
            d = f["data"][names[int(di)]]
            T = d["obs"][img_keys[0]].shape[0]
            valid = list(range(To - 1, T - H - 1))
            if not valid:
                continue
            ts = rng.choice(valid, size=min(n_windows, len(valid)), replace=False)
            actions = np.asarray(d["actions"]).astype(np.float32)
            for t in ts:
                for k in img_keys:
                    x = np.asarray(d["obs"][k])[t - To + 1:t + 1].astype(np.float32) / 255.
                    x = np.moveaxis(x, -1, 1)
                    obs_list[k].append(normalizer["obs"][k].normalize(x))
                    x = np.asarray(d["obs"][k])[t + H:t + H + 1].astype(np.float32) / 255.
                    x = np.moveaxis(x, -1, 1)
                    goal_list[k].append(normalizer["obs"][k].normalize(x))
                for k in low_keys:
                    x = np.asarray(d["obs"][k])[t - To + 1:t + 1].astype(np.float32)
                    obs_list[k].append(normalizer["obs"][k].normalize(x))
                    x = np.asarray(d["obs"][k])[t + H:t + H + 1].astype(np.float32)
                    goal_list[k].append(normalizer["obs"][k].normalize(x))
                a = actions[t:t + H]
                act_list.append(normalizer["action"].normalize(a))

    obs_arr  = {k: np.stack(v) for k, v in obs_list.items()}
    goal_arr = {k: np.stack(v) for k, v in goal_list.items()}
    act_arr  = np.stack(act_list)
    return obs_arr, goal_arr, act_arr


@torch.no_grad()
def _encode(encoder, arr_dict, device, chunk=64):
    """Stack-arrays (N, T, ...) per key -> (N, T, emb_dim)."""
    from tensordict import TensorDict
    N = next(iter(arr_dict.values())).shape[0]
    zs = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        td = TensorDict(
            {k: torch.from_numpy(v[s:e]).to(device) for k, v in arr_dict.items()},
            batch_size=e - s,
        )
        z = encoder(td, None)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


@torch.no_grad()
def _action_l2_at_noise(
    flow_map, z_t, slot_signal, slot_perturb, act_gt,
    num_steps, batch, device,
):
    """Run the action ODE with the third slot = slot_signal + slot_perturb.

    Args:
        flow_map: frozen IDM flow_map (FlowMap wrapping the action trunk).
        z_t:      (N, To, emb_dim) encoded obs.
        slot_signal: (N, 1, emb_dim) — what to put in the goal slot
            (true z_goal for goal-cond; or z_last_obs + delta for delta-cond
            which the IDM internally subtracts to get the delta back).
        slot_perturb: (N, 1, emb_dim) — additive noise.
        act_gt:  (N, H, A) ground-truth normalized actions.
    Returns: per-sample L2 array (N,).
    """
    N, H, A = act_gt.shape
    per_l2 = np.zeros(N)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        slot = torch.from_numpy(slot_signal[s:e] + slot_perturb[s:e]).to(device)
        obs_emb = torch.cat([torch.from_numpy(z_t[s:e]).to(device), slot], dim=1)
        act_s = torch.randn(e - s, H, A, device=device)
        t_grid = np.linspace(0, 1, num_steps + 1)
        for i in range(num_steps):
            s_val = float(t_grid[i]); t_val = float(t_grid[i + 1])
            s_t = torch.full((e - s,), s_val, device=device)
            v = flow_map.get_velocity(s_t, act_s, obs_emb)
            act_s = act_s + v * (t_val - s_val)
        gt = torch.from_numpy(act_gt[s:e]).to(device)
        per_l2[s:e] = (act_s - gt).reshape(e - s, -1).pow(2).mean(-1).sqrt().cpu().numpy()
    return per_l2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta_ckpt", default="logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_delta_cond_fdm1.0_gdp0.0_ac8/2026_05_03_22_35_04/models/model_step_300000.pt")
    p.add_argument("--goal_ckpt",  default="logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_fdm_aux/2026_04_27_01_03_55/models/model_step_300000.pt")
    p.add_argument("--task_config", default="tool_hang_ph_image_gp")
    p.add_argument("--config_dir",  default="examples/configs")
    p.add_argument("--dataset_path",default="data/robomimic/tool_hang/ph/image_v15.hdf5")
    p.add_argument("--n_demos",     type=int, default=30)
    p.add_argument("--n_windows_per_demo", type=int, default=20)
    p.add_argument("--num_steps",   type=int, default=20)
    p.add_argument("--batch",       type=int, default=64)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--out_dir",     default="viz/delta_vs_goal_idm_sensitivity")
    p.add_argument("--noise_levels", default="0,0.1,0.25,0.5,1.0,2.0",
                   help="comma list of ε values (noise std multipliers)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    eps_list = [float(x) for x in args.noise_levels.split(",")]
    out_dir = (Path(__file__).resolve().parents[1] / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- Load both IDMs (encoder + flow_map) and their normalizers ---
    enc_d, fm_d, cfg = _load_idm(args.delta_ckpt, "lbmidm_v2_delta",
                                  args.task_config, args.config_dir, args.device)
    nrm_d = _load_normalizer(args.delta_ckpt)
    enc_g, fm_g, _   = _load_idm(args.goal_ckpt,  "lbmidm_v2",
                                  args.task_config, args.config_dir, args.device)
    nrm_g = _load_normalizer(args.goal_ckpt)

    # --- Sample expert windows ---
    obs_d, goal_d, act_d = _build_windows(
        args.dataset_path, cfg.task, nrm_d, args.n_demos, args.n_windows_per_demo,
        args.seed,
    )
    obs_g, goal_g, act_g = _build_windows(
        args.dataset_path, cfg.task, nrm_g, args.n_demos, args.n_windows_per_demo,
        args.seed,
    )
    N = act_d.shape[0]
    loguru.logger.info(f"{N} windows, per-IDM (its own normalizer)")

    # --- Encode obs + goal under each IDM ---
    z_obs_d  = _encode(enc_d, obs_d,  args.device)        # (N, To, D)
    z_goal_d = _encode(enc_d, goal_d, args.device)        # (N, 1,  D)
    z_obs_g  = _encode(enc_g, obs_g,  args.device)
    z_goal_g = _encode(enc_g, goal_g, args.device)

    z_last_d = z_obs_d[:, -1:]
    delta_d  = z_goal_d - z_last_d                          # delta target

    # --- Per-dim std of the signals (computed across the sample) ---
    std_goal  = z_goal_g.reshape(-1, z_goal_g.shape[-1]).std(axis=0)  # (D,)
    std_delta = delta_d.reshape(-1, delta_d.shape[-1]).std(axis=0)    # (D,)
    print(f"\nmean per-dim std:  z_goal={std_goal.mean():.4f}   delta={std_delta.mean():.4f}")
    print(f"||z_goal|| mean:   {np.linalg.norm(z_goal_g.reshape(N,-1),axis=-1).mean():.3f}")
    print(f"||delta||  mean:   {np.linalg.norm(delta_d.reshape(N,-1),axis=-1).mean():.3f}")
    print(f"||noise floor|| (random act vs gt) mean L2: "
          f"{np.sqrt(((rng.standard_normal(act_d.shape).astype(np.float32) - act_d)**2).mean(axis=(1,2))).mean():.3f}")

    # --- Noise sweep ---
    results = {"delta": {}, "goal": {}}
    for eps in eps_list:
        # Match noise to slot scale per-dim. Same noise tensor for both
        # branches to control for random variation.
        n = rng.standard_normal(size=(N, 1, z_goal_g.shape[-1])).astype(np.float32)
        pert_goal  = n * eps * std_goal[None, None, :]
        pert_delta = n * eps * std_delta[None, None, :]

        # Goal-cond IDM: feed z_goal + pert_goal directly as the goal slot.
        l2_g = _action_l2_at_noise(
            fm_g, z_obs_g, slot_signal=z_goal_g, slot_perturb=pert_goal,
            act_gt=act_g, num_steps=args.num_steps, batch=args.batch,
            device=args.device,
        )
        # Delta-cond IDM: the slot we feed is z_last_obs + (delta + pert_delta);
        # the IDM internally subtracts to get (delta + pert_delta) → the third
        # AdaLN slot. So pert_delta carries directly through.
        l2_d = _action_l2_at_noise(
            fm_d, z_obs_d, slot_signal=z_last_d + delta_d, slot_perturb=pert_delta,
            act_gt=act_d, num_steps=args.num_steps, batch=args.batch,
            device=args.device,
        )
        results["goal"][eps]  = l2_g
        results["delta"][eps] = l2_d
        print(f"eps={eps:>5.2f}  goal L2 mean={l2_g.mean():.4f}   "
              f"delta L2 mean={l2_d.mean():.4f}   "
              f"ratio delta/goal={l2_d.mean()/max(l2_g.mean(),1e-6):.2f}")

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    xs = sorted(eps_list)
    mu_g  = [results["goal"][e].mean()  for e in xs]
    mu_d  = [results["delta"][e].mean() for e in xs]
    sd_g  = [results["goal"][e].std()   for e in xs]
    sd_d  = [results["delta"][e].std()  for e in xs]
    plt.errorbar(xs, mu_g, yerr=sd_g, marker="o", capsize=3,
                 label="goal-cond IDM (perturb z_goal)", color="C0")
    plt.errorbar(xs, mu_d, yerr=sd_d, marker="s", capsize=3,
                 label="delta-cond IDM (perturb delta)", color="C3")
    plt.xlabel("noise ε  (multiplier of per-dim std of the slot)")
    plt.ylabel("action L2 (normalized space)")
    plt.title("Calibrated sensitivity: action L2 vs noise injected into the AdaLN goal/delta slot")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_curve.png", dpi=140)
    plt.close()

    # Save raw numbers
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"per-dim std: z_goal={std_goal.mean():.4f}  delta={std_delta.mean():.4f}\n\n")
        f.write(f"{'eps':>6} {'goal L2':>10} {'delta L2':>10} {'ratio (delta/goal)':>18}\n")
        for e in xs:
            f.write(f"{e:>6.2f} {mu_g[xs.index(e)]:>10.4f} "
                    f"{mu_d[xs.index(e)]:>10.4f} {mu_d[xs.index(e)]/max(mu_g[xs.index(e)],1e-6):>18.2f}\n")
    loguru.logger.info(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
