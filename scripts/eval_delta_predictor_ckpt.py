"""Evaluate a trained DeltaPredictorDiTAgent checkpoint to diagnose whether
the predictor is the bottleneck behind the worse downstream action quality.

Two metric families:

  1) Predictor reconstruction (in encoder feature space)
       err_l2_norm  = mean ||delta_hat - delta_true|| / sigma_delta
       cos(delta_hat, delta_true)
       per-dim regression R^2 (linear-probe-style sanity)

  2) Downstream action quality (in normalized action space)
       L2 with predictor's delta_hat  (real pipeline)
       L2 with delta_true (oracle floor — already measured ~0.09)
       L2 with random delta (sanity ceiling)
       L2 with zero delta (no-info floor)

Splits demos into the same train/val partition the predictor saw via
``task.val_dataset_percentage`` so the reported numbers are on the actual
held-out set.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import h5py
import loguru
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _resolve_paths_from_predictor_ckpt(predictor_ckpt: str):
    ck = torch.load(predictor_ckpt, map_location="cpu", weights_only=False)
    idm_path = ck.get("idm_checkpoint_path")
    if idm_path is None:
        raise RuntimeError(
            f"{predictor_ckpt} has no 'idm_checkpoint_path' field — can't "
            f"resolve the IDM it was trained against."
        )
    return idm_path


def _detect_predictor_variant(predictor_ckpt):
    """Return one of {"dit", "ddt_ns"} based on saved state-dict keys."""
    ck = torch.load(predictor_ckpt, map_location="cpu", weights_only=False)
    sd = ck["goal_dit"]
    has_enc = any(k.startswith("enc_blocks.") for k in sd)
    return "ddt_ns" if has_enc else "dit"


def _infer_dit_hparams(predictor_ckpt):
    """Detect d_model + depth for plain-DiT predictor."""
    sd = torch.load(predictor_ckpt, map_location="cpu", weights_only=False)["goal_dit"]
    d_model = sd["blocks.0.attn.out_proj.weight"].shape[0]
    depth = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    return d_model, depth


def _infer_ddt_ns_hparams(predictor_ckpt):
    """Detect enc/dec d_model + depth for DDT-NS predictor."""
    sd = torch.load(predictor_ckpt, map_location="cpu", weights_only=False)["goal_dit"]
    d_model_enc = sd["enc_blocks.0.attn.out_proj.weight"].shape[0]
    d_model_dec = sd["dec_blocks.0.attn.out_proj.weight"].shape[0]
    enc_depth = max(int(k.split(".")[1]) for k in sd
                    if k.startswith("enc_blocks.")) + 1
    dec_depth = max(int(k.split(".")[1]) for k in sd
                    if k.startswith("dec_blocks.")) + 1
    return d_model_enc, d_model_dec, enc_depth, dec_depth


def _build_agent(predictor_ckpt, task_config, config_dir, device):
    import hydra
    from hydra import initialize_config_dir

    variant = _detect_predictor_variant(predictor_ckpt)
    loguru.logger.info(f"Detected predictor variant: {variant}")

    idm_path = _resolve_paths_from_predictor_ckpt(predictor_ckpt)
    idm_dir = os.path.dirname(idm_path)
    candidates = sorted(p for p in os.listdir(idm_dir) if p.startswith("delta_stats"))
    if not candidates:
        raise FileNotFoundError(
            f"No delta_stats_*.pt found next to IDM ckpt at {idm_dir}. "
            f"Run scripts/compute_delta_stats.py first."
        )
    delta_stats_path = os.path.join(idm_dir, candidates[-1])
    loguru.logger.info(f"Using delta stats: {delta_stats_path}")

    config_abs = str((Path(__file__).resolve().parents[1] / config_dir).resolve())

    if variant == "dit":
        d_model, depth = _infer_dit_hparams(predictor_ckpt)
        loguru.logger.info(f"Inferred goal_dit_d_model={d_model} depth={depth}")
        with initialize_config_dir(config_dir=config_abs, version_base=None):
            cfg = hydra.compose(config_name="main", overrides=[
                f"task={task_config}",
                "network=delta_predictor_dit",
                f"network.goal_dit_d_model={d_model}",
                f"network.goal_dit_depth={depth}",
                f"optimization.idm_checkpoint_path={idm_path}",
                f"optimization.delta_stats_path={delta_stats_path}",
                "optimization.use_compile=false",
                "optimization.use_cudagraphs=false",
            ])
        cfg.task.obs_dim = cfg.network.emb_dim
        from mip.agent_delta_predictor_dit import DeltaPredictorDiTAgent
        agent = DeltaPredictorDiTAgent(cfg)
    else:  # ddt_ns
        denc, ddec, edep, ddep = _infer_ddt_ns_hparams(predictor_ckpt)
        loguru.logger.info(
            f"Inferred d_model_enc={denc} d_model_dec={ddec} "
            f"enc_depth={edep} dec_depth={ddep}"
        )
        with initialize_config_dir(config_dir=config_abs, version_base=None):
            cfg = hydra.compose(config_name="main", overrides=[
                f"task={task_config}",
                "network=delta_predictor_ddt_ns",
                f"network.goal_ddt_d_model_enc={denc}",
                f"network.goal_ddt_d_model_dec={ddec}",
                f"network.goal_ddt_enc_depth={edep}",
                f"network.goal_ddt_dec_depth={ddep}",
                f"optimization.idm_checkpoint_path={idm_path}",
                f"optimization.delta_stats_path={delta_stats_path}",
                "optimization.use_compile=false",
                "optimization.use_cudagraphs=false",
            ])
        cfg.task.obs_dim = cfg.network.emb_dim
        from mip.agent_delta_predictor_ddt_ns import DeltaPredictorDDTNSAgent
        agent = DeltaPredictorDDTNSAgent(cfg)

    agent.load(predictor_ckpt)
    agent.eval()
    return agent, cfg


def _read_window(h5_demo, t, To, H, normalizer, img_keys, low_keys, device):
    obs_dict, goal_dict = {}, {}
    for k in img_keys:
        x = np.asarray(h5_demo["obs"][k])[t - To + 1:t + 1].astype(np.float32) / 255.
        x = np.moveaxis(x, -1, 1)
        obs_dict[k] = torch.from_numpy(normalizer["obs"][k].normalize(x))[None].to(device)
        x = np.asarray(h5_demo["obs"][k])[t + H:t + H + 1].astype(np.float32) / 255.
        x = np.moveaxis(x, -1, 1)
        goal_dict[k] = torch.from_numpy(normalizer["obs"][k].normalize(x))[None].to(device)
    for k in low_keys:
        x = np.asarray(h5_demo["obs"][k])[t - To + 1:t + 1].astype(np.float32)
        obs_dict[k] = torch.from_numpy(normalizer["obs"][k].normalize(x))[None].to(device)
        x = np.asarray(h5_demo["obs"][k])[t + H:t + H + 1].astype(np.float32)
        goal_dict[k] = torch.from_numpy(normalizer["obs"][k].normalize(x))[None].to(device)
    a = np.asarray(h5_demo["actions"])[t:t + H].astype(np.float32)
    a_norm = torch.from_numpy(normalizer["action"].normalize(a))[None].to(device)
    return obs_dict, goal_dict, a_norm


def _ode_action(flow_map, z_t, slot, act_dim, H, num_steps, device, sample_mode="stochastic"):
    """Run action ODE given a pre-built third-slot tensor (B, 1, D)."""
    obs_emb = torch.cat([z_t, slot], dim=1)
    B = z_t.shape[0]
    if sample_mode == "stochastic":
        act_s = torch.randn(B, H, act_dim, device=device)
    else:
        act_s = torch.zeros(B, H, act_dim, device=device)
    t_grid = np.linspace(0, 1, num_steps + 1)
    with torch.no_grad():
        for i in range(num_steps):
            s_v = float(t_grid[i]); t_v = float(t_grid[i + 1])
            s = torch.full((B,), s_v, device=device)
            v = flow_map.get_velocity(s, act_s, obs_emb)
            act_s = act_s + v * (t_v - s_v)
    return act_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictor_ckpt", required=True)
    p.add_argument("--task_config", default="tool_hang_ph_image_gp")
    p.add_argument("--network", default="delta_predictor_dit")
    p.add_argument("--config_dir", default="examples/configs")
    p.add_argument("--dataset_path", default="data/robomimic/tool_hang/ph/image_v15.hdf5")
    p.add_argument("--val_pct", type=float, default=0.6,
                   help="Match the predictor's training split. Train demos = first (1-val_pct), val demos = rest.")
    p.add_argument("--n_demos_each", type=int, default=30,
                   help="N demos to sample from each split")
    p.add_argument("--n_windows_per_demo", type=int, default=20)
    p.add_argument("--num_steps_idm", type=int, default=20)
    p.add_argument("--num_steps_goal", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    agent, cfg = _build_agent(args.predictor_ckpt, args.task_config,
                               args.config_dir, args.device)
    # Override goal_flow_num_steps from CLI (config default may differ).
    agent.config.optimization.goal_flow_num_steps = args.num_steps_goal

    # Normalizer sits next to the IDM ckpt.
    idm_dir = os.path.dirname(_resolve_paths_from_predictor_ckpt(args.predictor_ckpt))
    with open(os.path.join(idm_dir, "normalizer.pkl"), "rb") as f:
        normalizer = pickle.load(f)

    img_keys = sorted(k for k, v in cfg.task.shape_meta.obs.items() if v.type == "rgb")
    low_keys = sorted(k for k, v in cfg.task.shape_meta.obs.items() if v.type == "low_dim")
    To, H = cfg.task.obs_steps, cfg.task.horizon
    act_dim = cfg.task.act_dim
    device = args.device

    # --- Train/val split mirrors make_idm_dataset's slicing ---
    with h5py.File(args.dataset_path, "r") as f:
        all_names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        N_total = len(all_names)
        N_train = int(round(N_total * (1.0 - args.val_pct)))
        train_names = all_names[:N_train]
        val_names = all_names[N_train:]
        rng = np.random.default_rng(args.seed)

        splits = {
            "train": train_names[:args.n_demos_each],
            "val":   val_names[:args.n_demos_each],
        }

        results = {}
        for split_name, names in splits.items():
            loguru.logger.info(f"=== {split_name} split: {len(names)} demos "
                               f"(of {len(train_names if split_name=='train' else val_names)} total) ===")
            delta_true_list, delta_hat_list, z_last_list, z_goal_list = [], [], [], []
            obs_z_list, act_gt_list = [], []
            for name in tqdm(names, desc=f"encoding/predict {split_name}"):
                d = f["data"][name]
                T_d = d["obs"][img_keys[0]].shape[0]
                valid = list(range(To - 1, T_d - H - 1))
                if not valid:
                    continue
                ts = rng.choice(valid, size=min(args.n_windows_per_demo, len(valid)),
                                 replace=False)
                for t in ts:
                    obs_dict, goal_dict, act_gt = _read_window(
                        d, int(t), To, H, normalizer, img_keys, low_keys, device,
                    )
                    from tensordict import TensorDict
                    obs_td = TensorDict(obs_dict, batch_size=1)
                    goal_td = TensorDict(goal_dict, batch_size=1)

                    # Encode
                    with torch.no_grad():
                        z_t = agent._inner_encoder(obs_td, None)        # (1, To, D)
                        z_goal = agent._inner_encoder(goal_td, None)    # (1, 1, D)
                    z_last = z_t[:, -1:]
                    delta_true = (z_goal - z_last)                       # (1, 1, D)

                    # Predictor ODE → delta_hat (use wrapper, which returns
                    # cat([z_t, g_fake=z_last+delta_hat]) — so delta_hat = last - z_last).
                    with torch.no_grad():
                        obs_emb_pred = agent.wrapper_encoder_ema(obs_td, None)
                    g_fake_pred = obs_emb_pred[:, -1:]
                    delta_hat = g_fake_pred - z_last                     # (1, 1, D)

                    delta_true_list.append(delta_true.cpu())
                    delta_hat_list.append(delta_hat.cpu())
                    z_last_list.append(z_last.cpu())
                    z_goal_list.append(z_goal.cpu())
                    obs_z_list.append(z_t.cpu())
                    act_gt_list.append(act_gt.cpu())

            delta_true_t = torch.cat(delta_true_list, 0)   # (N,1,D)
            delta_hat_t  = torch.cat(delta_hat_list, 0)
            z_last_t     = torch.cat(z_last_list, 0)
            z_goal_t     = torch.cat(z_goal_list, 0)
            obs_z_t      = torch.cat(obs_z_list, 0)
            act_gt_t     = torch.cat(act_gt_list, 0)
            N = delta_true_t.shape[0]

            # --- Predictor reconstruction ---
            delta_true_v = delta_true_t.squeeze(1)
            delta_hat_v  = delta_hat_t.squeeze(1)
            err_l2 = (delta_hat_v - delta_true_v).norm(dim=-1)
            norm_target = delta_true_v.norm(dim=-1)
            sigma_delta = delta_true_v.std(dim=0)
            norm_err_per_dim = ((delta_hat_v - delta_true_v) / (sigma_delta + 1e-6)).norm(dim=-1) / np.sqrt(delta_true_v.shape[-1])
            cos = torch.nn.functional.cosine_similarity(delta_hat_v, delta_true_v, dim=-1)
            ss_res = ((delta_hat_v - delta_true_v) ** 2).sum()
            ss_tot = ((delta_true_v - delta_true_v.mean(dim=0, keepdim=True)) ** 2).sum()
            r2 = 1.0 - float(ss_res / max(float(ss_tot), 1e-12))

            # --- Action ODE under four slot constructions ---
            def batch_action_l2(slot_t, label):
                per_l2 = np.zeros(N)
                B = 64
                for s in range(0, N, B):
                    e = min(s + B, N)
                    slot = slot_t[s:e].to(device)
                    z = obs_z_t[s:e].to(device)
                    pred = _ode_action(
                        agent.flow_map, z, slot, act_dim, H,
                        args.num_steps_idm, device,
                        sample_mode="stochastic",
                    )
                    gt = act_gt_t[s:e].to(device)
                    per_l2[s:e] = (pred - gt).reshape(e - s, -1).pow(2).mean(-1).sqrt().cpu().numpy()
                return per_l2

            slot_true   = z_last_t + delta_true_t           # = z_goal (oracle)
            slot_hat    = z_last_t + delta_hat_t            # predictor's actual feed
            slot_zero   = z_last_t.clone()                   # "stay put"
            slot_random = z_last_t + torch.randn_like(delta_true_t) * sigma_delta

            l2_true   = batch_action_l2(slot_true,   "true delta")
            l2_hat    = batch_action_l2(slot_hat,    "predicted delta")
            l2_zero   = batch_action_l2(slot_zero,   "zero delta")
            l2_random = batch_action_l2(slot_random, "random delta")

            results[split_name] = {
                "N": N,
                "err_l2":          float(err_l2.mean()),
                "err_l2_norm":     float(err_l2.mean() / norm_target.mean()),
                "err_per_dim_std": float(norm_err_per_dim.mean()),
                "cos":             float(cos.mean()),
                "r2_predictor":    r2,
                "|delta_true|":    float(norm_target.mean()),
                "|delta_hat|":     float(delta_hat_v.norm(dim=-1).mean()),
                "l2_true":   l2_true.mean(),
                "l2_hat":    l2_hat.mean(),
                "l2_zero":   l2_zero.mean(),
                "l2_random": l2_random.mean(),
            }

    print("\n" + "=" * 80)
    print(f"{'split':>8} {'N':>5}  {'err_l2':>8} {'/||δ||':>8} {'/σ':>8} {'cos(δ̂,δ)':>10} {'R²':>7}")
    print("-" * 80)
    for split_name, r in results.items():
        print(f"{split_name:>8} {r['N']:>5}  {r['err_l2']:>8.4f} "
              f"{r['err_l2_norm']:>8.3f} {r['err_per_dim_std']:>8.3f} "
              f"{r['cos']:>10.3f} {r['r2_predictor']:>7.3f}")

    print(f"\n{'split':>8} {'||δ||':>8} {'||δ̂||':>8}  "
          f"{'L2_oracle':>10} {'L2_pred':>9} {'L2_zero':>9} {'L2_rand':>9}")
    print("-" * 80)
    for split_name, r in results.items():
        print(f"{split_name:>8} {r['|delta_true|']:>8.3f} {r['|delta_hat|']:>8.3f}  "
              f"{r['l2_true']:>10.4f} {r['l2_hat']:>9.4f} "
              f"{r['l2_zero']:>9.4f} {r['l2_random']:>9.4f}")
    print("=" * 80)
    print("Interpretation:")
    print("  - 'err_l2 / σ' near 1.0 = predictor at noise floor (no signal)")
    print("  -  R² < 0          = predictor worse than predicting the mean delta")
    print("  - L2_pred ≈ L2_oracle = predictor good enough, IDM unaffected")
    print("  - L2_pred ≫ L2_oracle but ≪ L2_zero = predictor partially helping")


if __name__ == "__main__":
    main()
