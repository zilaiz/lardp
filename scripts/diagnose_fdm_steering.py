"""Diagnostic for the FDM-steering idea.

Reports four residuals on a held-out batch from the IDM dataset, in the raw
encoder space the goal predictor + FDM both live in:

    (a) FDM(z_t, a_expert)            vs  encode(true_goal)
        -> Does FDM model the clean (s, a) -> s' map at all?
    (b) FDM(z_t, a_idm[true_goal])    vs  encode(true_goal)
        -> Does the IDM-sampled action satisfy FDM, decoupled from any
           goal-predictor error?
    (c) FDM(z_t, a_idm[pred_goal])    vs  s'_gp
        -> The actual gap that test-time steering toward s'_gp would close.
    (d) encode(true_goal)             vs  s'_gp
        -> Goal-predictor quality, for context.

Plus the per-element variance of the goal embedding so the residuals are
interpretable. Output also includes a rough R^2 = 1 - MSE / Var.

Usage:
    python scripts/diagnose_fdm_steering.py \
        --config_path outputs/2026-04-28/03-27-15/.hydra/config.yaml \
        --ckpt_path   logs/can_ph_image_goal_predictor_dit_None_lbmidm_v2_256_seed0_fdm_aux_200000/2026_04_28_03_27_15/models/model_best.pt \
        --num_batches 8
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import loguru
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mip.agent_goal_predictor_dit import GoalPredictorDiTAgent  # noqa: E402
from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.torch_utils import set_seed  # noqa: E402


def _to_device(td_or_dict, device):
    out = {}
    for k, v in td_or_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    bs = next(iter(out.values())).shape[0]
    return TensorDict(out, batch_size=bs)


def _idm_ode(flow_map, obs_emb, act_shape, num_steps, sample_mode, device):
    """Inline IDM ODE — equivalent to ``mip.samplers.ode_sampler`` but takes a
    pre-computed conditioning embedding instead of an encoder."""
    if sample_mode == "stochastic":
        act_s = torch.randn(act_shape, device=device)
    else:
        act_s = torch.zeros(act_shape, device=device)
    bs = act_shape[0]
    t_schedule = np.linspace(0, 1, num_steps + 1)
    for i in range(num_steps):
        s_val = float(t_schedule[i])
        t_val = float(t_schedule[i + 1])
        s = torch.full((bs,), s_val, device=device)
        v = flow_map.get_velocity(s, act_s, obs_emb)
        act_s = act_s + v * (t_val - s_val)
    return act_s


def _goal_ode(goal_flow_map, z_t, num_steps, device):
    """Run the goal-predictor DiT ODE in *normalized* z-scored goal space.
    Returns the un-denormalized output (B, 1, emb_dim)."""
    B, _, emb_dim = z_t.shape
    g_s = torch.randn(B, 1, emb_dim, device=device)
    t_schedule = np.linspace(0, 1, num_steps + 1)
    for i in range(num_steps):
        s_val = float(t_schedule[i])
        t_val = float(t_schedule[i + 1])
        s = torch.full((B,), s_val, device=device)
        v = goal_flow_map.get_velocity(s, g_s, z_t)
        g_s = g_s + v * (t_val - s_val)
    return g_s


def _residual_stats(pred, target, var_per_elem=None):
    """pred, target: (B, D). Returns dict with per-element MSE, sqrt-MSE per
    sample, and (optionally) implied R^2 against ``var_per_elem`` (a (D,)
    tensor of per-element variance over the dataset slice)."""
    diff = pred - target
    mse_elem = diff.pow(2).mean().item()                     # avg over B and D
    mse_sample = diff.pow(2).sum(dim=-1).mean().item()       # sum over D, mean over B
    out = {
        "mse_per_elem": mse_elem,
        "rmse_per_elem": float(np.sqrt(mse_elem)),
        "l2_sq_per_sample": mse_sample,
        "rms_l2_per_sample": float(np.sqrt(mse_sample)),
    }
    if var_per_elem is not None:
        # R^2 in the per-element averaged sense.
        v = float(var_per_elem.mean().item())
        out["var_per_elem"] = v
        out["r2"] = 1.0 - (mse_elem / max(v, 1e-12))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to .hydra/config.yaml from a goal_predictor_dit_v2 run")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Goal predictor DiT checkpoint (.pt)")
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps_idm", type=int, default=1,
                        help="ODE steps for the IDM action sampler")
    parser.add_argument("--num_steps_goal", type=int, default=None,
                        help="Override goal_flow_num_steps from config")
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val"],
                        help="Which dataset split to use")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalizer_path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(args.config_path)
    cfg.optimization.device = device
    cfg.optimization.use_compile = False
    cfg.optimization.use_cudagraphs = False
    cfg.optimization.num_steps = int(args.num_steps_idm)
    if args.num_steps_goal is not None:
        cfg.optimization.goal_flow_num_steps = int(args.num_steps_goal)

    # The agent's __init__ uses task.obs_dim only for non-image; for image
    # tasks the original training script sets it to network.emb_dim after
    # env.reset(). Mirror that without spinning up a sim.
    if cfg.task.obs_type == "image":
        cfg.task.obs_dim = cfg.network.emb_dim

    # --- Normalizer (so dataset matches the IDM's training stats) ---
    idm_path = cfg.optimization.idm_checkpoint_path
    norm_path = args.normalizer_path or os.path.join(
        os.path.dirname(idm_path), "normalizer.pkl"
    )
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"normalizer.pkl missing at {norm_path}")
    with open(norm_path, "rb") as f:
        idm_normalizer = pickle.load(f)
    loguru.logger.info(f"Loaded normalizer from {norm_path}")

    # --- Dataset ---
    dataset = make_idm_dataset(cfg.task, mode=args.mode, normalizer=idm_normalizer)
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

    # --- Agent (loads frozen IDM + builds goal DiT from cfg) ---
    agent = GoalPredictorDiTAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()
    loguru.logger.info(f"Loaded goal predictor DiT from {args.ckpt_path}")

    encoder = agent._inner_encoder            # frozen
    flow_map = agent.flow_map                 # frozen IDM (with .net = LBMDiTIDMv2)
    fdm_net = flow_map.net                    # has forward_predict
    goal_fm_ema = agent.goal_flow_map_ema     # EMA goal DiT

    # Sanity: confirm FDM head exists
    if not hasattr(fdm_net, "forward_predict"):
        raise RuntimeError("Loaded IDM net has no forward_predict (not lbmidm_v2?)")

    sample_mode = cfg.optimization.sample_mode
    horizon = cfg.task.horizon
    act_dim = cfg.task.act_dim
    goal_steps = cfg.optimization.goal_flow_num_steps

    # Accumulators (per-batch) for unweighted mean and variance
    accs = {k: [] for k in [
        "fdm_expert",        # (a)
        "fdm_idm_true",      # (b)
        "fdm_idm_pred",      # (c)
        "gp_vs_true",        # (d)
        "fdm_idm_pred_vs_true",   # bonus: does FDM(s, a_pred) reach the TRUE goal?
    ]}
    all_z_goal = []   # for variance scale

    n_done = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if n_done >= args.num_batches:
                break

            obs = _to_device(batch["obs"], device)
            goal_obs = _to_device(batch["goal_obs"], device)
            act_expert = batch["action"].to(device)
            act_expert = act_expert[:, :horizon, :]

            B = act_expert.shape[0]

            # 1. Encode
            z_t = encoder(obs, None)              # (B, To, D)
            z_g_true = encoder(goal_obs, None)    # (B, 1, D)

            # 2. Goal predictor s'_gp (in raw encoder space)
            g_norm = _goal_ode(goal_fm_ema, z_t, goal_steps, device)
            s_pred = agent._denormalize(g_norm)   # (B, 1, D)

            # 3. IDM actions for both goals
            cond_true = torch.cat([z_t, z_g_true], dim=1)
            cond_pred = torch.cat([z_t, s_pred], dim=1)
            act_shape = (B, horizon, act_dim)

            a_idm_true = _idm_ode(
                flow_map, cond_true, act_shape, args.num_steps_idm,
                sample_mode, device,
            )
            a_idm_pred = _idm_ode(
                flow_map, cond_pred, act_shape, args.num_steps_idm,
                sample_mode, device,
            )

            # 4. FDM forward_predict (only first To_obs frames of condition are
            #    used, so the goal slot is moot — pass the true-goal version).
            fdm_expert = fdm_net.forward_predict(cond_true, act_expert)         # (B, D)
            fdm_idm_true_pred = fdm_net.forward_predict(cond_true, a_idm_true)  # (B, D)
            fdm_idm_pred_pred = fdm_net.forward_predict(cond_true, a_idm_pred)  # (B, D)

            zg = z_g_true.squeeze(1)
            sg = s_pred.squeeze(1)

            # Variance scale collection (over time, accumulate across batches)
            all_z_goal.append(zg.detach().cpu())

            # Stash residuals (compute final stats once we have all batches)
            accs["fdm_expert"].append((fdm_expert.cpu(), zg.cpu()))
            accs["fdm_idm_true"].append((fdm_idm_true_pred.cpu(), zg.cpu()))
            accs["fdm_idm_pred"].append((fdm_idm_pred_pred.cpu(), sg.cpu()))
            accs["gp_vs_true"].append((sg.cpu(), zg.cpu()))
            accs["fdm_idm_pred_vs_true"].append((fdm_idm_pred_pred.cpu(), zg.cpu()))

            n_done += 1
            loguru.logger.info(f"batch {n_done}/{args.num_batches}  B={B}")

    # Per-element variance of the goal embedding (over the held-out slice)
    z_goal_cat = torch.cat(all_z_goal, dim=0)      # (N, D)
    var_per_elem = z_goal_cat.var(dim=0, unbiased=False)  # (D,)
    loguru.logger.info(
        f"goal embedding: N={z_goal_cat.shape[0]} D={z_goal_cat.shape[1]} "
        f"mean per-elem var={var_per_elem.mean().item():.4f}"
    )

    # Aggregate stats
    print("\n" + "=" * 78)
    print(f"FDM-steering diagnostic — mode={args.mode}, "
          f"num_steps_idm={args.num_steps_idm}, goal_steps={goal_steps}")
    print("=" * 78)
    for name, pairs in accs.items():
        preds = torch.cat([p for p, _ in pairs], dim=0)
        targets = torch.cat([t for _, t in pairs], dim=0)
        stats = _residual_stats(preds, targets, var_per_elem=var_per_elem)
        print(f"\n[{name}]  n={preds.shape[0]}")
        for k, v in stats.items():
            print(f"  {k:>22s}: {v:.5f}" if isinstance(v, float) else f"  {k:>22s}: {v}")

    print("\nLegend:")
    print("  fdm_expert            : FDM(s, a_expert) vs true z_goal           [does FDM model expert (s,a)->s'?]")
    print("  fdm_idm_true          : FDM(s, a_IDM[true_g]) vs true z_goal      [does IDM sample satisfy FDM?]")
    print("  fdm_idm_pred          : FDM(s, a_IDM[pred_g]) vs predicted s'_gp  [the gap steering would close]")
    print("  gp_vs_true            : s'_gp vs true z_goal                       [goal-predictor accuracy]")
    print("  fdm_idm_pred_vs_true  : FDM(s, a_IDM[pred_g]) vs true z_goal       [bonus: rollout-relevant gap]")
    print()
    print("R^2 ~ 1 means residual << embedding variance (predictor is sharp).")
    print("R^2 ~ 0 means residual is on the order of just predicting the mean.")


if __name__ == "__main__":
    main()
