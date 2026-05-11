"""Rollout-time variant of scripts/diagnose_fdm_steering.py.

Drives the goal_predictor_dit_v2 agent through the env with num_envs=1 and at
each control step records residuals that test whether FDM-toward-s'_gp
steering would have purchase on rollout-distribution states.

Per control step (call it step t, with the next step at t + act_steps):
  - residual_c:    ||FDM(z_t, a_idm) - s'_gp||^2  per-element MSE
                   The steering target. If small here, gradient steering has
                   nothing to do.
  - gp_step_size:  ||s'_gp - z_t[-1]||^2 per-elem
                   How far the goal predictor is asking us to go in latent
                   space. Useful as a scale.
  - dir_cos_sim:   cosine(actual_step_in_latent, gp_predicted_step), where
                   actual_step = z_{t+1}[-1] - z_t[-1] and
                   gp_predicted_step = s'_gp - z_t[-1].
                   Computed on the *previous* step once we have the next obs.
                   Tells us whether s'_gp pointed in the direction the policy
                   actually moved.
  - reach_residual_mse:
                   ||z_{t+1}[-1] - s'_gp||^2 per-elem on the *previous* step.
                   "How close did we actually land to where the goal predictor
                   said we'd land?" Strong signal for goal-predictor accuracy
                   under the action-step truncation.

Records are stratified by episode success/failure.

Usage:
    python scripts/diagnose_fdm_steering_rollout.py \
        --config_path outputs/2026-04-28/00-17-50/.hydra/config.yaml \
        --ckpt_path   logs/tool_hang_..._fdm_aux_200000/2026_04_28_00_17_50/models/model_best.pt \
        --num_episodes 20 --num_steps_idm 1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import loguru
import numpy as np
import torch
from omegaconf import OmegaConf

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mip.agent_goal_predictor_dit import GoalPredictorDiTAgent  # noqa: E402
from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.torch_utils import set_seed  # noqa: E402

ROBOMIMIC_TASKS = ("can", "lift", "square", "tool_hang", "transport")


def _idm_ode(flow_map, obs_emb, act_shape, num_steps, sample_mode, device):
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


def _normalize_obs(obs_raw, base_dataset, device):
    obs_dict = {}
    for k, v in obs_raw.items():
        v_np = v.astype(np.float32)
        v_norm = base_dataset.normalizer["obs"][k].normalize(v_np)
        obs_dict[k] = torch.tensor(v_norm, device=device, dtype=torch.float32)
    return obs_dict


def _summary(values, name=None):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _print_summary(label, values, indent="  "):
    s = _summary(values)
    if s is None:
        print(f"{indent}{label:>32s}: (no data)")
        return
    print(
        f"{indent}{label:>32s}: "
        f"mean={s['mean']:.5f}  std={s['std']:.5f}  "
        f"median={s['median']:.5f}  p10={s['p10']:.5f}  p90={s['p90']:.5f}  n={s['n']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--num_steps_idm", type=int, default=1)
    parser.add_argument("--num_steps_goal", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalizer_path", type=str, default=None)
    parser.add_argument("--records_out", type=str, default=None,
                        help="Optional .json path to dump per-step records")
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
    cfg.task.num_envs = 1
    if hasattr(cfg.task, "save_video"):
        cfg.task.save_video = False
    if cfg.task.obs_type == "image":
        cfg.task.obs_dim = cfg.network.emb_dim

    # --- Env ---
    envs = make_vec_env(cfg.task, seed=args.seed)
    obs, _ = envs.reset()
    loguru.logger.info("Env ready (num_envs=1)")

    # --- Normalizer + dataset (for normalizer + undo_transform_action) ---
    idm_path = cfg.optimization.idm_checkpoint_path
    norm_path = args.normalizer_path or os.path.join(
        os.path.dirname(idm_path), "normalizer.pkl"
    )
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"normalizer.pkl missing at {norm_path}")
    with open(norm_path, "rb") as f:
        idm_normalizer = pickle.load(f)
    dataset = make_idm_dataset(cfg.task, normalizer=idm_normalizer)
    base_dataset = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )
    loguru.logger.info("Loaded normalizer + dataset")

    # --- Agent ---
    agent = GoalPredictorDiTAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()
    loguru.logger.info(f"Loaded GP DiT from {args.ckpt_path}")

    encoder = agent._inner_encoder
    flow_map = agent.flow_map
    fdm_net = flow_map.net
    goal_fm = agent.goal_flow_map_ema
    if not hasattr(fdm_net, "forward_predict"):
        raise RuntimeError("Loaded IDM has no forward_predict (not lbmidm_v2?)")

    horizon = cfg.task.horizon
    act_dim = cfg.task.act_dim
    obs_steps = cfg.task.obs_steps
    act_steps = cfg.task.act_steps
    max_steps = cfg.task.max_episode_steps
    sample_mode = cfg.optimization.sample_mode
    goal_steps = cfg.optimization.goal_flow_num_steps
    abs_action_undo = (
        cfg.task.abs_action and cfg.task.env_name in ROBOMIMIC_TASKS
    )

    # --- Per-step records, stratified by success ---
    success_episodes = []
    fail_episodes = []
    all_records = []     # for optional dump

    for ep_idx in range(args.num_episodes):
        obs, _ = envs.reset()
        ep_reward = 0.0
        t = 0
        ep_records = []

        # State carried across iterations so we can do the t-1 -> t lookahead
        prev_z_last = None        # z_t[-1]   (1, D)
        prev_s_gp = None          # (1, D)
        prev_step_record = None

        while t < max_steps:
            obs_dict = _normalize_obs(obs, base_dataset, device)

            with torch.no_grad():
                z_t = encoder(obs_dict, None)                  # (1, To, D)
                z_t_last = z_t[:, -1]                          # (1, D)

                g_norm = _goal_ode(goal_fm, z_t, goal_steps, device)
                s_gp_full = agent._denormalize(g_norm)         # (1, 1, D)
                s_gp = s_gp_full.squeeze(1)                    # (1, D)

                cond = torch.cat([z_t, s_gp_full], dim=1)
                a_idm = _idm_ode(
                    flow_map, cond, (1, horizon, act_dim),
                    args.num_steps_idm, sample_mode, device,
                )

                fdm_pred = fdm_net.forward_predict(cond, a_idm)  # (1, D)

            residual_c_mse = ((fdm_pred - s_gp).pow(2).mean()).item()
            gp_step_mse = ((s_gp - z_t_last).pow(2).mean()).item()

            step_record = {
                "ep": ep_idx,
                "t": t,
                "residual_c_mse": residual_c_mse,
                "gp_step_mse": gp_step_mse,
                # filled in next step (after we observe what we landed on):
                "dir_cos_sim": None,
                "reach_residual_mse": None,
                "actual_step_mse": None,
            }

            # Lookahead fill-in for the *previous* step.
            if prev_z_last is not None and prev_s_gp is not None:
                actual_step = z_t_last - prev_z_last           # (1, D)
                pred_step = prev_s_gp - prev_z_last            # (1, D)
                a_norm = actual_step.norm(dim=-1).clamp_min(1e-8)
                p_norm = pred_step.norm(dim=-1).clamp_min(1e-8)
                cos_sim = ((actual_step * pred_step).sum(-1)
                           / (a_norm * p_norm)).item()
                reach_residual = ((z_t_last - prev_s_gp).pow(2).mean()).item()
                actual_mse = (actual_step.pow(2).mean()).item()
                prev_step_record["dir_cos_sim"] = cos_sim
                prev_step_record["reach_residual_mse"] = reach_residual
                prev_step_record["actual_step_mse"] = actual_mse

            ep_records.append(step_record)
            prev_step_record = step_record
            prev_z_last = z_t_last.detach()
            prev_s_gp = s_gp.detach()

            # Apply first act_steps of a_idm.
            act_normed = a_idm.detach().cpu().numpy()
            act = base_dataset.normalizer["action"].unnormalize(act_normed)
            start = obs_steps - 1
            end = start + act_steps
            act = act[:, start:end, :]
            if abs_action_undo:
                act = base_dataset.undo_transform_action(act)

            obs, reward, terminated, truncated, info = envs.step(act)
            ep_reward += float(np.sum(reward))
            t += act_steps
            if bool(terminated[0]) or bool(truncated[0]):
                break

        success = ep_reward > 0
        ep_summary = {
            "ep": ep_idx,
            "success": bool(success),
            "n_steps": t,
            "step_records": ep_records,
        }
        (success_episodes if success else fail_episodes).append(ep_summary)
        all_records.append(ep_summary)

        loguru.logger.info(
            f"[ep {ep_idx:03d}] success={int(success)}  steps={t}  "
            f"mean_residual_c={np.mean([r['residual_c_mse'] for r in ep_records]):.5f}"
        )

    # --- Aggregate ---
    def _flatten_field(eps, field):
        out = []
        for ep in eps:
            for r in ep["step_records"]:
                v = r.get(field)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    out.append(v)
        return out

    print("\n" + "=" * 78)
    print(f"Rollout-time FDM-steering diagnostic — env={cfg.task.env_name}, "
          f"num_steps_idm={args.num_steps_idm}, goal_steps={goal_steps}")
    print("=" * 78)
    print(f"  episodes: success {len(success_episodes)} / "
          f"fail {len(fail_episodes)} / total {args.num_episodes}")
    print(f"  per-elem var(z_goal) on val set was ~0.02 (tool_hang) / 0.04 (can)\n")

    for label, eps in [
        ("ALL EPISODES", success_episodes + fail_episodes),
        ("SUCCESS",      success_episodes),
        ("FAIL",         fail_episodes),
    ]:
        print(f"\n[{label}]  n_episodes={len(eps)}  "
              f"n_steps_total={sum(len(ep['step_records']) for ep in eps)}")
        if not eps:
            continue
        for field in [
            "residual_c_mse",         # FDM(s, a) vs s'_gp        (steering target)
            "gp_step_mse",            # ||s'_gp - z_t[-1]||^2     (scale)
            "actual_step_mse",        # ||z_{t+1}[-1] - z_t[-1]||^2
            "reach_residual_mse",     # ||z_{t+1}[-1] - s'_gp||^2 (GP reachability)
            "dir_cos_sim",            # cos(actual_step, gp_step)
        ]:
            _print_summary(field, _flatten_field(eps, field))

    print(
        "\nKey: residual_c_mse is the gradient-steering target. If it stays "
        "near the val-set floor in failure episodes, steering still has no "
        "purchase off-distribution. If it spikes in failures, FDM-IDM "
        "consistency breaks under stress and steering may help."
    )

    if args.records_out:
        out_path = Path(args.records_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_records, f, indent=2)
        loguru.logger.info(f"Wrote per-step records to {out_path}")


if __name__ == "__main__":
    main()
