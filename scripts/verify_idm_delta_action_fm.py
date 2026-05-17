"""Verify the new delta-cond IDM implementation by running action flow-matching
ODE on expert training data and comparing samples to ground-truth actions.

If ``LBMDiTIDMv2Delta._summarize`` is correct (third AdaLN slot = delta) and
the agent loads the checkpoint cleanly, the L2 between ODE-sampled actions
and the expert action chunk should be small. We also compute baselines:

  * sample with the goal frame *shuffled across the batch* — should be far worse
    (model uses the goal),
  * sample with the goal replaced by the encoder's ``uncond_emb`` — soft floor,
  * L2 of pure noise vs ground truth — hard floor,
  * for reference, the same on the goal-cond checkpoint loaded into the
    standard ``IDMFDMAgent`` with ``lbmidm_v2``.

Each L2 is reported in the dataset's *normalized* action space (same units the
network sees).
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


def make_agent(network_name: str, agent_cls, task_config: str,
               config_dir: str, device: str, ckpt: str):
    """Build agent + (conditionally) wrap encoder to match the ckpt's layout.

    Some IDM ckpts were saved with the encoder wrapped in
    ``GoalDropoutEncoder`` (state dict has ``uncond_emb`` and
    ``encoder.`` prefix); others were saved unwrapped. We detect which by
    peeking at the ckpt and wrap only when needed.
    """
    import hydra
    from hydra import initialize_config_dir

    config_abs = str((Path(__file__).resolve().parents[1] / config_dir).resolve())
    overrides = [f"task={task_config}", f"network={network_name}",
                 "optimization.use_compile=false",
                 "optimization.use_cudagraphs=false"]
    with initialize_config_dir(config_dir=config_abs, version_base=None):
        cfg = hydra.compose(config_name="main", overrides=overrides)
    cfg.task.obs_dim = cfg.network.emb_dim
    cfg.optimization.device = device

    agent = agent_cls(cfg)

    # Peek at the ckpt to decide whether to wrap.
    sd_peek = torch.load(ckpt, map_location="cpu", weights_only=False)["encoder"]
    has_wrapper = "uncond_emb" in sd_peek or any(
        k.startswith("encoder.") for k in sd_peek
    )
    if has_wrapper:
        from mip.encoders import GoalDropoutEncoder
        enc_out_dim = cfg.network.encoder_out_dim or cfg.network.emb_dim
        agent.encoder = GoalDropoutEncoder(
            agent.encoder, enc_out_dim, cfg.task.obs_steps,
            cfg.optimization.goal_dropout_prob,
        ).to(device)
        agent.encoder_ema = GoalDropoutEncoder(
            agent.encoder_ema, enc_out_dim, cfg.task.obs_steps,
            cfg.optimization.goal_dropout_prob,
        ).to(device)
        agent.optimizer.add_param_group({"params": [agent.encoder.uncond_emb]})
    agent.__compile__()
    return agent, cfg, has_wrapper


def load_normalizer(ckpt: str):
    with open(Path(ckpt).parent / "normalizer.pkl", "rb") as f:
        return pickle.load(f)


def build_windows(h5_path: str, task_cfg, normalizer,
                  n_demos: int, n_windows_per_demo: int,
                  seed: int):
    """Return:
       obs_td: dict[k] -> (N, To+1, ...) normalized stack
       act_normed: (N, H, A) normalized ground-truth actions
       num_total: N
    """
    rng = np.random.default_rng(seed)
    img_keys = sorted(k for k, v in task_cfg.shape_meta.obs.items()
                      if v.type == "rgb")
    lowdim_keys = sorted(k for k, v in task_cfg.shape_meta.obs.items()
                          if v.type == "low_dim")
    To = task_cfg.obs_steps
    H = task_cfg.horizon

    obs_list = {k: [] for k in img_keys + lowdim_keys}
    act_list = []

    with h5py.File(h5_path, "r") as f:
        all_names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        chosen = rng.choice(len(all_names),
                            size=min(n_demos, len(all_names)),
                            replace=False)
        for di in chosen:
            d = f["data"][all_names[int(di)]]
            T = d["obs"][img_keys[0]].shape[0]
            valid_t = list(range(To - 1, T - H - 1))
            if not valid_t:
                continue
            ts = rng.choice(valid_t,
                            size=min(n_windows_per_demo, len(valid_t)),
                            replace=False)
            actions = np.asarray(d["actions"]).astype(np.float32)
            for t in ts:
                idx = list(range(t - To + 1, t + 1)) + [t + H]   # To obs + 1 goal
                for k in img_keys:
                    x = np.asarray(d["obs"][k])[idx].astype(np.float32) / 255.
                    x = np.moveaxis(x, -1, 1)
                    obs_list[k].append(normalizer["obs"][k].normalize(x))
                for k in lowdim_keys:
                    x = np.asarray(d["obs"][k])[idx].astype(np.float32)
                    obs_list[k].append(normalizer["obs"][k].normalize(x))
                a = actions[t:t + H]
                act_list.append(normalizer["action"].normalize(a))

    obs_arr = {k: np.stack(v) for k, v in obs_list.items()}      # (N, To+1, ...)
    act_arr = np.stack(act_list)                                  # (N, H, A)
    return obs_arr, act_arr


@torch.no_grad()
def sample_l2(agent, obs_arr, act_arr, device, num_steps: int, batch: int,
              goal_mode: str = "true"):
    """Run agent.sample on every window. Returns per-element L2 to GT.

    goal_mode:
      "true"     -> use the real goal frame (the (To+1)-th in obs_arr)
      "shuffled" -> shuffle the goal frame across the batch (counterfactual)
      "uncond"   -> replace goal with encoder.uncond_emb-equivalent: pass only
                    To obs frames so the wrapper auto-pads with uncond_emb.
    """
    from tensordict import TensorDict
    N, H, A = act_arr.shape
    per_window_l2 = np.zeros(N, dtype=np.float64)
    rng = np.random.default_rng(0)

    if goal_mode == "shuffled":
        perm = rng.permutation(N)

    for s in range(0, N, batch):
        e = min(s + batch, N)
        B = e - s
        obs_chunk = {}
        for k, v in obs_arr.items():
            arr = v[s:e].copy()                              # (B, To+1, ...)
            if goal_mode == "shuffled":
                arr[:, -1] = v[perm[s:e], -1]
            obs_chunk[k] = torch.from_numpy(arr).to(device)
        if goal_mode == "uncond":
            # Drop the goal frame; GoalDropoutEncoder auto-pads with uncond_emb.
            # contiguous() is required — MultiImageObsEncoder calls .view().
            for k in obs_chunk:
                obs_chunk[k] = obs_chunk[k][:, :-1].contiguous()
        obs_td = TensorDict(obs_chunk, batch_size=B)
        act_0 = torch.randn(B, H, A, device=device)
        act_pred = agent.sample(act_0, obs_td, num_steps=num_steps, use_ema=True)
        gt = torch.from_numpy(act_arr[s:e]).to(device)
        per_window_l2[s:e] = (
            (act_pred - gt).reshape(B, -1).pow(2).mean(-1).sqrt().cpu().numpy()
        )

    return per_window_l2


def evaluate_ckpt(tag, ckpt, network_name, agent_cls, args):
    loguru.logger.info(f"=== {tag} ({network_name}) ===")
    agent, cfg, has_wrapper = make_agent(
        network_name, agent_cls, args.task_config,
        args.config_dir, args.device, ckpt,
    )
    loguru.logger.info(f"[{tag}] encoder wrapper present in ckpt: {has_wrapper}")
    agent.load(ckpt, load_optimizer=False)
    normalizer = load_normalizer(ckpt)

    obs_arr, act_arr = build_windows(
        args.dataset_path, cfg.task, normalizer,
        n_demos=args.n_demos, n_windows_per_demo=args.n_windows_per_demo,
        seed=args.seed,
    )
    N = act_arr.shape[0]
    loguru.logger.info(
        f"[{tag}] {N} windows from {args.n_demos} demos, "
        f"To={cfg.task.obs_steps}, H={cfg.task.horizon}, A={cfg.task.act_dim}"
    )

    # Hard floor — pure noise vs GT
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(act_arr.shape).astype(np.float32)
    l2_noise = np.sqrt(((noise - act_arr) ** 2).mean(axis=(1, 2)))

    # Action ODE samples with: true goal / shuffled goal / uncond goal.
    # uncond requires the GoalDropoutEncoder wrapper (auto-pads with uncond_emb).
    results = {
        "noise_vs_gt": l2_noise,
        "true_goal": sample_l2(agent, obs_arr, act_arr, args.device,
                                num_steps=args.num_steps,
                                batch=args.batch, goal_mode="true"),
        "shuffled_goal": sample_l2(agent, obs_arr, act_arr, args.device,
                                    num_steps=args.num_steps,
                                    batch=args.batch, goal_mode="shuffled"),
    }
    if has_wrapper:
        results["uncond_goal"] = sample_l2(
            agent, obs_arr, act_arr, args.device,
            num_steps=args.num_steps, batch=args.batch, goal_mode="uncond",
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta_ckpt", default="logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_delta_cond_fdm1.0_gdp0.0_ac8_ndp_0/2026_05_03_22_36_46/models/model_step_300000.pt")
    p.add_argument("--goal_ckpt",  default="logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_fdm_aux/2026_04_27_01_03_55/models/model_step_300000.pt")
    # Optional: compare two delta-cond ckpts (e.g., dropout=0 vs dropout>0).
    p.add_argument("--delta_ckpt2", default=None,
                   help="If set, also evaluate this as a second delta-cond ckpt and skip the goal-cond run.")
    p.add_argument("--delta_tag",  default="delta",  help="Tag for first delta ckpt")
    p.add_argument("--delta_tag2", default="delta2", help="Tag for second delta ckpt")
    p.add_argument("--task_config", default="tool_hang_ph_image_gp")
    p.add_argument("--config_dir", default="examples/configs")
    p.add_argument("--dataset_path", default="data/robomimic/tool_hang/ph/image_v15.hdf5")
    p.add_argument("--n_demos", type=int, default=20)
    p.add_argument("--n_windows_per_demo", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    from mip.agent import TrainingAgent
    from mip.agent_idm_fdm_delta import IDMFDMAgentDelta

    out = {}
    out[args.delta_tag] = evaluate_ckpt(
        args.delta_tag, args.delta_ckpt, "lbmidm_v2_delta",
        IDMFDMAgentDelta, args,
    )
    if args.delta_ckpt2:
        out[args.delta_tag2] = evaluate_ckpt(
            args.delta_tag2, args.delta_ckpt2, "lbmidm_v2_delta",
            IDMFDMAgentDelta, args,
        )
    else:
        out["goal"] = evaluate_ckpt(
            "goal", args.goal_ckpt, "lbmidm_v2",
            TrainingAgent, args,
        )

    # Print summary table
    print("\n" + "=" * 72)
    print(f"{'condition':<20} {'mean L2':>10} {'median L2':>12} {'std L2':>10}")
    print("-" * 72)
    for tag, res in out.items():
        print(f"\n  ckpt: {tag}")
        for cond in ["noise_vs_gt", "uncond_goal", "shuffled_goal", "true_goal"]:
            if cond not in res:
                continue
            v = res[cond]
            print(f"  {cond:<20} {v.mean():>10.4f} {np.median(v):>12.4f} {v.std():>10.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
