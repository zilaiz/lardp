"""Trunk-side test: open-loop action accuracy of the FULL policies.

The encoders are decision-equivalent (probing). This tests whether the TRUNK
turns that (equivalent) representation into more expert-like actions. For each
full policy (encoder+trunk+sampler) we sample N action chunks per state, un-
normalize to physical units, and compare to the ground-truth EXPERT action chunk.

States: held-out UNSEEN expert demos (optimal labels) + successful-rollout
pre-success states (good recovery actions). Same states for all three policies.

Metrics (physical action units, lower=better):
  chunk_mean   mean over N samples of ||a_pred - a_gt|| (avg closeness)
  chunk_bestN  min over N (does the policy put MASS near the expert action)
  first_mean / first_bestN  same on the executed (first) action
  spread       mean sample std (multimodality / dispersion)

Usage: python scripts/trunk_action.py --task can
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

import scripts.repr_extract as X
from scripts.repr_unseen import sample_subset
from mip.dataset_utils import MinMaxNormalizer
from mip.agent import TrainingAgent
from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent

N_SAMP = 16
CAP = 600


def fit_action_norm(path, demo_names):
    with h5py.File(path, "r") as f:
        acts = [f["data"][dm]["actions"][:].astype(np.float32) for dm in demo_names]
    return MinMaxNormalizer(np.concatenate(acts, 0))


def merge_act(a, b):
    return MinMaxNormalizer(np.stack([np.minimum(a.min, b.min), np.maximum(a.max, b.max)]))


def build_agent(task, setting):
    cfg = OmegaConf.create(X.CFGS[f"{task}/{setting}"])
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    OmegaConf.update(cfg, "optimization.device", X.DEVICE, merge=False)
    if setting == "DP":
        agent = TrainingAgent(cfg)
    else:
        agent = LBMDiTJointPTAgent(cfg)
    agent.load(X.CKPTS[f"{task}/{setting}"]["ckpt"], load_optimizer=False)
    agent.eval() if hasattr(agent, "eval") else None
    return agent, cfg


def build_obs(data, idx, normd, To):
    ob = {}
    for k in X.RGB:
        ob[k] = torch.from_numpy(X._img_norm(data["img"][k][idx])).to(X.DEVICE)
    for k in X.LOWDIM:
        ob[k] = torch.from_numpy(normd[k].normalize(data["low"][k][idx])).to(X.DEVICE)
    return ob


@torch.no_grad()
def sampled_actions(agent, data, idxs, obs_norm, act_norm, To, H, A, num_steps, bs=32):
    """Return physical sampled actions (len(idxs), N, H, A)."""
    out = []
    for b in range(0, len(idxs), bs):
        chunk = idxs[b:b + bs]
        B = len(chunk)
        ob = build_obs(data, chunk, obs_norm, To)
        ob = {k: v.repeat_interleave(N_SAMP, dim=0) for k, v in ob.items()}  # (B*N,...)
        a0 = torch.randn(B * N_SAMP, H, A, device=X.DEVICE)
        a = agent.sample(act_0=a0, obs=ob, num_steps=num_steps, use_ema=True)
        a = a.detach().cpu().numpy().reshape(B, N_SAMP, H, A)
        a = act_norm.unnormalize(a)  # physical
        out.append(a)
    return np.concatenate(out, 0)


def errors(pred, gt):
    """pred (M,N,H,A) physical, gt (M,H,A). Returns dict of error metrics."""
    d = pred - gt[:, None]                          # (M,N,H,A)
    chunk_l2 = np.linalg.norm(d.reshape(*d.shape[:2], -1), axis=2)  # (M,N)
    first_l2 = np.linalg.norm(d[:, :, 0, :], axis=2)               # (M,N)
    r = {
        "chunk_mean": float(chunk_l2.mean()),
        "chunk_bestN": float(chunk_l2.min(1).mean()),
        "first_mean": float(first_l2.mean()),
        "first_bestN": float(first_l2.min(1).mean()),
        "spread": float(pred[:, :, 0, :].std(1).mean()),
    }
    # per-component first-action mean error
    comp = {"pos": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7)}
    for nm, sl in comp.items():
        r[f"first_{nm}"] = float(np.linalg.norm(d[:, :, 0, sl], axis=2).mean())
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True)
    task = ap.parse_args().task
    cfg = X._cfg(task, "EP")
    sm = cfg.task.shape_meta.obs
    X.RGB = [k for k, v in sm.items() if v.get("type") == "rgb"]
    X.LOWDIM = [k for k, v in sm.items() if v.get("type", "low_dim") == "low_dim"]
    exp_p, roll_p = cfg.task.dataset_paths[0], cfg.task.dataset_paths[1]
    To, H, A = int(cfg.task.obs_steps), int(cfg.task.horizon), int(cfg.task.act_dim)
    val_pct = float(cfg.task.val_dataset_percentage)
    num_steps = int(cfg.optimization.joint_num_steps)

    with h5py.File(exp_p, "r") as f:
        total = len(f["data"])
    train_count = total - int(total * val_pct)
    unseen = [f"demo_{i}" for i in range(train_count, total)]

    # normalizers
    N_exp = X.fit_lowdim_normalizers(exp_p)
    N_mix = X.merge_lowdim(N_exp, X.fit_lowdim_normalizers(roll_p))
    obs_norm = {"DP": N_exp, "EO": N_exp, "EP": N_mix}
    act_exp = fit_action_norm(exp_p, [f"demo_{i}" for i in range(train_count)])
    with h5py.File(roll_p, "r") as f:
        roll_demos = list(f["data"].keys())
    act_roll = fit_action_norm(roll_p, roll_demos)
    act_mix = merge_act(act_exp, act_roll)
    act_norm = {"DP": act_exp, "EO": act_exp, "EP": act_mix}

    from scripts.repr_success import sample_rollout
    exp_data = sample_subset(exp_p, unseen, To, H, CAP, seed=20)
    R = sample_rollout(roll_p, To, H, 4000, seed=21)
    pre = R["pre"]                                     # successful pre-success states
    roll_data = {"img": {k: R["img"][k][pre] for k in X.RGB},
                 "low": {k: R["low"][k][pre] for k in X.LOWDIM}, "act": R["act"][pre]}
    sources = {"unseen_exp": exp_data, "succ_roll": roll_data}
    print(f"[{task}] unseen-exp={len(exp_data['act'])} succ-roll={len(roll_data['act'])} "
          f"N={N_SAMP} steps={num_steps}")

    out = {"task": task, "settings": {}}
    for s in ["DP", "EO", "EP"]:
        agent, _ = build_agent(task, s)
        rec = {}
        for src, data in sources.items():
            gt = data["act"]; idxs = np.arange(len(gt))
            pred = sampled_actions(agent, data, idxs, obs_norm[s], act_norm[s], To, H, A, num_steps)
            rec[src] = errors(pred, gt)
        out["settings"][s] = rec
        del agent; torch.cuda.empty_cache()
        print(f"  {s}: done")

    json.dump(out, open(f"{X.OUT_ROOT}/{task}/trunk.json", "w"), indent=2)
    for src in sources:
        print(f"\n===== {task} [{src}] action error vs target (physical; lower=better) =====")
        print(f"{'metric':<14}{'DP':>10}{'EO':>10}{'EP':>10}")
        for m in ["chunk_mean", "chunk_bestN", "first_mean", "first_bestN",
                  "first_pos", "first_rot", "first_grip", "spread"]:
            print(f"{m:<14}" + "".join(f"{out['settings'][s][src][m]:>10.4f}" for s in ["DP", "EO", "EP"]))
    print(f"saved {X.OUT_ROOT}/{task}/trunk.json")


if __name__ == "__main__":
    main()
