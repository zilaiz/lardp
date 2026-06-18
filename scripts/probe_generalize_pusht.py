"""Seen->unseen linear probing on PushT, for DP / joint(expert) / joint(expert+10%play).

PushT obs = [image(3,96,96), agent_pos(2)], action 2-D, H=16. State (5-D) =
[agent_x, agent_y, block_x, block_y, block_angle]; agent_pos = robot, block = object.
Data is a single expert zarr; demos split by val_pct=0.8 (train=first 20%, val=rest).

Two linear probes (ridge, fit on seen-demo windows / R^2 on unseen-demo windows;
post-LN embeddings for the joint models):
  action-decodability   : emb            -> action chunk (H*2)
  forward-predictability : [emb, chunk]  -> s_{t+H}   (agent_pos; and full state incl block)

Emits repr_analysis/probe_generalize_pusht{,_object}.png + .json.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.repr_extract as X
from scripts.repr_probe2 import lin_r2
from mip.network_utils import get_encoder
from mip.dataset_utils import ReplayBuffer, MinMaxNormalizer
from mip.datasets.pusht_dataset import _split_demo_indices

CFGS = json.load(open("/tmp/pusht_cfgs.json"))
CKPTS = json.load(open("/tmp/pusht_ckpts.json"))
SETTINGS = ["DP", "EO", "EP"]
LABELS = {"DP": "vanilla DP", "EO": "joint (expert)", "EP": "joint (expert+10%play)"}
COLORS = {"DP": "#888888", "EO": "#4C72B0", "EP": "#C44E52"}
ZARR = "data/pusht/pusht_cchi_v7_replay.zarr"
CAP = 2500


def build_encoder(setting):
    cfg = OmegaConf.create(CFGS[setting])
    enc = get_encoder(cfg.network, cfg.task).to(X.DEVICE)
    sd = torch.load(CKPTS[setting]["ckpt"], map_location=X.DEVICE, weights_only=False)
    enc.load_state_dict(sd["encoder_ema"]); enc.eval()
    ln = None
    if "target_ln_ema" in sd:
        d = sd["target_ln_ema"]["weight"].shape[0]
        ln = torch.nn.LayerNorm(d, elementwise_affine=True).to(X.DEVICE)
        ln.load_state_dict(sd["target_ln_ema"]); ln.eval()
    return enc, ln


def sample_pusht(imgs, states, actions, starts, ends, episodes, To, H, cap, seed):
    rng = np.random.RandomState(seed)
    eps = [i for i in episodes if ends[i] - starts[i] >= To + H]
    iw, aw, act, tgt = [], [], [], []
    for _ in range(cap):
        i = eps[rng.randint(len(eps))]
        s, e = starts[i], ends[i]
        t = rng.randint(s + To - 1, e - H)          # t+H <= e-1
        iw.append(imgs[t - To + 1:t + 1])
        aw.append(states[t - To + 1:t + 1, :2])
        act.append(actions[t:t + H])
        tgt.append(states[t + H, :5])               # full state at s_{t+H}
    data = {"img": {"image": np.stack(iw)},
            "low": {"agent_pos": np.stack(aw).astype(np.float32)},
            "goal_img": {"image": np.stack(iw)[:, :1]},
            "goal_low": {"agent_pos": np.stack(aw)[:, :1]},
            "cur": np.zeros((len(act), 1), np.float32)}
    return data, np.stack(act).astype(np.float32), np.stack(tgt).astype(np.float32)


def main():
    X.RGB = ["image"]; X.LOWDIM = ["agent_pos"]
    cfg = OmegaConf.create(CFGS["EO"])
    To, H = int(cfg.task.obs_steps), int(cfg.task.horizon)
    val_pct = float(cfg.task.val_dataset_percentage)

    rb = ReplayBuffer.copy_from_path(ZARR, keys=["state", "action", "img"])
    imgs = np.asarray(rb["img"]); states = np.asarray(rb["state"]).astype(np.float32)
    actions = np.asarray(rb["action"]).astype(np.float32)
    ends = np.asarray(rb.episode_ends); starts = np.concatenate([[0], ends[:-1]])
    n = rb.n_episodes
    seen = _split_demo_indices(n, val_pct, "train")
    unseen = _split_demo_indices(n, val_pct, "val")
    seen_glob = np.concatenate([np.arange(starts[i], ends[i]) for i in seen])
    agent_norm = {"agent_pos": MinMaxNormalizer(states[seen_glob, :2])}  # train-split normalizer
    print(f"[pusht] To={To} H={H} val_pct={val_pct} seen={len(seen)} unseen={len(unseen)} demos")

    s_data, s_act, s_full = sample_pusht(imgs, states, actions, starts, ends, seen, To, H, CAP, 0)
    u_data, u_act, u_full = sample_pusht(imgs, states, actions, starts, ends, unseen, To, H, CAP, 1)
    s_chunk = s_act.reshape(len(s_act), -1); u_chunk = u_act.reshape(len(u_act), -1)

    res = {m: {} for m in ["action_decode", "forward_agent", "forward_full"]}
    for s in SETTINGS:
        e, ln = build_encoder(s)
        es = X.encode(e, ln, s_data, agent_norm, To)[1]    # post-LN, seen
        eu = X.encode(e, ln, u_data, agent_norm, To)[1]    # post-LN, unseen
        del e, ln; torch.cuda.empty_cache()
        res["action_decode"][s] = lin_r2(es, s_chunk, eu, u_chunk)
        Xs = np.concatenate([es, s_chunk], 1); Xu = np.concatenate([eu, u_chunk], 1)
        res["forward_agent"][s] = lin_r2(Xs, s_full[:, :2], Xu, u_full[:, :2])
        res["forward_full"][s] = lin_r2(Xs, s_full, Xu, u_full)
        print(f"  {s}: act={res['action_decode'][s]:.3f} "
              f"fwd_agent={res['forward_agent'][s]:.3f} fwd_full={res['forward_full'][s]:.3f}")

    json.dump(res, open(f"{X.OUT_ROOT}/probe_generalize_pusht.json", "w"), indent=2)

    def chart(fwd_key, fwd_title, fname):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, (metric, title) in zip(
                axes, [("action_decode", "Action-decodability (emb → action chunk)"),
                       (fwd_key, fwd_title)]):
            vals = [res[metric][s] for s in SETTINGS]
            bars = ax.bar(np.arange(3), vals, color=[COLORS[s] for s in SETTINGS], width=0.6)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)
            ax.set_xticks(np.arange(3)); ax.set_xticklabels([LABELS[s] for s in SETTINGS], fontsize=8)
            ax.set_ylim(0, 1.0); ax.set_ylabel("R²  (fit on seen, eval on unseen)")
            ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        fig.suptitle("PushT linear probing: seen→unseen expert generalization", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{X.OUT_ROOT}/{fname}", dpi=150)
        print(f"saved {X.OUT_ROOT}/{fname}")

    chart("forward_agent", "Forward-predictability ([emb,chunk] → agent_pos)",
          "probe_generalize_pusht.png")
    chart("forward_full", "Forward-predictability ([emb,chunk] → full state incl block)",
          "probe_generalize_pusht_object.png")


if __name__ == "__main__":
    main()
