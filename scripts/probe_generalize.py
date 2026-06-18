"""Linear probing, fit on SEEN expert states / eval on UNSEEN expert states,
for DP / EO (joint-expert) / EP (joint-expert+play) on 4 robomimic tasks.

Two probes (linear ridge, fit on seen-demo windows, R^2 evaluated on unseen-demo
windows; post-LN embeddings = LN treated as part of the encoder for EO/EP):
  action-decodability : embedding              -> action chunk (H*A)
  forward-predictability: [embedding, chunk]   -> next physical state (eef+grip+object)

Produces repr_analysis/probe_generalize.{json,png}. PNG = 2 subplots
(action-decodability | forward-predictability), each grouped bars of the 3
encoders across the 4 tasks.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.repr_extract as X
from scripts.repr_probe2 import lin_r2

TASKS = ["can", "square", "transport", "tool_hang"]
SETTINGS = ["DP", "EO", "EP"]
LABELS = {"DP": "vanilla DP", "EO": "joint (expert)", "EP": "joint (expert+play)"}


def sample_fwd(path, To, H, cap, seed, demo_names, include_object=False):
    """Window around CURRENT frame t: obs = frames [t-To+1 .. t]; action chunk =
    actions[t : t+H] (the H actions executed from t); forward target = state at
    t+H (s_{t+H}, after the full chunk). If include_object, the hidden object pose
    is appended to the forward target."""
    rng = np.random.RandomState(seed)
    with h5py.File(path, "r") as f:
        demos = [d for d in demo_names if f["data"][d]["actions"].shape[0] >= To + H]
        has_obj = include_object and ("object" in f["data"][demos[0]]["obs"])
        picks = []
        for _ in range(cap):
            dm = demos[rng.randint(len(demos))]
            T = f["data"][dm]["actions"].shape[0]
            t = rng.randint(To - 1, T - H)            # current frame; t+H <= T-1
            picks.append((dm, t))
        img = {k: [] for k in X.RGB}; low = {k: [] for k in X.LOWDIM}
        act, nxt = [], []
        for dm, t in picks:
            g = f["data"][dm]["obs"]
            for k in X.RGB:
                img[k].append(g[k][t - To + 1:t + 1])          # To frames ending at t
            for k in X.LOWDIM:
                low[k].append(g[k][t - To + 1:t + 1].astype(np.float32))
            st = [g[k][t + H].astype(np.float32) for k in X.LOWDIM]   # s_{t+H}: eef+grip
            if has_obj:
                st.append(g["object"][t + H].astype(np.float32))      # + hidden object pose
            nxt.append(np.concatenate(st))
            act.append(f["data"][dm]["actions"][t:t + H].astype(np.float32))  # a_t..a_{t+H-1}
    N = len(act)
    data = {"img": {k: np.stack(img[k]) for k in X.RGB},
            "low": {k: np.stack(low[k]) for k in X.LOWDIM},
            "goal_img": {k: np.stack(img[k])[:, :1] for k in X.RGB},
            "goal_low": {k: np.stack(low[k])[:, :1] for k in X.LOWDIM},
            "cur": np.zeros((N, 1), np.float32)}     # dummy: X.encode reads cur only for N
    return data, np.stack(act).astype(np.float32), np.stack(nxt).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-object", action="store_true",
                    help="include hidden object pose in the forward target")
    obj = ap.parse_args().with_object
    suffix = "_object" if obj else ""
    res = {m: {s: [] for s in SETTINGS} for m in ["action_decode", "forward_pred"]}
    for task in TASKS:
        cfg = X._cfg(task, "EP")
        sm = cfg.task.shape_meta.obs
        X.RGB = [k for k, v in sm.items() if v.get("type") == "rgb"]
        X.LOWDIM = [k for k, v in sm.items() if v.get("type", "low_dim") == "low_dim"]
        exp_p = cfg.task.dataset_paths[0]
        To, H = int(cfg.task.obs_steps), int(cfg.task.horizon)
        val_pct = float(cfg.task.val_dataset_percentage)
        N_exp = X.fit_lowdim_normalizers(exp_p)
        with h5py.File(exp_p, "r") as f:
            total = len(f["data"])
        tc = total - int(total * val_pct)
        seen = [f"demo_{i}" for i in range(tc)]
        unseen = [f"demo_{i}" for i in range(tc, total)]

        s_data, s_act, s_nxt = sample_fwd(exp_p, To, H, X.CAP, 0, seen, obj)
        u_data, u_act, u_nxt = sample_fwd(exp_p, To, H, X.CAP, 1, unseen, obj)
        s_chunk = s_act.reshape(len(s_act), -1); u_chunk = u_act.reshape(len(u_act), -1)

        for s in SETTINGS:
            e, ln = X.build_encoder(task, s)
            es = X.encode(e, ln, s_data, N_exp, To)[1]     # post-LN, seen
            eu = X.encode(e, ln, u_data, N_exp, To)[1]     # post-LN, unseen
            del e, ln; torch.cuda.empty_cache()
            # action-decodability: emb -> action chunk  (fit seen, eval unseen)
            res["action_decode"][s].append(lin_r2(es, s_chunk, eu, u_chunk))
            # forward-predictability: [emb, chunk] -> next physical state
            Xs = np.concatenate([es, s_chunk], 1); Xu = np.concatenate([eu, u_chunk], 1)
            res["forward_pred"][s].append(lin_r2(Xs, s_nxt, Xu, u_nxt))
        print(f"[{task}] tc={tc} unseen={len(unseen)} "
              f"act={[round(res['action_decode'][s][-1],3) for s in SETTINGS]} "
              f"fwd={[round(res['forward_pred'][s][-1],3) for s in SETTINGS]}")

    json.dump(res, open(f"{X.OUT_ROOT}/probe_generalize{suffix}.json", "w"), indent=2)

    # ---- bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fwd_t = "Forward-predictability ([emb, chunk] → next state" + (
        " incl. object)" if obj else ")")
    titles = {"action_decode": "Action-decodability (emb → action chunk)",
              "forward_pred": fwd_t}
    colors = {"DP": "#888888", "EO": "#4C72B0", "EP": "#C44E52"}
    x = np.arange(len(TASKS)); w = 0.26
    for ax, metric in zip(axes, ["action_decode", "forward_pred"]):
        for i, s in enumerate(SETTINGS):
            vals = res[metric][s]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=LABELS[s], color=colors[s])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(TASKS)
        ax.set_ylabel("R²  (fit on seen, eval on unseen)")
        ax.set_title(titles[metric]); ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Linear probing: seen→unseen expert generalization", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{X.OUT_ROOT}/probe_generalize{suffix}.png", dpi=150)
    print(f"\nsaved {X.OUT_ROOT}/probe_generalize{suffix}.json and .png")


if __name__ == "__main__":
    main()
