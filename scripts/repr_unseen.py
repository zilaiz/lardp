"""Generalization test: probe on COMPLETE UNSEEN expert demos.

The encoders trained on only the first `train_count` expert demos
(train_count = total - int(total*val_pct); can=20/200, tool_hang=40%, ...).
The rest are held out and NEVER seen. We probe decision-relevant decodability on
SEEN (train) vs UNSEEN (held-out) expert states — both have optimal labels and
lie on the expert manifold. If a joint encoder generalizes better from few demos
(small SEEN->UNSEEN drop, higher unseen decodability), that's a clean reason
joint_pt > DP. Linear + MLP, each encoder fed its own normalizer.

Usage: python scripts/repr_unseen.py --task can
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

import scripts.repr_extract as X
from scripts.repr_probe2 import lin_r2, mlp_r2, split


def sample_subset(path, demo_names, To, horizon, cap, seed):
    rng = np.random.RandomState(seed)
    with h5py.File(path, "r") as f:
        lens = {dm: f["data"][dm]["actions"].shape[0] for dm in demo_names
                if f["data"][dm]["actions"].shape[0] >= horizon + 1}
        names = list(lens.keys())
        picks = [(names[rng.randint(len(names))], None) for _ in range(cap)]
        picks = [(dm, rng.randint(0, lens[dm] - horizon)) for dm, _ in picks]
        img = {k: [] for k in X.RGB}; gimg = {k: [] for k in X.RGB}
        low = {k: [] for k in X.LOWDIM}; glow = {k: [] for k in X.LOWDIM}
        cur, nxt, act = [], [], []
        for dm, p in picks:
            g = f["data"][dm]["obs"]
            for k in X.RGB:
                img[k].append(g[k][p:p + To]); gimg[k].append(g[k][p + horizon][None])
            for k in X.LOWDIM:
                low[k].append(g[k][p:p + To].astype(np.float32))
                glow[k].append(g[k][p + horizon][None].astype(np.float32))
            cur.append(np.concatenate([g[k][p + To - 1] for k in X.LOWDIM]))
            nxt.append(np.concatenate([g[k][p + horizon] for k in X.LOWDIM]))
            act.append(f["data"][dm]["actions"][p:p + horizon].astype(np.float32))
    return {
        "img": {k: np.stack(img[k]) for k in X.RGB},
        "goal_img": {k: np.stack(gimg[k]) for k in X.RGB},
        "low": {k: np.stack(low[k]) for k in X.LOWDIM},
        "goal_low": {k: np.stack(glow[k]) for k in X.LOWDIM},
        "cur": np.stack(cur).astype(np.float32), "nxt": np.stack(nxt).astype(np.float32),
        "act": np.stack(act).astype(np.float32),
    }


def probe_set(emb, data):
    n = len(emb); tr, va = split(n, 3)
    chunk = data["act"].reshape(n, -1)
    out = {}
    out["actchunk_lin"] = lin_r2(emb[tr], chunk[tr], emb[va], chunk[va])
    out["actchunk_mlp"] = mlp_r2(emb[tr], chunk[tr], emb[va], chunk[va])
    out["nxt_lin"] = lin_r2(emb[tr], data["nxt"][tr], emb[va], data["nxt"][va])
    out["nxt_mlp"] = mlp_r2(emb[tr], data["nxt"][tr], emb[va], data["nxt"][va])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True)
    task = ap.parse_args().task
    cfg = X._cfg(task, "EP")
    sm = cfg.task.shape_meta.obs
    X.RGB = [k for k, v in sm.items() if v.get("type") == "rgb"]
    X.LOWDIM = [k for k, v in sm.items() if v.get("type", "low_dim") == "low_dim"]
    exp_p, roll_p = cfg.task.dataset_paths[0], cfg.task.dataset_paths[1]
    To, H = int(cfg.task.obs_steps), int(cfg.task.horizon)
    val_pct = float(cfg.task.val_dataset_percentage)

    with h5py.File(exp_p, "r") as f:
        total = len(f["data"])
    train_count = total - int(total * val_pct)
    seen = [f"demo_{i}" for i in range(train_count)]
    unseen = [f"demo_{i}" for i in range(train_count, total)]
    print(f"[{task}] total={total} train_count={train_count} "
          f"({100*train_count/total:.0f}% seen) unseen={len(unseen)} demos")

    N_exp = X.fit_lowdim_normalizers(exp_p)
    N_mix = X.merge_lowdim(N_exp, X.fit_lowdim_normalizers(roll_p))
    norm_of = {"DP": N_exp, "EO": N_exp, "EP": N_mix}

    seen_data = sample_subset(exp_p, seen, To, H, min(1500, train_count * 60), seed=10)
    unseen_data = sample_subset(exp_p, unseen, To, H, 2500, seed=11)

    out = {"task": task, "train_count": train_count, "total": total, "settings": {}}
    for s in ["DP", "EO", "EP"]:
        enc, ln = X.build_encoder(task, s)
        es = X.encode(enc, ln, seen_data, norm_of[s], To)[0]
        eu = X.encode(enc, ln, unseen_data, norm_of[s], To)[0]
        del enc, ln
        import torch; torch.cuda.empty_cache()
        out["settings"][s] = {"seen": probe_set(es, seen_data),
                              "unseen": probe_set(eu, unseen_data)}

    json.dump(out, open(f"{X.OUT_ROOT}/{task}/unseen.json", "w"), indent=2)
    print(f"\n===== {task}  SEEN vs UNSEEN expert decodability (mlp R^2; gap=seen-unseen) =====")
    print(f"{'probe':<14}{'DP seen/uns/gap':>22}{'EO seen/uns/gap':>22}{'EP seen/uns/gap':>22}")
    for p in ["actchunk_mlp", "nxt_mlp"]:
        row = f"{p:<14}"
        for s in ["DP", "EO", "EP"]:
            sv = out["settings"][s]["seen"][p]; uv = out["settings"][s]["unseen"][p]
            row += f"{sv:>7.2f}/{uv:.2f}/{sv-uv:+.2f}   "
        print(row)
    print(f"saved {X.OUT_ROOT}/{task}/unseen.json")


if __name__ == "__main__":
    main()
