"""Does s2e=true hurt STATE IDENTIFIABILITY (map distinct states to similar embeddings)?

Distinct from decodability: a probe can regress an average target even under partial
collapse. Identifiability = are distinct physical states kept distinct in latent space.

On expert states (where the hidden OBJECT pose is available), for DP / EO (s2e=F) /
EP / EOs2e (s2e=T), using post-LN embeddings (LN = part of encoder):
  cur_decode   R^2 decoding the FULL current physical state (eef+gripper+object)
               from the embedding (lin+mlp). Lower => identity info lost.
  nn_ratio     median[ phys_dist(latent nearest-neighbor) / phys_dist(physical NN) ].
               1.0 = latent preserves local neighborhoods; >>1 = distinct states
               collapse to nearby embeddings (LOSS of identifiability).
  dist_rho     Spearman(latent pairwise dist, physical pairwise dist). Lower => worse.
  erank        effective rank (collapse magnitude).

Usage: python scripts/identifiability.py
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

import glob

import scripts.repr_extract as X
from scripts.repr_probe2 import lin_r2, mlp_r2, split
from scripts.repr_analyze import effective_rank
from scripts.s2e_compare import find_s2e_ckpt, build_s2e_encoder, VALPCT

TASKS = ["can", "square", "transport", "tool_hang"]


def find_eof0_ckpt(task):
    """seed-0 s2e=FALSE EO (seed-matched to the seed-0 s2e=true ckpt)."""
    pat = (f"logs/{task}_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_tsdiagonal_"
           f"ys1.0_ya1.0_ed256_d256_L8_h10_{VALPCT[task]}_s2efalse_elr1.0_esfnull_cs0.0_"
           f"ccadd_spx1_apvelocity_sw1.0_panfalse_pt/*/models/model_best.pt")
    c = sorted(glob.glob(pat))
    return c[-1] if c else None


def sample_expert_with_object(path, To, horizon, cap, seed, demo_names=None):
    rng = np.random.RandomState(seed)
    with h5py.File(path, "r") as f:
        pool = demo_names if demo_names is not None else list(f["data"].keys())
        demos = [d for d in pool
                 if f["data"][d]["actions"].shape[0] >= horizon + 1]
        has_obj = "object" in f["data"][demos[0]]["obs"]
        picks = []
        for _ in range(cap):
            dm = demos[rng.randint(len(demos))]
            T = f["data"][dm]["actions"].shape[0]
            picks.append((dm, rng.randint(0, T - horizon)))
        img = {k: [] for k in X.RGB}; low = {k: [] for k in X.LOWDIM}
        phys = []
        for dm, p in picks:
            g = f["data"][dm]["obs"]; cf = p + To - 1
            for k in X.RGB:
                img[k].append(g[k][p:p + To])
            for k in X.LOWDIM:
                low[k].append(g[k][p:p + To].astype(np.float32))
            st = [g[k][cf].astype(np.float32) for k in X.LOWDIM]
            if has_obj:
                st.append(g["object"][cf].astype(np.float32))
            phys.append(np.concatenate(st))
    data = {"img": {k: np.stack(img[k]) for k in X.RGB},
            "low": {k: np.stack(low[k]) for k in X.LOWDIM},
            "goal_img": {k: np.stack(img[k])[:, :1] for k in X.RGB},   # unused dummies
            "goal_low": {k: np.stack(low[k])[:, :1] for k in X.LOWDIM},
            "cur": np.stack(phys).astype(np.float32)}
    return data, np.stack(phys).astype(np.float32)


def nn_ratio(emb, phys, nq=600, seed=0):
    """median phys-dist to latent-NN / phys-dist to physical-NN."""
    e = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
    p = (phys - phys.mean(0)) / (phys.std(0) + 1e-6)
    rng = np.random.RandomState(seed)
    q = rng.choice(len(e), min(nq, len(e)), replace=False)
    te, tp = cKDTree(e), cKDTree(p)
    # latent NN (exclude self)
    _, li = te.query(e[q], k=2)
    lnn = li[:, 1]
    # physical NN floor
    _, pi = tp.query(p[q], k=2)
    pnn = pi[:, 1]
    phys_to_lnn = np.linalg.norm(p[q] - p[lnn], axis=1)
    phys_to_pnn = np.linalg.norm(p[q] - p[pnn], axis=1)
    return float(np.median(phys_to_lnn / (phys_to_pnn + 1e-9)))


def dist_rho(emb, phys, npair=8000, seed=0):
    e = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
    p = (phys - phys.mean(0)) / (phys.std(0) + 1e-6)
    rng = np.random.RandomState(seed)
    i = rng.randint(0, len(e), npair); j = rng.randint(0, len(e), npair)
    de = np.linalg.norm(e[i] - e[j], axis=1)
    dp = np.linalg.norm(p[i] - p[j], axis=1)
    ra = np.argsort(np.argsort(de)); rb = np.argsort(np.argsort(dp))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-9))


def main():
    out = []
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
        seen_demos = [f"demo_{i}" for i in range(tc)]
        unseen_demos = [f"demo_{i}" for i in range(tc, total)]

        encs = {}
        for s in ["DP", "EO", "EP"]:
            encs[s] = X.build_encoder(task, s)
        encs["EOf0"] = build_s2e_encoder(task, find_eof0_ckpt(task))   # seed0 s2e=FALSE
        encs["EOs2e"] = build_s2e_encoder(task, find_s2e_ckpt(task))   # seed0 s2e=TRUE

        for srcname, dlist in [("seen", seen_demos), ("unseen", unseen_demos)]:
            data, phys = sample_expert_with_object(exp_p, To, H, X.CAP, seed=0,
                                                   demo_names=dlist)
            rec = {"task": task, "src": srcname, "phys_dim": int(phys.shape[1])}
            n = len(phys); tr, va = split(n, 0)
            for s, (e, ln) in encs.items():
                emb = X.encode(e, ln, data, N_exp, To)[1]   # post-LN
                ez = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
                rec[f"{s}_cur_mlp"] = mlp_r2(ez[tr], phys[tr], ez[va], phys[va])
                rec[f"{s}_nn_ratio"] = nn_ratio(emb, phys)
                rec[f"{s}_dist_rho"] = dist_rho(emb, phys)
                rec[f"{s}_erank"] = effective_rank(emb)[0]
            out.append(rec)
        torch.cuda.empty_cache()
        print(f"[{task}] seen={len(seen_demos)} unseen={len(unseen_demos)} done")

    json.dump(out, open(f"{X.OUT_ROOT}/identifiability.json", "w"), indent=2)
    SET = ["DP", "EO", "EP", "EOf0", "EOs2e"]   # EOf0 vs EOs2e = seed-matched s2e F-vs-T
    for srcname in ["seen", "unseen"]:
        for metric, hint in [("cur_mlp", "state recover R2, higher=better"),
                             ("nn_ratio", "latent-NN collision, 1=ideal HIGHER=WORSE")]:
            print(f"\n### [{srcname}] {metric}  ({hint}) ###")
            print(f"{'task':<11}" + "".join(f"{s:>9}" for s in SET))
            for r in [x for x in out if x["src"] == srcname]:
                print(f"{r['task']:<11}" + "".join(f"{r[f'{s}_{metric}']:>9.3f}" for s in SET))
    print(f"\nsaved {X.OUT_ROOT}/identifiability.json")


if __name__ == "__main__":
    main()
