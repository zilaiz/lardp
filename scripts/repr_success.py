"""Why does joint_pt (EO/EP) beat diffusion policy (DP)? Offline tests on
SUCCESS-aware rollout data.

The rollout hdf5 is a mix (~72% of episodes eventually succeed). We use the
reward to split it and run decision-honest probes:

  (1) recovery-action decode  -- on SUCCESSFUL pre-success states (actions there
      are good recovery behaviors): decode action chunk + physical next-state.
      For EO-vs-DP this is CLEAN: neither trained on rollout, labels are good,
      states are off-manifold. A joint_pt win here = genuine off-manifold
      decision-representation advantage.
  (2) value / recoverability  -- predict eventual SUCCESS (AUC) and time-to-
      success (R^2) from the embedding, over all rollout states. A rep that
      linearly separates recoverable from doomed states is decision-useful.

linear (ridge) + MLP for every probe. Each encoder fed its own normalizer.

Usage: python scripts/repr_success.py --task can
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

import scripts.repr_extract as X
from scripts.repr_probe2 import lin_r2, mlp_r2, split

CAP = 4000


def _auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def lin_auc(Xtr, ytr, Xva, yva):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    d = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(d), Xtr.T @ (ytr - 0.5))
    return _auc(yva, Xva @ w)


def mlp_auc(Xtr, ytr, Xva, yva, epochs=300, h=256):
    import torch
    import torch.nn as nn
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=X.DEVICE)
    yt = torch.tensor(ytr[:, None], dtype=torch.float32, device=X.DEVICE)
    xv = torch.tensor((Xva - mu) / sd, dtype=torch.float32, device=X.DEVICE)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], h), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)).to(X.DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best = 0.5
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        lossf(net(xt), yt).backward(); opt.step()
        if ep % 15 == 0 or ep == epochs - 1:
            net.eval()
            import torch as T
            with T.no_grad():
                best = max(best, _auc(yva, net(xv).cpu().numpy().ravel()))
    return best


def sample_rollout(path, To, horizon, cap, seed):
    rng = np.random.RandomState(seed)
    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())
        meta = {}
        for dm in demos:
            r = f["data"][dm]["rewards"][:]
            succ = bool(r.max() > 0)
            fs = int(np.argmax(r > 0)) if succ else 10**9
            meta[dm] = (f["data"][dm]["actions"].shape[0], succ, fs)
        picks = []
        for _ in range(cap):
            dm = demos[rng.randint(len(demos))]
            T, _, _ = meta[dm]
            if T < horizon + 1:
                continue
            p = rng.randint(0, T - horizon)
            picks.append((dm, p))
        img = {k: [] for k in X.RGB}; gimg = {k: [] for k in X.RGB}
        low = {k: [] for k in X.LOWDIM}; glow = {k: [] for k in X.LOWDIM}
        cur, nxt, act, succ, tts, pre = [], [], [], [], [], []
        for dm, p in picks:
            g = f["data"][dm]["obs"]; T, s, fs = meta[dm]
            cf = p + To - 1
            for k in X.RGB:
                img[k].append(g[k][p:p + To]); gimg[k].append(g[k][p + horizon][None])
            for k in X.LOWDIM:
                low[k].append(g[k][p:p + To].astype(np.float32))
                glow[k].append(g[k][p + horizon][None].astype(np.float32))
            cur.append(np.concatenate([g[k][cf] for k in X.LOWDIM]))
            nxt.append(np.concatenate([g[k][p + horizon] for k in X.LOWDIM]))
            act.append(f["data"][dm]["actions"][p:p + horizon].astype(np.float32))
            succ.append(s); tts.append(min(fs - cf, 500) if s else -1)
            pre.append(s and cf < fs)
    return {
        "img": {k: np.stack(img[k]) for k in X.RGB},
        "goal_img": {k: np.stack(gimg[k]) for k in X.RGB},
        "low": {k: np.stack(low[k]) for k in X.LOWDIM},
        "goal_low": {k: np.stack(glow[k]) for k in X.LOWDIM},
        "cur": np.stack(cur).astype(np.float32), "nxt": np.stack(nxt).astype(np.float32),
        "act": np.stack(act).astype(np.float32),
        "succ": np.array(succ), "tts": np.array(tts, np.float32), "pre": np.array(pre),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True)
    task = ap.parse_args().task
    cfg = X._cfg(task, "EP")
    sm = cfg.task.shape_meta.obs
    X.RGB = [k for k, v in sm.items() if v.get("type") == "rgb"]
    X.LOWDIM = [k for k, v in sm.items() if v.get("type", "low_dim") == "low_dim"]
    exp_p, roll_p = cfg.task.dataset_paths[0], cfg.task.dataset_paths[1]
    To, H = int(cfg.task.obs_steps), int(cfg.task.horizon)

    N_exp = X.fit_lowdim_normalizers(exp_p)
    N_mix = X.merge_lowdim(N_exp, X.fit_lowdim_normalizers(roll_p))
    norm_of = {"DP": N_exp, "EO": N_exp, "EP": N_mix}
    R = sample_rollout(roll_p, To, H, CAP, seed=2)
    print(f"[{task}] rollout windows={len(R['succ'])} succ={R['succ'].mean():.2f} "
          f"pre-success={R['pre'].mean():.2f}")

    out = {"task": task, "frac_succ": float(R["succ"].mean()), "settings": {}}
    for s in ["DP", "EO", "EP"]:
        enc, ln = X.build_encoder(task, s)
        emb = X.encode(enc, ln, R, norm_of[s], To)[0]   # (N,256) raw last-frame
        del enc, ln
        r = {}
        # (2) value / recoverability over ALL rollout states
        y = R["succ"].astype(float)
        tr, va = split(len(emb), 7)
        r["succ_auc_lin"] = lin_auc(emb[tr], y[tr], emb[va], y[va])
        r["succ_auc_mlp"] = mlp_auc(emb[tr], y[tr], emb[va], y[va])
        # (1) on SUCCESSFUL pre-success states only
        m = R["pre"]
        em, am, nm, cm, tm = emb[m], R["act"][m], R["nxt"][m], R["cur"][m], R["tts"][m]
        n = len(em); ptr, pva = split(n, 8)
        chunk = am.reshape(n, -1)
        r["recov_actchunk_lin"] = lin_r2(em[ptr], chunk[ptr], em[pva], chunk[pva])
        r["recov_actchunk_mlp"] = mlp_r2(em[ptr], chunk[ptr], em[pva], chunk[pva])
        r["recov_nxt_lin"] = lin_r2(em[ptr], nm[ptr], em[pva], nm[pva])
        r["recov_nxt_mlp"] = mlp_r2(em[ptr], nm[ptr], em[pva], nm[pva])
        r["tts_r2_lin"] = lin_r2(em[ptr], tm[ptr, None], em[pva], tm[pva, None])
        r["tts_r2_mlp"] = mlp_r2(em[ptr], tm[ptr, None], em[pva], tm[pva, None])
        out["settings"][s] = r
        import torch; torch.cuda.empty_cache()

    json.dump(out, open(f"{X.OUT_ROOT}/{task}/success.json", "w"), indent=2)
    print(f"\n===== {task}  success-aware probes (lin | mlp) =====")
    print(f"{'probe':<20}{'DP':>14}{'EO':>14}{'EP':>14}")
    for nm in ["succ_auc", "recov_actchunk", "recov_nxt", "tts_r2"]:
        row = f"{nm:<20}"
        for s in ["DP", "EO", "EP"]:
            lin = out["settings"][s][f"{nm}_lin"]; mlp = out["settings"][s][f"{nm}_mlp"]
            row += f"  {lin:>5.2f}|{mlp:>5.2f} "
        print(row)
    print(f"saved {X.OUT_ROOT}/{task}/success.json")


if __name__ == "__main__":
    main()
