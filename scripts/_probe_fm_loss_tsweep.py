"""Sweep the flow-time t for the one-step FM-loss conditioning discriminator.

Δ(t) = L_FM(a | obs, expert; t) - L_FM(a | obs, null; t), per window, for action
and state streams. Encode each window once (ResNet is t-independent); reuse the
embeddings across all t with cheap trunk-only forwards. Report, per t, the
ROC-AUC of per-rollout Δ for separating good (high demo score) from bad rollouts.
"""
import os
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, "scripts")
from _probe_mixed_vs_expert import build_config, load_agent, MIXED_CKPT, DEVICE  # noqa: E402

ROLLOUT = "data/pusht/image_rollouts.hdf5"
T_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_NOISE = 8
K_PER_DEMO = 2
X1_EPS = 0.05
SEED = 0
CHUNK = 512


def gather(cfg):
    import h5py
    H, OB = cfg.task.horizon, cfg.task.obs_steps
    rng = np.random.default_rng(SEED)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    r = dict(oi=[], oa=[], gi=[], ga=[], ac=[], ds=[], demo=[])
    with h5py.File(ROLLOUT, "r", locking=False) as f:
        d = f["data"]; n = int(d.attrs["num_demos"])
        lens = np.array([int(d[f"demo_{i}"].attrs["num_samples"]) for i in range(n)])
        sc = np.array([float(d[f"demo_{i}"]["rewards"][:].max()) for i in range(n)])
        for di in range(n):
            if lens[di] <= H + OB:
                continue
            g = d[f"demo_{di}"]; im = g["obs"]["image"]; ap = g["obs"]["agent_pos"]; ac = g["actions"]; T = lens[di]
            for _ in range(K_PER_DEMO):
                t = int(rng.integers(0, T - H - 1))
                r["oi"].append(im[t:t+OB]); r["oa"].append(ap[t:t+OB])
                r["gi"].append(im[t+H:t+H+1]); r["ga"].append(ap[t+H:t+H+1])
                r["ac"].append(ac[t:t+H]); r["ds"].append(sc[di]); r["demo"].append(di)
    return r


def auc(score, label):
    order = np.argsort(score, kind="mergesort"); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score)+1)
    pos = label == 1; np_, nn = int(pos.sum()), int((~pos).sum())
    return float("nan") if np_ == 0 or nn == 0 else (ranks[pos].sum() - np_*(np_+1)/2)/(np_*nn)


def main():
    cfg = build_config(); cfg.optimization.device = DEVICE
    from mip.datasets.pusht_dataset import make_pusht_goal_dataset
    eds = make_pusht_goal_dataset(cfg.task, mode="train")
    nrm = (eds.datasets[0] if hasattr(eds, "datasets") else eds).normalizer
    r = gather(cfg); nwin = len(r["ac"])
    ni = lambda a: nrm["obs"]["image"].normalize(np.moveaxis(np.stack(a), -1, 2).astype(np.float32)/255.0)
    na = lambda a: nrm["obs"]["agent_pos"].normalize(np.stack(a).astype(np.float32))
    obs_i, obs_a = ni(r["oi"]), na(r["oa"]); g_i, g_a = ni(r["gi"]), na(r["ga"])
    a_pl = nrm["action"].normalize(np.stack(r["ac"]).astype(np.float32))
    ds = np.array(r["ds"]); demo = np.array(r["demo"])

    agent = load_agent(MIXED_CKPT, cfg); net = agent.net_ema
    enc, tln = agent._eval_encoder_modules(True)
    D = cfg.network.encoder_out_dim

    # encode each window ONCE (t-independent)
    z_all = torch.empty(nwin, cfg.task.obs_steps, D, device=DEVICE)
    tgt_all = torch.empty(nwin, 1, D, device=DEVICE)
    a_all = torch.tensor(a_pl, device=DEVICE)
    with torch.no_grad():
        for c0 in range(0, nwin, CHUNK):
            c1 = min(c0+CHUNK, nwin)
            obs = {"image": torch.tensor(obs_i[c0:c1], device=DEVICE), "agent_pos": torch.tensor(obs_a[c0:c1], device=DEVICE)}
            goal = {"image": torch.tensor(g_i[c0:c1], device=DEVICE), "agent_pos": torch.tensor(g_a[c0:c1], device=DEVICE)}
            z_all[c0:c1] = tln(enc(obs, None)); tgt_all[c0:c1] = tln(enc(goal, None))

    @torch.no_grad()
    def deltas(t):
        denom = max(1.0-t, X1_EPS)
        La_e=La_n=Ls_e=Ls_n=None
        out = {k: np.zeros(nwin) for k in ("La_e","La_n","Ls_e","Ls_n")}
        for c0 in range(0, nwin, CHUNK):
            c1=min(c0+CHUNK,nwin); B=c1-c0
            z=z_all[c0:c1]; tgt=tgt_all[c0:c1]; ap=a_all[c0:c1]
            tb=torch.full((B,), t, device=DEVICE)
            for oidx,(le,ls) in [(agent.net.EXPERT_IDX,("La_e","Ls_e")),(agent.net.NULL_IDX,("La_n","Ls_n"))]:
                opt=torch.full((B,), oidx, device=DEVICE, dtype=torch.long)
                gen=torch.Generator(device=DEVICE).manual_seed(0)
                La=torch.zeros(B,device=DEVICE); Ls=torch.zeros(B,device=DEVICE)
                for _ in range(N_NOISE):
                    an=torch.randn(ap.shape,generator=gen,device=DEVICE); sn=torch.randn(tgt.shape,generator=gen,device=DEVICE)
                    a_t=(1-t)*an+t*ap; s_t=(1-t)*sn+t*tgt
                    s_head,a_head,_=net(s_t,a_t,tb,tb,z,opt)
                    La+=(a_head-(ap-an)).pow(2).mean((1,2)); Ls+=((s_head-tgt)/denom).pow(2).mean((1,2))
                out[le][c0:c1]=(La/N_NOISE).cpu().numpy(); out[ls][c0:c1]=(Ls/N_NOISE).cpu().numpy()
        return out["La_e"]-out["La_n"], out["Ls_e"]-out["Ls_n"]

    udemo=np.unique(demo); sdemo=np.array([ds[demo==u][0] for u in udemo])
    def perdemo(x): return np.array([x[demo==u].mean() for u in udemo])
    print(f"windows={nwin}, demos={len(udemo)}, n_noise={N_NOISE}")
    print(f"\n{'t':>5s} | {'AUC dA@.95':>11s}{'AUC dS@.95':>11s}{'AUC comb@.95':>13s} | {'AUC dS@.5':>10s}  corr(score,dS)")
    best=(-1,None)
    for t in T_LIST:
        dA,dS=deltas(t); dAd,dSd=perdemo(dA),perdemo(dS)
        lab95=(sdemo>=0.95).astype(int); lab50=(sdemo>=0.5).astype(int)
        comb=-(dAd/(dAd.std()+1e-9)+dSd/(dSd.std()+1e-9))
        aA,aS,aC=auc(-dAd,lab95),auc(-dSd,lab95),auc(comb,lab95)
        aS50=auc(-dSd,lab50); cor=np.corrcoef(sdemo,dSd)[0,1]
        if aS>best[0]: best=(aS,t)
        print(f"{t:>5.1f} | {aA:>11.3f}{aS:>11.3f}{aC:>13.3f} | {aS50:>10.3f}  {cor:+.3f}")
    print(f"\nbest AUC(dS, good>=0.95) = {best[0]:.3f} at t={best[1]}")


if __name__ == "__main__":
    main()
