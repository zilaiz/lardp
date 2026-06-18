"""One-step FM-loss conditioning probe + good/bad-rollout separability.

For each rollout window we score the RECORDED action chunk and the LN'd goal
embedding directly under the model's training loss at a FIXED flow time t
(hparam, default 0.6), under expert- and null-conditioning. No multi-step
sampling -> a clean likelihood-style fit, averaged over a few noise draws.

  action (velocity param): a_t = (1-t)noise + t*a;  loss = MSE(a_head, a - noise)
  state  (x1 param):       s_t = (1-t)noise + t*tgt; v=(s_head-s_t)/max(1-t,eps),
                           loss = MSE(v - v_gt) = MSE((s_head-tgt)/denom)

Delta = L_expert - L_null (per window). Negative => expert-cond fits better
(=> more expert-like / "good"). We bin by demo score and test whether a
threshold on Delta separates good from bad rollouts (ROC-AUC + Youden).
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
T_FLOW = 0.6        # <-- the flow-time hyperparameter
N_NOISE = 8         # noise draws averaged at the fixed t
K_PER_DEMO = 2      # windows sampled per demo (for per-rollout aggregation)
X1_EPS = 0.05
SEED = 0
CHUNK = 512


def gather_windows(cfg):
    import h5py
    H, OB = cfg.task.horizon, cfg.task.obs_steps
    rng = np.random.default_rng(SEED)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    rec = dict(oi=[], oa=[], gi=[], ga=[], ac=[], pr=[], ds=[], demo=[])
    with h5py.File(ROLLOUT, "r", locking=False) as f:
        d = f["data"]; n = int(d.attrs["num_demos"])
        lens = np.array([int(d[f"demo_{i}"].attrs["num_samples"]) for i in range(n)])
        scores = np.array([float(d[f"demo_{i}"]["rewards"][:].max()) for i in range(n)])
        for di in range(n):
            if lens[di] <= H + OB:
                continue
            g = d[f"demo_{di}"]; im = g["obs"]["image"]; ap = g["obs"]["agent_pos"]
            ac = g["actions"]; rw = g["rewards"]; T = lens[di]
            for _ in range(K_PER_DEMO):
                t = int(rng.integers(0, T - H - 1))
                rec["oi"].append(im[t:t+OB]); rec["oa"].append(ap[t:t+OB])
                rec["gi"].append(im[t+H:t+H+1]); rec["ga"].append(ap[t+H:t+H+1])
                rec["ac"].append(ac[t:t+H]); rec["pr"].append(float(rw[t+OB-1]))
                rec["ds"].append(scores[di]); rec["demo"].append(di)
    return rec


@torch.no_grad()
def fm_losses(agent, obs, goal, a_play, t, opt_idx, gen):
    """Per-window action+state FM loss at fixed t, averaged over N_NOISE draws."""
    net = agent.net_ema
    enc, tln = agent._eval_encoder_modules(True)
    z = tln(enc(obs, None))                 # condition (encode once)
    tgt = tln(enc(goal, None))              # state target x1
    B = a_play.shape[0]
    tb = torch.full((B,), t, device=DEVICE)
    denom = max(1.0 - t, X1_EPS)
    La = torch.zeros(B, device=DEVICE); Ls = torch.zeros(B, device=DEVICE)
    for _ in range(N_NOISE):
        an = torch.randn(a_play.shape, generator=gen, device=DEVICE)
        sn = torch.randn(tgt.shape, generator=gen, device=DEVICE)
        a_t = (1 - t) * an + t * a_play
        s_t = (1 - t) * sn + t * tgt
        s_head, a_head, _ = net(s_t, a_t, tb, tb, z, opt_idx)
        La += (a_head - (a_play - an)).pow(2).mean((1, 2))            # velocity param
        Ls += ((s_head - tgt) / denom).pow(2).mean((1, 2))            # x1 param -> v form
    return (La / N_NOISE).cpu().numpy(), (Ls / N_NOISE).cpu().numpy()


def auc(score, label):  # score higher => more likely positive(label==1)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = label == 1; npos = int(pos.sum()); nneg = int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def best_threshold(delta, label):
    # predict good (label 1) if delta < tau; scan candidate taus
    taus = np.quantile(delta, np.linspace(0.01, 0.99, 99))
    best = (-1, None, None, None)
    for tau in taus:
        pred = delta < tau
        tp = np.sum(pred & (label == 1)); fp = np.sum(pred & (label == 0))
        fn = np.sum(~pred & (label == 1)); tn = np.sum(~pred & (label == 0))
        tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best[0]:
            best = (j, tau, tpr, fpr)
    return best


def main():
    cfg = build_config(); cfg.optimization.device = DEVICE
    nrm = None
    from mip.datasets.pusht_dataset import make_pusht_goal_dataset
    eds = make_pusht_goal_dataset(cfg.task, mode="train")
    nrm = (eds.datasets[0] if hasattr(eds, "datasets") else eds).normalizer

    rec = gather_windows(cfg)
    nwin = len(rec["ac"])
    def ni(arr): x = np.moveaxis(np.stack(arr), -1, 2).astype(np.float32) / 255.0; return nrm["obs"]["image"].normalize(x)
    def na(arr): return nrm["obs"]["agent_pos"].normalize(np.stack(arr).astype(np.float32))
    obs_i = ni(rec["oi"]); obs_a = na(rec["oa"]); g_i = ni(rec["gi"]); g_a = na(rec["ga"])
    a_pl = nrm["action"].normalize(np.stack(rec["ac"]).astype(np.float32))
    pr = np.array(rec["pr"]); ds = np.array(rec["ds"]); demo = np.array(rec["demo"])

    agent = load_agent(MIXED_CKPT, cfg)
    La_e = np.zeros(nwin); La_n = np.zeros(nwin); Ls_e = np.zeros(nwin); Ls_n = np.zeros(nwin)
    for c0 in range(0, nwin, CHUNK):
        c1 = min(c0 + CHUNK, nwin)
        obs = {"image": torch.tensor(obs_i[c0:c1], device=DEVICE), "agent_pos": torch.tensor(obs_a[c0:c1], device=DEVICE)}
        goal = {"image": torch.tensor(g_i[c0:c1], device=DEVICE), "agent_pos": torch.tensor(g_a[c0:c1], device=DEVICE)}
        ap = torch.tensor(a_pl[c0:c1], device=DEVICE)
        eidx = torch.full((c1 - c0,), agent.net.EXPERT_IDX, device=DEVICE, dtype=torch.long)
        nidx = torch.full((c1 - c0,), agent.net.NULL_IDX, device=DEVICE, dtype=torch.long)
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        La_e[c0:c1], Ls_e[c0:c1] = fm_losses(agent, obs, goal, ap, T_FLOW, eidx, gen)
        gen = torch.Generator(device=DEVICE).manual_seed(0)  # same noise across cond
        La_n[c0:c1], Ls_n[c0:c1] = fm_losses(agent, obs, goal, ap, T_FLOW, nidx, gen)

    dA = La_e - La_n; dS = Ls_e - Ls_n
    print(f"\nt={T_FLOW}, n_noise={N_NOISE}, windows={nwin}\n")

    # binned by demo score
    order = np.argsort(ds); q = np.array_split(order, 5)
    print(f"{'demo score':>12s}{'n':>5s} | {'La:exp':>8s}{'La:null':>8s}{'A_win':>7s} | {'Ls:exp':>8s}{'Ls:null':>8s}{'S_win':>7s}")
    for b in q:
        ae, an = La_e[b].mean(), La_n[b].mean(); se, sn = Ls_e[b].mean(), Ls_n[b].mean()
        print(f"{ds[b].min():.2f}-{ds[b].max():.2f}{len(b):>7d} | {ae:>8.4f}{an:>8.4f}{('EXP' if ae<an else 'play'):>7s} | {se:>8.4f}{sn:>8.4f}{('EXP' if se<sn else 'play'):>7s}")
    print(f"  corr(score, dA)={np.corrcoef(ds,dA)[0,1]:+.3f}  corr(score, dS)={np.corrcoef(ds,dS)[0,1]:+.3f}")

    # per-demo aggregation -> good/bad classification
    udemo = np.unique(demo)
    dA_demo = np.array([dA[demo == u].mean() for u in udemo])
    dS_demo = np.array([dS[demo == u].mean() for u in udemo])
    score_demo = np.array([ds[demo == u][0] for u in udemo])
    print(f"\nper-rollout separability ({len(udemo)} demos):")
    for thr in [0.95, 0.7, 0.5]:
        lab = (score_demo >= thr).astype(int)
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        aA = auc(-dA_demo, lab); aS = auc(-dS_demo, lab)
        comb = -(dA_demo / (dA_demo.std() + 1e-9) + dS_demo / (dS_demo.std() + 1e-9))
        aC = auc(comb, lab)
        jA, tauA, tprA, fprA = best_threshold(dA_demo, lab)
        jS, tauS, tprS, fprS = best_threshold(dS_demo, lab)
        print(f"  good = demo_score>={thr} ({lab.sum()}/{len(lab)}): "
              f"AUC dA={aA:.3f} dS={aS:.3f} comb={aC:.3f} | "
              f"best dS thr: dS<{tauS:+.4f} -> TPR={tprS:.2f} FPR={fprS:.2f} (J={jS:.2f})")


if __name__ == "__main__":
    main()
