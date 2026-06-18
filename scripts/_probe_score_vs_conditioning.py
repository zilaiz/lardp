"""Does conditioning-vs-error depend on a rollout's score?

Hypothesis: high-score (expert-like) play transitions are predicted better under
EXPERT conditioning; low-score (junk) ones better under PLAY(null) conditioning.

For many rollout windows spanning the score range we predict the action chunk
and the next-state embedding under expert- and null-conditioning (same noise),
measure MSE to the recorded play action / LN'd goal embedding, and bin by:
  (a) per-step reward (coverage at the current frame), and
  (b) demo peak score (whole-rollout success).
We report, per bin, mean action/state MSE under each conditioning and the
crossover.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, "scripts")
from _probe_mixed_vs_expert import (  # noqa: E402
    build_config, load_agent, sample_with_opt, MIXED_CKPT, NUM_STEPS, DEVICE,
)

ROLLOUT = "data/pusht/image_rollouts.hdf5"
N_WIN = 640          # windows sampled across demos/timesteps
SEED = 0


def main():
    import h5py
    from mip.datasets.pusht_dataset import make_pusht_goal_dataset
    cfg = build_config(); cfg.optimization.device = DEVICE
    H, OB = cfg.task.horizon, cfg.task.obs_steps

    exp_ds = make_pusht_goal_dataset(cfg.task, mode="train")
    if hasattr(exp_ds, "datasets"):
        exp_ds = exp_ds.datasets[0]
    nrm = exp_ds.normalizer

    rng = np.random.default_rng(SEED)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # gather windows: (img_obs, ap_obs, img_goal, ap_goal, action, per_step_r, demo_score)
    obs_img=[]; obs_ap=[]; g_img=[]; g_ap=[]; acts=[]; pstep=[]; dscore=[]
    with h5py.File(ROLLOUT, "r", locking=False) as f:
        d=f["data"]; n=int(d.attrs["num_demos"])
        # precompute demo lengths & scores
        lens=np.array([int(d[f"demo_{i}"].attrs["num_samples"]) for i in range(n)])
        scores=np.array([float(d[f"demo_{i}"]["rewards"][:].max()) for i in range(n)])
        valid=[i for i in range(n) if lens[i] > H + OB]
        for _ in range(N_WIN):
            di=int(rng.choice(valid)); T=lens[di]
            t=int(rng.integers(0, T - H - 1))
            g=d[f"demo_{di}"]; im=g["obs"]["image"]; ap=g["obs"]["agent_pos"]; ac=g["actions"]; rw=g["rewards"]
            obs_img.append(im[t:t+OB]); obs_ap.append(ap[t:t+OB])
            g_img.append(im[t+H:t+H+1]); g_ap.append(ap[t+H:t+H+1])
            acts.append(ac[t:t+H])
            pstep.append(float(rw[t+OB-1])); dscore.append(scores[di])

    def norm_img(arr):  # (B,To,H,W,C)uint8 -> (B,To,C,H,W) in [-1,1]
        x=np.moveaxis(np.stack(arr),-1,2).astype(np.float32)/255.0
        return torch.tensor(nrm["obs"]["image"].normalize(x), device=DEVICE)
    def norm_ap(arr):
        x=np.stack(arr).astype(np.float32)
        return torch.tensor(nrm["obs"]["agent_pos"].normalize(x), device=DEVICE)
    obs={"image":norm_img(obs_img),"agent_pos":norm_ap(obs_ap)}
    goal={"image":norm_img(g_img),"agent_pos":norm_ap(g_ap)}
    a_play=torch.tensor(nrm["action"].normalize(np.stack(acts).astype(np.float32)), device=DEVICE)
    pstep=np.array(pstep); dscore=np.array(dscore)

    agent=load_agent(MIXED_CKPT, cfg)
    enc,tln=agent._eval_encoder_modules(True)
    B=a_play.shape[0]; D=cfg.network.encoder_out_dim
    g=torch.Generator(device=DEVICE).manual_seed(0)
    x0=torch.randn(B,1,D,generator=g,device=DEVICE)
    a0=torch.randn(B,H,cfg.task.act_dim,generator=g,device=DEVICE)
    with torch.no_grad():
        z=tln(enc(obs,None)); s_tgt=tln(enc(goal,None))
        a_exp,s_exp=sample_with_opt(agent,z,x0,a0,agent.net.EXPERT_IDX,NUM_STEPS)
        a_nul,s_nul=sample_with_opt(agent,z,x0,a0,agent.net.NULL_IDX,NUM_STEPS)
    # per-window MSE (mean over dims)
    am_exp=(a_exp-a_play).pow(2).mean((1,2)).cpu().numpy()
    am_nul=(a_nul-a_play).pow(2).mean((1,2)).cpu().numpy()
    sm_exp=(s_exp-s_tgt).pow(2).mean((1,2)).cpu().numpy()
    sm_nul=(s_nul-s_tgt).pow(2).mean((1,2)).cpu().numpy()

    def report(score, label):
        order=np.argsort(score); q=np.array_split(order,5)
        print(f"\n=== binned by {label} (5 quantile bins, low->high) ===")
        print(f"{'score range':>16s}{'n':>5s} | {'act:exp':>8s}{'act:null':>9s}{'a_better':>9s} | {'st:exp':>8s}{'st:null':>9s}{'s_better':>9s}")
        for b in q:
            sr=f"{score[b].min():.2f}-{score[b].max():.2f}"
            ae,an=am_exp[b].mean(),am_nul[b].mean()
            se,sn=sm_exp[b].mean(),sm_nul[b].mean()
            ab="EXPERT" if ae<an else "play"
            sb="EXPERT" if se<sn else "play"
            print(f"{sr:>16s}{len(b):>5d} | {ae:>8.4f}{an:>9.4f}{ab:>9s} | {se:>8.4f}{sn:>9.4f}{sb:>9s}")
        # correlation of (exp - null) error with score: negative => expert better at high score
        from numpy import corrcoef
        print(f"  corr(score, act_MSE_exp - act_MSE_null) = {corrcoef(score, am_exp-am_nul)[0,1]:+.3f}  "
              f"(neg => expert-cond relatively better as score rises)")
        print(f"  corr(score, st_MSE_exp  - st_MSE_null)  = {corrcoef(score, sm_exp-sm_nul)[0,1]:+.3f}")

    print(f"sampled {B} rollout windows | action scale mean|a|={a_play.abs().mean():.3f}")
    report(pstep, "per-step reward (current-frame coverage)")
    report(dscore, "demo peak score (whole-rollout success)")


if __name__ == "__main__":
    main()
