"""Decisive tests for what s2e=true does to the representation.

(1) k-fold cross-demo held-out R^2 for THREE targets, per encoder:
      - action chunk            (action-relevance)
      - future proprio @ t+H    (forward-dynamics / state content -- what FDM trains)
      - current proprio @ t     (raw state content)
    Double dissociation prediction (if FDM turns it into a state encoder):
      s2e=true LOSES on action, WINS on future/current proprio.

(2) Latent-OOD on REAL rollout states: standardize z by expert stats, build expert
    PCA(30) whitening; report mean Mahalanobis^2 (in-subspace) + residual energy
    (out-of-subspace) for held-out EXPERT vs ROLLOUT z. Ratio rollout/expert is
    scale-invariant => how much further off-manifold each encoder pushes the same
    off-expert states.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import h5py, numpy as np
import scripts.viz_encoder_compare as V

NORM=str(ROOT/"viz/can_joint_pt_compare/normalizer.pkl")
EXP="data/robomimic/can/ph/image_v15.hdf5"; ROLL="data/robomimic/can/image_rollouts_clipped.hdf5"
LOG="logs/can_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_stduniform_atduniform_tsdiagonal_ys1.0_ya1.0_ed256_d256_L8_h10_0.9"
CKPTS={
 "s2efalse_all":  f"{LOG}_s2efalse_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_20/models/model_latest.pt",
 "s2etrue_all":   f"{LOG}_s2etrue_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_04/models/model_latest.pt",
 "s2efalse_play": f"{LOG}_s2efalse_schnoisier_play_elr1.0_esf0.8_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_24_56/models/model_latest.pt",
}
PROP=["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos"]

def r2(X,Y,Xv,Yv,lam=1e-1):
    mu=X.mean(0,keepdims=True); sd=np.clip(X.std(0,keepdims=True),1e-6,None)
    ty=Yv  # standardize targets per-dim by train stats
    muy=Y.mean(0,keepdims=True); sdy=np.clip(Y.std(0,keepdims=True),1e-6,None)
    Ys=(Y-muy)/sdy; Yvs=(Yv-muy)/sdy
    Xs=np.c_[(X-mu)/sd,np.ones((len(X),1))]; Xvs=np.c_[(Xv-mu)/sd,np.ones((len(Xv),1))]
    W=np.linalg.solve(Xs.T@Xs+lam*np.eye(Xs.shape[1]),Xs.T@Ys)
    res=((Xvs@W-Yvs)**2).sum(); tot=((Yvs-Ys.mean(0,keepdims=True))**2).sum()
    return 1-res/max(tot,1e-12)

def main():
    tc,nc=V._compose_cfg(ROOT,"examples/configs","can_ph_image","lbmdit_joint_pt")
    ik=[k for k,v in tc.shape_meta.obs.items() if v.type=="rgb"]; ld=[k for k,v in tc.shape_meta.obs.items() if v.type=="low_dim"]
    norm,_=V._load_normalizer(CKPTS["s2efalse_all"],NORM); H=int(tc.horizon)
    rng=np.random.default_rng(0)
    with h5py.File(EXP,"r") as f:
        ed=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        enames=[ed[i] for i in sorted(rng.choice(len(ed),40,replace=False).tolist())]
        earr={n:V._read_demo(f["data"][n],norm,ik,ld) for n in enames}
        eact={n:norm["action"].normalize(np.asarray(f["data"][n]["actions"]).astype(np.float32)) for n in enames}
    with h5py.File(ROLL,"r") as f:
        rd=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        rnames=[rd[i] for i in sorted(rng.choice(len(rd),25,replace=False).tolist())]
        rarr={n:V._read_demo(f["data"][n],norm,ik,ld) for n in rnames}

    def proprio(arrs,n,off=0):
        z=arrs[n]; T=next(iter(z.values())).shape[0]
        idx=np.minimum(np.arange(T)+off,T-1)
        return np.concatenate([arrs[n][k][idx] for k in PROP],1)

    print(f"{'encoder':<14} | {'R2 action':>10} {'R2 futProp':>11} {'R2 curProp':>11} | "
          f"{'expMah2':>8} {'rollMah2':>9} {'ratio':>6} | {'expResid':>9} {'rollResid':>10} {'ratio':>6}")
    for tag,ck in CKPTS.items():
        enc,_=V._load_inner_encoder(ck,nc,tc,"cuda",True,tag)
        Ze={n:V._encode_demo(enc,earr[n],"cuda",64) for n in enames}
        Zr={n:V._encode_demo(enc,rarr[n],"cuda",64) for n in rnames}
        # ---- (1) 5-fold cross-demo probes ----
        folds=np.array_split(np.array(enames),5)
        accs={"act":[],"fut":[],"cur":[]}
        def build(ns,kind):
            Xs,Ys=[],[]
            for n in ns:
                z=Ze[n]
                if kind=="act": y=V._build_action_chunks(eact[n],H)
                elif kind=="fut": y=proprio(earr,n,H)
                else: y=proprio(earr,n,0)
                m=min(len(z),len(y)); Xs.append(z[:m]); Ys.append(y[:m])
            return np.concatenate(Xs),np.concatenate(Ys)
        for i in range(5):
            va=list(folds[i]); tr=[n for n in enames if n not in va]
            for kind,key in [("act","act"),("fut","fut"),("cur","cur")]:
                Xt,Yt=build(tr,kind); Xv,Yv=build(va,kind)
                accs[key].append(r2(Xt,Yt,Xv,Yv))
        ra,rf,rc=np.mean(accs["act"]),np.mean(accs["fut"]),np.mean(accs["cur"])
        # ---- (2) latent OOD vs real rollout states ----
        Zall=np.concatenate([Ze[n] for n in enames],0)
        mu=Zall.mean(0,keepdims=True); sd=np.clip(Zall.std(0,keepdims=True),1e-6,None)
        Zs=(Zall-mu)/sd
        # split expert into fit/heldout for baseline
        nfit=int(0.8*len(Zs)); perm=rng.permutation(len(Zs))
        Zfit=Zs[perm[:nfit]]; Zexp_ho=Zs[perm[nfit:]]
        U,S,Vt=np.linalg.svd(Zfit-Zfit.mean(0,keepdims=True),full_matrices=False)
        k=30; comps=Vt[:k]; ev=(S[:k]**2)/max(len(Zfit)-1,1)
        def mah_resid(Z):
            Zc=Z-Zfit.mean(0,keepdims=True)
            proj=Zc@comps.T               # (n,k)
            mah=((proj**2)/np.clip(ev,1e-9,None)[None]).sum(1).mean()
            recon=proj@comps; resid=((Zc-recon)**2).sum(1).mean()
            return mah,resid
        eM,eR=mah_resid(Zexp_ho)
        Zr_all=(np.concatenate([Zr[n] for n in rnames],0)-mu)/sd
        rM,rR=mah_resid(Zr_all)
        print(f"{tag:<14} | {ra:>10.3f} {rf:>11.3f} {rc:>11.3f} | "
              f"{eM:>8.1f} {rM:>9.1f} {rM/eM:>6.2f} | {eR:>9.1f} {rR:>10.1f} {rR/eR:>6.2f}")

if __name__=="__main__": main()
