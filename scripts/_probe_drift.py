"""Within-run drift test: does s2e=true's encoder get WORSE for action / further
OOD as training continues (best->latest), while s2e=false stays stable?
Catches the self-referential drift in the act."""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import h5py, numpy as np
import scripts.viz_encoder_compare as V

NORM=str(ROOT/"viz/can_joint_pt_compare/normalizer.pkl")
EXP="data/robomimic/can/ph/image_v15.hdf5"; ROLL="data/robomimic/can/image_rollouts_clipped.hdf5"
LOG="logs/can_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_stduniform_atduniform_tsdiagonal_ys1.0_ya1.0_ed256_d256_L8_h10_0.9"
D1=f"{LOG}_s2efalse_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_20/models"
D2=f"{LOG}_s2etrue_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_04/models"
CKPTS={
 "s2efalse@best":  f"{D1}/model_best.pt","s2efalse@latest":f"{D1}/model_latest.pt",
 "s2etrue@best":   f"{D2}/model_best.pt","s2etrue@latest": f"{D2}/model_latest.pt",
}
PROP=["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos"]
def r2(X,Y,Xv,Yv,lam=1e-1):
    mu=X.mean(0,keepdims=True); sd=np.clip(X.std(0,keepdims=True),1e-6,None)
    muy=Y.mean(0,keepdims=True); sdy=np.clip(Y.std(0,keepdims=True),1e-6,None)
    Ys=(Y-muy)/sdy; Yvs=(Yv-muy)/sdy
    Xs=np.c_[(X-mu)/sd,np.ones((len(X),1))]; Xvs=np.c_[(Xv-mu)/sd,np.ones((len(Xv),1))]
    W=np.linalg.solve(Xs.T@Xs+lam*np.eye(Xs.shape[1]),Xs.T@Ys)
    return 1-((Xvs@W-Yvs)**2).sum()/max(((Yvs-Ys.mean(0,keepdims=True))**2).sum(),1e-12)
def main():
    tc,nc=V._compose_cfg(ROOT,"examples/configs","can_ph_image","lbmdit_joint_pt")
    ik=[k for k,v in tc.shape_meta.obs.items() if v.type=="rgb"]; ld=[k for k,v in tc.shape_meta.obs.items() if v.type=="low_dim"]
    norm,_=V._load_normalizer(CKPTS["s2efalse@best"],NORM); H=int(tc.horizon); rng=np.random.default_rng(0)
    with h5py.File(EXP,"r") as f:
        ed=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        enames=[ed[i] for i in sorted(rng.choice(len(ed),40,replace=False).tolist())]
        earr={n:V._read_demo(f["data"][n],norm,ik,ld) for n in enames}
        eact={n:norm["action"].normalize(np.asarray(f["data"][n]["actions"]).astype(np.float32)) for n in enames}
    with h5py.File(ROLL,"r") as f:
        rd=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        rnames=[rd[i] for i in sorted(rng.choice(len(rd),25,replace=False).tolist())]
        rarr={n:V._read_demo(f["data"][n],norm,ik,ld) for n in rnames}
    print(f"{'ckpt':<16} | {'rawStd':>7} {'R2 act':>7} {'R2 fut':>7} | {'rollResidRatio':>14}")
    for tag,ck in CKPTS.items():
        enc,_=V._load_inner_encoder(ck,nc,tc,"cuda",True,tag)
        Ze={n:V._encode_demo(enc,earr[n],"cuda",64) for n in enames}
        folds=np.array_split(np.array(enames),5); aa,af=[],[]
        def build(ns,kind):
            Xs,Ys=[],[]
            for n in ns:
                z=Ze[n]
                if kind=="act": y=V._build_action_chunks(eact[n],H)
                else:
                    T=z.shape[0]; idx=np.minimum(np.arange(T)+H,T-1)
                    y=np.concatenate([earr[n][k][idx] for k in PROP],1)
                m=min(len(z),len(y)); Xs.append(z[:m]); Ys.append(y[:m])
            return np.concatenate(Xs),np.concatenate(Ys)
        for i in range(5):
            va=list(folds[i]); tr=[n for n in enames if n not in va]
            Xt,Yt=build(tr,"act"); Xv,Yv=build(va,"act"); aa.append(r2(Xt,Yt,Xv,Yv))
            Xt,Yt=build(tr,"fut"); Xv,Yv=build(va,"fut"); af.append(r2(Xt,Yt,Xv,Yv))
        Zall=np.concatenate([Ze[n] for n in enames],0); raw_std=Zall.std(0).mean()
        mu=Zall.mean(0,keepdims=True); sd=np.clip(Zall.std(0,keepdims=True),1e-6,None); Zs=(Zall-mu)/sd
        nfit=int(0.8*len(Zs)); perm=rng.permutation(len(Zs)); Zfit=Zs[perm[:nfit]]; Zho=Zs[perm[nfit:]]
        U,S,Vt=np.linalg.svd(Zfit-Zfit.mean(0,keepdims=True),full_matrices=False); k=30; comps=Vt[:k]
        def resid(Z):
            Zc=Z-Zfit.mean(0,keepdims=True); proj=Zc@comps.T; return ((Zc-proj@comps)**2).sum(1).mean()
        eR=resid(Zho); Zr=(np.concatenate([Ze if False else V._encode_demo(enc,rarr[n],'cuda',64) for n in rnames],0)-mu)/sd
        rR=resid(Zr)
        print(f"{tag:<16} | {raw_std:>7.3f} {np.mean(aa):>7.3f} {np.mean(af):>7.3f} | {rR/eR:>14.2f}")
if __name__=="__main__": main()
