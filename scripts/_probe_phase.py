"""Phase-resolved cross-demo action R^2: is s2e=true's action deficit uniform,
or concentrated at a task-critical phase (e.g. grasp)?"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import h5py, numpy as np
import scripts.viz_encoder_compare as V
NORM=str(ROOT/"viz/can_joint_pt_compare/normalizer.pkl"); EXP="data/robomimic/can/ph/image_v15.hdf5"
LOG="logs/can_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_stduniform_atduniform_tsdiagonal_ys1.0_ya1.0_ed256_d256_L8_h10_0.9"
CKPTS={
 "s2efalse_all": f"{LOG}_s2efalse_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_20/models/model_latest.pt",
 "s2etrue_all":  f"{LOG}_s2etrue_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_04/models/model_latest.pt",
 "s2efalse_play":f"{LOG}_s2efalse_schnoisier_play_elr1.0_esf0.8_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_24_56/models/model_latest.pt",
}
NB=5
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
    norm,_=V._load_normalizer(CKPTS["s2efalse_all"],NORM); H=int(tc.horizon); rng=np.random.default_rng(0)
    with h5py.File(EXP,"r") as f:
        ed=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        enames=[ed[i] for i in sorted(rng.choice(len(ed),40,replace=False).tolist())]
        earr={n:V._read_demo(f["data"][n],norm,ik,ld) for n in enames}
        eact={n:norm["action"].normalize(np.asarray(f["data"][n]["actions"]).astype(np.float32)) for n in enames}
    folds=np.array_split(np.array(enames),5)
    print(f"{'encoder':<14} | "+" ".join(f'ph{b}/{NB}' for b in range(NB))+"   (cross-demo action R2 by phase bin)")
    for tag,ck in CKPTS.items():
        enc,_=V._load_inner_encoder(ck,nc,tc,"cuda",True,tag)
        Z={n:V._encode_demo(enc,earr[n],"cuda",64) for n in enames}
        A={n:V._build_action_chunks(eact[n],H) for n in enames}
        Ph={n:(np.arange(len(Z[n]))/max(len(Z[n])-1,1)) for n in enames}
        binr=[[] for _ in range(NB)]
        for i in range(5):
            va=list(folds[i]); tr=[n for n in enames if n not in va]
            def stack(ns):
                Xs,Ys,Ps=[],[],[]
                for n in ns:
                    m=min(len(Z[n]),len(A[n])); Xs.append(Z[n][:m]); Ys.append(A[n][:m]); Ps.append(Ph[n][:m])
                return np.concatenate(Xs),np.concatenate(Ys),np.concatenate(Ps)
            Xt,Yt,Pt=stack(tr); Xv,Yv,Pv=stack(va)
            for b in range(NB):
                lo,hi=b/NB,(b+1)/NB
                mt=(Pt>=lo)&(Pt<hi if b<NB-1 else Pt<=hi); mv=(Pv>=lo)&(Pv<hi if b<NB-1 else Pv<=hi)
                if mt.sum()>50 and mv.sum()>20:
                    binr[b].append(r2(Xt[mt],Yt[mt],Xv[mv],Yv[mv]))
        print(f"{tag:<14} | "+" ".join(f'{np.mean(b):>5.2f}' if b else '  -- ' for b in binr))
if __name__=="__main__": main()
