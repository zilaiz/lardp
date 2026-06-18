"""Decisive test: what fraction of the encoder's variance is action-relevant?
Regress z on the action chunk (predict z FROM action) and report R^2 = fraction
of z-variance linearly explained by the action. Low R^2 => more action-irrelevant
(nuisance) variance => lower action SNR => worse downstream generalization.
Also report held-out R^2 of the forward map (action from z) for reference.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import h5py, numpy as np
import scripts.viz_encoder_compare as V

NORM=str(ROOT/"viz/can_joint_pt_compare/normalizer.pkl"); DS="data/robomimic/can/ph/image_v15.hdf5"
LOG="logs/can_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_stduniform_atduniform_tsdiagonal_ys1.0_ya1.0_ed256_d256_L8_h10_0.9"
CKPTS={
 "s2efalse_all":  f"{LOG}_s2efalse_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_20/models/model_latest.pt",
 "s2etrue_all":   f"{LOG}_s2etrue_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_04/models/model_latest.pt",
 "s2efalse_play": f"{LOG}_s2efalse_schnoisier_play_elr1.0_esf0.8_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_24_56/models/model_latest.pt",
}
def ridge_r2(X,Y,Xv,Yv,lam=1e-2):
    mu=X.mean(0,keepdims=True); sd=np.clip(X.std(0,keepdims=True),1e-6,None)
    Xs=np.c_[(X-mu)/sd,np.ones((len(X),1))]; Xvs=np.c_[(Xv-mu)/sd,np.ones((len(Xv),1))]
    W=np.linalg.solve(Xs.T@Xs+lam*np.eye(Xs.shape[1]),Xs.T@Y)
    res=((Xvs@W-Yv)**2).sum(); tot=((Yv-Y.mean(0,keepdims=True))**2).sum()
    return 1-res/max(tot,1e-12)
def main():
    tc,nc=V._compose_cfg(ROOT,"examples/configs","can_ph_image","lbmdit_joint_pt")
    ik=[k for k,v in tc.shape_meta.obs.items() if v.type=="rgb"]; ld=[k for k,v in tc.shape_meta.obs.items() if v.type=="low_dim"]
    norm,_=V._load_normalizer(CKPTS["s2efalse_all"],NORM); H=int(tc.horizon)
    rng=np.random.default_rng(0)
    with h5py.File(DS,"r") as f:
        ds=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        names=[ds[i] for i in sorted(rng.choice(len(ds),10,replace=False).tolist())]
        arrs={n:V._read_demo(f["data"][n],norm,ik,ld) for n in names}
        acts={n:norm["action"].normalize(np.asarray(f["data"][n]["actions"]).astype(np.float32)) for n in names}
    tr=names[:8]; va=names[8:]
    print(f"{'encoder':<14} | {'R2(z|action)':>13}  {'R2 heldout':>11} | {'R2(action|z)':>13} {'heldout':>9}")
    for tag,ck in CKPTS.items():
        enc,_=V._load_inner_encoder(ck,nc,tc,"cuda",True,tag)
        Z={n:V._encode_demo(enc,arrs[n],"cuda",64) for n in names}
        A={n:V._build_action_chunks(acts[n],H) for n in names}
        def stack(ns):
            Xs,Ys=[],[]
            for n in ns:
                m=min(len(Z[n]),len(A[n])); Xs.append(Z[n][:m]); Ys.append(A[n][:m])
            return np.concatenate(Xs),np.concatenate(Ys)
        Ztr,Atr=stack(tr); Zva,Ava=stack(va); Zall,Aall=stack(names)
        # fraction of z-variance explained by action (in-sample structural + held-out)
        r2_za_in=ridge_r2(Aall,Zall,Aall,Zall)
        r2_za_ho=ridge_r2(Atr,Ztr,Ava,Zva)
        # forward map action-from-z (held-out generalization)
        r2_az_in=ridge_r2(Zall,Aall,Zall,Aall)
        r2_az_ho=ridge_r2(Ztr,Atr,Zva,Ava)
        print(f"{tag:<14} | {r2_za_in:>13.3f}  {r2_za_ho:>11.3f} | {r2_az_in:>13.3f} {r2_az_ho:>9.3f}")
if __name__=="__main__": main()
