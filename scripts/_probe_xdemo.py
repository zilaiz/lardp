"""Cross-demo same-phase similarity, raw vs DC-removed, to test whether s2e=true
genuinely spreads same-phase states apart (action-invariance lost) or whether the
raw-cosine gap is just the DC-offset confound.
Also: nearest-training-neighbor distance for HELD-OUT demos in a scale-invariant
(z-standardized) space -- do unseen states land closer to training z for s2e=false?
"""
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
PH=[0.0,0.25,0.5,0.75,1.0]
def offdiag_cos(M):
    Mn=M/np.clip(np.linalg.norm(M,axis=1,keepdims=True),1e-9,None)
    S=Mn@Mn.T; n=len(M); return S[np.triu_indices(n,1)].mean()
def main():
    tc,nc=V._compose_cfg(ROOT,"examples/configs","can_ph_image","lbmdit_joint_pt")
    ik=[k for k,v in tc.shape_meta.obs.items() if v.type=="rgb"]; ld=[k for k,v in tc.shape_meta.obs.items() if v.type=="low_dim"]
    norm,_=V._load_normalizer(CKPTS["s2efalse_all"],NORM); rng=np.random.default_rng(0)
    with h5py.File(EXP,"r") as f:
        ed=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        names=[ed[i] for i in sorted(rng.choice(len(ed),16,replace=False).tolist())]
        arr={n:V._read_demo(f["data"][n],norm,ik,ld) for n in names}
    print(f"{'encoder':<14} | {'xdemo cos RAW':>13} {'xdemo cos DC-removed':>20} | {'NN-dist tr':>11} {'NN-dist held':>12} {'ratio':>6}")
    for tag,ck in CKPTS.items():
        enc,_=V._load_inner_encoder(ck,nc,tc,"cuda",True,tag)
        Z={n:V._encode_demo(enc,arr[n],"cuda",64) for n in names}
        allz=np.concatenate([Z[n] for n in names],0); gmean=allz.mean(0,keepdims=True)
        raw,dc=[],[]
        for ph in PH:
            M=np.stack([Z[n][min(int(round(ph*(len(Z[n])-1))),len(Z[n])-1)] for n in names],0)
            raw.append(offdiag_cos(M)); dc.append(offdiag_cos(M-gmean))
        # scale-invariant nearest-neighbor: standardize by train stats, held-out demos -> nearest train frame
        tr=names[:12]; ho=names[12:]
        Ztr=np.concatenate([Z[n] for n in tr],0); mu=Ztr.mean(0,keepdims=True); sd=np.clip(Ztr.std(0,keepdims=True),1e-6,None)
        Ztr_s=(Ztr-mu)/sd
        def nn(zset):
            zs=(zset-mu)/sd; d=np.sqrt(((zs[:,None,:]-Ztr_s[None,:,:])**2).sum(-1)); return d.min(1).mean()
        # subsample train frames for speed
        idx=rng.choice(len(Ztr_s),min(2000,len(Ztr_s)),replace=False); Ztr_s=Ztr_s[idx]
        nn_tr=nn(np.concatenate([Z[n] for n in tr[:3]],0))   # train-vs-train baseline
        nn_ho=nn(np.concatenate([Z[n] for n in ho],0))       # heldout-vs-train
        print(f"{tag:<14} | {np.mean(raw):>13.3f} {np.mean(dc):>20.3f} | {nn_tr:>11.2f} {nn_ho:>12.2f} {nn_ho/nn_tr:>6.2f}")
if __name__=="__main__": main()
