"""Is the state loss trivially satisfiable by copying the conditioning?
target = LN(enc(obs_{t+H})), conditioning z_t = LN(enc(obs_t)).  A 'copy z_t'
baseline predicts target by z_t. R2_copy = how much target variance copying explains.
High / inflated-by-s2e=true => the state loss admits (and the encoder was shaped
toward) a degenerate self-copy shortcut, not genuine dynamics."""
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
H=10
def main():
    tc,nc=V._compose_cfg(ROOT,"examples/configs","can_ph_image","lbmdit_joint_pt")
    ik=[k for k,v in tc.shape_meta.obs.items() if v.type=="rgb"]; ld=[k for k,v in tc.shape_meta.obs.items() if v.type=="low_dim"]
    norm,_=V._load_normalizer(CKPTS["s2efalse_all"],NORM); rng=np.random.default_rng(0)
    with h5py.File(EXP,"r") as f:
        ed=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        names=[ed[i] for i in sorted(rng.choice(len(ed),16,replace=False).tolist())]
        arr={n:V._read_demo(f["data"][n],norm,ik,ld) for n in names}
    print(f"{'encoder':<14} | {'R2_copy(goal|zt)':>16} {'cos(zt,goal)':>13} {'rel gap||dz_H||/spread':>22}")
    for tag,ck in CKPTS.items():
        enc,ckd=V._load_inner_encoder(ck,nc,tc,"cuda",True,tag)
        tln=V._maybe_load_target_ln(ckd,nc,"cuda",True,tag)
        cur,goal=[],[]
        for n in names:
            z=V._encode_demo(enc,arr[n],"cuda",64,target_ln=tln)  # post-LN, matches loss space
            T=len(z)
            if T<=H: continue
            cur.append(z[:T-H]); goal.append(z[H:])
        cur=np.concatenate(cur,0); goal=np.concatenate(goal,0)
        # R2 of copy baseline: predict goal by cur
        ss_res=((goal-cur)**2).sum()
        ss_tot=((goal-goal.mean(0,keepdims=True))**2).sum()
        r2_copy=1-ss_res/ss_tot
        cn=cur/np.clip(np.linalg.norm(cur,axis=1,keepdims=True),1e-9,None)
        gn=goal/np.clip(np.linalg.norm(goal,axis=1,keepdims=True),1e-9,None)
        cos=(cn*gn).sum(1).mean()
        spread=np.sqrt(((goal-goal.mean(0,keepdims=True))**2).sum(1).mean())  # RMS radius
        relgap=np.sqrt(((goal-cur)**2).sum(1)).mean()/spread
        print(f"{tag:<14} | {r2_copy:>16.3f} {cos:>13.3f} {relgap:>22.3f}")
if __name__=="__main__": main()
