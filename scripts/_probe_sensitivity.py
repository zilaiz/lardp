"""Test off-manifold fragility: how much does z move under a small input
perturbation, relative to the natural frame-to-frame motion? A higher ratio =
higher-gain encoder = more fragile to the distribution shift of BC rollout.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import h5py, numpy as np, torch
import scripts.viz_encoder_compare as V

NORM = str(ROOT / "viz/can_joint_pt_compare/normalizer.pkl")
DS = "data/robomimic/can/ph/image_v15.hdf5"
LOG = "logs/can_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_stduniform_atduniform_tsdiagonal_ys1.0_ya1.0_ed256_d256_L8_h10_0.9"
CKPTS = {
 "s2efalse_all":  f"{LOG}_s2efalse_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_20/models/model_latest.pt",
 "s2etrue_all":   f"{LOG}_s2etrue_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_04/models/model_latest.pt",
 "s2efalse_play": f"{LOG}_s2efalse_schnoisier_play_elr1.0_esf0.8_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_24_56/models/model_latest.pt",
}

def encode(enc, arrs, dev, noise=0.0, gen=None):
    T = next(iter(arrs.values())).shape[0]
    out=[]
    for s in range(0,T,64):
        e=min(s+64,T); ch={}
        for k,v in arrs.items():
            x=torch.from_numpy(v[s:e][None]).to(dev)
            if noise>0:
                x=x+noise*torch.randn(x.shape, generator=gen, device=dev)
            ch[k]=x
        with torch.no_grad(): z=enc(ch,None)
        out.append(z.squeeze(0).cpu().numpy())
    return np.concatenate(out,0)

def main():
    dev="cuda"
    task_cfg, net_cfg = V._compose_cfg(ROOT,"examples/configs","can_ph_image","lbmdit_joint_pt")
    ik=[k for k,v in task_cfg.shape_meta.obs.items() if v.type=="rgb"]
    ld=[k for k,v in task_cfg.shape_meta.obs.items() if v.type=="low_dim"]
    norm,_=V._load_normalizer(CKPTS["s2efalse_all"],NORM)
    rng=np.random.default_rng(0)
    with h5py.File(DS,"r") as f:
        ds=sorted(f["data"].keys(),key=lambda s:int(s.split("_")[-1]))
        names=[ds[i] for i in sorted(rng.choice(len(ds),10,replace=False).tolist())]
        arrs={n:V._read_demo(f["data"][n],norm,ik,ld) for n in names}
    gen=torch.Generator(device=dev)
    print(f"{'encoder':<14} | {'||dz_time||':>11} | "+" | ".join(f'sig={s}:dz/dz_t' for s in (0.03,0.08)))
    for tag,ck in CKPTS.items():
        enc,_=V._load_inner_encoder(ck,net_cfg,task_cfg,dev,True,tag)
        dz_time=[]; z0={}
        for n in names:
            z=encode(enc,arrs[n],dev); z0[n]=z
            dz_time.append(np.linalg.norm(np.diff(z,axis=0),axis=-1))
        mt=np.concatenate(dz_time).mean()
        row=f"{tag:<14} | {mt:>11.4f} | "
        for sig in (0.03,0.08):
            ratios=[]
            for n in names:
                gen.manual_seed(123)
                zp=encode(enc,arrs[n],dev,noise=sig,gen=gen)
                dzn=np.linalg.norm(zp-z0[n],axis=-1)
                ratios.append(dzn)
            mn=np.concatenate(ratios).mean()
            row+=f"{mn:>7.4f} ({mn/mt:>5.2f}x)  "
        print(row)

if __name__=="__main__": main()
