"""Decompose feature-std inflation and measure scale/offset-invariant smoothness
for the three can_ph_image joint_pt encoders (s2efalse_all / s2etrue_all /
s2efalse_play). Reuses the loaders from viz_encoder_compare.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

import scripts.viz_encoder_compare as V

NORM = str(ROOT / "viz/can_joint_pt_compare/normalizer.pkl")
DS = "data/robomimic/can/ph/image_v15.hdf5"
LOG = "logs/can_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_stduniform_atduniform_tsdiagonal_ys1.0_ya1.0_ed256_d256_L8_h10_0.9"
CKPTS = {
    "s2efalse_all":  f"{LOG}_s2efalse_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_20/models/model_latest.pt",
    "s2etrue_all":   f"{LOG}_s2etrue_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_04/models/model_latest.pt",
    "s2efalse_play": f"{LOG}_s2efalse_schnoisier_play_elr1.0_esf0.8_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_24_56/models/model_latest.pt",
}


def seg_dir_cos(z):
    """cos(dz_t, dz_{t+1}) — segment-to-segment direction consistency.
    Scale/offset invariant. ~1 => straight & smooth; lower => sharp turns/jerk."""
    dz = np.diff(z, axis=0)
    a, b = dz[:-1], dz[1:]
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.clip(den, 1e-12, None)


def centered_consec_cos(z):
    """cos(z_t-mu, z_{t+1}-mu): consecutive cosine after removing the per-demo
    DC offset, so it is not inflated by a large mean ||z||."""
    zc = z - z.mean(0, keepdims=True)
    a, b = zc[:-1], zc[1:]
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.clip(den, 1e-12, None)


def main():
    task_cfg, network_cfg = V._compose_cfg(ROOT, "examples/configs",
                                           "can_ph_image", "lbmdit_joint_pt")
    image_keys = [k for k, v in task_cfg.shape_meta.obs.items() if v.type == "rgb"]
    lowdim_keys = [k for k, v in task_cfg.shape_meta.obs.items() if v.type == "low_dim"]
    norm, _ = V._load_normalizer(CKPTS["s2efalse_all"], NORM)

    rng = np.random.default_rng(0)
    with h5py.File(DS, "r") as f:
        all_demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        chosen = sorted(rng.choice(len(all_demos), size=10, replace=False).tolist())
        names = [all_demos[i] for i in chosen]
        arrs = {n: V._read_demo(f["data"][n], norm, image_keys, lowdim_keys) for n in names}

    print(f"{'encoder':<14} | {'||gamma||2':>9} {'mean|g|':>8} {'mean|b|':>8} | "
          f"{'raw |z|':>8} {'raw std':>8} {'raw |mean|':>9} {'std/|z|':>8} | "
          f"{'postLN std':>10} | {'segDirCos':>9} {'ctrConsCos':>10}")
    for tag, ck in CKPTS.items():
        enc, ckd = V._load_inner_encoder(ck, network_cfg, task_cfg, "cuda", True, tag)
        tln = V._maybe_load_target_ln(ckd, network_cfg, "cuda", True, tag)
        gamma = ckd["target_ln_ema"]["weight"].float().cpu().numpy()
        beta = ckd["target_ln_ema"]["bias"].float().cpu().numpy()

        raw_norms, raw_z, post_z, sdc, ccc = [], [], [], [], []
        for n in names:
            z = V._encode_demo(enc, arrs[n], "cuda", 64, target_ln=None)      # raw
            zp = V._encode_demo(enc, arrs[n], "cuda", 64, target_ln=tln)      # post-LN
            raw_norms.append(np.linalg.norm(z, axis=-1))                      # per-frame ||z||
            raw_z.append(z); post_z.append(zp)
            sdc.append(seg_dir_cos(z)); ccc.append(centered_consec_cos(z))
        Z = np.concatenate(raw_z, 0)
        ZP = np.concatenate(post_z, 0)
        raw_std = Z.std(0).mean()
        raw_dc = np.abs(Z.mean(0)).mean()        # magnitude of the mean (DC offset) per dim
        mean_norm = np.concatenate(raw_norms).mean()
        post_std = ZP.std(0).mean()
        print(f"{tag:<14} | {np.linalg.norm(gamma):>9.3f} {np.abs(gamma).mean():>8.3f} "
              f"{np.abs(beta).mean():>8.3f} | {mean_norm:>8.3f} {raw_std:>8.3f} "
              f"{raw_dc:>9.3f} {raw_std/mean_norm:>8.4f} | {post_std:>10.3f} | "
              f"{np.concatenate(sdc).mean():>9.4f} {np.concatenate(ccc).mean():>10.4f}")


if __name__ == "__main__":
    main()
