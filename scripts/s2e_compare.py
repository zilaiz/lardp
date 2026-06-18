"""Does joint_state_loss_to_encoder=True change the learned representation?

Builds the s2e=TRUE joint_pt expert-only encoder (FDM/state loss flows INTO the
encoder) for each task, encodes the SAME windows as features.npz, and compares it
to DP, EO (s2e=false), EP via:
  - probing (action chunk + physical next-state, lin + MLP) on expert states
  - similarity (xpred_R2, CKA) vs DP / EO / EP

Key question: under s2e=false the encoder gradient is action-only (like DP). With
s2e=true the FDM target directly shapes the encoder, so EO_s2e should be MORE
different from DP (lower CKA) and/or decode different content.

Usage: python scripts/s2e_compare.py
"""
from __future__ import annotations

import glob
import json

import numpy as np
import torch

import scripts.repr_extract as X
from scripts.repr_probe2 import lin_r2, mlp_r2, split
from scripts.repr_similarity import xpred_r2, cka
from mip.network_utils import get_encoder

TASKS = ["can", "square", "transport", "tool_hang"]
VALPCT = {"can": "0.9", "square": "0.75", "transport": "0.8", "tool_hang": "0.6"}


def find_s2e_ckpt(task):
    pat = (f"logs/{task}_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_tsdiagonal_"
           f"ys1.0_ya1.0_ed256_d256_L8_h10_{VALPCT[task]}_s2etrue_esfnull_cs0.0_ccadd_"
           f"spx1_pt/*/models/model_best.pt")
    c = sorted(glob.glob(pat))
    return c[-1] if c else None


def build_s2e_encoder(task, ckpt):
    cfg = X._cfg(task, "EO")  # encoder arch identical to EO
    enc = get_encoder(cfg.network, cfg.task).to(X.DEVICE)
    sd = torch.load(ckpt, map_location=X.DEVICE, weights_only=False)
    enc.load_state_dict(sd["encoder_ema"]); enc.eval()
    out_dim = sd["target_ln_ema"]["weight"].shape[0]
    ln = torch.nn.LayerNorm(out_dim, elementwise_affine=True).to(X.DEVICE)
    ln.load_state_dict(sd["target_ln_ema"]); ln.eval()
    return enc, ln


def main():
    sim_rows = []
    probe_rows = []
    for task in TASKS:
        ckpt = find_s2e_ckpt(task)
        if ckpt is None:
            print(f"[{task}] no s2e=true ckpt found"); continue
        cfg = X._cfg(task, "EP")
        sm = cfg.task.shape_meta.obs
        X.RGB = [k for k, v in sm.items() if v.get("type") == "rgb"]
        X.LOWDIM = [k for k, v in sm.items() if v.get("type", "low_dim") == "low_dim"]
        exp_p, roll_p = cfg.task.dataset_paths[0], cfg.task.dataset_paths[1]
        To, H = int(cfg.task.obs_steps), int(cfg.task.horizon)
        N_exp = X.fit_lowdim_normalizers(exp_p)

        # same windows as features.npz (seed 0 exp / seed 1 roll)
        exp = X.sample_windows(exp_p, To, H, X.CAP, seed=0)
        roll = X.sample_windows(roll_p, To, H, X.CAP, seed=1)
        enc, ln = build_s2e_encoder(task, ckpt)
        # post-LN (LN treated as part of the encoder for joint settings); index [1]=h_ln
        s2e_exp = X.encode(enc, ln, exp, N_exp, To)[1]
        s2e_roll = X.encode(enc, ln, roll, N_exp, To)[1]
        del enc, ln; torch.cuda.empty_cache()

        d = np.load(f"{X.OUT_ROOT}/{task}/features.npz")
        emb = {"DP": d["DP_exp_ln"], "EO": d["EO_exp_ln"], "EP": d["EP_exp_ln"],
               "EOs2e": s2e_exp}
        embr = {"DP": d["DP_roll_ln"], "EO": d["EO_roll_ln"], "EP": d["EP_roll_ln"],
                "EOs2e": s2e_roll}

        # similarity of EO_s2e vs others (exp), plus DP-EO ref
        sim = {"task": task}
        for other in ["DP", "EO", "EP"]:
            sim[f"EOs2e-{other}_xpred"] = 0.5 * (xpred_r2(s2e_exp, emb[other]) +
                                                 xpred_r2(emb[other], s2e_exp))
            sim[f"EOs2e-{other}_cka"] = cka(s2e_exp, emb[other])
        sim["DP-EO_cka"] = cka(emb["DP"], emb["EO"])
        sim_rows.append(sim)

        # probing on expert: action chunk + next-state (lin+mlp)
        n = len(s2e_exp); tr, va = split(n, 0)
        chunk = exp["act"].reshape(n, -1); nxt = exp["nxt"]
        pr = {"task": task}
        for nm, S in [("DP", emb["DP"]), ("EO", emb["EO"]), ("EP", emb["EP"]), ("EOs2e", s2e_exp)]:
            Sz = (S - S.mean(0)) / (S.std(0) + 1e-6)
            pr[f"{nm}_act"] = mlp_r2(Sz[tr], chunk[tr], Sz[va], chunk[va])
            pr[f"{nm}_nxt"] = mlp_r2(Sz[tr], nxt[tr], Sz[va], nxt[va])
        probe_rows.append(pr)
        print(f"[{task}] done  ckpt={ckpt.split('/')[1][:40]}...")

    print("\n############ s2e=TRUE encoder: similarity to others (expert; CKA / xpred_R2) ############")
    print(f"{'task':<11}{'EOs2e-DP':>14}{'EOs2e-EO':>14}{'EOs2e-EP':>14}{'(ref)DP-EO':>12}")
    for s in sim_rows:
        print(f"{s['task']:<11}"
              f"{s['EOs2e-DP_cka']:.2f}/{s['EOs2e-DP_xpred']:.2f}   "
              f"{s['EOs2e-EO_cka']:.2f}/{s['EOs2e-EO_xpred']:.2f}   "
              f"{s['EOs2e-EP_cka']:.2f}/{s['EOs2e-EP_xpred']:.2f}   "
              f"{s['DP-EO_cka']:>10.2f}")

    print("\n############ probing (expert, MLP R^2): action chunk / next-state ############")
    print(f"{'task':<11}{'DP':>14}{'EO':>14}{'EP':>14}{'EOs2e':>14}")
    for p in probe_rows:
        print(f"{p['task']:<11}" + "".join(
            f"{p[s+'_act']:.2f}/{p[s+'_nxt']:.2f}    " for s in ["DP", "EO", "EP", "EOs2e"]))

    json.dump({"sim": sim_rows, "probe": probe_rows},
              open(f"{X.OUT_ROOT}/s2e_compare.json", "w"), indent=2)
    print(f"\nsaved {X.OUT_ROOT}/s2e_compare.json")


if __name__ == "__main__":
    main()
