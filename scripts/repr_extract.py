"""Extract frozen-encoder state representations for the DP/EO/EP comparison.

Lightweight h5py reader (no full replay-buffer load): samples a fixed number of
interior windows per source, reads only the needed frames, and encodes them with
each of the three EMA encoders.

For one robomimic task, dumps to repr_analysis/<task>/features.npz:
  <S>_<src>_raw  (N,256)  last-obs-frame embedding   (S in DP/EO/EP, src in exp/roll)
  <S>_<src>_ln   (N,256)  target_ln(raw)  (=raw for DP)
  <S>_<src>_goal (N,256)  encoder(goal frame)        (FDM-target embedding)
  exp_cur/nxt (N,9)  eef_pos(3)+eef_quat(4)+gripper(2) at last-obs / goal frame
  exp_act (N,H,7) raw action chunk        (roll_* likewise)

Images use ImageNormalizer (x*2-1), identical for all models, so they are read &
normalized ONCE. Only the 9 low-dim dims are re-normalized per model:
DP/EO use the expert MinMax normalizer, EP uses the expert+rollout merged one.
Encoders run in eval() (center crop, deterministic).

Usage: python scripts/repr_extract.py --task can
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from mip.dataset_utils import MinMaxNormalizer
from mip.network_utils import get_encoder

CFGS = json.load(open("/tmp/full_cfgs.json"))
CKPTS = json.load(open("/tmp/ckpt_paths.json"))
OUT_ROOT = "/oscar/data/csun45/zzeng28/repo/lardp/repr_analysis"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAP = 2000
LOWDIM = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
RGB = ["agentview_image", "robot0_eye_in_hand_image"]


def _cfg(task, setting):
    return OmegaConf.create(CFGS[f"{task}/{setting}"])


def build_encoder(task, setting):
    cfg = _cfg(task, setting)
    enc = get_encoder(cfg.network, cfg.task).to(DEVICE)
    sd = torch.load(CKPTS[f"{task}/{setting}"]["ckpt"], map_location=DEVICE,
                    weights_only=False)
    enc.load_state_dict(sd["encoder_ema"])
    enc.eval()
    ln = None
    if "target_ln_ema" in sd:
        out_dim = sd["target_ln_ema"]["weight"].shape[0]
        affine = bool(cfg.optimization.get("joint_target_ln_affine", True))
        ln = torch.nn.LayerNorm(out_dim, elementwise_affine=affine).to(DEVICE)
        ln.load_state_dict(sd["target_ln_ema"])
        ln.eval()
    return enc, ln


def fit_lowdim_normalizers(path):
    """One MinMaxNormalizer per low-dim key, fit over all demos in the file."""
    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())
        cols = {k: [] for k in LOWDIM}
        for dm in demos:
            obs = f["data"][dm]["obs"]
            for k in LOWDIM:
                cols[k].append(obs[k][:].astype(np.float32))
    return {k: MinMaxNormalizer(np.concatenate(cols[k], 0)) for k in LOWDIM}


def merge_lowdim(na, nb):
    out = {}
    for k in LOWDIM:
        gmin = np.minimum(na[k].min, nb[k].min)
        gmax = np.maximum(na[k].max, nb[k].max)
        out[k] = MinMaxNormalizer(np.stack([gmin, gmax]))
    return out


def sample_windows(path, To, horizon, cap, seed):
    """Read raw frames+labels for `cap` interior windows. Returns dict of arrays."""
    rng = np.random.RandomState(seed)
    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())
        # build (demo, p) candidates, then subsample
        cand = []
        lens = {}
        for dm in demos:
            T = f["data"][dm]["actions"].shape[0]
            lens[dm] = T
            if T >= horizon + 1:
                cand.append(dm)
        # sample windows uniformly over demos*positions
        picks = []
        for _ in range(cap):
            dm = cand[rng.randint(len(cand))]
            p = rng.randint(0, lens[dm] - horizon)  # frames [p .. p+horizon] exist
            picks.append((dm, p))
        img = {k: [] for k in RGB}
        goal_img = {k: [] for k in RGB}
        low = {k: [] for k in LOWDIM}
        goal_low = {k: [] for k in LOWDIM}
        cur, nxt, act = [], [], []
        for dm, p in picks:
            g = f["data"][dm]
            obs = g["obs"]
            idx_obs = list(range(p, p + To))          # To obs frames
            for k in RGB:
                img[k].append(obs[k][idx_obs])         # (To,84,84,3) uint8
                goal_img[k].append(obs[k][p + horizon][None])  # (1,84,84,3)
            st = []
            for k in LOWDIM:
                arr = obs[k][idx_obs].astype(np.float32)   # (To,d)
                low[k].append(arr)
                goal_low[k].append(obs[k][p + horizon][None].astype(np.float32))
                st.append(obs[k])  # full for cur/nxt below
            curv = np.concatenate([obs[k][p + To - 1] for k in LOWDIM])
            nxtv = np.concatenate([obs[k][p + horizon] for k in LOWDIM])
            cur.append(curv); nxt.append(nxtv)
            act.append(g["actions"][p:p + horizon].astype(np.float32))
    out = {
        "img": {k: np.stack(img[k]) for k in RGB},          # (N,To,84,84,3)
        "goal_img": {k: np.stack(goal_img[k]) for k in RGB},  # (N,1,84,84,3)
        "low": {k: np.stack(low[k]) for k in LOWDIM},        # (N,To,d)
        "goal_low": {k: np.stack(goal_low[k]) for k in LOWDIM},
        "cur": np.stack(cur).astype(np.float32),
        "nxt": np.stack(nxt).astype(np.float32),
        "act": np.stack(act).astype(np.float32),
    }
    return out


def _img_norm(arr_uint8):
    # (B,T,84,84,3) uint8 -> (B,T,3,84,84) float in [-1,1]
    x = np.moveaxis(arr_uint8.astype(np.float32), -1, 2) / 255.0
    return x * 2.0 - 1.0


@torch.no_grad()
def encode(enc, ln, data, normd, To, bs=64):
    """Encode windows; images shared, low-dim normalized by `normd`."""
    N = data["cur"].shape[0]
    H_raw, H_ln, H_goal = [], [], []
    # pre-normalize images once (model-independent)
    obs_img = {k: _img_norm(data["img"][k]) for k in RGB}
    goal_img = {k: _img_norm(data["goal_img"][k]) for k in RGB}
    for b in range(0, N, bs):
        sl = slice(b, b + bs)
        obs, goal = {}, {}
        for k in RGB:
            obs[k] = torch.from_numpy(obs_img[k][sl]).to(DEVICE)
            goal[k] = torch.from_numpy(goal_img[k][sl]).to(DEVICE)
        for k in LOWDIM:
            obs[k] = torch.from_numpy(normd[k].normalize(data["low"][k][sl])).to(DEVICE)
            goal[k] = torch.from_numpy(normd[k].normalize(data["goal_low"][k][sl])).to(DEVICE)
        h = enc(obs, None)
        hg = enc(goal, None)
        h_last = h[:, -1, :]
        H_raw.append(h_last.cpu().numpy())
        H_goal.append(hg[:, -1, :].cpu().numpy())
        H_ln.append((ln(h_last) if ln is not None else h_last).cpu().numpy())
    return np.concatenate(H_raw), np.concatenate(H_ln), np.concatenate(H_goal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    task = args.task
    cfg = _cfg(task, "EP")
    # rgb / low-dim keys are task-specific (camera names, # robots) — derive
    # them from shape_meta rather than assuming the can/square layout.
    global RGB, LOWDIM
    sm = cfg.task.shape_meta.obs
    RGB = [k for k, v in sm.items() if v.get("type") == "rgb"]
    LOWDIM = [k for k, v in sm.items() if v.get("type", "low_dim") == "low_dim"]
    print(f"[{task}] rgb={RGB}\n[{task}] low={LOWDIM}")
    expert_path = os.path.expanduser(cfg.task.dataset_paths[0])
    rollout_path = os.path.expanduser(cfg.task.dataset_paths[1])
    To, horizon = int(cfg.task.obs_steps), int(cfg.task.horizon)
    print(f"[{task}] To={To} horizon={horizon}\n  expert={expert_path}\n  rollout={rollout_path}")

    N_exp = fit_lowdim_normalizers(expert_path)
    N_roll = fit_lowdim_normalizers(rollout_path)
    N_mix = merge_lowdim(N_exp, N_roll)
    print(f"[{task}] normalizers fit. eef_pos exp-range={N_exp['robot0_eef_pos'].range}, "
          f"mix-range={N_mix['robot0_eef_pos'].range}")

    exp = sample_windows(expert_path, To, horizon, CAP, seed=0)
    roll = sample_windows(rollout_path, To, horizon, CAP, seed=1)
    print(f"[{task}] sampled exp={exp['cur'].shape[0]} roll={roll['cur'].shape[0]} windows")

    out = {
        "exp_cur": exp["cur"], "exp_nxt": exp["nxt"], "exp_act": exp["act"],
        "roll_cur": roll["cur"], "roll_nxt": roll["nxt"], "roll_act": roll["act"],
        "meta": np.array([To, horizon]),
    }
    norm_of = {"DP": N_exp, "EO": N_exp, "EP": N_mix}
    for s in ["DP", "EO", "EP"]:
        enc, ln = build_encoder(task, s)
        for src, data in [("exp", exp), ("roll", roll)]:
            hr, hl, hg = encode(enc, ln, data, norm_of[s], To)
            out[f"{s}_{src}_raw"], out[f"{s}_{src}_ln"], out[f"{s}_{src}_goal"] = hr, hl, hg
        del enc, ln
        torch.cuda.empty_cache()
        print(f"[{task}] {s}: exp_raw std={out[f'{s}_exp_raw'].std():.4f} "
              f"ln std={out[f'{s}_exp_ln'].std():.4f}")

    os.makedirs(f"{OUT_ROOT}/{task}", exist_ok=True)
    np.savez_compressed(f"{OUT_ROOT}/{task}/features.npz", **out)
    print(f"[{task}] saved {OUT_ROOT}/{task}/features.npz")


if __name__ == "__main__":
    main()
