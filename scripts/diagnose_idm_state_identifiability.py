"""Diagnose whether an IDM-pretrained encoder is state-identifiable, or
whether it has collapsed states that share the same action chunk.

For each encoder we run two probes on held-out frames from the expert dataset:

  1. Linear probe (Ridge) R² for:
        z -> lowdim_state   (eef_pos + eef_quat + gripper_qpos)
        z -> action_chunk   (horizon-length flattened action)
     Hypothesis: if the encoder is action-collapsed, R²(state) << R²(action).

  2. Distance correlation (Spearman) on random frame pairs:
        rho( ||z_i - z_j||,  ||state_i - state_j|| )       — state-identifiability
        rho( ||z_i - z_j||,  ||action_i - action_j|| )      — action-alignment
     Hypothesis: a collapsed encoder ranks pairs more by action-chunk distance
     than by state distance.

Outputs go to ``viz/idm_state_identifiability/`` (configurable):
   linear_probe_r2.png
   distance_correlation.png
   summary.txt
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import h5py
import loguru
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Loader helpers (mirror viz_idm_latent_trajectories.py)
# ---------------------------------------------------------------------------

def _load_inner_encoder(ckpt_path, task_cfg, network_cfg, device, use_ema=True):
    from mip.network_utils import get_encoder
    encoder = get_encoder(network_cfg, task_cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd_key = "encoder_ema" if (use_ema and "encoder_ema" in ck) else "encoder"
    sd = ck[sd_key]
    if any(k.startswith("encoder.") for k in sd):
        sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    encoder.load_state_dict(sd, strict=False)
    encoder.eval().requires_grad_(False)
    return encoder


def _load_normalizer(ckpt_path):
    with open(Path(ckpt_path).parent / "normalizer.pkl", "rb") as f:
        return pickle.load(f)


def _read_demo(h5_demo, normalizer, image_keys, lowdim_keys):
    out = {}
    for k in image_keys:
        x = np.asarray(h5_demo["obs"][k]).astype(np.float32) / 255.0
        x = np.moveaxis(x, -1, 1)
        out[k] = normalizer["obs"][k].normalize(x)
    for k in lowdim_keys:
        x = np.asarray(h5_demo["obs"][k]).astype(np.float32)
        out[k] = normalizer["obs"][k].normalize(x)
    return out


def _encode_demo(encoder, obs_arrays, device, chunk=64):
    T = next(iter(obs_arrays.values())).shape[0]
    zs = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        obs = {k: torch.from_numpy(v[s:e][None]).to(device)
               for k, v in obs_arrays.items()}
        with torch.no_grad():
            z = encoder(obs, None).squeeze(0).cpu().numpy()
        zs.append(z)
    return np.concatenate(zs, axis=0)


# ---------------------------------------------------------------------------

def gather_features(args, device):
    """Encode every frame from N_demos and gather aligned (z, state, action)."""
    import hydra
    from hydra import initialize_config_dir

    repo_root = Path(__file__).resolve().parents[1]
    config_abs = str((repo_root / args.config_dir).resolve())
    overrides = [f"task={args.task_config}", f"network={args.network_config}"]
    with initialize_config_dir(config_dir=config_abs, version_base=None):
        cfg = hydra.compose(config_name="main", overrides=overrides)
    task_cfg, network_cfg = cfg.task, cfg.network

    image_keys = [k for k, v in task_cfg.shape_meta.obs.items()
                  if v.type == "rgb"]
    lowdim_keys = [k for k, v in task_cfg.shape_meta.obs.items()
                   if v.type == "low_dim"]

    # Pick lowdim probe target: concat of all lowdim keys (raw, not normalized,
    # so distances are in physical units rather than normalizer-scaled).
    state_keys = lowdim_keys
    H = task_cfg.horizon

    models = {}
    for tag, ckpt in [("delta", args.delta_ckpt), ("goal", args.goal_ckpt)]:
        enc = _load_inner_encoder(ckpt, task_cfg, network_cfg, device,
                                  args.use_ema)
        norm = _load_normalizer(ckpt)
        models[tag] = (enc, norm)

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.dataset_path, "r") as f:
        all_demos = sorted(f["data"].keys(),
                            key=lambda s: int(s.split("_")[-1]))
        n_pick = min(args.n_demos, len(all_demos))
        chosen = sorted(rng.choice(len(all_demos), size=n_pick,
                                    replace=False).tolist())
        names = [all_demos[i] for i in chosen]
        loguru.logger.info(f"Probing on {len(names)} demos: {names[:8]}...")

        feats = {tag: [] for tag in models}
        states_list, actions_list, demo_ids = [], [], []
        for di, name in enumerate(tqdm(names, desc="encoding")):
            d = f["data"][name]
            T = d["obs"][image_keys[0]].shape[0]
            T_valid = T - H  # need horizon-length action chunk
            if T_valid <= 0:
                continue
            for tag, (enc, norm) in models.items():
                arrs = _read_demo(d, norm, image_keys, lowdim_keys)
                z = _encode_demo(enc, arrs, device, args.chunk)  # (T, D)
                feats[tag].append(z[:T_valid])
            # State (concat of raw lowdim keys at frame t)
            state = np.concatenate(
                [np.asarray(d["obs"][k]).astype(np.float32)[:T_valid]
                 for k in state_keys], axis=-1,
            )
            # Action chunk (raw, not normalized): a[t : t+H], flattened
            actions = np.asarray(d["actions"]).astype(np.float32)  # (T, A)
            chunks = np.stack([actions[t:t + H] for t in range(T_valid)], 0)
            states_list.append(state)
            actions_list.append(chunks.reshape(T_valid, -1))
            demo_ids.append(np.full(T_valid, di, dtype=np.int32))

    Z = {tag: np.concatenate(feats[tag], 0) for tag in feats}
    S = np.concatenate(states_list, 0)         # (N, state_dim)
    A = np.concatenate(actions_list, 0)        # (N, H*act_dim)
    demo_ids = np.concatenate(demo_ids, 0)     # (N,)

    loguru.logger.info(
        f"Gathered {S.shape[0]} frames | "
        f"state_dim={S.shape[1]} ({state_keys}) | action_dim={A.shape[1]}"
    )
    return Z, S, A, demo_ids, state_keys


# ---------------------------------------------------------------------------
# Probe 1: Linear probe (Ridge) R²
# ---------------------------------------------------------------------------

def linear_probe(Z_train, Z_test, Y_train, Y_test, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(Z_train, Y_train)
    Y_pred = model.predict(Z_test)
    r2_overall = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    r2_per_dim = r2_score(Y_test, Y_pred, multioutput="raw_values")
    return r2_overall, r2_per_dim


# ---------------------------------------------------------------------------
# Probe 2: Distance correlation
# ---------------------------------------------------------------------------

def distance_correlation(Z, S, A, n_pairs, rng):
    """Sample random frame pairs; report Spearman of feat-dist with
    state-dist and action-dist."""
    N = Z.shape[0]
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    feat = np.linalg.norm(Z[i] - Z[j], axis=-1)
    state = np.linalg.norm(S[i] - S[j], axis=-1)
    action = np.linalg.norm(A[i] - A[j], axis=-1)
    rho_state = float(spearmanr(feat, state).statistic)
    rho_action = float(spearmanr(feat, action).statistic)
    return rho_state, rho_action, feat, state, action


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta_ckpt", default="logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_delta_cond_fdm1.0_gdp0.0_ac8_ndp_0/2026_05_03_22_36_46/models/model_step_300000.pt")
    p.add_argument("--goal_ckpt",  default="logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_fdm_aux/2026_04_27_01_03_55/models/model_step_300000.pt")
    p.add_argument("--task_config", default="tool_hang_ph_image_gp")
    p.add_argument("--network_config", default="lbmidm_v2")
    p.add_argument("--config_dir", default="examples/configs")
    p.add_argument("--dataset_path", default="data/robomimic/tool_hang/ph/image_v15.hdf5")
    p.add_argument("--n_demos", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--out_dir", default="viz/idm_state_identifiability")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--n_pairs", type=int, default=50000)
    args = p.parse_args()

    out_dir = (Path(__file__).resolve().parents[1] / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    Z, S, A, demo_ids, state_keys = gather_features(args, args.device)

    rng = np.random.default_rng(args.seed)
    # Train/test split by demo to avoid leakage between adjacent frames
    unique_demos = np.unique(demo_ids)
    train_demos, test_demos = train_test_split(
        unique_demos, test_size=0.2, random_state=args.seed,
    )
    train_mask = np.isin(demo_ids, train_demos)
    test_mask = np.isin(demo_ids, test_demos)
    loguru.logger.info(
        f"Split: train {train_mask.sum()} frames "
        f"({len(train_demos)} demos), test {test_mask.sum()} frames "
        f"({len(test_demos)} demos)"
    )

    # --- Probe 1: linear probe R² ---
    probe_results = {}  # tag -> dict
    for tag in ["delta", "goal"]:
        Z_tr, Z_te = Z[tag][train_mask], Z[tag][test_mask]
        S_tr, S_te = S[train_mask], S[test_mask]
        A_tr, A_te = A[train_mask], A[test_mask]
        r2_state, r2_state_per = linear_probe(
            Z_tr, Z_te, S_tr, S_te, alpha=args.ridge_alpha,
        )
        r2_action, r2_action_per = linear_probe(
            Z_tr, Z_te, A_tr, A_te, alpha=args.ridge_alpha,
        )
        probe_results[tag] = {
            "r2_state": r2_state,
            "r2_state_per_dim": r2_state_per,
            "r2_action": r2_action,
            "r2_action_per_dim": r2_action_per,
        }

    # --- Probe 2: distance correlation ---
    dist_results = {}
    for tag in ["delta", "goal"]:
        rho_s, rho_a, feat, state, action = distance_correlation(
            Z[tag][test_mask], S[test_mask], A[test_mask],
            n_pairs=args.n_pairs, rng=rng,
        )
        dist_results[tag] = {
            "rho_state": rho_s, "rho_action": rho_a,
            "feat": feat, "state": state, "action": action,
        }

    # --- Build per-dim labels for the state probe ---
    state_dim_labels = []
    for k in state_keys:
        if "eef_pos" in k:
            for ax in "xyz":
                state_dim_labels.append(f"eef_pos.{ax}")
        elif "eef_quat" in k:
            for ax in "wxyz":
                state_dim_labels.append(f"eef_quat.{ax}")
        elif "gripper" in k:
            state_dim_labels.append("gripper.0")
            state_dim_labels.append("gripper.1")
        else:
            # generic: just enumerate
            for i in range(probe_results["delta"]["r2_state_per_dim"].shape[0]
                           - len(state_dim_labels)):
                state_dim_labels.append(f"{k}.{i}")

    # ---- Figure 1: linear probe R² (state per-dim + action overall) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    x = np.arange(len(state_dim_labels))
    w = 0.4
    axes[0].bar(x - w / 2, probe_results["delta"]["r2_state_per_dim"], w,
                label="delta", color="C0")
    axes[0].bar(x + w / 2, probe_results["goal"]["r2_state_per_dim"], w,
                label="goal", color="C1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(state_dim_labels, rotation=45, ha="right")
    axes[0].set_ylabel("R²  (held-out demos)")
    axes[0].set_title("Linear probe  z → state  (per-dim)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    tags = ["delta", "goal"]
    state_r2 = [probe_results[t]["r2_state"] for t in tags]
    action_r2 = [probe_results[t]["r2_action"] for t in tags]
    x2 = np.arange(2)
    axes[1].bar(x2 - w / 2, state_r2, w, label="z → state", color="C2")
    axes[1].bar(x2 + w / 2, action_r2, w, label="z → action_chunk", color="C3")
    axes[1].set_xticks(x2); axes[1].set_xticklabels(tags)
    axes[1].set_ylabel("R² (held-out demos)")
    axes[1].set_title("Linear probe — overall R²")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    for xi, vs, va in zip(x2, state_r2, action_r2):
        axes[1].text(xi - w / 2, vs + 0.01, f"{vs:.3f}", ha="center", fontsize=9)
        axes[1].text(xi + w / 2, va + 0.01, f"{va:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "linear_probe_r2.png", dpi=140)
    plt.close()

    # ---- Figure 2: distance correlation ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    sample = 4000
    for col, tag in enumerate(tags):
        d = dist_results[tag]
        idx = rng.choice(d["feat"].shape[0], size=min(sample, d["feat"].shape[0]),
                         replace=False)
        # feat vs state
        ax = axes[0][col]
        ax.scatter(d["state"][idx], d["feat"][idx], s=2, alpha=0.3)
        ax.set_xlabel("‖state_i - state_j‖  (physical units)")
        ax.set_ylabel("‖z_i - z_j‖  (feature space)")
        ax.set_title(
            f"{tag} — feat-dist vs state-dist  "
            f"ρ = {d['rho_state']:.3f}"
        )
        ax.grid(True, alpha=0.3)
        # feat vs action
        ax = axes[1][col]
        ax.scatter(d["action"][idx], d["feat"][idx], s=2, alpha=0.3,
                   color="C3")
        ax.set_xlabel("‖action_chunk_i - action_chunk_j‖")
        ax.set_ylabel("‖z_i - z_j‖")
        ax.set_title(
            f"{tag} — feat-dist vs action-dist  "
            f"ρ = {d['rho_action']:.3f}"
        )
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "distance_correlation.png", dpi=140)
    plt.close()

    # ---- Summary ----
    lines = [f"Frames used (train/test): {train_mask.sum()}/{test_mask.sum()}",
             f"State dim ({len(state_dim_labels)}): {state_dim_labels}",
             f"Action-chunk dim: {A.shape[1]}", ""]
    for tag in tags:
        pr = probe_results[tag]; dr = dist_results[tag]
        lines.append(
            f"[{tag}]  R²(state)={pr['r2_state']:.3f}  "
            f"R²(action)={pr['r2_action']:.3f}  "
            f"|  ρ(feat,state)={dr['rho_state']:.3f}  "
            f"ρ(feat,action)={dr['rho_action']:.3f}"
        )
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)
    loguru.logger.info(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
