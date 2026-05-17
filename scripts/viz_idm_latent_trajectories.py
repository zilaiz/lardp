"""Compare encoder latent trajectories of a delta-cond vs goal-cond IDM.

Loads two IDM checkpoints (same encoder architecture, different conditioning
target during pretraining), encodes the same set of expert demos with each
encoder, and renders:

  1) PCA-2D latent trajectories per model (one panel per model, all sampled
     demos overlaid, color = time within demo).
  2) Per-step consecutive cosine similarity ``cos(z_t, z_{t+1})`` vs time.
  3) Per-step normalized distance to final embedding ``||z_t-z_T||/||z_0-z_T||``.
  4) Self-similarity heatmap ``cos(z_i, z_j)`` for one example demo per model.
  5) Cross-encoder linear CKA between the two encoders (single scalar).

Defaults target tool_hang/lbmidm_v2 (the only pair of checkpoints we have).
Network and task configs are inferred from the standard hydra group names —
no per-checkpoint hydra config is needed.

Usage:
    python scripts/viz_idm_latent_trajectories.py \
        --delta_ckpt logs/.../delta.../models/model_step_300000.pt \
        --goal_ckpt  logs/.../goal.../models/model_step_300000.pt \
        --task_config tool_hang_ph_image_gp \
        --network_config lbmidm_v2 \
        --out_dir viz/idm_latent_delta_vs_goal
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
from sklearn.decomposition import PCA
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_inner_encoder(ckpt_path: str, task_cfg, network_cfg, device: str,
                        use_ema: bool):
    """Load the bare MultiImageObsEncoder from a checkpoint.

    Strips any ``GoalDropoutEncoder`` wrapping so we get back the same module
    on both branches (delta and goal). EMA weights preferred if available.
    """
    from mip.network_utils import get_encoder

    encoder = get_encoder(network_cfg, task_cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd_key = "encoder_ema" if (use_ema and "encoder_ema" in ck) else "encoder"
    sd = ck[sd_key]

    is_wrapped = any(k.startswith("encoder.") for k in sd)
    if is_wrapped:
        sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    loguru.logger.info(
        f"[{Path(ckpt_path).parent.parent.name}] loaded {sd_key} "
        f"(wrapped={is_wrapped}) | missing={len(missing)} unexpected={len(unexpected)}"
    )
    encoder.eval().requires_grad_(False)
    return encoder


def _load_normalizer(ckpt_path: str):
    p = Path(ckpt_path).parent / "normalizer.pkl"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        return pickle.load(f)


def _read_demo(h5_demo, normalizer, image_keys: list[str],
               lowdim_keys: list[str]) -> dict[str, np.ndarray]:
    """Read + normalize a single demo's obs streams. Returns dict of arrays
    shaped (T, *), with images already CHW and normalized."""
    out = {}
    for k in image_keys:
        x = np.asarray(h5_demo["obs"][k]).astype(np.float32) / 255.0  # (T, H, W, C)
        x = np.moveaxis(x, -1, 1)                                    # (T, C, H, W)
        x = normalizer["obs"][k].normalize(x)
        out[k] = x
    for k in lowdim_keys:
        x = np.asarray(h5_demo["obs"][k]).astype(np.float32)         # (T, D)
        x = normalizer["obs"][k].normalize(x)
        out[k] = x
    return out


def _encode_demo(encoder, obs_arrays: dict[str, np.ndarray],
                 device: str, chunk: int = 64) -> np.ndarray:
    """Encode an entire demo. Returns z of shape (T, emb_dim)."""
    T = next(iter(obs_arrays.values())).shape[0]
    z_chunks = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        obs_chunk = {}
        for k, v in obs_arrays.items():
            obs_chunk[k] = torch.from_numpy(v[s:e][None]).to(device)  # (1, t, *)
        with torch.no_grad():
            z = encoder(obs_chunk, None)  # (1, t, emb_dim) if keep_horizon_dims
        z = z.squeeze(0).detach().cpu().numpy()
        z_chunks.append(z)
    return np.concatenate(z_chunks, axis=0)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centered-kernel alignment between two (N, D) feature sets."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / max(den, 1e-12))


def _consecutive_cos_sim(z: np.ndarray) -> np.ndarray:
    a = z[:-1]
    b = z[1:]
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.clip(den, 1e-12, None)


def _normalized_distance_to_goal(z: np.ndarray) -> np.ndarray:
    """||z_t - z_T|| / ||z_0 - z_T||, with z_T = mean of last 5 frames."""
    z_goal = z[-5:].mean(0)
    d = np.linalg.norm(z - z_goal[None], axis=-1)
    d0 = max(d[0], 1e-12)
    return d / d0


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta_ckpt", default="logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_delta_cond_fdm1.0_gdp0.0_ac8_ndp_0/2026_05_03_22_36_46/models/model_step_300000.pt")
    p.add_argument("--goal_ckpt",  default="logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_fdm_aux/2026_04_27_01_03_55/models/model_step_300000.pt")
    p.add_argument("--task_config", default="tool_hang_ph_image_gp")
    p.add_argument("--network_config", default="lbmidm_v2")
    p.add_argument("--config_dir", default="examples/configs")
    p.add_argument("--dataset_path", default="data/robomimic/tool_hang/ph/image_v15.hdf5")
    p.add_argument("--n_demos", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--out_dir", default="viz/idm_latent_delta_vs_goal")
    p.add_argument("--chunk", type=int, default=64)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Hydra: build task_cfg + network_cfg once (shared between both) ---
    import hydra
    from hydra import initialize_config_dir
    config_abs = str((repo_root / args.config_dir).resolve())
    overrides = [f"task={args.task_config}", f"network={args.network_config}"]
    with initialize_config_dir(config_dir=config_abs, version_base=None):
        cfg = hydra.compose(config_name="main", overrides=overrides)
    task_cfg, network_cfg = cfg.task, cfg.network

    # --- Pick demos to encode ---
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.dataset_path, "r") as f:
        all_demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        chosen = sorted(rng.choice(len(all_demos), size=args.n_demos,
                                    replace=False).tolist())
        names = [all_demos[i] for i in chosen]
        loguru.logger.info(f"Picked demos: {names}")

        # --- Load both encoders + their normalizers ---
        models = {}
        for tag, ckpt in [("delta", args.delta_ckpt), ("goal", args.goal_ckpt)]:
            enc = _load_inner_encoder(ckpt, task_cfg, network_cfg,
                                       args.device, args.use_ema)
            norm = _load_normalizer(ckpt)
            models[tag] = (enc, norm)

        image_keys = [k for k, v in task_cfg.shape_meta.obs.items()
                      if v.type == "rgb"]
        lowdim_keys = [k for k, v in task_cfg.shape_meta.obs.items()
                       if v.type == "low_dim"]

        # --- Encode every chosen demo with both models ---
        latents = {tag: {} for tag in models}  # tag -> {name: (T, D)}
        for name in tqdm(names, desc="encoding"):
            for tag, (enc, norm) in models.items():
                arrs = _read_demo(f["data"][name], norm, image_keys, lowdim_keys)
                latents[tag][name] = _encode_demo(enc, arrs, args.device, args.chunk)

    # --- 1) PCA-2D trajectories per model ---
    plt.figure(figsize=(12, 5))
    pcas = {}
    for col, tag in enumerate(["delta", "goal"]):
        all_z = np.concatenate([latents[tag][n] for n in names], axis=0)
        pca = PCA(n_components=2).fit(all_z)
        pcas[tag] = pca
        ax = plt.subplot(1, 2, col + 1)
        cmap = plt.colormaps["viridis"]
        for i, n in enumerate(names):
            z2 = pca.transform(latents[tag][n])
            T = z2.shape[0]
            colors = cmap(np.linspace(0, 1, T))
            ax.scatter(z2[:, 0], z2[:, 1], c=colors, s=3, alpha=0.5)
            ax.plot(z2[:, 0], z2[:, 1], lw=0.4, alpha=0.4,
                    color=cmap(0.5))
            ax.scatter(z2[0, 0], z2[0, 1], marker="o", s=40,
                       edgecolor="black", facecolor=cmap(0.0))
            ax.scatter(z2[-1, 0], z2[-1, 1], marker="*", s=80,
                       edgecolor="black", facecolor=cmap(1.0))
        ax.set_title(f"{tag} encoder — PCA-2D latent trajectories "
                     f"(EV={pca.explained_variance_ratio_.sum():.2f})")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.suptitle("o=start  *=end  (color = time)")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_trajectories.png", dpi=140)
    plt.close()

    # --- 2) Per-step consecutive cos_sim vs time ---
    plt.figure(figsize=(12, 4))
    for col, tag in enumerate(["delta", "goal"]):
        ax = plt.subplot(1, 2, col + 1)
        for n in names:
            cs = _consecutive_cos_sim(latents[tag][n])
            ax.plot(cs, lw=0.8, alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{tag} encoder — cos(z_t, z_t+1)")
        ax.set_xlabel("t"); ax.set_ylabel("cos sim")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "consecutive_cosine.png", dpi=140)
    plt.close()

    # --- 3) Normalized distance-to-final ---
    plt.figure(figsize=(12, 4))
    for col, tag in enumerate(["delta", "goal"]):
        ax = plt.subplot(1, 2, col + 1)
        for n in names:
            d = _normalized_distance_to_goal(latents[tag][n])
            ax.plot(d, lw=0.8, alpha=0.6)
        ax.axhline(1.0, color="k", lw=0.4, ls="--")
        ax.axhline(0.0, color="k", lw=0.4, ls="--")
        ax.set_title(f"{tag} encoder — ||z_t - z_T||  (normalized by t=0)")
        ax.set_xlabel("t"); ax.set_ylabel("rel L2")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "distance_to_final.png", dpi=140)
    plt.close()

    # --- 4a) Per-demo side-by-side PCA-2D (delta | goal columns, demo per row) ---
    n = len(names)
    # Precompute shared axes limits per encoder so panels are comparable
    limits = {}
    for tag in ["delta", "goal"]:
        all_z2 = pcas[tag].transform(
            np.concatenate([latents[tag][m] for m in names], 0)
        )
        xmin, xmax = all_z2[:, 0].min(), all_z2[:, 0].max()
        ymin, ymax = all_z2[:, 1].min(), all_z2[:, 1].max()
        pad = 0.05 * max(xmax - xmin, ymax - ymin)
        limits[tag] = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)

    cmap = plt.colormaps["viridis"]
    fig, axes = plt.subplots(n, 2, figsize=(8.5, 3.6 * n), squeeze=False)
    for i, name in enumerate(names):
        for col, tag in enumerate(["delta", "goal"]):
            ax = axes[i][col]
            z2 = pcas[tag].transform(latents[tag][name])
            T = z2.shape[0]
            colors = cmap(np.linspace(0, 1, T))
            ax.scatter(z2[:, 0], z2[:, 1], c=colors, s=5, alpha=0.7)
            ax.plot(z2[:, 0], z2[:, 1], lw=0.5, alpha=0.35, color=cmap(0.5))
            ax.scatter(z2[0, 0], z2[0, 1], marker="o", s=60,
                       edgecolor="black", facecolor=cmap(0.0))
            ax.scatter(z2[-1, 0], z2[-1, 1], marker="*", s=110,
                       edgecolor="black", facecolor=cmap(1.0))
            xmin, xmax, ymin, ymax = limits[tag]
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_title(f"{name} — {tag}  (T={T})", fontsize=10)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Per-demo PCA-2D (shared per-encoder basis)  "
        f"EV[delta]={pcas['delta'].explained_variance_ratio_.sum():.2f}  "
        f"EV[goal]={pcas['goal'].explained_variance_ratio_.sum():.2f}",
        y=1.0,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "pca_per_demo_compare.png", dpi=140)
    plt.close()

    # Paginated version (4 demos per page) for easier viewing
    page_size = 4
    for page_idx in range(0, n, page_size):
        page_names = names[page_idx:page_idx + page_size]
        fig, axes = plt.subplots(len(page_names), 2,
                                 figsize=(8.5, 3.6 * len(page_names)),
                                 squeeze=False)
        for i, name in enumerate(page_names):
            for col, tag in enumerate(["delta", "goal"]):
                ax = axes[i][col]
                z2 = pcas[tag].transform(latents[tag][name])
                T = z2.shape[0]
                colors = cmap(np.linspace(0, 1, T))
                ax.scatter(z2[:, 0], z2[:, 1], c=colors, s=5, alpha=0.7)
                ax.plot(z2[:, 0], z2[:, 1], lw=0.5, alpha=0.35,
                        color=cmap(0.5))
                ax.scatter(z2[0, 0], z2[0, 1], marker="o", s=60,
                           edgecolor="black", facecolor=cmap(0.0))
                ax.scatter(z2[-1, 0], z2[-1, 1], marker="*", s=110,
                           edgecolor="black", facecolor=cmap(1.0))
                xmin, xmax, ymin, ymax = limits[tag]
                ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
                ax.set_title(f"{name} — {tag}  (T={T})", fontsize=10)
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"pca_per_demo_compare_page{page_idx // page_size}.png",
                    dpi=140)
        plt.close()

    # --- 4b) Per-demo self-similarity grid: 1 row per demo, 2 cols (delta, goal) ---
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.2 * n), squeeze=False)
    for i, name in enumerate(names):
        for col, tag in enumerate(["delta", "goal"]):
            z = latents[tag][name]
            zn = z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12, None)
            sim = zn @ zn.T
            ax = axes[i][col]
            im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
            ax.set_title(f"{tag} — cos(z_i, z_j)  {name}", fontsize=10)
            ax.set_xlabel("j"); ax.set_ylabel("i")
            fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.savefig(out_dir / "self_similarity_all_demos.png", dpi=130)
    plt.close()

    # Also keep a compact version: 2 cols × n rows split into pages of 4 demos
    page_size = 4
    for page_idx in range(0, n, page_size):
        page_names = names[page_idx:page_idx + page_size]
        fig, axes = plt.subplots(len(page_names), 2,
                                 figsize=(10, 4.2 * len(page_names)),
                                 squeeze=False)
        for i, name in enumerate(page_names):
            for col, tag in enumerate(["delta", "goal"]):
                z = latents[tag][name]
                zn = z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True),
                                 1e-12, None)
                sim = zn @ zn.T
                ax = axes[i][col]
                im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
                ax.set_title(f"{tag} — cos(z_i, z_j)  {name}", fontsize=10)
                ax.set_xlabel("j"); ax.set_ylabel("i")
                fig.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        plt.savefig(out_dir / f"self_similarity_page{page_idx // page_size}.png",
                    dpi=130)
        plt.close()

    # --- 5) Linear CKA between the two encoders ---
    X = np.concatenate([latents["delta"][n] for n in names], axis=0)
    Y = np.concatenate([latents["goal"][n] for n in names], axis=0)
    cka = _linear_cka(X, Y)

    # Per-model summary stats
    summary_lines = [f"linear CKA(delta, goal) = {cka:.4f}", ""]
    for tag in ["delta", "goal"]:
        cos_means = [_consecutive_cos_sim(latents[tag][n]).mean() for n in names]
        z_all = np.concatenate([latents[tag][n] for n in names], axis=0)
        std_per_dim = z_all.std(0).mean()

        # Effective rank from the eigenspectrum of the feature covariance.
        Z = z_all - z_all.mean(0, keepdims=True)
        s = np.linalg.svd(Z, compute_uv=False)
        lam = (s ** 2) / max(Z.shape[0] - 1, 1)          # eigenvalues of cov
        pr_cov = float((lam.sum() ** 2) / max((lam ** 2).sum(), 1e-12))
        # Roy & Vetterli entropy-based effective rank
        p = lam / max(lam.sum(), 1e-12)
        p_pos = p[p > 0]
        eff_rank_entropy = float(np.exp(-(p_pos * np.log(p_pos)).sum()))
        # Diagonal-only (what we had before — kept for comparison)
        pr_diag = float(
            (z_all.var(0).sum()) ** 2
            / max((z_all.var(0) ** 2).sum(), 1e-12)
        )
        # 90%/99% variance thresholds
        cum = np.cumsum(lam) / max(lam.sum(), 1e-12)
        n_for = lambda thr: int(np.searchsorted(cum, thr) + 1)  # noqa: E731

        summary_lines.append(
            f"[{tag}] mean consecutive cos = {np.mean(cos_means):.4f} "
            f"| feature std (mean) = {std_per_dim:.4f}"
        )
        summary_lines.append(
            f"       PR(cov-eigs) = {pr_cov:.2f}  "
            f"PR(diag-only) = {pr_diag:.2f}  "
            f"eff_rank(entropy) = {eff_rank_entropy:.2f}  "
            f"#dims for 90%/99% var = {n_for(0.90)} / {n_for(0.99)}"
        )
    summary = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)

    loguru.logger.info(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
