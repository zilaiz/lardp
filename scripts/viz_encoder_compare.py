"""Compare encoder latent trajectories across DP / IDM / joint-DDT checkpoints.

Generalizes ``viz_idm_latent_trajectories.py`` (delta vs goal IDM) to an
arbitrary set of named encoders. Default supports the three training recipes
we care about:

  * ``dp``         — vanilla flow-matching policy (``train_robomimic.py``),
                     plain ``MultiImageObsEncoder``, flat state dict.
  * ``idm``        — IDM with ``GoalDropoutEncoder`` wrapping
                     (``train_robomimic_idm{,_fdm}.py``); state dict has
                     ``encoder.*`` keys + ``uncond_emb``.
  * ``joint_ddt``  — joint DDT trunk with target LayerNorm
                     (``train_robomimic_lbmdit_joint_ddt.py``,
                     ``slurm/sweep_robomimic_lbmdit_joint_ddt.sh``); state
                     dict is flat, also stores ``target_ln_ema`` which is
                     applied in-trunk on top of the encoder output.

For each provided checkpoint we encode the same expert demos and render:

  1) PCA-2D latent trajectories, one panel per encoder.
  2) Per-step ``cos(z_t, z_{t+1})`` vs time, one panel per encoder.
  3) Per-step ``||z_t - z_T|| / ||z_0 - z_T||``, one panel per encoder.
  4) Per-demo PCA-2D side-by-side (rows=demos, cols=encoders), shared
     per-encoder axis limits.
  5) Per-demo self-similarity heatmap ``cos(z_i, z_j)`` (rows=demos,
     cols=encoders).
  6) Pairwise linear CKA between every pair of encoders.

Usage (any subset of the three; omit a flag to skip that encoder):

    python scripts/viz_encoder_compare.py \\
        --dp_ckpt        logs/.../dp/.../models/model_step_300000.pt \\
        --dp_network     lbmdit \\
        --idm_ckpt       logs/.../idm/.../models/model_step_300000.pt \\
        --idm_network    lbmidm_v2 \\
        --joint_ddt_ckpt logs/.../joint_ddt/.../models/model_step_300000.pt \\
        --joint_ddt_network lbmdit_joint_ddt \\
        --task_config    tool_hang_ph_image_gp \\
        --dataset_path   data/robomimic/tool_hang/ph/image_v15.hdf5 \\
        --out_dir        viz/encoder_compare
"""

from __future__ import annotations

import argparse
import itertools
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_inner_encoder(ckpt_path: str, network_cfg, task_cfg, device: str,
                        use_ema: bool, tag: str):
    """Load the bare ``MultiImageObsEncoder`` from a checkpoint.

    Handles all three save formats:
      * IDM (``GoalDropoutEncoder`` wrapping): state dict has ``encoder.*``
        keys and a top-level ``uncond_emb`` — we slice off the prefix and
        drop the uncond embedding.
      * DP / joint_ddt: flat state dict on the bare encoder.

    EMA weights preferred when available (``encoder_ema``).
    """
    from mip.network_utils import get_encoder

    encoder = get_encoder(network_cfg, task_cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    sd_key = "encoder_ema" if (use_ema and "encoder_ema" in ck) else "encoder"
    if sd_key not in ck:
        raise KeyError(
            f"[{tag}] checkpoint missing '{sd_key}' key. "
            f"Available: {list(ck.keys())}"
        )
    sd = ck[sd_key]

    # Detect GoalDropoutEncoder wrapping: any key starts with 'encoder.'.
    is_wrapped = any(k.startswith("encoder.") for k in sd)
    if is_wrapped:
        sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    loguru.logger.info(
        f"[{tag}] loaded {sd_key} (wrapped={is_wrapped}) | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    encoder.eval().requires_grad_(False)
    return encoder, ck


def _maybe_load_target_ln(ck: dict, network_cfg, device: str, use_ema: bool,
                          tag: str):
    """For joint_ddt checkpoints, build a frozen LayerNorm matching the
    saved ``target_ln_ema`` (or ``target_ln``) and return it. Returns None
    when the checkpoint has no target_ln (DP / IDM).
    """
    ln_key = "target_ln_ema" if (use_ema and "target_ln_ema" in ck) \
        else ("target_ln" if "target_ln" in ck else None)
    if ln_key is None:
        return None

    sd = ck[ln_key]
    obs_dim = network_cfg.get("encoder_out_dim", None) or network_cfg.emb_dim
    # joint_ddt LayerNorm: elementwise_affine flag is implicit in the SD.
    elementwise_affine = "weight" in sd
    ln = torch.nn.LayerNorm(obs_dim, elementwise_affine=elementwise_affine).to(device)
    missing, unexpected = ln.load_state_dict(sd, strict=False)
    loguru.logger.info(
        f"[{tag}] loaded {ln_key} (affine={elementwise_affine}) | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    ln.eval().requires_grad_(False)
    return ln


def _load_normalizer(ckpt_path: str, shared_path: str | None = None):
    """Load normalizer.pkl next to the ckpt; fall back to ``shared_path``.

    Only ``train_robomimic_idm{,_fdm}.py`` saves ``normalizer.pkl`` to disk;
    ``train_robomimic.py`` (DP) and ``train_robomimic_lbmdit_joint_ddt.py``
    do not. The normalizer is a deterministic function of the dataset, so
    reusing the IDM run's normalizer.pkl is safe when all three trained on
    the same task config / dataset.
    """
    p = Path(ckpt_path).parent / "normalizer.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f), str(p)
    if shared_path and Path(shared_path).exists():
        with open(shared_path, "rb") as f:
            return pickle.load(f), shared_path
    raise FileNotFoundError(
        f"No normalizer.pkl at {p} and no usable --shared_normalizer; "
        f"DP and joint_ddt runs don't save normalizers — point "
        f"--shared_normalizer at an IDM run's normalizer.pkl for the same task."
    )


def _compose_cfg(repo_root: Path, config_dir: str, task_config: str,
                 network_config: str):
    """Build a fresh hydra cfg with the requested task/network groups."""
    import hydra
    from hydra import initialize_config_dir
    config_abs = str((repo_root / config_dir).resolve())
    overrides = [f"task={task_config}", f"network={network_config}"]
    with initialize_config_dir(config_dir=config_abs, version_base=None):
        cfg = hydra.compose(config_name="main", overrides=overrides)
    return cfg.task, cfg.network


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _read_demo(h5_demo, normalizer, image_keys: list[str],
               lowdim_keys: list[str]) -> dict[str, np.ndarray]:
    """Read + normalize a single demo's obs streams. Images already CHW
    and normalized. Returns dict of arrays shaped (T, *)."""
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
                 device: str, chunk: int, target_ln=None) -> np.ndarray:
    """Encode an entire demo. Returns z of shape (T, emb_dim).

    If ``target_ln`` is provided (joint_ddt), apply it to every embedding
    so the comparison matches what the trunk actually consumes downstream.
    """
    T = next(iter(obs_arrays.values())).shape[0]
    z_chunks = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        obs_chunk = {}
        for k, v in obs_arrays.items():
            obs_chunk[k] = torch.from_numpy(v[s:e][None]).to(device)  # (1, t, *)
        with torch.no_grad():
            z = encoder(obs_chunk, None)  # (1, t, emb_dim) with keep_horizon_dims
            if target_ln is not None:
                z = target_ln(z)
        z = z.squeeze(0).detach().cpu().numpy()
        z_chunks.append(z)
    return np.concatenate(z_chunks, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centered-kernel alignment between two (N, D) feature sets.

    Invariant to scale and orthogonal rotation, so it is a fair comparison
    even when D differs between encoders.
    """
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
    """``||z_t - z_T|| / ||z_0 - z_T||`` with ``z_T = mean(last 5 frames)``."""
    z_goal = z[-5:].mean(0)
    d = np.linalg.norm(z - z_goal[None], axis=-1)
    d0 = max(d[0], 1e-12)
    return d / d0


# ---------------------------------------------------------------------------
# Linear / MLP probes
# ---------------------------------------------------------------------------

def _build_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Per-frame action chunks ``actions[t : t+horizon]``, with end-of-demo
    padding by replicating the last action. Returns ``(T, horizon * act_dim)``.
    """
    T, A = actions.shape
    pad = np.repeat(actions[-1:], horizon - 1, axis=0)
    a_pad = np.concatenate([actions, pad], axis=0)
    chunks = np.stack([a_pad[t:t + horizon] for t in range(T)], axis=0)
    return chunks.reshape(T, -1)


def _stack_probe_data(z_per_demo: dict[str, np.ndarray],
                      actions_per_demo: dict[str, np.ndarray],
                      names: list[str], horizon: int):
    """Concatenate (z, action_chunk) pairs over the given demo names."""
    Xs, Ys = [], []
    for name in names:
        z = z_per_demo[name]
        y = _build_action_chunks(actions_per_demo[name], horizon)
        n = min(z.shape[0], y.shape[0])
        Xs.append(z[:n]); Ys.append(y[:n])
    return np.concatenate(Xs, 0), np.concatenate(Ys, 0)


def _stack_paired_probe_data(z_per_demo: dict[str, np.ndarray],
                             actions_per_demo: dict[str, np.ndarray],
                             names: list[str], horizon: int,
                             goal_offset: int):
    """Pair each frame t with a future-obs encoding z_goal = z[t+goal_offset]
    (clamped to last frame), so X = [z_obs[t], z_goal[t]] in R^{2D}, Y is
    the action chunk at t. Mirrors the IDM training-time conditioning
    structure, just used as input to a linear probe."""
    Xs, Ys = [], []
    for name in names:
        z = z_per_demo[name]
        T = z.shape[0]
        idx = np.minimum(np.arange(T) + goal_offset, T - 1)
        z_goal = z[idx]
        x = np.concatenate([z, z_goal], axis=1)
        y = _build_action_chunks(actions_per_demo[name], horizon)
        n = min(x.shape[0], y.shape[0])
        Xs.append(x[:n]); Ys.append(y[:n])
    return np.concatenate(Xs, 0), np.concatenate(Ys, 0)


def _linear_probe(X_tr, Y_tr, X_va, Y_va, ridge: float = 1e-3):
    """Closed-form ridge regression on standardized features."""
    mu = X_tr.mean(0, keepdims=True)
    sd = np.clip(X_tr.std(0, keepdims=True), 1e-6, None)
    Xt = np.concatenate([(X_tr - mu) / sd, np.ones((X_tr.shape[0], 1))], 1)
    Xv = np.concatenate([(X_va - mu) / sd, np.ones((X_va.shape[0], 1))], 1)
    XtX = Xt.T @ Xt
    XtY = Xt.T @ Y_tr
    W = np.linalg.solve(XtX + ridge * np.eye(Xt.shape[1]), XtY)
    train_mse = float(((Xt @ W - Y_tr) ** 2).mean())
    val_mse = float(((Xv @ W - Y_va) ** 2).mean())
    return train_mse, val_mse


def _mlp_probe(X_tr, Y_tr, X_va, Y_va, hidden: int = 256, depth: int = 2,
               steps: int = 3000, batch_size: int = 256, lr: float = 1e-3,
               device: str = "cpu"):
    """Small MLP probe with Adam. Reports best val MSE seen during training
    (to remove undertraining and overfitting as confounders)."""
    import torch.nn as nn
    mu = X_tr.mean(0, keepdims=True)
    sd = np.clip(X_tr.std(0, keepdims=True), 1e-6, None)
    Xt = torch.from_numpy(((X_tr - mu) / sd).astype(np.float32)).to(device)
    Xv = torch.from_numpy(((X_va - mu) / sd).astype(np.float32)).to(device)
    Yt = torch.from_numpy(Y_tr.astype(np.float32)).to(device)
    Yv = torch.from_numpy(Y_va.astype(np.float32)).to(device)

    layers, d = [], Xt.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.GELU()]
        d = hidden
    layers += [nn.Linear(d, Yt.shape[1])]
    mlp = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    N = Xt.shape[0]
    best_val = float("inf")
    last_train = float("inf")
    for step in range(steps):
        idx = torch.randperm(N, device=device)[:batch_size]
        loss = ((mlp(Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                v = float(((mlp(Xv) - Yv) ** 2).mean().item())
                t = float(((mlp(Xt) - Yt) ** 2).mean().item())
            best_val = min(best_val, v)
            last_train = t
    return last_train, best_val


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    # Per-encoder ckpt / network config. Any can be omitted to skip.
    p.add_argument("--dp_ckpt",         default=None)
    p.add_argument("--dp_network",      default="lbmdit")
    p.add_argument("--idm_ckpt",        default=None)
    p.add_argument("--idm_network",     default="lbmidm_v2")
    p.add_argument("--joint_ddt_ckpt",  default=None)
    p.add_argument("--joint_ddt_network", default="lbmdit_joint_ddt")
    p.add_argument("--joint_ddt2_ckpt", default=None,
                   help="Optional second joint_ddt checkpoint. Treated "
                        "identically to --joint_ddt_ckpt (target_ln applied, "
                        "solo probe input). Pair with --joint_ddt2_tag to "
                        "label it in plots.")
    p.add_argument("--joint_ddt2_network", default="lbmdit_joint_ddt")
    p.add_argument("--joint_ddt_tag",  default="joint_ddt",
                   help="Display tag for the first joint_ddt ckpt.")
    p.add_argument("--joint_ddt2_tag", default="joint_ddt2",
                   help="Display tag for the second joint_ddt ckpt.")

    # Shared task / dataset.
    p.add_argument("--task_config", default="tool_hang_ph_image_gp")
    p.add_argument("--config_dir",  default="examples/configs")
    p.add_argument("--dataset_path",
                   default="data/robomimic/tool_hang/ph/image_v15.hdf5")

    p.add_argument("--n_demos", type=int, default=8)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema",  dest="use_ema", action="store_false")
    # joint_ddt applies target_ln(encoder(obs)) in-trunk; default to applying
    # it here too so the comparison matches downstream usage. Flip off if you
    # want to compare the raw encoder outputs only.
    p.add_argument("--apply_target_ln", action="store_true", default=True)
    p.add_argument("--no_target_ln", dest="apply_target_ln",
                   action="store_false")
    p.add_argument("--out_dir", default="viz/encoder_compare")
    p.add_argument("--chunk",   type=int, default=64)
    p.add_argument("--shared_normalizer", default=None,
                   help="Fallback normalizer.pkl for ckpts without one "
                        "(DP / joint_ddt). Defaults to the IDM ckpt's "
                        "normalizer.pkl when --idm_ckpt is provided.")
    args = p.parse_args()

    # Auto-discover shared normalizer from the IDM run if not given.
    if args.shared_normalizer is None and args.idm_ckpt:
        cand = Path(args.idm_ckpt).parent / "normalizer.pkl"
        if cand.exists():
            args.shared_normalizer = str(cand)
            loguru.logger.info(
                f"Using IDM run's normalizer as fallback: {args.shared_normalizer}"
            )

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect requested encoders in display order ---
    requested = [
        ("dp",                  args.dp_ckpt,         args.dp_network),
        ("idm",                 args.idm_ckpt,        args.idm_network),
        (args.joint_ddt_tag,    args.joint_ddt_ckpt,  args.joint_ddt_network),
        (args.joint_ddt2_tag,   args.joint_ddt2_ckpt, args.joint_ddt2_network),
    ]
    requested = [(t, c, n) for (t, c, n) in requested if c]
    if len(requested) < 2:
        raise SystemExit(
            "Need at least two of --dp_ckpt / --idm_ckpt / --joint_ddt_ckpt "
            "/ --joint_ddt2_ckpt to do a comparison."
        )
    tags = [t for (t, _, _) in requested]
    loguru.logger.info(f"Comparing encoders: {tags}")

    # --- Pick demos (one shared HDF5 handle for all encoders) ---
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.dataset_path, "r") as f:
        all_demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        chosen = sorted(rng.choice(len(all_demos), size=args.n_demos,
                                    replace=False).tolist())
        names = [all_demos[i] for i in chosen]
        loguru.logger.info(f"Picked demos: {names}")

        # --- Load each (encoder, target_ln?, normalizer) ---
        models = {}
        task_cfg_shared = None
        for tag, ckpt, network_config in requested:
            task_cfg, network_cfg = _compose_cfg(
                repo_root, args.config_dir, args.task_config, network_config,
            )
            if task_cfg_shared is None:
                task_cfg_shared = task_cfg
            enc, ck = _load_inner_encoder(
                ckpt, network_cfg, task_cfg, args.device, args.use_ema, tag,
            )
            tln = None
            if tag.startswith("joint_ddt") and args.apply_target_ln:
                tln = _maybe_load_target_ln(
                    ck, network_cfg, args.device, args.use_ema, tag,
                )
            norm, norm_src = _load_normalizer(ckpt, args.shared_normalizer)
            loguru.logger.info(f"[{tag}] normalizer <- {norm_src}")
            models[tag] = dict(enc=enc, tln=tln, norm=norm)

        image_keys = [k for k, v in task_cfg_shared.shape_meta.obs.items()
                      if v.type == "rgb"]
        lowdim_keys = [k for k, v in task_cfg_shared.shape_meta.obs.items()
                       if v.type == "low_dim"]

        # --- Encode every chosen demo with every model ---
        latents: dict[str, dict[str, np.ndarray]] = {t: {} for t in tags}
        for name in tqdm(names, desc="encoding"):
            for tag in tags:
                m = models[tag]
                arrs = _read_demo(f["data"][name], m["norm"],
                                  image_keys, lowdim_keys)
                latents[tag][name] = _encode_demo(
                    m["enc"], arrs, args.device, args.chunk,
                    target_ln=m["tln"],
                )

        # --- Per-demo normalized actions for linear/MLP probes ---
        # Use any model's normalizer (assumed shared / functionally identical
        # across encoders trained on the same dataset).
        action_norm = next(iter(models.values()))["norm"]["action"]
        demo_actions: dict[str, np.ndarray] = {}
        for name in names:
            a = np.asarray(f["data"][name]["actions"]).astype(np.float32)
            demo_actions[name] = action_norm.normalize(a)

    n_models = len(tags)
    n = len(names)

    # ---------- 1) PCA-2D trajectories, one panel per encoder ----------
    pcas = {}
    plt.figure(figsize=(5.5 * n_models, 5))
    for col, tag in enumerate(tags):
        all_z = np.concatenate([latents[tag][m] for m in names], axis=0)
        pca = PCA(n_components=2).fit(all_z)
        pcas[tag] = pca
        ax = plt.subplot(1, n_models, col + 1)
        cmap = plt.colormaps["viridis"]
        for name in names:
            z2 = pca.transform(latents[tag][name])
            T = z2.shape[0]
            colors = cmap(np.linspace(0, 1, T))
            ax.scatter(z2[:, 0], z2[:, 1], c=colors, s=3, alpha=0.5)
            ax.plot(z2[:, 0], z2[:, 1], lw=0.4, alpha=0.4, color=cmap(0.5))
            ax.scatter(z2[0, 0], z2[0, 1], marker="o", s=40,
                       edgecolor="black", facecolor=cmap(0.0))
            ax.scatter(z2[-1, 0], z2[-1, 1], marker="*", s=80,
                       edgecolor="black", facecolor=cmap(1.0))
        ax.set_title(f"{tag} — PCA-2D  (EV={pca.explained_variance_ratio_.sum():.2f})")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.suptitle("o=start  *=end  (color = time)")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_trajectories.png", dpi=140)
    plt.close()

    # ---------- 2) Per-step consecutive cos-sim vs time ----------
    plt.figure(figsize=(5.5 * n_models, 4))
    for col, tag in enumerate(tags):
        ax = plt.subplot(1, n_models, col + 1)
        for name in names:
            cs = _consecutive_cos_sim(latents[tag][name])
            ax.plot(cs, lw=0.8, alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{tag} — cos(z_t, z_t+1)")
        ax.set_xlabel("t"); ax.set_ylabel("cos sim")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "consecutive_cosine.png", dpi=140)
    plt.close()

    # ---------- 3) Normalized distance-to-final ----------
    plt.figure(figsize=(5.5 * n_models, 4))
    for col, tag in enumerate(tags):
        ax = plt.subplot(1, n_models, col + 1)
        for name in names:
            d = _normalized_distance_to_goal(latents[tag][name])
            ax.plot(d, lw=0.8, alpha=0.6)
        ax.axhline(1.0, color="k", lw=0.4, ls="--")
        ax.axhline(0.0, color="k", lw=0.4, ls="--")
        ax.set_title(f"{tag} — ||z_t - z_T|| (normalized)")
        ax.set_xlabel("t"); ax.set_ylabel("rel L2")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "distance_to_final.png", dpi=140)
    plt.close()

    # ---------- 4) Per-demo side-by-side PCA-2D ----------
    # Precompute shared axis limits per encoder so panels are comparable.
    limits = {}
    for tag in tags:
        all_z2 = pcas[tag].transform(
            np.concatenate([latents[tag][m] for m in names], 0)
        )
        xmin, xmax = all_z2[:, 0].min(), all_z2[:, 0].max()
        ymin, ymax = all_z2[:, 1].min(), all_z2[:, 1].max()
        pad = 0.05 * max(xmax - xmin, ymax - ymin)
        limits[tag] = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)

    cmap = plt.colormaps["viridis"]

    def _draw_pca_grid(page_names, path):
        fig, axes = plt.subplots(
            len(page_names), n_models,
            figsize=(4.5 * n_models, 3.6 * len(page_names)),
            squeeze=False,
        )
        for i, name in enumerate(page_names):
            for col, tag in enumerate(tags):
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
                ax.set_title(f"{name} — {tag} (T={T})", fontsize=10)
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()

    _draw_pca_grid(names, out_dir / "pca_per_demo_compare.png")
    page_size = 4
    for page_idx in range(0, n, page_size):
        page_names = names[page_idx:page_idx + page_size]
        _draw_pca_grid(
            page_names,
            out_dir / f"pca_per_demo_compare_page{page_idx // page_size}.png",
        )

    # ---------- 5) Self-similarity heatmaps ----------
    def _draw_sim_grid(page_names, path):
        fig, axes = plt.subplots(
            len(page_names), n_models,
            figsize=(5.0 * n_models, 4.2 * len(page_names)),
            squeeze=False,
        )
        for i, name in enumerate(page_names):
            for col, tag in enumerate(tags):
                z = latents[tag][name]
                zn = z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True),
                                 1e-12, None)
                sim = zn @ zn.T
                ax = axes[i][col]
                im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1,
                               origin="lower")
                ax.set_title(f"{tag} — cos(z_i, z_j)  {name}", fontsize=10)
                ax.set_xlabel("j"); ax.set_ylabel("i")
                fig.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()

    _draw_sim_grid(names, out_dir / "self_similarity_all_demos.png")
    for page_idx in range(0, n, page_size):
        page_names = names[page_idx:page_idx + page_size]
        _draw_sim_grid(
            page_names,
            out_dir / f"self_similarity_page{page_idx // page_size}.png",
        )

    # ---------- 6) Cross-demo same-phase similarity ----------
    # If the encoder encodes "phase / progress" but not absolute state, then
    # frames at the same normalized phase in *different* demos should look
    # nearly identical (cos sim ≈ 1 across demos). If the encoder preserves
    # state, those off-diagonal entries should be much smaller — different
    # initial conditions / object poses at the same phase look different.
    #
    # The within-demo block structure in `self_similarity_*.png` is consistent
    # with *either* (state-preserving or phase-only), so this panel is the
    # disambiguator.
    phases = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(
        len(phases), n_models,
        figsize=(3.2 * n_models + 0.5, 2.8 * len(phases)),
        squeeze=False,
    )
    phase_off_diag_means: dict[str, list[float]] = {t: [] for t in tags}
    for row, ph in enumerate(phases):
        for col, tag in enumerate(tags):
            zs = []
            for name in names:
                z = latents[tag][name]
                T = z.shape[0]
                idx = min(int(round(ph * (T - 1))), T - 1)
                zs.append(z[idx])
            Z = np.stack(zs, axis=0)  # (N_demos, D)
            Zn = Z / np.clip(np.linalg.norm(Z, axis=-1, keepdims=True),
                              1e-12, None)
            sim = Zn @ Zn.T  # (N_demos, N_demos)
            ax = axes[row][col]
            im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
            ax.set_title(f"{tag} — phase={ph:.2f}", fontsize=10)
            ax.set_xlabel("demo j"); ax.set_ylabel("demo i")
            ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8)
            N = sim.shape[0]
            if N >= 2:
                ut = sim[np.triu_indices(N, k=1)]
                phase_off_diag_means[tag].append(float(ut.mean()))
            else:
                phase_off_diag_means[tag].append(float("nan"))
    plt.suptitle(
        "Cross-demo same-phase cos sim "
        "(off-diag ≈ 1  ⇒  encoder collapses different states sharing a phase)",
        y=1.0,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "cross_demo_same_phase_similarity.png", dpi=130)
    plt.close()

    # Compact summary curve: mean off-diagonal cos sim vs phase, per encoder.
    plt.figure(figsize=(6.5, 4))
    for tag in tags:
        plt.plot(phases, phase_off_diag_means[tag], marker="o", label=tag)
    plt.xlabel("normalized phase  t / T")
    plt.ylabel("mean cross-demo cos sim (off-diagonal)")
    plt.title("Cross-demo collapse at matched phase\n"
              "(higher = encoder gives same z to different states at same phase)")
    plt.ylim(-0.05, 1.05)
    plt.axhline(1.0, color="k", lw=0.4, ls="--")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cross_demo_phase_collapse.png", dpi=140)
    plt.close()

    # ---------- 7) Linear / MLP probes:  z_obs -> action_chunk ----------
    # Tests *representation alignment* (linear) vs *information content* (MLP):
    #   linear MSE: how directly is the action predictable from z_obs?
    #                A low number means features are pre-aligned with the
    #                action manifold (DP's training regime).
    #   MLP MSE:    how predictable is the action with non-linear capacity?
    #                If MLP << linear for a given encoder, the info is there
    #                but rotated / non-linearly entangled — consistent with
    #                IDM (which trained z_obs to be combined with z_goal).
    horizon = int(task_cfg_shared.horizon)
    n_train = max(1, int(round(0.75 * n)))
    train_names = names[:n_train]
    val_names = names[n_train:] if n_train < n else names[-1:]
    loguru.logger.info(
        f"Probe split: {len(train_names)} train demos, {len(val_names)} val demos"
    )

    # Per-encoder probe input mode — matches how each encoder was *used* during
    # training. IDM was always combined with z_goal, so probing IDM with z_obs
    # alone undersells it. DP / joint_ddt take only z_obs at training time, so
    # we probe them solo.
    probe_mode = {"dp": "solo", "idm": "paired", "joint_ddt": "solo"}

    def _build_probe_xy(tag, demo_names):
        if probe_mode.get(tag, "solo") == "paired":
            return _stack_paired_probe_data(
                latents[tag], demo_actions, demo_names, horizon,
                goal_offset=horizon,
            )
        return _stack_probe_data(
            latents[tag], demo_actions, demo_names, horizon,
        )

    probe_results: dict[str, dict[str, tuple[float, float]]] = {}
    for tag in tags:
        X_tr, Y_tr = _build_probe_xy(tag, train_names)
        X_va, Y_va = _build_probe_xy(tag, val_names)
        lin_tr, lin_va = _linear_probe(X_tr, Y_tr, X_va, Y_va)
        mlp_tr, mlp_va = _mlp_probe(X_tr, Y_tr, X_va, Y_va, device=args.device)
        probe_results[tag] = {
            "linear": (lin_tr, lin_va),
            "mlp":    (mlp_tr, mlp_va),
            "mode":   probe_mode.get(tag, "solo"),
        }
        loguru.logger.info(
            f"[{tag}] input={probe_results[tag]['mode']}  "
            f"linear: train={lin_tr:.4f} val={lin_va:.4f}  | "
            f"mlp: train={mlp_tr:.4f} val={mlp_va:.4f}"
        )

    # Bar chart of val MSE per (encoder, probe-type) using the deployment-
    # matched input for each encoder (DP = z_obs only, IDM = [z_obs, z_goal]).
    fig, ax = plt.subplots(figsize=(2.4 * n_models + 2.5, 4.2))
    xs = np.arange(n_models)
    width = 0.35
    xtick_labels = [
        f"{t}\n({probe_results[t]['mode']})" for t in tags
    ]
    lin_vals = [probe_results[t]["linear"][1] for t in tags]
    mlp_vals = [probe_results[t]["mlp"][1]    for t in tags]
    bars_l = ax.bar(xs - width / 2, lin_vals, width, label="linear probe")
    bars_m = ax.bar(xs + width / 2, mlp_vals, width, label="MLP probe")
    for b, v in zip(bars_l, lin_vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    for b, v in zip(bars_m, mlp_vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("val MSE  (normalized actions)")
    ax.set_title(f"Deployment-matched probe val MSE -> action chunk  "
                 f"(horizon={horizon}, {len(train_names)}/{len(val_names)} demos)\n"
                 f"DP: input = z_obs   |   IDM: input = [z_obs, z_goal]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "probe_action_mse.png", dpi=140)
    plt.close()

    # ---------- 8) Pairwise linear CKA + per-model effective-rank stats ----------
    pooled = {tag: np.concatenate([latents[tag][m] for m in names], axis=0)
              for tag in tags}

    summary_lines = ["Pairwise linear CKA:"]
    cka_pairs = list(itertools.combinations(tags, 2))
    cka_matrix = np.eye(n_models)
    tag_idx = {t: i for i, t in enumerate(tags)}
    for a, b in cka_pairs:
        cka = _linear_cka(pooled[a], pooled[b])
        cka_matrix[tag_idx[a], tag_idx[b]] = cka
        cka_matrix[tag_idx[b], tag_idx[a]] = cka
        summary_lines.append(f"  CKA({a:>9}, {b:>9}) = {cka:.4f}")
    summary_lines.append("")

    # Render CKA matrix as a small heatmap too.
    fig, ax = plt.subplots(figsize=(0.9 * n_models + 1.5, 0.9 * n_models + 1.0))
    im = ax.imshow(cka_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n_models), tags)
    ax.set_yticks(range(n_models), tags)
    for i in range(n_models):
        for j in range(n_models):
            ax.text(j, i, f"{cka_matrix[i, j]:.2f}", ha="center",
                    va="center", color="white" if cka_matrix[i, j] < 0.6 else "black",
                    fontsize=10)
    ax.set_title("Pairwise linear CKA")
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.savefig(out_dir / "cka_matrix.png", dpi=140)
    plt.close()

    for tag in tags:
        cos_means = [_consecutive_cos_sim(latents[tag][m]).mean() for m in names]
        z_all = pooled[tag]
        std_per_dim = z_all.std(0).mean()

        Z = z_all - z_all.mean(0, keepdims=True)
        s = np.linalg.svd(Z, compute_uv=False)
        lam = (s ** 2) / max(Z.shape[0] - 1, 1)
        pr_cov = float((lam.sum() ** 2) / max((lam ** 2).sum(), 1e-12))
        p = lam / max(lam.sum(), 1e-12)
        p_pos = p[p > 0]
        eff_rank_entropy = float(np.exp(-(p_pos * np.log(p_pos)).sum()))
        pr_diag = float(
            (z_all.var(0).sum()) ** 2
            / max((z_all.var(0) ** 2).sum(), 1e-12)
        )
        cum = np.cumsum(lam) / max(lam.sum(), 1e-12)
        n_for = lambda thr: int(np.searchsorted(cum, thr) + 1)  # noqa: E731

        summary_lines.append(
            f"[{tag}] D={z_all.shape[1]}  "
            f"mean consecutive cos = {np.mean(cos_means):.4f}  "
            f"feature std (mean) = {std_per_dim:.4f}"
        )
        summary_lines.append(
            f"       PR(cov-eigs) = {pr_cov:.2f}  "
            f"PR(diag-only) = {pr_diag:.2f}  "
            f"eff_rank(entropy) = {eff_rank_entropy:.2f}  "
            f"#dims for 90%/99% var = {n_for(0.90)} / {n_for(0.99)}"
        )

    # Cross-demo same-phase collapse stats.
    summary_lines.append("")
    summary_lines.append(
        "Cross-demo same-phase cos sim (mean off-diagonal)  "
        "— high = encoder collapses different states sharing a phase:"
    )
    header = "  phase  " + "  ".join(f"{t:>10}" for t in tags)
    summary_lines.append(header)
    for i, ph in enumerate(phases):
        row = f"  {ph:>4.2f}  " + "  ".join(
            f"{phase_off_diag_means[t][i]:>10.4f}" for t in tags
        )
        summary_lines.append(row)
    summary_lines.append(
        "  mean  " + "  ".join(
            f"{float(np.nanmean(phase_off_diag_means[t])):>10.4f}" for t in tags
        )
    )

    # Linear / MLP probe results (deployment-matched input per encoder).
    summary_lines.append("")
    summary_lines.append(
        f"-> action_chunk probe val MSE  (horizon={horizon}, "
        f"{len(train_names)}/{len(val_names)} train/val demos, "
        f"each encoder probed with its training-time input):"
    )
    summary_lines.append(
        "  encoder       input               linear-val   mlp-val   gap(lin-mlp)"
    )
    input_label = {"solo": "z_obs", "paired": "[z_obs, z_goal]"}
    for tag in tags:
        _, lin_va = probe_results[tag]["linear"]
        _, mlp_va = probe_results[tag]["mlp"]
        mode = probe_results[tag]["mode"]
        gap = lin_va - mlp_va
        summary_lines.append(
            f"  {tag:<12}  {input_label[mode]:<18}  "
            f"{lin_va:>9.4f}   {mlp_va:>7.4f}   {gap:>+8.4f}"
        )
    summary_lines.append(
        "  Each encoder is probed with the input form it was trained for. "
        "Large gap = info present but non-linearly entangled."
    )

    summary = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)

    loguru.logger.info(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
