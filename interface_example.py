"""Policy inference server for the lardp Franka coffee-pod policy.

Tailored from the diffusion_policy interface_example.py to host an lbmdit
flow-matching policy trained via `examples/train_franka.py`. The Flask
endpoints (`/predict`, `/reset`, `/health`) are kept compatible with the
existing real-robot client; observations and actions are still exchanged
as base64-encoded numpy arrays.

Usage
-----
    python interface_example.py \
        --ckpt_path        logs/<exp>/<ts>/models/model_latest.pt \
        --config_path      outputs/<date>/<time>/.hydra/config.yaml \
        --normalizer_path  checkpoints/<...>_normalizer.npz \
        --port 5000

For the 2026_04_21_02_56_39 lbmdit run:
    --config_path     outputs/2026-04-21/02-56-39/.hydra/config.yaml
    --normalizer_path checkpoints/franka_coffee_pod_cog_lbmdit_normalizer.npz

The normalizer .npz is produced once by `scripts/export_franka_normalizer.py`,
so the inference host doesn't need access to the training HDF5.

Wire format (per-key in the JSON body to /predict)
--------------------------------------------------
    {
      "<obs_key>": {"data": <base64>, "dtype": "<np dtype>", "shape": [...]},
      ...
      "_img_policy": true     # optional; defaults to True since obs has images
    }

When `_img_policy=True`, each value is expected with the time dim already
stacked (shape `(obs_steps, ...)` per key). Otherwise the server adds a
singleton time dim. Images must be (T, 3, H, W) float32 in [0, 1] (CHW,
NOT HWC); lowdim values are sent in their raw (un-normalized) units.
"""

from __future__ import annotations

import base64
import sys
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from omegaconf import OmegaConf

# >>> EDIT ME <<< Absolute path to the lardp repo root on this machine.
# This file is meant to live OUTSIDE the lardp repo (e.g. next to the
# real-robot driver), so we cannot auto-detect the repo location.
LARDP_PATH = "/path/to/lardp"
if LARDP_PATH not in sys.path:
    sys.path.append(LARDP_PATH)

from mip.agent import TrainingAgent
from mip.dataset_utils import MinMaxNormalizer, dict_apply
from mip.franka_inference import decode_action, decode_delta_action

app = Flask(__name__)

# Globals populated by `initialize_policy`
agent: TrainingAgent | None = None
config = None
device: torch.device | None = None
normalizer: dict | None = None  # {"action": MinMaxNormalizer, "obs": {key: Normalizer or None}}

# Visualization
save_obs_dir: str | None = None        # if set, dump received obs as PNGs here
save_obs_max: int = 50                 # cap how many requests we save (avoid filling disk)
_save_obs_count: int = 0


# ---------------------------------------------------------------------------
# Normalizer construction
# ---------------------------------------------------------------------------
def _populate_minmax(target: MinMaxNormalizer, mn, mx, rng):
    """Hydrate a MinMaxNormalizer from precomputed arrays without re-scanning data."""
    target.min = np.asarray(mn, dtype=np.float32)
    target.max = np.asarray(mx, dtype=np.float32)
    target.range = np.asarray(rng, dtype=np.float32)
    return target


def _load_normalizer_from_npz(path: str) -> dict:
    """Load action + low-dim normalizers from a .npz produced by
    `scripts/export_franka_normalizer.py`.

    Image normalization is hardcoded as `x*2-1` (the standard ImageNormalizer)
    and applied inline in `_normalize_obs`, so it doesn't appear here.
    """
    data = np.load(path, allow_pickle=True)
    keys = [str(k) for k in data["keys"]]

    # MinMaxNormalizer.__init__ requires a real array — give it a dummy and
    # then overwrite the stats with the precomputed values.
    def _make(prefix: str) -> MinMaxNormalizer:
        n = MinMaxNormalizer(np.zeros((1, data[f"{prefix}_min"].shape[0]),
                                       dtype=np.float32))
        return _populate_minmax(
            n, data[f"{prefix}_min"], data[f"{prefix}_max"], data[f"{prefix}_range"]
        )

    norm = {"obs": {}, "action": _make("action")}
    for key in keys:
        norm["obs"][key] = _make(f"obs__{key}")

    print(f"[normalizer] loaded from {path}: action range shape="
          f"{norm['action'].range.shape}, lowdim keys={keys}")
    return norm


# ---------------------------------------------------------------------------
# Policy initialization
# ---------------------------------------------------------------------------
def initialize_policy(
    ckpt_path: str,
    config_path: str,
    normalizer_path: str,
    num_steps: int | None = None,
    sample_mode: str | None = None,
    act_steps: int | None = None,
):
    """Build the TrainingAgent, load the checkpoint, and load the normalizer."""
    global agent, config, device, normalizer

    print(f"Loading hydra config: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Inference-time overrides — the trained model is unchanged, but we may
    # want a different number of ODE steps or action chunk size at deploy.
    if num_steps is not None:
        OmegaConf.update(cfg, "optimization.num_steps", int(num_steps), merge=False)
    if sample_mode is not None:
        OmegaConf.update(cfg, "optimization.sample_mode", str(sample_mode), merge=False)
    if act_steps is not None:
        OmegaConf.update(cfg, "task.act_steps", int(act_steps), merge=False)

    # Force inference-friendly settings.
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)

    config = cfg
    device = torch.device(cfg.optimization.device)
    print(f"Device: {device} | num_steps: {cfg.optimization.num_steps} | "
          f"sample_mode: {cfg.optimization.sample_mode} | "
          f"horizon: {cfg.task.horizon} | obs_steps: {cfg.task.obs_steps} | "
          f"act_steps: {cfg.task.act_steps}")

    print(f"Building TrainingAgent({cfg.network.network_type}, "
          f"emb_dim={cfg.network.emb_dim})...")
    agent = TrainingAgent(cfg)

    print(f"Loading checkpoint: {ckpt_path}")
    agent.load(ckpt_path, load_optimizer=False)
    agent.eval()

    print(f"Loading normalizer: {normalizer_path}")
    normalizer = _load_normalizer_from_npz(normalizer_path)

    print("Policy initialized successfully")
    return cfg


# ---------------------------------------------------------------------------
# Per-request helpers
# ---------------------------------------------------------------------------
def _decode_obs_payload(data: dict) -> dict[str, np.ndarray]:
    """base64 -> np.ndarray for each key in the JSON body."""
    obs = {}
    for key, value in data.items():
        array_bytes = base64.b64decode(value["data"])
        arr = np.frombuffer(array_bytes, dtype=value["dtype"]).reshape(value["shape"])
        obs[key] = arr
    return obs


def _save_io_for_inspection(
    obs_np: dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    """Dump received obs + returned action to disk for visual/numeric inspection.

    For each /predict call (up to `save_obs_max` calls):
      - Save each RGB obs key's `obs_steps` time steps as PNG.
      - Append a per-request block to `io_log.txt` with:
          * obs ranges + low-dim values
          * the (act_steps, 10) action chunk, decomposed into pos / rot6d / grip

    Image shape detection:
      - (T, 3, H, W) float [0, 1]  → CHW float, convert to uint8 HWC
      - (T, H, W, 3) uint8         → save directly
      - (T, H, W, 3) float [0, 1]  → scale to uint8
    """
    global _save_obs_count
    if save_obs_dir is None or _save_obs_count >= save_obs_max:
        return

    import os
    from PIL import Image as PILImage

    os.makedirs(save_obs_dir, exist_ok=True)
    idx = _save_obs_count

    log_lines = [f"\n=== request {idx} ==="]
    log_lines.append("[obs]")
    for key, arr in sorted(obs_np.items()):
        # Image keys (4-D): save PNGs, no log entry.
        if arr.ndim == 4:
            for t in range(arr.shape[0]):
                frame = arr[t]
                if frame.shape[0] == 3 and frame.dtype != np.uint8:
                    img = (np.clip(frame, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
                elif frame.shape[-1] == 3 and frame.dtype == np.uint8:
                    img = frame
                elif frame.shape[-1] == 3:
                    img = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                else:
                    continue
                fname = f"req_{idx:04d}_{key}_t{t}.png"
                PILImage.fromarray(img).save(os.path.join(save_obs_dir, fname))
        # Low-dim keys: log values only.
        else:
            log_lines.append(
                f"  {key:25s} shape={arr.shape} dtype={arr.dtype} "
                f"values={arr.tolist()}"
            )

    # action layout is shape-dependent: 10 = abs [pos, rot6d, grip],
    # 7 = chunk-relative delta [pos_d, axis_angle_d, grip].
    is_delta_action = action.shape[-1] == 7
    train_act_min = normalizer["action"].min
    train_act_max = normalizer["action"].max
    if is_delta_action:
        log_lines.append(
            "[action] shape=(act_steps, 7) = [pos_delta(3), axis_angle_delta(3), grip(1)]"
        )
        for t, a in enumerate(action):
            log_lines.append(
                f"  t={t}: pos_d=[{a[0]:+.4f},{a[1]:+.4f},{a[2]:+.4f}]  "
                f"aa_d=[{a[3]:+.4f},{a[4]:+.4f},{a[5]:+.4f}]  "
                f"grip={a[6]:.2f}"
            )
        names = ['pos_dx', 'pos_dy', 'pos_dz',
                 'aa_x', 'aa_y', 'aa_z',
                 'grip']
    else:
        log_lines.append(
            "[action] shape=(act_steps, 10) = [pos(3), rot6d(6), grip(1)]"
        )
        for t, a in enumerate(action):
            log_lines.append(
                f"  t={t}: pos=[{a[0]:+.4f},{a[1]:+.4f},{a[2]:+.4f}]  "
                f"rot6d=[{a[3]:+.4f},{a[4]:+.4f},{a[5]:+.4f},"
                f"{a[6]:+.4f},{a[7]:+.4f},{a[8]:+.4f}]  "
                f"grip={a[9]:.2f}"
            )
        names = ['pos_x', 'pos_y', 'pos_z',
                 'r6d_0', 'r6d_1', 'r6d_2', 'r6d_3', 'r6d_4', 'r6d_5',
                 'grip']

    # Per-channel OOD summary (whole chunk) vs training range
    log_lines.append("[action OOD vs training]")
    a_min = action.min(axis=0)
    a_max = action.max(axis=0)
    for i, nm in enumerate(names):
        flag = ""
        if a_min[i] < train_act_min[i] - 1e-3:
            flag += "↓"
        if a_max[i] > train_act_max[i] + 1e-3:
            flag += "↑"
        log_lines.append(
            f"  {nm:<6} train=[{train_act_min[i]:+.4f},{train_act_max[i]:+.4f}]  "
            f"live=[{a_min[i]:+.4f},{a_max[i]:+.4f}] {flag}"
        )

    with open(os.path.join(save_obs_dir, "io_log.txt"), "a") as f:
        f.write("\n".join(log_lines) + "\n")

    _save_obs_count += 1
    print(f"[save_io] dumped request {idx} to {save_obs_dir} "
          f"({_save_obs_count}/{save_obs_max})", flush=True)


def _normalize_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply train-time normalization, mirroring `FrankaImageDataset.__getitem__`.

    Accepts the obs in the same layout as the source HDF5 (per
    `examples/process_dataset/convert_franka_coffee_pod.py`):
      - RGB:  (T, H, W, 3) uint8  -- HWC, raw 0..255
      - lowdim: (T, dim)   float32 in raw units; (T,) is auto-promoted to (T, 1)

    CHW float32 in [0, 1] is also accepted; it will be detected from shape
    and dtype.

    Drops any key not in `shape_meta["obs"]` (e.g. `is_contact`, extras).
    """
    shape_meta_obs = config.task.shape_meta["obs"]
    out = {}
    for key, arr in obs.items():
        if key not in shape_meta_obs:
            continue
        meta = shape_meta_obs[key]
        meta_type = meta.get("type", "low_dim")
        per_step_expected = tuple(meta["shape"])

        if meta_type == "rgb":
            a = arr
            # Per-step shape = drop the leading time axis (T, ...).
            per_step = a.shape[1:]
            chw = per_step_expected                                     # (3, H, W)
            hwc = (per_step_expected[1], per_step_expected[2], per_step_expected[0])
            if per_step == chw:
                pass
            elif per_step == hwc:
                a = np.moveaxis(a, -1, -3)                              # HWC -> CHW
            else:
                raise ValueError(
                    f"image '{key}' per-step shape {per_step} does not match "
                    f"CHW {chw} or HWC {hwc}"
                )
            a = a.astype(np.float32, copy=False)
            # Scale to [0, 1]: uint8 came in as 0..255; floats may be 0..255 or 0..1.
            if arr.dtype == np.uint8 or float(a.max()) > 1.5:
                a = a / 255.0
            out[key] = (a * 2.0 - 1.0).astype(np.float32)               # ImageNormalizer
        else:
            # Low-dim: auto-promote (T,) -> (T, 1) when the expected per-step
            # is a single scalar (e.g. robot0_gripper_qpos).
            a = arr.astype(np.float32, copy=False)
            if a.ndim == 1 and per_step_expected == (1,):
                a = a[:, None]
            out[key] = normalizer["obs"][key].normalize(a)              # MinMax
    return out


@app.route("/predict", methods=["POST"])
def predict():
    """Sample an action chunk for the given observation."""
    global agent, config, device, normalizer

    if agent is None:
        return jsonify({"error": "Policy not initialized"}), 500

    try:
        data = request.get_json()
        # Default True since the Franka pipeline is image-based and the client
        # naturally has obs_steps frames stacked.
        img_policy = bool(data.pop("_img_policy", True))

        obs_np = _decode_obs_payload(data)
        # Snapshot received obs (pre-normalize) — saved together with the action below.
        obs_np_raw = {k: v.copy() for k, v in obs_np.items()}
        obs_np = _normalize_obs(obs_np)

        with torch.no_grad():
            t0 = time.time()
            if img_policy:
                # Already (obs_steps, ...) — only add batch dim.
                obs_torch = dict_apply(
                    obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                )
            else:
                # Add both batch and singleton time dim.
                obs_torch = dict_apply(
                    obs_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device),
                )

            act_0 = torch.randn(
                (1, config.task.horizon, config.task.act_dim),
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_to_gpu = time.time() - t0

            t0 = time.time()
            act_normed = agent.sample(act_0=act_0, obs=obs_torch, use_ema=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_predict = time.time() - t0

            t0 = time.time()
            act_normed_np = act_normed[0].detach().to("cpu").numpy()  # (horizon, 10)
            act = normalizer["action"].unnormalize(act_normed_np)

            # Slice the executable window: drop pre-obs frames, take act_steps.
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            action = act[start:end].astype(np.float32)            # (act_steps, 7 or 10)

            # Detect action format from the task config.
            # - abs  (10-dim) [pos(3), rot6d(6), gripper(1)]      → decode_action
            # - delta (7-dim) [pos_d(3), axis_angle_d(3), grip(1)] → decode_delta_action
            #                  anchored at the LAST obs frame (un-normalized).
            is_delta = (
                getattr(config.task, "delta_action_anchor", None) == "current_obs"
            )
            if is_delta:
                # Gripper is dim 6 in the 7-dim delta format; binarize same way.
                action[:, 6] = (action[:, 6] > 0.5).astype(np.float32)
                # Anchor: un-normalized EEF pose at the last obs frame
                # (matches the dataset's load-time delta transform anchor).
                anchor_pos = obs_np_raw["robot0_eef_pos"][-1].astype(np.float32)
                anchor_quat_xyzw = obs_np_raw["robot0_eef_quat"][-1].astype(np.float32)
                decoded = decode_delta_action(action, anchor_pos, anchor_quat_xyzw)
            else:
                # Binarize the gripper channel to {0, 1} so the controller's
                # equality-based smoothing (`grasps[i] != grasps[i-1]`) actually
                # fires. Training data was binary; the policy outputs a soft
                # scalar that the dataset's MinMax round-trip leaves continuous.
                action[:, 9] = (action[:, 9] > 0.5).astype(np.float32)
                decoded = decode_action(action)                   # pos / quat_xyzw / gripper
            t_to_cpu = time.time() - t0

        # Diagnostic dump (no-op if --save_obs_dir not set)
        _save_io_for_inspection(obs_np_raw, action)

        action_b64 = base64.b64encode(action.tobytes()).decode("utf-8")
        response = {
            "action": {
                "data": action_b64,
                "dtype": str(action.dtype),
                "shape": list(action.shape),
            },
            # Driver-friendly view; harmless if the client ignores it.
            "decoded": {
                "pos":       decoded["pos"].tolist(),
                "quat_xyzw": decoded["quat_xyzw"].tolist(),
                "gripper":   decoded["gripper"].tolist(),
            },
            "timing": {
                "to_gpu":  t_to_gpu  * 1000,
                "predict": t_predict * 1000,
                "to_cpu":  t_to_cpu  * 1000,
            },
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """No-op: flow-matching policy is stateless across calls."""
    if agent is None:
        return jsonify({"error": "Policy not initialized"}), 500
    return jsonify({"status": "success"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "policy_loaded": agent is not None,
        # Kept for client compatibility — lbmdit does not predict contact.
        "predict_contact": False,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to TrainingAgent checkpoint (.pt)")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to .hydra/config.yaml from the training run")
    parser.add_argument("--normalizer_path", type=str, required=True,
                        help="Path to .npz produced by "
                             "scripts/export_franka_normalizer.py")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override optimization.num_steps for inference")
    parser.add_argument("--sample_mode", type=str, default=None,
                        choices=["stochastic", "zero"],
                        help="Override optimization.sample_mode "
                             "(action ODE source: stochastic noise vs zeros)")
    parser.add_argument("--act_steps", type=int, default=None,
                        help="Override task.act_steps for inference")
    parser.add_argument("--save_obs_dir", type=str, default=None,
                        help="If set, dump received obs (PNGs + log) to this dir "
                             "for visual inspection. Useful for debugging.")
    parser.add_argument("--save_obs_max", type=int, default=50,
                        help="Cap on how many requests' obs we save (default 50)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    save_obs_dir = args.save_obs_dir
    save_obs_max = int(args.save_obs_max)

    initialize_policy(
        ckpt_path=args.ckpt_path,
        config_path=args.config_path,
        normalizer_path=args.normalizer_path,
        num_steps=args.num_steps,
        sample_mode=args.sample_mode,
        act_steps=args.act_steps,
    )

    print(f"Starting policy server on port {args.port}")
    app.run(host="localhost", port=args.port, threaded=False)
