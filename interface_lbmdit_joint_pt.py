"""Policy inference server for the lardp Franka LBMDiTJointPT (E2E) policy.

Sibling of ``interface_lbmdit_joint_ddt.py``. Same Flask endpoints
(``/predict``, ``/reset``, ``/health``), same wire format, same joint Euler
ODE sampler — the only difference is the **joint trunk is the single-stack
PT variant** (one width ``emb_dim``, one depth ``num_layers``, no
encoder/decoder split). All training-time knobs (decoupled t, t-schedule,
SD3 shifts, CFG, target_ln + EMA, optimality / null) are inherited from
the DDT agent and exposed here unchanged.

  - Loads ``LBMDiTJointPTAgent``: trainable encoder + trainable joint PT
    trunk over ``(next_state_embedding, action_chunk)``. The encoder ckpt,
    encoder EMA, joint trunk, trunk EMA, ``target_ln``, and ``target_ln_ema``
    are all restored from the saved ``model_*.pt``.
  - ``optimization.idm_checkpoint_path`` is OPTIONAL — only relevant if the
    training run warm-started the encoder from an IDM ckpt. At deploy time
    we don't reload that init; the checkpoint already has the final encoder
    weights. Override only if your training-time config path is no longer
    valid on this machine and the agent ``__init__`` would otherwise fail.
  - The data normalizer is NOT auto-saved by the joint training pipeline.
    ``--normalizer_path`` is required at deploy time; point it at any
    compatible ``normalizer.pkl``: the sibling of the IDM ckpt the run was
    warm-started from (most common), or one exported from the same HDF5 the
    run trained on. Wrong normalizer = wrong obs/action scale at inference.

Inference knobs (all optional; defaults come from the saved hydra config):
  --joint_num_steps        ODE steps for the joint sampler.
  --joint_sample_mode      "stochastic" | "zero" — initial action stream.
  --joint_t_schedule       "diagonal" | "state_first" | "pyramid" |
                           "action_only" — (t_state, t_action) inference
                           trajectory. "action_only" pins t_state at eps so
                           only the action stream walks (state token stays
                           noisy; use joint_sample_mode=stochastic).
  --joint_pyramid_offset   Offset for the "pyramid" schedule (ignored else).
  --joint_t_eps            Clamp schedule endpoints to [eps, 1-eps].
  --joint_t_shift_state    SD3-style time shift for the state stream.
  --joint_t_shift_action   SD3-style time shift for the action stream.
  --joint_cfg_scale        CFG strength w; v_guided = (1+w)v_cond - w v_uncond.
                           0 = plain conditional sampling.
  --idm_checkpoint_path    Override optimization.idm_checkpoint_path (only
                           matters if the agent __init__ touches the path
                           on this machine; weights themselves come from
                           --ckpt_path).
  --act_steps              Override task.act_steps for the executable slice.

Usage
-----
    python interface_lbmdit_joint_pt.py \\
        --ckpt_path        logs/<exp>/<ts>/models/model_latest.pt \\
        --config_path      outputs/<date>/<time>/.hydra/config.yaml \\
        --normalizer_path  <idm>/normalizer.pkl \\
        --joint_num_steps  5 \\
        --joint_t_schedule diagonal \\
        --port 5000

Wire format: same as interface_example.py — controller code unchanged.
"""

from __future__ import annotations

import base64
import pickle
import sys
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from omegaconf import OmegaConf

# >>> EDIT ME <<< Absolute path to the lardp repo root on this machine.
LARDP_PATH = "/path/to/lardp"
if LARDP_PATH not in sys.path:
    sys.path.append(LARDP_PATH)

from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
from mip.dataset_utils import dict_apply
from mip.franka_inference import decode_action

app = Flask(__name__)

# Globals populated by `initialize_policy`
agent: LBMDiTJointPTAgent | None = None
config = None
device: torch.device | None = None
normalizer: dict | None = None

# Visualization
save_obs_dir: str | None = None
save_obs_max: int = 50
_save_obs_count: int = 0


# ---------------------------------------------------------------------------
# Normalizer loading (pickle from IDM training)
# ---------------------------------------------------------------------------
def _load_normalizer_from_pickle(path: str) -> dict:
    """Load a normalizer pickle.

    Format: ``{"obs": {key: Normalizer}, "action": Normalizer}`` where the
    per-key normalizers are ``MinMaxNormalizer`` (low-dim) or
    ``ImageNormalizer`` (rgb).
    """
    with open(path, "rb") as f:
        norm = pickle.load(f)
    obs_keys = sorted(norm["obs"].keys()) if "obs" in norm else []
    print(f"[normalizer] loaded from {path}: obs keys={obs_keys}, "
          f"action range shape={getattr(norm['action'], 'range', np.zeros(0)).shape}")
    return norm


# ---------------------------------------------------------------------------
# Policy initialization
# ---------------------------------------------------------------------------
def initialize_policy(
    ckpt_path: str,
    config_path: str,
    normalizer_path: str,
    idm_checkpoint_path: str | None = None,
    joint_num_steps: int | None = None,
    joint_sample_mode: str | None = None,
    joint_t_schedule: str | None = None,
    joint_pyramid_offset: float | None = None,
    joint_t_eps: float | None = None,
    joint_t_shift_state: float | None = None,
    joint_t_shift_action: float | None = None,
    joint_cfg_scale: float | None = None,
    act_steps: int | None = None,
):
    """Build the joint-PT-E2E agent, load the checkpoint + normalizer.

    Hydra overrides are applied BEFORE constructing the agent so the agent's
    internal scalar caches (``_num_steps``, ``_sample_mode``, ``_t_schedule``,
    ``_shift_state``/``_shift_action``, ``_cfg_scale``, ...) pick up the
    new values. Order matters: the agent reads these once in ``__init__``.
    """
    global agent, config, device, normalizer

    print(f"Loading hydra config: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Optional override for the IDM warm-start path. The agent only touches
    # this path in __init__ (and only when it points to a real file); at
    # deploy time the final weights come from --ckpt_path, so passing this
    # is usually unnecessary. Override only if the training-time path is no
    # longer valid on this machine.
    if idm_checkpoint_path is not None:
        OmegaConf.update(cfg, "optimization.idm_checkpoint_path",
                         idm_checkpoint_path, merge=False)

    # Joint-trunk inference knobs.
    if joint_num_steps is not None:
        OmegaConf.update(cfg, "optimization.joint_num_steps",
                         int(joint_num_steps), merge=False)
    if joint_sample_mode is not None:
        OmegaConf.update(cfg, "optimization.joint_sample_mode",
                         str(joint_sample_mode), merge=False)
    if joint_t_schedule is not None:
        OmegaConf.update(cfg, "optimization.joint_t_schedule",
                         str(joint_t_schedule), merge=False)
    if joint_pyramid_offset is not None:
        OmegaConf.update(cfg, "optimization.joint_pyramid_offset",
                         float(joint_pyramid_offset), merge=False)
    if joint_t_eps is not None:
        OmegaConf.update(cfg, "optimization.joint_t_eps",
                         float(joint_t_eps), merge=False)
    if joint_t_shift_state is not None:
        OmegaConf.update(cfg, "optimization.joint_t_shift_state",
                         float(joint_t_shift_state), merge=False)
    if joint_t_shift_action is not None:
        OmegaConf.update(cfg, "optimization.joint_t_shift_action",
                         float(joint_t_shift_action), merge=False)
    if joint_cfg_scale is not None:
        OmegaConf.update(cfg, "optimization.joint_cfg_scale",
                         float(joint_cfg_scale), merge=False)
    if act_steps is not None:
        OmegaConf.update(cfg, "task.act_steps", int(act_steps), merge=False)

    # Force inference-friendly settings.
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)

    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)

    config = cfg
    device = torch.device(cfg.optimization.device)
    print(
        f"Device: {device} | joint_num_steps: {cfg.optimization.joint_num_steps} | "
        f"sample_mode: {cfg.optimization.joint_sample_mode} | "
        f"schedule: {cfg.optimization.joint_t_schedule} | "
        f"pyramid_offset: {cfg.optimization.joint_pyramid_offset} | "
        f"t_eps: {cfg.optimization.joint_t_eps} | "
        f"shift_state: {cfg.optimization.joint_t_shift_state} | "
        f"shift_action: {cfg.optimization.joint_t_shift_action} | "
        f"cfg_scale: {cfg.optimization.joint_cfg_scale} | "
        f"horizon: {cfg.task.horizon} | obs_steps: {cfg.task.obs_steps} | "
        f"act_steps: {cfg.task.act_steps}"
    )

    print(
        f"Building LBMDiTJointPTAgent({cfg.network.network_type}, "
        f"d_model={cfg.network.emb_dim}, depth={cfg.network.num_layers}, "
        f"cond_compose={cfg.network.joint_cond_compose}) — "
        f"encoder trains end-to-end; target_ln replaces offline goal stats..."
    )
    agent = LBMDiTJointPTAgent(cfg)

    print(f"Loading joint trunk + encoder + target_ln checkpoint: {ckpt_path}")
    agent.load(ckpt_path, load_optimizer=False)
    agent.eval()

    print(f"Loading normalizer (pickle): {normalizer_path}")
    normalizer = _load_normalizer_from_pickle(normalizer_path)

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


def _normalize_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply train-time normalization, mirroring ``RobomimicImageIDMDataset.__getitem__``.

    Per-key normalizers come from the loaded pickle. The image path shapes
    the input into CHW float in [0, 1] and then calls ``.normalize(...)``
    (``ImageNormalizer`` is ``x*2 - 1``). Drops any key not in
    ``shape_meta["obs"]``.
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
            per_step = a.shape[1:]
            chw = per_step_expected
            hwc = (per_step_expected[1], per_step_expected[2], per_step_expected[0])
            if per_step == chw:
                pass
            elif per_step == hwc:
                a = np.moveaxis(a, -1, -3)
            else:
                raise ValueError(
                    f"image '{key}' per-step shape {per_step} does not match "
                    f"CHW {chw} or HWC {hwc}"
                )
            a = a.astype(np.float32, copy=False)
            if arr.dtype == np.uint8 or float(a.max()) > 1.5:
                a = a / 255.0
            out[key] = normalizer["obs"][key].normalize(a).astype(np.float32)
        else:
            a = arr.astype(np.float32, copy=False)
            if a.ndim == 1 and per_step_expected == (1,):
                a = a[:, None]
            out[key] = normalizer["obs"][key].normalize(a).astype(np.float32)
    return out


def _save_io_for_inspection(
    obs_np: dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    """Dump received obs + returned action to disk for inspection (no-op if disabled)."""
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
        else:
            log_lines.append(
                f"  {key:25s} shape={arr.shape} dtype={arr.dtype} "
                f"values={arr.tolist()}"
            )

    log_lines.append("[action] shape=(act_steps, 10) = [pos(3), rot6d(6), grip(1)]")
    train_act_min = normalizer["action"].min
    train_act_max = normalizer["action"].max
    for t, a in enumerate(action):
        log_lines.append(
            f"  t={t}: pos=[{a[0]:+.4f},{a[1]:+.4f},{a[2]:+.4f}]  "
            f"rot6d=[{a[3]:+.4f},{a[4]:+.4f},{a[5]:+.4f},"
            f"{a[6]:+.4f},{a[7]:+.4f},{a[8]:+.4f}]  "
            f"grip={a[9]:.2f}"
        )

    log_lines.append("[action OOD vs training]")
    a_min = action.min(axis=0)
    a_max = action.max(axis=0)
    names = ['pos_x', 'pos_y', 'pos_z',
             'r6d_0', 'r6d_1', 'r6d_2', 'r6d_3', 'r6d_4', 'r6d_5',
             'grip']
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Sample an action chunk via the joint PT sampler."""
    global agent, config, device, normalizer

    if agent is None:
        return jsonify({"error": "Policy not initialized"}), 500

    try:
        data = request.get_json()
        img_policy = bool(data.pop("_img_policy", True))

        obs_np = _decode_obs_payload(data)
        obs_np_raw = {k: v.copy() for k, v in obs_np.items()}
        obs_np = _normalize_obs(obs_np)

        with torch.no_grad():
            t0 = time.time()
            if img_policy:
                obs_torch = dict_apply(
                    obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                )
            else:
                obs_torch = dict_apply(
                    obs_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device),
                )

            # Initial action noise. The agent draws state noise internally
            # (no `state_0` is needed); `_sample_mode` controls whether it's
            # stochastic or zero.
            act_0 = torch.randn(
                (1, config.task.horizon, config.task.act_dim),
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_to_gpu = time.time() - t0

            t0 = time.time()
            # Pass num_steps explicitly so it composes with whatever was set
            # at agent build time (the agent default is `_num_steps` from cfg).
            act_normed = agent.sample(
                act_0=act_0,
                obs=obs_torch,
                num_steps=int(config.optimization.joint_num_steps),
                use_ema=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_predict = time.time() - t0

            t0 = time.time()
            act_normed_np = act_normed[0].detach().to("cpu").numpy()  # (horizon, 10)
            act = normalizer["action"].unnormalize(act_normed_np)

            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            action = act[start:end].astype(np.float32)            # (act_steps, 10)

            # Same gripper binarization as the other franka interfaces — the
            # controller's smoothing block requires {0, 1}.
            action[:, 9] = (action[:, 9] > 0.5).astype(np.float32)

            decoded = decode_action(action)
            t_to_cpu = time.time() - t0

        _save_io_for_inspection(obs_np_raw, action)

        action_b64 = base64.b64encode(action.tobytes()).decode("utf-8")
        response = {
            "action": {
                "data": action_b64,
                "dtype": str(action.dtype),
                "shape": list(action.shape),
            },
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
    """No-op: joint-PT policy is stateless across calls."""
    if agent is None:
        return jsonify({"error": "Policy not initialized"}), 500
    return jsonify({"status": "success"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "policy_loaded": agent is not None,
        "predict_contact": False,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to LBMDiTJointPTAgent checkpoint (.pt)")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to .hydra/config.yaml from the training run")
    parser.add_argument("--normalizer_path", type=str, required=True,
                        help="Path to normalizer.pkl (sibling of the IDM ckpt "
                             "the run was warm-started from, or one exported "
                             "from the same HDF5)")

    # Optional override for the IDM warm-start path. Only matters if the
    # training-time path is no longer valid on this machine; the final
    # weights come from --ckpt_path.
    parser.add_argument("--idm_checkpoint_path", type=str, default=None,
                        help="Override optimization.idm_checkpoint_path")

    # Joint-trunk sampler knobs.
    parser.add_argument("--joint_num_steps", type=int, default=None,
                        help="Override optimization.joint_num_steps "
                             "(Euler steps for the joint sampler)")
    parser.add_argument("--joint_sample_mode", type=str, default=None,
                        choices=["stochastic", "zero"],
                        help="Override optimization.joint_sample_mode "
                             "(initial action/state stream)")
    parser.add_argument("--joint_t_schedule", type=str, default=None,
                        choices=["diagonal", "state_first", "pyramid",
                                 "action_only"],
                        help="Override optimization.joint_t_schedule "
                             "((t_state, t_action) inference trajectory). "
                             "'action_only' pins t_state at eps; pair with "
                             "joint_sample_mode=stochastic.")
    parser.add_argument("--joint_pyramid_offset", type=float, default=None,
                        help="Override optimization.joint_pyramid_offset "
                             "(only used when schedule == 'pyramid')")
    parser.add_argument("--joint_t_eps", type=float, default=None,
                        help="Override optimization.joint_t_eps "
                             "(clamp schedule endpoints to [eps, 1-eps])")
    parser.add_argument("--joint_t_shift_state", type=float, default=None,
                        help="Override optimization.joint_t_shift_state "
                             "(SD3-style shift on the state stream)")
    parser.add_argument("--joint_t_shift_action", type=float, default=None,
                        help="Override optimization.joint_t_shift_action "
                             "(SD3-style shift on the action stream)")
    parser.add_argument("--joint_cfg_scale", type=float, default=None,
                        help="Override optimization.joint_cfg_scale "
                             "(CFG strength w; 0 = plain conditional)")

    parser.add_argument("--act_steps", type=int, default=None,
                        help="Override task.act_steps for the executable slice")

    parser.add_argument("--save_obs_dir", type=str, default=None,
                        help="If set, dump received obs (PNGs + log) to this dir")
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
        idm_checkpoint_path=args.idm_checkpoint_path,
        joint_num_steps=args.joint_num_steps,
        joint_sample_mode=args.joint_sample_mode,
        joint_t_schedule=args.joint_t_schedule,
        joint_pyramid_offset=args.joint_pyramid_offset,
        joint_t_eps=args.joint_t_eps,
        joint_t_shift_state=args.joint_t_shift_state,
        joint_t_shift_action=args.joint_t_shift_action,
        joint_cfg_scale=args.joint_cfg_scale,
        act_steps=args.act_steps,
    )

    print(f"Starting policy server on port {args.port}")
    app.run(host="localhost", port=args.port, threaded=False)
