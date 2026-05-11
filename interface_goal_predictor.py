"""Policy inference server for the lardp Franka goal_predictor_dit_v2 policy.

Mirrors `interface_example.py` (which serves an lbmdit checkpoint) but loads a
`GoalPredictorDiTAgent` instead of `TrainingAgent`. The Flask endpoints
(`/predict`, `/reset`, `/health`) are kept identical so the existing
`PolicyClient` on the controller side works unchanged.

What's different vs interface_example.py:
  - Loads `GoalPredictorDiTAgent` (frozen IDM + trainable goal-predictor DiT).
  - The agent's `__init__` itself loads the frozen IDM checkpoint and
    (optionally) the precomputed goal stats — both paths are read from the
    saved hydra `optimization.{idm_checkpoint_path, goal_stats_path}`.
  - Normalizer is a pickle (`normalizer.pkl`) saved by the IDM training,
    not the .npz used by the lbmdit pipeline. It contains live normalizer
    objects (MinMaxNormalizer / ImageNormalizer), so we just load and call
    `.normalize(...)` / `.unnormalize(...)` directly.
  - One extra inference knob: `--goal_flow_num_steps` (ODE steps for the
    goal predictor's diffusion). `--num_steps` still controls the IDM action
    sampler.

Usage
-----
    python interface_goal_predictor.py \
        --ckpt_path        logs/<exp>/<ts>/models/model_latest.pt \
        --config_path      outputs/<date>/<time>/.hydra/config.yaml \
        --normalizer_path  <path>/normalizer.pkl \
        --port 5000

Optional overrides (otherwise read from the saved hydra config):
    --idm_checkpoint_path  <path to IDM .pt>
    --goal_stats_path      <path to goal stats .pt>
    --num_steps            1
    --goal_flow_num_steps  5
    --act_steps            8
    --save_obs_dir         /tmp/policy_io
    --save_obs_max         50

Wire format (same as interface_example.py — controller code unchanged).
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

from mip.agent_goal_predictor_dit import GoalPredictorDiTAgent
from mip.dataset_utils import dict_apply
from mip.franka_inference import decode_action

app = Flask(__name__)

# Globals populated by `initialize_policy`
agent: GoalPredictorDiTAgent | None = None
config = None
device: torch.device | None = None
normalizer: dict | None = None  # {"obs": {key: Normalizer}, "action": MinMaxNormalizer}

# Visualization
save_obs_dir: str | None = None
save_obs_max: int = 50
_save_obs_count: int = 0


# ---------------------------------------------------------------------------
# Normalizer loading (pickle from IDM training)
# ---------------------------------------------------------------------------
def _load_normalizer_from_pickle(path: str) -> dict:
    """Load the IDM normalizer (saved as `normalizer.pkl` next to the IDM ckpt).

    Format: dict with structure `{"obs": {key: Normalizer}, "action": Normalizer}`
    where the per-key normalizers are `MinMaxNormalizer` (low-dim) or
    `ImageNormalizer` (rgb), each exposing `.normalize(...)` / `.unnormalize(...)`.
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
    goal_stats_path: str | None = None,
    num_steps: int | None = None,
    goal_flow_num_steps: int | None = None,
    sample_mode: str | None = None,
    goal_sample_mode: str | None = None,
    act_steps: int | None = None,
):
    """Build the GoalPredictorDiTAgent, load the goal-DiT ckpt, load normalizer.

    Sample-mode overrides are applied BEFORE constructing the agent. The
    goal-predictor's ``__init__`` caches ``_goal_sample_mode`` and wires it
    into the encoder wrappers, so a late override would be ignored; the IDM
    action ``sample_mode`` is read at sample time and is also picked up via
    the OmegaConf-updated config.
    """
    global agent, config, device, normalizer

    print(f"Loading hydra config: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Inference-time overrides.
    if idm_checkpoint_path is not None:
        OmegaConf.update(cfg, "optimization.idm_checkpoint_path",
                         idm_checkpoint_path, merge=False)
    if goal_stats_path is not None:
        OmegaConf.update(cfg, "optimization.goal_stats_path",
                         goal_stats_path, merge=False)
    if num_steps is not None:
        OmegaConf.update(cfg, "optimization.num_steps", int(num_steps), merge=False)
    if goal_flow_num_steps is not None:
        OmegaConf.update(cfg, "optimization.goal_flow_num_steps",
                         int(goal_flow_num_steps), merge=False)
    if sample_mode is not None:
        OmegaConf.update(cfg, "optimization.sample_mode",
                         str(sample_mode), merge=False)
    if goal_sample_mode is not None:
        OmegaConf.update(cfg, "optimization.goal_sample_mode",
                         str(goal_sample_mode), merge=False)
    if act_steps is not None:
        OmegaConf.update(cfg, "task.act_steps", int(act_steps), merge=False)

    # Force inference-friendly settings.
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)

    # Goal-predictor agent reads obs_dim from network.emb_dim for image obs.
    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)

    config = cfg
    device = torch.device(cfg.optimization.device)
    # Resolve goal_sample_mode the same way the agent does (None -> inherit
    # IDM action sample_mode) for accurate logging.
    effective_goal_sample_mode = (
        cfg.optimization.goal_sample_mode
        if cfg.optimization.goal_sample_mode is not None
        else cfg.optimization.sample_mode
    )
    print(f"Device: {device} | num_steps: {cfg.optimization.num_steps} | "
          f"goal_flow_num_steps: {cfg.optimization.goal_flow_num_steps} | "
          f"sample_mode: {cfg.optimization.sample_mode} | "
          f"goal_sample_mode: {effective_goal_sample_mode} | "
          f"horizon: {cfg.task.horizon} | obs_steps: {cfg.task.obs_steps} | "
          f"act_steps: {cfg.task.act_steps}")

    print(f"Building GoalPredictorDiTAgent({cfg.network.network_type}, "
          f"emb_dim={cfg.network.emb_dim}) — also loads frozen IDM internally...")
    agent = GoalPredictorDiTAgent(cfg)

    print(f"Loading goal predictor DiT checkpoint: {ckpt_path}")
    agent.load(ckpt_path, load_optimizer=False)
    agent.eval()

    print(f"Loading IDM normalizer (pickle): {normalizer_path}")
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
    """Apply train-time normalization, mirroring `RobomimicImageIDMDataset.__getitem__`.

    Per-key normalizers come from the loaded pickle (`MinMaxNormalizer` or
    `ImageNormalizer`). The image path here just shapes the input into the
    layout the normalizer expects (CHW float in [0, 1]) and then calls
    `.normalize(...)` (which for `ImageNormalizer` is `x*2 - 1`).

    Drops any key not in `shape_meta["obs"]` (e.g. `is_contact`, `pcd`).
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
    """Save received obs + returned action to disk for inspection (no-op if disabled)."""
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
    """Sample an action chunk for the given observation."""
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

            act_0 = torch.randn(
                (1, config.task.horizon, config.task.act_dim),
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_to_gpu = time.time() - t0

            t0 = time.time()
            # GoalPredictorDiTAgent.sample uses `num_steps` (IDM ODE steps); the
            # goal predictor's own ODE step count comes from
            # config.optimization.goal_flow_num_steps.
            act_normed = agent.sample(act_0=act_0, obs=obs_torch, use_ema=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_predict = time.time() - t0

            t0 = time.time()
            act_normed_np = act_normed[0].detach().to("cpu").numpy()  # (horizon, 10)
            act = normalizer["action"].unnormalize(act_normed_np)

            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            action = act[start:end].astype(np.float32)            # (act_steps, 10)

            # Same gripper binarization as interface_example.py — controller
            # smoothing block requires {0, 1}.
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
    """No-op: goal-predictor + flow-matching policy is stateless across calls."""
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
                        help="Path to GoalPredictorDiTAgent checkpoint (.pt)")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to .hydra/config.yaml from the training run")
    parser.add_argument("--normalizer_path", type=str, required=True,
                        help="Path to normalizer.pkl saved by IDM training "
                             "(usually a sibling of the IDM checkpoint)")
    parser.add_argument("--idm_checkpoint_path", type=str, default=None,
                        help="Override optimization.idm_checkpoint_path")
    parser.add_argument("--goal_stats_path", type=str, default=None,
                        help="Override optimization.goal_stats_path")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override optimization.num_steps "
                             "(IDM action sampler ODE steps)")
    parser.add_argument("--goal_flow_num_steps", type=int, default=None,
                        help="Override optimization.goal_flow_num_steps "
                             "(goal predictor DiT ODE steps)")
    parser.add_argument("--sample_mode", type=str, default=None,
                        choices=["stochastic", "zero"],
                        help="Override optimization.sample_mode "
                             "(IDM action ODE source)")
    parser.add_argument("--goal_sample_mode", type=str, default=None,
                        choices=["stochastic", "zero"],
                        help="Override optimization.goal_sample_mode "
                             "(goal-predictor ODE source; if unset, the "
                             "agent inherits the IDM action sample_mode)")
    parser.add_argument("--act_steps", type=int, default=None,
                        help="Override task.act_steps for inference")
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
        goal_stats_path=args.goal_stats_path,
        num_steps=args.num_steps,
        goal_flow_num_steps=args.goal_flow_num_steps,
        sample_mode=args.sample_mode,
        goal_sample_mode=args.goal_sample_mode,
        act_steps=args.act_steps,
    )

    print(f"Starting policy server on port {args.port}")
    app.run(host="localhost", port=args.port, threaded=False)
