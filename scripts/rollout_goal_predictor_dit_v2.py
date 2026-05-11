"""Roll out a goal_predictor_dit_v2 checkpoint and save each episode as an MP4 (H.264).

Loads a Goal Predictor DiT agent from a saved hydra config + checkpoint, runs
N episodes in the environment with `num_envs=1`, and writes one MP4 per
episode to `--out_dir`. Files are renamed with a success/fail suffix once the
episode finishes.

Usage:
    python scripts/rollout_goal_predictor_dit_v2.py \
        --ckpt_path     logs/<exp>/<ts>/models/model_best.pt \
        --config_path   outputs/<date>/<time>/.hydra/config.yaml \
        --out_dir       viz/dit_v2_rollouts \
        --n_demos       20 \
        --num_steps     1
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# Set MuJoCo rendering backend before importing robomimic/mujoco modules
os.environ["MUJOCO_GL"] = "egl"

import loguru  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mip.agent_goal_predictor_dit import GoalPredictorDiTAgent  # noqa: E402
from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.env_utils import VideoRecordingWrapper  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.torch_utils import set_seed  # noqa: E402

ROBOMIMIC_TASKS = ("can", "lift", "square", "tool_hang", "transport")


def _find_video_wrapper(env):
    """Walk through nested wrappers to find the VideoRecordingWrapper."""
    while not isinstance(env, VideoRecordingWrapper):
        env = env.env
    return env


def _rollout_one_episode(
    envs,
    agent,
    base_dataset,
    cfg,
    num_steps: int,
    device: str,
):
    """Run a single episode with the loaded agent. Returns (success, total_reward, steps)."""
    horizon = cfg.task.horizon
    obs_steps = cfg.task.obs_steps
    act_steps = cfg.task.act_steps
    act_dim = cfg.task.act_dim
    max_episode_steps = cfg.task.max_episode_steps
    num_envs = cfg.task.num_envs  # 1 for this script

    obs, _ = envs.reset()
    ep_reward = 0.0
    t = 0

    while t < max_episode_steps:
        obs_dict = {}
        for k in obs:
            obs_k = obs[k].astype(np.float32)
            obs_k = base_dataset.normalizer["obs"][k].normalize(obs_k)
            obs_dict[k] = torch.tensor(obs_k, device=device, dtype=torch.float32)

        act_0 = torch.randn((num_envs, horizon, act_dim), device=device)
        act_normed = agent.sample(
            act_0=act_0, obs=obs_dict, num_steps=num_steps, use_ema=True
        )
        act_normed = act_normed.detach().cpu().numpy()
        act = base_dataset.normalizer["action"].unnormalize(act_normed)

        start = obs_steps - 1
        end = start + act_steps
        act = act[:, start:end, :]

        if cfg.task.abs_action and cfg.task.env_name in ROBOMIMIC_TASKS:
            act = base_dataset.undo_transform_action(act)

        obs, reward, terminated, truncated, info = envs.step(act)
        ep_reward += float(np.sum(reward))
        t += act_steps

        # Stop early on done so SyncVectorEnv doesn't auto-reset and corrupt
        # the in-progress mp4 mid-stream.
        if bool(terminated[0]) or bool(truncated[0]):
            break

    success = ep_reward > 0
    return success, ep_reward, t


def main():
    parser = argparse.ArgumentParser(
        description="Roll out goal_predictor_dit_v2 and save per-episode MP4s.",
    )
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to goal predictor DiT checkpoint (.pt)")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to .hydra/config.yaml from the training run")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to write per-episode MP4s")
    parser.add_argument("--n_demos", type=int, default=10,
                        help="Number of episodes to roll out")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="ODE solver steps for the IDM action sampler")
    parser.add_argument("--goal_flow_num_steps", type=int, default=None,
                        help="Override optimization.goal_flow_num_steps "
                             "(ODE steps for the goal predictor DiT). "
                             "Default: keep the value from the training config.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Env seed (also seeds torch/numpy)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device (cuda or cpu)")
    parser.add_argument("--idm_checkpoint_path", type=str, default=None,
                        help="Override optimization.idm_checkpoint_path "
                             "(defaults to value in --config_path)")
    parser.add_argument("--goal_stats_path", type=str, default=None,
                        help="Override optimization.goal_stats_path "
                             "(defaults to value in --config_path)")
    parser.add_argument("--normalizer_path", type=str, default=None,
                        help="Override path to normalizer.pkl "
                             "(defaults to next to the IDM checkpoint)")
    parser.add_argument("--render_h", type=int, default=512,
                        help="Video frame height in pixels (re-renders from the "
                             "robosuite sim instead of using the 84x84 obs image)")
    parser.add_argument("--render_w", type=int, default=512,
                        help="Video frame width in pixels")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config_path)

    # --- Inference-friendly overrides ---
    if args.idm_checkpoint_path is not None:
        cfg.optimization.idm_checkpoint_path = args.idm_checkpoint_path
    if args.goal_stats_path is not None:
        cfg.optimization.goal_stats_path = args.goal_stats_path
    cfg.task.num_envs = 1
    cfg.task.save_video = True
    cfg.optimization.use_compile = False
    cfg.optimization.use_cudagraphs = False
    cfg.optimization.num_steps = int(args.num_steps)
    if args.goal_flow_num_steps is not None:
        cfg.optimization.goal_flow_num_steps = int(args.goal_flow_num_steps)
    if not torch.cuda.is_available():
        cfg.optimization.device = "cpu"
    else:
        cfg.optimization.device = args.device

    set_seed(args.seed)
    device = cfg.optimization.device

    # --- Env setup ---
    envs = make_vec_env(cfg.task, seed=args.seed)
    obs, _ = envs.reset()
    if cfg.task.obs_type == "image":
        cfg.task.obs_dim = cfg.network.emb_dim
    else:
        cfg.task.obs_dim = obs.shape[-1]
    loguru.logger.info("Finished setting up env")

    # --- Load IDM normalizer (used for obs/action normalize/unnormalize) ---
    idm_path = cfg.optimization.idm_checkpoint_path
    if idm_path is None:
        raise ValueError("optimization.idm_checkpoint_path must be set in config")
    normalizer_path = args.normalizer_path or os.path.join(
        os.path.dirname(idm_path), "normalizer.pkl"
    )
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            f"normalizer.pkl not found at {normalizer_path}. "
            "Pass --normalizer_path explicitly."
        )
    with open(normalizer_path, "rb") as f:
        idm_normalizer = pickle.load(f)
    loguru.logger.info(f"Loaded IDM normalizer from {normalizer_path}")

    # --- Build dataset (for normalizer + undo_transform_action) ---
    dataset = make_idm_dataset(cfg.task, normalizer=idm_normalizer)
    base_dataset = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )

    # --- Build & load goal predictor DiT agent ---
    agent = GoalPredictorDiTAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()
    loguru.logger.info(f"Loaded goal predictor DiT from {args.ckpt_path}")

    # --- Locate the VideoRecordingWrapper to set per-episode mp4 paths ---
    video_env = _find_video_wrapper(envs.envs[0])

    # --- Re-render at higher resolution by patching the image wrapper to ask
    # the underlying robosuite sim for a fresh frame each time, instead of
    # returning the cached 84x84 obs image.
    image_wrapper = video_env.env  # RobomimicImageWrapper
    robosuite_env = image_wrapper.env  # robomimic EnvRobosuite
    camera_name = image_wrapper.render_obs_key
    if camera_name.endswith("_image"):
        camera_name = camera_name[: -len("_image")]
    render_h = int(args.render_h)
    render_w = int(args.render_w)

    def _hires_render(mode="rgb_array", **_kwargs):
        return robosuite_env.render(
            mode="rgb_array",
            height=render_h,
            width=render_w,
            camera_name=camera_name,
        )

    image_wrapper.render = _hires_render
    loguru.logger.info(
        f"Rendering video at {render_w}x{render_h} from camera '{camera_name}'"
    )

    # --- Rollout loop ---
    success_count = 0
    rollout_records = []

    for ep_idx in range(args.n_demos):
        # Make sure any prior recording is finalized to disk before we set a
        # new file_path; otherwise the previous mp4 could end up empty.
        video_env.video_recoder.stop()
        ep_path = out_dir / f"rollout_{ep_idx:04d}.mp4"
        video_env.file_path = str(ep_path)

        success, ep_reward, n_steps = _rollout_one_episode(
            envs=envs,
            agent=agent,
            base_dataset=base_dataset,
            cfg=cfg,
            num_steps=args.num_steps,
            device=device,
        )

        # Finalize the mp4 so the file is fully written before renaming.
        video_env.video_recoder.stop()
        video_env.file_path = None

        suffix = "success" if success else "fail"
        final_path = out_dir / f"rollout_{ep_idx:04d}_{suffix}.mp4"
        if ep_path.exists():
            ep_path.rename(final_path)
        else:
            loguru.logger.warning(
                f"[ep {ep_idx:03d}] expected mp4 at {ep_path} but file is missing"
            )
            final_path = ep_path

        success_count += int(success)
        rollout_records.append({
            "episode": ep_idx,
            "success": bool(success),
            "reward": ep_reward,
            "steps": n_steps,
            "path": str(final_path),
        })
        loguru.logger.info(
            f"[ep {ep_idx:03d}] success={int(success)} reward={ep_reward:.2f} "
            f"steps={n_steps} -> {final_path}"
        )

    loguru.logger.info(
        f"Done. success rate = {success_count}/{args.n_demos} "
        f"= {success_count / args.n_demos:.2%}"
    )

    # Sidecar summary so the rollout outcomes are easy to script against.
    summary_path = out_dir / "rollouts.json"
    import json
    with summary_path.open("w") as f:
        json.dump(
            {
                "ckpt_path": args.ckpt_path,
                "config_path": args.config_path,
                "n_demos": args.n_demos,
                "success_count": success_count,
                "success_rate": success_count / args.n_demos,
                "num_steps": args.num_steps,
                "seed": args.seed,
                "rollouts": rollout_records,
            },
            f,
            indent=2,
        )
    loguru.logger.info(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
