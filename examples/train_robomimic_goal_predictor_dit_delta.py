"""Training pipeline for the delta-target DiT goal predictor on robomimic.

Same training/eval loops as `train_robomimic_goal_predictor_dit.py` — the only
difference is the agent class. The DiT learns a velocity field over normalized
deltas `g - last_obs`, then at inference the predicted delta is denormalized
and added to the encoded last observation to produce the absolute goal
embedding consumed by the frozen IDM.

`config.optimization.goal_stats_path` must point at a stats file produced by
`scripts/compute_goal_delta_stats.py` (containing `delta_mean`/`delta_var`).
"""

import os

# Set MuJoCo rendering backend before importing any robomimic/mujoco modules
os.environ["MUJOCO_GL"] = "egl"  # noqa: E402

import hydra  # noqa: E402
import loguru  # noqa: E402
import torch  # noqa: E402

from examples.train_robomimic_goal_predictor_dit import eval as eval_fn  # noqa: E402
from examples.train_robomimic_goal_predictor_dit import train as train_fn  # noqa: E402
from mip.agent_goal_predictor_dit_delta import GoalPredictorDiTDeltaAgent  # noqa: E402
from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.logger import Logger  # noqa: E402
from mip.samplers import get_default_step_list  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline for delta-target goal predictor DiT training."""
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("warn")
        torch.set_float32_matmul_precision("high")

    set_seed(config.optimization.seed)
    limit_threads(1)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        config.task.obs_dim = obs.shape[-1]
    loguru.logger.info("Finished setting up env")

    # Load normalizer from pretrained IDM to ensure consistent normalization
    import pickle
    idm_normalizer = None
    if config.optimization.idm_checkpoint_path:
        normalizer_path = os.path.join(
            os.path.dirname(config.optimization.idm_checkpoint_path), "normalizer.pkl"
        )
        if os.path.exists(normalizer_path):
            with open(normalizer_path, "rb") as f:
                idm_normalizer = pickle.load(f)
            loguru.logger.info(f"Loaded IDM normalizer from {normalizer_path}")
        else:
            loguru.logger.warning(
                f"IDM normalizer not found at {normalizer_path}, will compute new one"
            )

    dataset = make_idm_dataset(config.task, normalizer=idm_normalizer)
    loguru.logger.info("Finished setting up IDM dataset")

    agent = GoalPredictorDiTDeltaAgent(config)
    resume_state = None

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading delta goal predictor DiT from {config.optimization.model_path}"
        )
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)

    if config.mode == "train":
        train_fn(config, envs, dataset, agent, logger, resume_state=resume_state)
    elif config.mode == "eval":
        agent.eval()
        num_steps_list = get_default_step_list(config.optimization.loss_type)
        for num_steps in num_steps_list:
            metrics = {"step": num_steps}
            metrics.update(eval_fn(config, envs, dataset, agent, logger, num_steps))
            logger.log(metrics, category="eval")

        for key, val in metrics.items():
            if "mean_success" in key:
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
