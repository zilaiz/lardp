"""Training entry for LBMDiTJointPTFrozenTargetAgent on PushT.

State-representation ablation of the PushT joint_pt pipeline: reuses the
``train`` / ``evaluate`` loop (and PushT coverage eval) from
``train_pusht_lbmdit_joint_pt.py`` but swaps the agent so the FM state-flow
target is a **frozen external encoder** (``dp_checkpoint_path``) z-scored by
precomputed per-dim goal stats (``goal_stats_path``), instead of the trainable
``target_ln`` self-target. The input encoder and the
``joint_state_loss_to_encoder`` knob are unchanged.

Goal stats for PushT must come from a PushT-dataset export (the robomimic
``scripts/compute_goal_stats_dp.py`` uses ``make_idm_dataset``); see
``scripts/compute_goal_stats_dp_pusht.py``.

Author: Zilai Zeng
"""

import os

import hydra
import loguru
import torch

os.environ.setdefault("MUJOCO_GL", "egl")  # noqa: E402

from examples.train_pusht_lbmdit_joint_pt import (  # noqa: E402
    evaluate,
    train,
)
from mip.agent_lbmdit_joint_pt_frozentgt import (  # noqa: E402
    LBMDiTJointPTFrozenTargetAgent,
)
from mip.datasets.pusht_dataset import make_pusht_goal_dataset  # noqa: E402
from mip.envs.pusht import make_vec_env  # noqa: E402
from mip.logger import Logger  # noqa: E402
from mip.samplers import get_default_step_list  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
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

    dataset = make_pusht_goal_dataset(config.task)
    loguru.logger.info("Finished setting up PushT goal dataset")

    agent = LBMDiTJointPTFrozenTargetAgent(config)
    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading LBMDiTJointPTFrozenTarget from {config.optimization.model_path}"
        )
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
    elif config.mode == "eval":
        agent.eval()
        num_steps_list = (
            config.optimization.eval_num_steps
            or get_default_step_list(config.optimization.loss_type)
        )
        for num_steps in num_steps_list:
            metrics = {"step": num_steps}
            metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps))
            logger.log(metrics, category="eval")
        for key, val in metrics.items():
            if "mean_success" in key:
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
