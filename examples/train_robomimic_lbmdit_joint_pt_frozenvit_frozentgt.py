"""Training entry for LBMDiTJointPTFrozenViTTargetAgent on robomimic.

Frozen-ViT input + shared-backbone frozen target ablation: identical to
``train_robomimic_lbmdit_joint_pt_frozentgt.py`` (reuses the E2E ``train`` /
``eval`` loops, the frozen-target agent surface, and per-dim goal z-score) but
the input encoder is ``encoder_type=frozen_vit`` and the FM state-flow target
REUSES that same frozen backbone (DINOv2 CLS / SigLIP pooler_output) instead of
loading a separate target encoder. See ``mip/agent_lbmdit_joint_pt_frozenvit_
frozentgt.py``.

Export the goal stats first with ``scripts/compute_goal_stats_target.py`` (the
same stats as the scratch-input ``frozentgt`` arm — same descriptor), then pass
``optimization.goal_stats_path=<stats.pt>`` and the matching
``optimization.target_encoder_image_key=<camera>`` to this script.
"""

import os
import pickle

import hydra
import loguru
import torch

# Set MuJoCo rendering backend before importing any robomimic/mujoco modules
os.environ["MUJOCO_GL"] = "egl"  # noqa: E402

from examples.train_robomimic_lbmdit_joint_e2e import (  # noqa: E402
    eval as eval_fn,
)
from examples.train_robomimic_lbmdit_joint_e2e import (
    train,
)
from mip.agent_lbmdit_joint_pt_frozenvit_frozentgt import (  # noqa: E402
    LBMDiTJointPTFrozenViTTargetAgent,
)
from mip.datasets.robomimic_dataset import make_idm_dataset  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
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
    obs, info = envs.reset()
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        config.task.obs_dim = obs.shape[-1]
    loguru.logger.info("Finished setting up env")

    # Reuse IDM normalizer when warm-starting the input encoder so action / obs
    # scaling matches the pretrained encoder.
    idm_normalizer = None
    if config.optimization.idm_checkpoint_path:
        normalizer_path = os.path.join(
            os.path.dirname(config.optimization.idm_checkpoint_path),
            "normalizer.pkl",
        )
        if os.path.exists(normalizer_path):
            with open(normalizer_path, "rb") as f:
                idm_normalizer = pickle.load(f)
            loguru.logger.info(f"Loaded IDM normalizer from {normalizer_path}")
        else:
            loguru.logger.warning(
                f"IDM normalizer not found at {normalizer_path}; computing fresh"
            )

    dataset = make_idm_dataset(config.task, normalizer=idm_normalizer)
    loguru.logger.info("Finished setting up IDM dataset")

    agent = LBMDiTJointPTFrozenViTTargetAgent(config)
    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading LBMDiTJointPTFrozenViTTarget from "
            f"{config.optimization.model_path}"
        )
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
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
