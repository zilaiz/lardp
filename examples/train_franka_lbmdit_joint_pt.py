"""Franka LBMDiTJointPT training pipeline (real-robot, no sim env).

Single-trunk PT variant of the Franka joint pipeline. Mirrors
``train_franka_lbmdit_joint_ddt.py`` in dataset/eval plumbing but swaps
the joint trunk to ``LBMDiTJointPTAgent`` (one width, one depth — no
encoder/decoder split).

Reuses the DDT version's ``train`` and ``compute_val_loss`` helpers via
import. The agent classes are subclass-compatible (``LBMDiTJointPTAgent``
inherits from ``LBMDiTJointDDTAgent`` and only swaps the trunk), so all
update / sample / EMA / save / load logic carries over unchanged.
"""

import os
import pickle

import hydra
import loguru
import torch

from examples.train_franka_lbmdit_joint_ddt import (
    compute_val_loss,  # noqa: F401  (re-exported for parity)
    train,
)
from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
from mip.config import Config
from mip.datasets.robomimic_dataset import RobomimicImageIDMDataset, make_idm_dataset
from mip.logger import Logger
from mip.torch_utils import limit_threads, set_seed

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs/", config_name="main_franka")
def main(config: Config):
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("warn")
        torch.set_float32_matmul_precision("high")

    set_seed(config.optimization.seed)
    limit_threads(1)
    logger = Logger(config)
    loguru.logger.info("Logger ready")

    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        raise NotImplementedError(
            "Franka LBMDiTJointPT training only supports image obs"
        )

    # Optional IDM warm-start. When provided, also reuse the IDM normalizer
    # so observations/actions match the encoder's training scale. When None,
    # the encoder trains from scratch and a fresh normalizer is computed.
    idm_path = config.optimization.idm_checkpoint_path
    idm_normalizer = None
    if idm_path is not None and idm_path != "None" and idm_path != "null":
        normalizer_path = os.path.join(os.path.dirname(idm_path), "normalizer.pkl")
        if os.path.exists(normalizer_path):
            with open(normalizer_path, "rb") as f:
                idm_normalizer = pickle.load(f)
            loguru.logger.info(f"Loaded IDM normalizer from {normalizer_path}")
        else:
            loguru.logger.warning(
                f"IDM normalizer not found at {normalizer_path}; computing a fresh one"
            )
    else:
        loguru.logger.info(
            "No idm_checkpoint_path — encoder trains from scratch with a "
            "freshly computed normalizer"
        )

    dataset = make_idm_dataset(config.task, mode="train", normalizer=idm_normalizer)
    loguru.logger.info(f"Train IDM dataset: {len(dataset)} samples")

    base_dataset = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )

    val_dataset = None
    if config.task.val_dataset_percentage > 0.0:
        primary_path = (
            config.task.dataset_paths[0]
            if config.task.dataset_paths
            else config.task.dataset_path
        )
        val_dataset = RobomimicImageIDMDataset(
            dataset_dir=os.path.expanduser(primary_path),
            shape_meta=config.task.shape_meta,
            n_obs_steps=config.task.obs_steps,
            horizon=config.task.horizon,
            pad_before=config.task.obs_steps - 1,
            pad_after=config.task.act_steps - 1,
            abs_action=config.task.abs_action,
            val_dataset_percentage=config.task.val_dataset_percentage,
            mode="val",
            normalizer=base_dataset.normalizer,
        )
        loguru.logger.info(f"Val IDM dataset: {len(val_dataset)} samples")

    agent = LBMDiTJointPTAgent(config)
    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading LBMDiTJointPT from {config.optimization.model_path}"
        )
        resume_state = agent.load(
            config.optimization.model_path, load_optimizer=True,
        )

    if config.mode == "train":
        train(config, dataset, val_dataset, agent, logger, resume_state=resume_state)
    else:
        raise ValueError(
            f"Franka joint PT pipeline only supports mode='train' "
            f"(got {config.mode!r})"
        )


if __name__ == "__main__":
    main()
