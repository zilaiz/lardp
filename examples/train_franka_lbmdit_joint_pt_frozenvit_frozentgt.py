"""Franka frozen-ViT input + shared-backbone FROZEN-TARGET training pipeline.

Franka analog of ``train_robomimic_lbmdit_joint_pt_frozenvit_frozentgt.py``, and
the frozen-ViT-input variant of ``train_franka_lbmdit_joint_pt_frozentgt.py``.
The input/condition encoder is ``encoder_type=frozen_vit`` (frozen DINOv2/SigLIP
backbone + trainable per-view MAP adapter) and the FM **state-flow target** is
the SAME frozen backbone's native pooled descriptor (DINOv2 CLS / SigLIP
``pooler_output``), z-scored by precomputed goal stats — built by
``LBMDiTJointPTFrozenViTTargetAgent`` (which reuses the input encoder's backbone;
no second load).

Franka has no simulator, so the only periodic eval is the **val FM loss**. The
custom ``compute_val_loss`` / ``train`` from
``train_franka_lbmdit_joint_pt_frozentgt.py`` are reused VERBATIM: they are
agent-class-agnostic (they call ``agent.target_encoder.embed`` +
``agent._normalize_goal`` and divide the state loss by the trunk ``state_dim``),
so they work unchanged for the FrozenViT-target subclass. Only ``main`` is
redefined here to instantiate the subclass agent.

Prerequisites: export goal stats once per (task, backbone) with
``scripts/compute_goal_stats_target.py`` (the SAME descriptor as the scratch-
input Franka frozentgt arm — reuse ``goal_stats/<short>_<statsfx>.pt`` as-is),
then pass ``optimization.goal_stats_path``, ``optimization.target_encoder_image_key``
(front_cam_image) and ``network.state_target_dim=<D>`` to this script. The
frozen_vit input requires ``network.frozen_vit_path`` + ``frozen_vit_backbone``
matching that backbone, and ``network.frozen_vit_expose_pooled=true``.

Author: Zilai Zeng
"""

import os
import pickle

import hydra
import loguru
import torch

# Reuse the frozen-target Franka train loop + val FM loss verbatim (they only use
# the agent interface that the FrozenViT-target subclass also satisfies).
from examples.train_franka_lbmdit_joint_pt_frozentgt import train
from mip.agent_lbmdit_joint_pt_frozenvit_frozentgt import (
    LBMDiTJointPTFrozenViTTargetAgent,
)
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
            "Franka LBMDiTJointPTFrozenViTTarget training only supports image obs"
        )

    # INPUT encoder warm-start (the target reuses the frozen backbone, separate).
    # Keep idm_checkpoint_path null for frozen_vit (an IDM ckpt carries a ResNet
    # encoder state dict that shape-mismatches the frozen_vit encoder); when given
    # it is only used to reuse the matching normalizer.
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
            "No idm_checkpoint_path — input encoder trains from scratch with a "
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
            # Must match the train dataset's action representation, otherwise the
            # shared normalizer (fit on 7-dim deltas in delta mode) will mismatch
            # the 10-dim raw actions returned by __getitem__.
            delta_action_anchor=getattr(config.task, "delta_action_anchor", None),
        )
        loguru.logger.info(f"Val IDM dataset: {len(val_dataset)} samples")

    agent = LBMDiTJointPTFrozenViTTargetAgent(config)
    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(
            f"Loading LBMDiTJointPTFrozenViTTarget from "
            f"{config.optimization.model_path}"
        )
        resume_state = agent.load(
            config.optimization.model_path,
            load_optimizer=True,
        )

    if config.mode == "train":
        train(config, dataset, val_dataset, agent, logger, resume_state=resume_state)
    else:
        raise ValueError(
            f"Franka joint PT frozen-ViT-target pipeline only supports "
            f"mode='train' (got {config.mode!r})"
        )


if __name__ == "__main__":
    main()
