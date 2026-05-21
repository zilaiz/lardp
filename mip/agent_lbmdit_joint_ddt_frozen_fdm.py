"""LBMDiTJointDDTFrozenFDMAgent: joint DDT trunk with a BYOL-FDM-pretrained encoder.

Functionally identical to ``LBMDiTJointDDTFrozenAgent`` — same DDT trunk,
same state+action joint flow loss, same goal-stats normalization, same
freeze/fine-tune modes. The only difference is at initialization:

- The ``idm_checkpoint_path`` config field points to an FDM checkpoint
  produced by ``FDMAgent`` (BYOL-style forward-dynamics encoder
  pretraining). The state-dict layout is identical to IDM checkpoints
  (``encoder`` + ``encoder_ema`` + ``flow_map`` + ``flow_map_ema``), so
  the parent's loading path works as-is for ``state_dict["encoder"]``.
- When ``optimization.joint_use_encoder_ema_for_init=True``, the input
  encoder is re-initialized from ``state_dict["encoder_ema"]`` (the BYOL
  EMA target) instead of ``state_dict["encoder"]`` (the online encoder).
  For BYOL-style pretraining the EMA target is sometimes the smoother
  / preferred downstream artifact; both options are typically close at
  convergence.

The agent file exists as a separate class purely for semantic clarity
(distinguishing IDM-pretrained from FDM-pretrained runs in logs, save
paths, and downstream code-search). All inherited behavior is unchanged.
"""

from __future__ import annotations

import loguru
import torch

from mip.agent_lbmdit_joint_ddt_frozen import LBMDiTJointDDTFrozenAgent
from mip.config import Config


class LBMDiTJointDDTFrozenFDMAgent(LBMDiTJointDDTFrozenAgent):
    """Joint DDT trunk + BYOL-FDM-pretrained encoder."""

    def __init__(self, config: Config):
        super().__init__(config)
        if config.optimization.joint_use_encoder_ema_for_init:
            self._reload_encoder_from_ema()

    def _reload_encoder_from_ema(self) -> None:
        """Re-load the input encoder from ``state_dict["encoder_ema"]``.

        The parent ``__init__`` has already loaded from ``state_dict["encoder"]``.
        Here we overwrite those weights with the BYOL EMA target. Also refreshes
        ``target_encoder`` if it's a separate copy (split-target mode).
        """
        device = self.config.optimization.device
        ckpt_path = self.config.optimization.idm_checkpoint_path
        loguru.logger.info(
            f"Re-loading input encoder from BYOL EMA target in {ckpt_path}"
        )

        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "encoder_ema" not in state_dict:
            raise KeyError(
                "joint_use_encoder_ema_for_init=True but checkpoint has no "
                "'encoder_ema' key. Use a checkpoint saved by FDMAgent (or any "
                "TrainingAgent subclass — they all save encoder_ema)."
            )
        encoder_ema_sd = state_dict["encoder_ema"]

        # Defensive: handle the GoalDropoutEncoder-wrapped layout. FDM
        # checkpoints don't have this wrap, but a future variant might.
        has_goal_dropout = any(k.startswith("encoder.") for k in encoder_ema_sd)
        if has_goal_dropout:
            inner_sd = {
                k.removeprefix("encoder."): v
                for k, v in encoder_ema_sd.items()
                if k.startswith("encoder.")
            }
            self.encoder.load_state_dict(inner_sd)
        else:
            self.encoder.load_state_dict(encoder_ema_sd)

        # In split-target mode, target_encoder is a separate deepcopy that
        # holds the originally-loaded (online) weights. Sync it to the new
        # EMA-initialized input encoder so both branches start from the same
        # point. If target_encoder is just an alias (frozen / e2e modes),
        # this branch is skipped — the alias already sees the new weights.
        if self.target_encoder is not self.encoder:
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            self.target_encoder.eval()

        loguru.logger.info(
            "Input encoder re-initialized from BYOL EMA target "
            "(state_dict['encoder_ema'])"
        )
