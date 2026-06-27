"""LBMDiTJointPTFrozenViTTargetAgent: frozen-ViT input + shared-backbone target.

Ablation of the frozen-ViT joint_pt agent. The input side is unchanged from the
frozen_vit pipeline: a frozen pretrained backbone (DINOv2 / SigLIP) + a trainable
per-camera attentive-pool (MAP) adapter feeds the AdaLN condition. The ONLY swap
is the FM state-flow TARGET: instead of the EMA of that trainable adapter
(self-distillation, the default frozen_vit target), the target is the frozen
backbone's NATIVE pooled descriptor (DINOv2 CLS / SigLIP pooler_output) z-scored
by precomputed goal stats — i.e. the ``frozentgt`` design, but the target
encoder REUSES the backbone the input encoder already loaded instead of loading
a second copy.

Concretely this is ``LBMDiTJointPTFrozenTargetAgent`` (which already swaps in a
frozen ``TargetEncoder`` + per-dim z-score and routes the rest of the joint
pipeline unchanged), with two specializations:
  1. the input encoder is ``encoder_type=frozen_vit`` (set via config), and
  2. ``_build_target_encoder`` returns a ``SharedBackboneTargetEncoder`` wrapping
     ``self.encoder.backbone`` — the SAME frozen backbone instance — rather than
     ``get_target_encoder(config)`` (which would ``from_pretrained`` a 2nd copy).

Because the shared target's pooling/preprocessing match the standalone
``DINOv2TargetEncoder`` / ``SiglipTargetEncoder`` exactly, the goal z-score stats
exported for the scratch-input ``frozentgt`` arms are reusable here, and the
three cells form a clean comparison:
  - frozen_vit             : frozen_vit input  + EMA-adapter target
  - frozentgt (scratch in) : scratch ResNet in + frozen pooled target
  - THIS                   : frozen_vit input  + frozen pooled target
sharing the input with the first and the target with the second.

Dims decouple as in ``frozentgt``: the target dim is the backbone hidden size
(DINOv2-S 384 / SigLIP-B 768), so set ``network.state_target_dim`` = that dim and
``network.emb_dim`` >= it (RAE); ``encoder_out_dim`` (the condition width) stays
256. SigLIP additionally needs ``network.frozen_vit_expose_pooled=true`` so the
shared backbone keeps its pooler head.

Author: Zilai Zeng
"""

from __future__ import annotations

import loguru

from mip.agent_lbmdit_joint_pt_frozentgt import LBMDiTJointPTFrozenTargetAgent
from mip.config import Config
from mip.encoders import FrozenViTMultiObsEncoder
from mip.target_encoders import SharedBackboneTargetEncoder, TargetEncoder


class LBMDiTJointPTFrozenViTTargetAgent(LBMDiTJointPTFrozenTargetAgent):
    """Frozen-target joint_pt agent whose FM target reuses the input encoder's
    frozen ViT backbone (no second backbone load). See module docstring.
    """

    def _build_target_encoder(self, config: Config) -> TargetEncoder:
        # super().__init__ has already built self.encoder via get_encoder, which
        # honors network.encoder_type. This agent requires the frozen_vit input
        # encoder so there is a backbone to share.
        if not isinstance(self.encoder, FrozenViTMultiObsEncoder):
            raise ValueError(
                "LBMDiTJointPTFrozenViTTargetAgent requires the frozen_vit input "
                "encoder (network.encoder_type=frozen_vit); got "
                f"{type(self.encoder).__name__}. Use "
                "LBMDiTJointPTFrozenTargetAgent for a standalone target encoder."
            )
        image_key = TargetEncoder._resolve_image_key(config, config.optimization)
        loguru.logger.info(
            "Building shared-backbone FM target (reusing the input frozen_vit "
            f"backbone; target camera={image_key!r})"
        )
        return SharedBackboneTargetEncoder(self.encoder.backbone, image_key)
