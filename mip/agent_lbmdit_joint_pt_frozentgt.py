"""LBMDiTJointPTFrozenTargetAgent: joint_pt with a FROZEN external target encoder.

State-representation ablation for the joint_pt framework. Everything on the
INPUT / condition side is the unchanged single-trunk PT agent
(``LBMDiTJointPTAgent``): a trainable image encoder + learnable ``target_ln``
feed the AdaLN condition, and ``joint_state_loss_to_encoder`` still routes the
state-flow loss into (or away from) that input encoder. The ONLY thing swapped
is the **FM state-flow target**: instead of ``target_ln(encoder(goal_obs))``
(self-referential, learnable), the next-state target is the output of a
**frozen, pretrained encoder** — a pluggable ``TargetEncoder`` (see
``mip/target_encoders.py``) selected by ``optimization.target_encoder_type`` —
z-scored by **precomputed per-dimension goal statistics** (``goal_stats_path``).

This isolates "what state representation is the policy asked to denoise toward"
as a single knob, holding the rest of the joint_pt pipeline (trunk, decoupled
t, schedules, CFG, EMA of the *input* rep, the s2e knob) fixed. Swapping the
target representation (DP/LBMDiT, LeWM, DINOv2, ...) is a ``target_encoder_type``
change, not agent surgery.

Why precomputed z-score (not a learnable LayerNorm) on the frozen target:
standard practice for denoising a pretrained latent (LDM's fixed scalar,
l-DAE's per-dim standardization). A learnable shared LN would (a) drift as the
condition encoder trains — reintroducing a moving target — and (b) confound a
fair comparison across target encoders by reshaping each one's geometry
differently. A fixed per-dim z-score is a global, invertible transform that
preserves the representation being ablated and puts every encoder's target on
a comparable ~unit-variance scale. Export stats with
``scripts/compute_goal_stats_dp.py`` (robomimic).

The target encoder is never EMA'd, runs in eval (deterministic embeddings,
matching the EO/EP baseline whose target came from the eval EMA encoder), and is
used at TRAINING ONLY — inference is byte-identical to ``LBMDiTJointPTAgent``
(the target encoder is not touched in ``sample`` / ``sample_joint``), so the
deployed policy structure is unchanged across ablation arms.

Note: the frozen target encoder + z-score stats are NOT serialized into the
agent checkpoint (they are deterministic and reconstructed in ``__init__``).
The target-encoder source (e.g. ``dp_checkpoint_path``) and ``goal_stats_path``
must therefore stay valid for resume; deploy does not need them (inference never
calls the target encoder).

Author: Zilai Zeng
"""

from __future__ import annotations

import loguru
import torch

from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
from mip.config import Config
from mip.target_encoders import get_target_encoder


class LBMDiTJointPTFrozenTargetAgent(LBMDiTJointPTAgent):
    """joint_pt single-trunk agent with a frozen, pluggable FM-target encoder
    + precomputed per-dim z-score. See module docstring.
    """

    def __init__(self, config: Config):
        # Build the full joint_pt agent first (trainable input encoder +
        # target_ln + EMA + PT trunk + optimizer + all the joint_* caches).
        super().__init__(config)
        device = config.optimization.device
        opt = config.optimization

        # --- Frozen target encoder (pluggable; see mip/target_encoders.py) ---
        # Produces the FM state-flow target. "dp" = frozen DP/LBMDiT encoder;
        # other types (lewm, dinov2, ...) swap the target representation behind
        # the same interface. It is frozen (requires_grad off) and eval-always.
        self.target_encoder = get_target_encoder(config)

        # Guardrail: the trunk's state-stream dim must equal the target dim. The
        # trunk was built (in super().__init__) with state_dim = state_target_dim
        # or obs_dim, so a foreign target encoder requires network.state_target_dim
        # to be set to its output dim.
        obs_dim = config.network.encoder_out_dim or config.network.emb_dim
        trunk_state_dim = getattr(config.network, "state_target_dim", None) or obs_dim
        if self.target_encoder.output_dim != trunk_state_dim:
            raise ValueError(
                f"target encoder output_dim={self.target_encoder.output_dim} != "
                f"trunk state dim={trunk_state_dim}. Set "
                f"network.state_target_dim={self.target_encoder.output_dim} "
                f"(the target encoder's output dim)."
            )
        # RAE (arXiv 2510.11690): the diffusion trunk width (emb_dim / d_model)
        # must be >= the target feature dim, else the flow-matching loss has an
        # irreducible floor (the tail eigenvalues) and a wide target is unfairly
        # penalized. Non-blocking warning so a mis-set sweep emb_dim is caught.
        d_model = config.network.emb_dim
        if d_model < self.target_encoder.output_dim:
            loguru.logger.warning(
                f"network.emb_dim={d_model} < target dim "
                f"{self.target_encoder.output_dim}: per RAE the diffusion trunk "
                f"width should be >= the target feature dim, or the FM loss has "
                f"an irreducible floor. Raise network.emb_dim to >= "
                f"{self.target_encoder.output_dim}."
            )

        # --- Optionally warm-start the LIVE input encoder from the SAME source
        # encoder, so the condition and the frozen target start in the same
        # representation space (isolates whether s2e=True still hurts when there
        # is no foreign-manifold gap at init). Input encoder stays trainable;
        # only the target is frozen. Only valid when the target encoder can
        # produce a state_dict compatible with our MultiImageObsEncoder (e.g.
        # the DP target); foreign ViT targets return None and error here.
        if opt.init_input_encoder_from_dp:
            if opt.idm_checkpoint_path is not None:
                raise ValueError(
                    "init_input_encoder_from_dp=True conflicts with "
                    "idm_checkpoint_path (two competing input-encoder warm-"
                    "starts). Set optimization.idm_checkpoint_path=null."
                )
            init_sd = self.target_encoder.input_encoder_init_state_dict()
            if init_sd is None:
                raise ValueError(
                    "init_input_encoder_from_dp=True but target_encoder_type="
                    f"{opt.target_encoder_type!r} cannot initialize the input "
                    "encoder (architecturally incompatible with the "
                    "MultiImageObsEncoder). Use target_encoder_type='dp', or "
                    "set init_input_encoder_from_dp=false."
                )
            self.encoder.load_state_dict(init_sd)
            self.encoder.requires_grad_(True)  # stays live (finetune)
            # encoder_ema was deepcopied from the pre-warm-start (scratch)
            # encoder in the parent __init__; re-sync it so the EMA and the
            # eval-time condition start aligned with the warm-started weights
            # (mirrors the idm-warmstart path, which deepcopies AFTER loading).
            self.encoder_ema.load_state_dict(self.encoder.state_dict())
            loguru.logger.info(
                "init_input_encoder_from_dp=True: input encoder warm-started "
                "from the target encoder's source weights (LIVE / finetuned); "
                "encoder_ema re-synced to match"
            )

        # --- Precomputed per-dim z-score stats for the frozen target ---
        # Reuses the inherited _normalize_goal (reads self._goal_mean /
        # self._goal_var, nulled by the parent __init__).
        stats_path = opt.goal_stats_path
        if stats_path is None:
            loguru.logger.warning(
                "No goal_stats_path set — the frozen target will be denoised "
                "in RAW encoder space (not zero-mean / unit-var). Export stats "
                "with scripts/compute_goal_stats_dp.py for a fair comparison."
            )
            self._goal_mean = None
            self._goal_var = None
        else:
            loguru.logger.info(f"Loading goal z-score stats from {stats_path}")
            stats = torch.load(stats_path, map_location=device, weights_only=False)
            self._goal_mean = stats["mean"].to(device)
            self._goal_var = stats["var"].to(device)

        # EMA-target is meaningless for a frozen target (and its inherited code
        # path would silently use the *condition* encoder's EMA, not this one).
        # Force it off.
        if self._use_ema_target:
            loguru.logger.info(
                "Disabling joint_use_ema_target: the target encoder is frozen "
                "and external; EMA-target would reference the condition encoder."
            )
        self._use_ema_target = False

    # ------------------------------------------------------------------
    # The one swapped piece: the FM state-flow target.
    # ------------------------------------------------------------------
    def _encode_condition_target(self, obs, goal_obs):
        """Condition from the trainable input encoder (+ target_ln, live grad);
        target from the FROZEN ``TargetEncoder``, z-scored by precomputed stats.

        The condition path is byte-identical to the parent so the encoder's
        gradient channel and the ``joint_state_loss_to_encoder`` routing in
        ``update`` are unchanged. The target is fully detached (frozen encoder
        under no_grad), so the input encoder is never shaped by being a target.
        """
        z_t = self.target_ln(self.encoder(obs, None))
        with torch.no_grad():
            target = self._normalize_goal(self.target_encoder.embed(goal_obs))
        return z_t, target

    # ------------------------------------------------------------------
    # Modes: keep the frozen target encoder deterministic (eval) always.
    # save/load are inherited — the target encoder + stats are reconstructed
    # in __init__ from the configured source / goal_stats_path, so they need no
    # checkpoint round-trip.
    # ------------------------------------------------------------------
    def eval(self):
        super().eval()
        self.target_encoder.eval()

    def train(self):
        super().train()
        # Frozen target stays in eval: deterministic goal embeddings (no
        # CropRandomizer / dropout / BN-batch noise in the regression target).
        self.target_encoder.eval()
