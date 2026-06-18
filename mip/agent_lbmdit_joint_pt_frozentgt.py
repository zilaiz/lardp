"""LBMDiTJointPTFrozenTargetAgent: joint_pt with a FROZEN external target encoder.

State-representation ablation for the joint_pt framework. Everything on the
INPUT / condition side is the unchanged single-trunk PT agent
(``LBMDiTJointPTAgent``): a trainable image encoder + learnable ``target_ln``
feed the AdaLN condition, and ``joint_state_loss_to_encoder`` still routes the
state-flow loss into (or away from) that input encoder. The ONLY thing swapped
is the **FM state-flow target**: instead of ``target_ln(encoder(goal_obs))``
(self-referential, learnable), the next-state target is the output of a
**frozen, pretrained encoder** loaded from a separate checkpoint, z-scored by
**precomputed per-dimension goal statistics** (``goal_stats_path``).

This isolates "what state representation is the policy asked to denoise toward"
as a single knob, holding the rest of the joint_pt pipeline (trunk, decoupled
t, schedules, CFG, EMA of the *input* rep, the s2e knob) fixed.

Why precomputed z-score (not a learnable LayerNorm) on the frozen target:
standard practice for denoising a pretrained latent (LDM's fixed scalar,
l-DAE's per-dim standardization). A learnable shared LN would (a) drift as the
condition encoder trains — reintroducing a moving target — and (b) confound a
fair comparison across target encoders by reshaping each one's geometry
differently. A fixed per-dim z-score is a global, invertible transform that
preserves the representation being ablated and puts every encoder's target on
a comparable ~unit-variance scale. Export stats with
``scripts/compute_goal_stats_dp.py`` (robomimic).

Encoder source: a pretrained **LBMDiT (DP)** checkpoint (``dp_checkpoint_path``,
top-level ``encoder`` / ``encoder_ema``; pick via ``dp_use_encoder_ema``). The
loader mirrors ``LBMDiTJointDDTFrozenDPAgent``. The target encoder is never
EMA'd, runs in eval (deterministic center-crop embeddings, matching the EO/EP
baseline whose target came from the eval EMA encoder), and is used at TRAINING
ONLY — inference is byte-identical to ``LBMDiTJointPTAgent`` (the target encoder
is not touched in ``sample`` / ``sample_joint``), so the deployed policy
structure is unchanged across ablation arms.

Note: the frozen target encoder + z-score stats are NOT serialized into the
agent checkpoint (they are deterministic and reconstructed in ``__init__``).
``dp_checkpoint_path`` and ``goal_stats_path`` must therefore stay valid for
resume; deploy does not need them (inference never calls the target encoder).

Author: Zilai Zeng
"""

from __future__ import annotations

import loguru
import torch

from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
from mip.config import Config
from mip.network_utils import get_encoder


class LBMDiTJointPTFrozenTargetAgent(LBMDiTJointPTAgent):
    """joint_pt single-trunk agent with a frozen, externally-sourced FM-target
    encoder + precomputed per-dim z-score. See module docstring.
    """

    def __init__(self, config: Config):
        # Build the full joint_pt agent first (trainable input encoder +
        # target_ln + EMA + PT trunk + optimizer + all the joint_* caches).
        super().__init__(config)
        device = config.optimization.device
        opt = config.optimization

        # --- Frozen target encoder from a pretrained LBMDiT (DP) checkpoint ---
        dp_path = opt.dp_checkpoint_path
        if dp_path is None:
            raise ValueError(
                "optimization.dp_checkpoint_path must be set for "
                "LBMDiTJointPTFrozenTargetAgent — it is the frozen encoder "
                "that produces the FM state-flow target."
            )
        loguru.logger.info(
            f"Loading FROZEN target encoder from {dp_path} (DP/LBMDiT)"
        )
        state_dict = torch.load(dp_path, map_location=device, weights_only=False)
        encoder_key = "encoder_ema" if opt.dp_use_encoder_ema else "encoder"
        if encoder_key not in state_dict:
            available = sorted(k for k in state_dict if "encoder" in k)
            raise KeyError(
                f"DP checkpoint {dp_path} has no '{encoder_key}' key. "
                f"Available encoder-like keys: {available}. Toggle "
                f"optimization.dp_use_encoder_ema or pick another checkpoint."
            )
        encoder_sd = state_dict[encoder_key]
        # LBMDiT does not wrap the encoder; a leading 'encoder.' prefix or an
        # 'uncond_emb' means this is an IDM / GoalDropout checkpoint by mistake.
        if "uncond_emb" in encoder_sd or any(
            k.startswith("encoder.") for k in encoder_sd
        ):
            raise RuntimeError(
                f"DP checkpoint {dp_path} key '{encoder_key}' looks like a "
                f"GoalDropoutEncoder / IDM state dict (has 'encoder.' prefix "
                f"or 'uncond_emb'). Point dp_checkpoint_path at an LBMDiT (DP) "
                f"checkpoint instead."
            )

        # Build with the SAME network/task config as the trainable encoder so
        # the output dim equals obs_dim (no trunk change needed). load_state_dict
        # is the architecture guardrail — it errors loudly on any mismatch
        # (e.g. a DP checkpoint trained at a different encoder_out_dim).
        self.target_encoder = get_encoder(config.network, config.task).to(device)
        self.target_encoder.load_state_dict(encoder_sd)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        loguru.logger.info(
            f"Loaded {encoder_key} into frozen target encoder "
            f"({sum(v.numel() for v in encoder_sd.values()):,} params)"
        )

        # --- Optionally warm-start the LIVE input encoder from the SAME DP
        # encoder, so the condition and the frozen target start in the same
        # representation space (isolates whether s2e=True still hurts when
        # there is no foreign-manifold gap to cross at init). The input encoder
        # stays trainable / finetuned; only the target encoder is frozen.
        if opt.init_input_encoder_from_dp:
            if opt.idm_checkpoint_path is not None:
                raise ValueError(
                    "init_input_encoder_from_dp=True conflicts with "
                    "idm_checkpoint_path (two competing input-encoder warm-"
                    "starts). Set optimization.idm_checkpoint_path=null."
                )
            # Same arch as the target encoder (both get_encoder(network, task)),
            # which already load_state_dict'd encoder_sd — so this is safe and
            # makes the input encoder identical to the frozen target at step 0.
            self.encoder.load_state_dict(encoder_sd)
            self.encoder.requires_grad_(True)  # stays live (finetune)
            # encoder_ema was deepcopied from the pre-warm-start (scratch)
            # encoder in the parent __init__; re-sync it so the EMA and the
            # eval-time condition start aligned with the warm-started weights
            # (mirrors the idm-warmstart path, which deepcopies AFTER loading).
            self.encoder_ema.load_state_dict(self.encoder.state_dict())
            loguru.logger.info(
                f"init_input_encoder_from_dp=True: input encoder warm-started "
                f"from {encoder_key} of {dp_path} (LIVE / finetuned); "
                f"encoder_ema re-synced to match"
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
        target from the FROZEN external encoder, z-scored by precomputed stats.

        The condition path is byte-identical to the parent so the encoder's
        gradient channel and the ``joint_state_loss_to_encoder`` routing in
        ``update`` are unchanged. The target is fully detached (frozen encoder
        under no_grad), so the input encoder is never shaped by being a target.
        """
        z_t = self.target_ln(self.encoder(obs, None))
        with torch.no_grad():
            z_goal = self.target_encoder(goal_obs, None)
            target = self._normalize_goal(z_goal)
        return z_t, target

    # ------------------------------------------------------------------
    # Modes: keep the frozen target encoder deterministic (eval) always.
    # save/load are inherited — the target encoder + stats are reconstructed
    # in __init__ from dp_checkpoint_path / goal_stats_path, so they need no
    # checkpoint round-trip.
    # ------------------------------------------------------------------
    def eval(self):
        super().eval()
        self.target_encoder.eval()

    def train(self):
        super().train()
        # Frozen target stays in eval: deterministic goal embeddings (no
        # CropRandomizer / dropout noise in the regression target).
        self.target_encoder.eval()
