"""LBMDiTJointDDTFrozenDPAgent: DDT-trunk joint flow matching with a frozen
(or optionally fine-tuned) **LBMDiT/DP-pretrained** obs encoder.

Identical machinery to ``LBMDiTJointDDTFrozenAgent`` (per-token DDT trunk,
decoupled (t_state, t_action), offline goal-stats z-scoring of the FM state
target, optimality + CFG) — the only difference is where the encoder weights
come from:

  - parent (``LBMDiTJointDDTFrozenAgent``): IDM checkpoint, may be wrapped in
    ``GoalDropoutEncoder`` (``encoder.<inner>`` + ``uncond_emb``).
  - this class: pretrained LBMDiT/DP checkpoint (``mip/agent.py:save``), saved
    with separate ``encoder`` and ``encoder_ema`` keys, *not* wrapped in
    ``GoalDropoutEncoder``. We load whichever of those two the user picked.

The downstream encoder architecture must still match the LBMDiT checkpoint's
encoder architecture — pass the same ``network`` config the LBMDiT was trained
under (or a compatible one) so ``get_encoder`` produces the right module shape
for ``load_state_dict``.

Author: Zilai Zeng
"""

from __future__ import annotations

from copy import deepcopy

import loguru
import torch

from mip.agent_lbmdit_joint_ddt_frozen import LBMDiTJointDDTFrozenAgent
from mip.config import Config
from mip.interpolant import Interpolant
from mip.network_utils import get_encoder
from mip.networks.lbmdit_joint_ddt import LBMDiTJointDDT
from mip.torch_utils import report_parameters


class LBMDiTJointDDTFrozenDPAgent(LBMDiTJointDDTFrozenAgent):
    """DDT-trunk joint agent with encoder loaded from a pretrained LBMDiT/DP.

    Inherits ``update`` / ``sample`` / ``sample_joint`` / ``_build_schedule``
    / time helpers / save / load / eval / train / EMA from the IDM-frozen
    parent (``LBMDiTJointDDTFrozenAgent`` → ``LBMDiTJointAgent``). Only the
    encoder-loading path is overridden.
    """

    def __init__(self, config: Config):
        # Skip parent __init__ — it requires idm_checkpoint_path. Re-do the
        # standalone setup with DP checkpoint loading instead.
        self.config = config
        device = config.optimization.device

        # --- Encoder: instantiate, then load weights from LBMDiT checkpoint. ---
        self.encoder = get_encoder(config.network, config.task).to(device)

        dp_path = config.optimization.dp_checkpoint_path
        if dp_path is None:
            raise ValueError(
                "optimization.dp_checkpoint_path must be set for "
                "LBMDiTJointDDTFrozenDPAgent so the encoder is initialized "
                "from a pretrained LBMDiT (DP) checkpoint."
            )
        loguru.logger.info(f"Loading pretrained encoder from {dp_path} (DP)")
        state_dict = torch.load(dp_path, map_location=device, weights_only=False)

        use_ema = config.optimization.dp_use_encoder_ema
        encoder_key = "encoder_ema" if use_ema else "encoder"
        if encoder_key not in state_dict:
            available = sorted(k for k in state_dict.keys() if "encoder" in k)
            raise KeyError(
                f"DP checkpoint at {dp_path} has no '{encoder_key}' key. "
                f"Available encoder-like keys: {available}. "
                f"Toggle optimization.dp_use_encoder_ema or pick a different "
                f"checkpoint."
            )
        encoder_sd = state_dict[encoder_key]

        # Sanity-check this isn't a GoalDropoutEncoder state dict — LBMDiT
        # doesn't wrap, so any leading ``encoder.`` prefix here is unexpected
        # and probably means the user pointed us at an IDM checkpoint by
        # mistake. Fail loudly rather than silently strip.
        if "uncond_emb" in encoder_sd or any(
            k.startswith("encoder.") for k in encoder_sd
        ):
            raise RuntimeError(
                f"DP checkpoint at {dp_path} key '{encoder_key}' looks like a "
                f"GoalDropoutEncoder state dict (has 'encoder.' prefixed keys "
                f"or 'uncond_emb'). LBMDiT does not wrap the encoder; you may "
                f"have passed an IDM checkpoint. Use LBMDiTJointDDTFrozenAgent "
                f"(idm_checkpoint_path) for that case."
            )

        self.encoder.load_state_dict(encoder_sd)
        loguru.logger.info(
            f"Loaded {encoder_key} weights from DP checkpoint "
            f"({sum(v.numel() for v in encoder_sd.values()):,} params)"
        )

        if config.optimization.joint_freeze_encoder:
            self.encoder.requires_grad_(False)
            loguru.logger.info("Encoder frozen for DDT joint training")
        else:
            loguru.logger.info(
                "Encoder will be fine-tuned during DDT joint training"
            )

        # --- Goal normalization stats (same as IDM-frozen path) ---
        self._norm_eps = 1e-5
        stats_path = config.optimization.goal_stats_path
        if stats_path is not None:
            loguru.logger.info(
                f"Loading goal normalization stats from {stats_path}"
            )
            stats = torch.load(stats_path, map_location=device, weights_only=False)
            self._goal_mean = stats["mean"].to(device)  # (emb_dim,)
            self._goal_var = stats["var"].to(device)    # (emb_dim,)
        else:
            loguru.logger.warning(
                "No goal_stats_path set — state-flow will operate in raw "
                "encoder space (target is not zero-mean / unit-var)."
            )
            self._goal_mean = None
            self._goal_var = None

        # --- Joint DDT trunk + EMA (same as parent) ---
        obs_dim = config.network.encoder_out_dim or config.network.emb_dim
        enc_hidden = config.network.joint_ddt_d_model_enc or config.network.emb_dim
        dec_hidden = config.network.joint_ddt_d_model_dec or 2 * enc_hidden
        self.net = LBMDiTJointDDT(
            act_dim=config.task.act_dim,
            Ta=config.task.horizon,
            obs_dim=obs_dim,
            To=config.task.obs_steps,
            enc_hidden=enc_hidden,
            enc_depth=config.network.joint_ddt_enc_depth,
            enc_n_heads=config.network.joint_ddt_n_heads_enc,
            dec_hidden=dec_hidden,
            dec_depth=config.network.joint_ddt_dec_depth,
            dec_n_heads=config.network.joint_ddt_n_heads_dec,
            dropout=config.network.dropout,
            timestep_emb_type=config.network.timestep_emb_type,
            timestep_emb_dim=config.network.timestep_emb_dim,
            opt_emb_dim=config.network.joint_opt_emb_dim,
        ).to(device)
        report_parameters(self.net, model_name="LBMDiTJointDDT")

        self.net_ema = deepcopy(self.net).requires_grad_(False)
        self.net_ema.eval()

        # --- Interpolant ---
        self.interpolant = Interpolant(config.optimization.interp_type)

        # --- Optimizer ---
        params = list(self.net.parameters())
        if not config.optimization.joint_freeze_encoder:
            params += list(self.encoder.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

        # --- Scalar caches: parent fields + DDT-specific (mirror parent) ---
        self._w_state = config.optimization.joint_state_loss_weight
        self._w_action = config.optimization.joint_action_loss_weight
        self._cfg_dropout_prob = config.optimization.joint_cfg_dropout_prob
        self._cfg_scale = config.optimization.joint_cfg_scale
        self._sample_mode = config.optimization.joint_sample_mode
        self._num_steps = config.optimization.joint_num_steps

        self._decouple_t = config.optimization.joint_decouple_t
        self._t_schedule = config.optimization.joint_t_schedule
        self._pyramid_offset = config.optimization.joint_pyramid_offset
        self._t_eps = config.optimization.joint_t_eps
        self._shift_state = float(config.optimization.joint_t_shift_state)
        self._shift_action = float(config.optimization.joint_t_shift_action)
        self._t_dist = config.optimization.joint_t_dist
        self._t_dist_mu = float(config.optimization.joint_t_dist_mu)
        self._t_dist_sigma = float(config.optimization.joint_t_dist_sigma)
        if self._t_dist not in ("uniform", "logit_normal"):
            raise ValueError(
                f"joint_t_dist must be 'uniform' or 'logit_normal'; "
                f"got {self._t_dist!r}"
            )
