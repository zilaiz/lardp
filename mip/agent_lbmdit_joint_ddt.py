"""LBMDiTJointDDTAgent: joint flow matching with decoupled state/action time.

DDT-style joint trunk (encoder + decoder width split, per-token AdaLN cond)
combined with the E2E target-LN encoder pattern from LBMDiTJointE2EAgent.

Differences from ``LBMDiTJointE2EAgent``:

- ``self.net`` is ``LBMDiTJointDDT`` (per-token AdaLN, encoder+decoder split)
  instead of ``LBMDiTJoint`` (uniform-width DiT, global cond).
- During training, the state token and action chunk can have **independent
  flow times** ``t_state`` and ``t_action`` (configurable via
  ``optimization.joint_decouple_t``). When decoupled, each batch sample
  draws ``t_state, t_action ~ Uniform[eps, 1-eps]`` independently.
- During inference, the (t_state, t_action) trajectory through [0,1]^2 is
  configurable via ``optimization.joint_t_schedule``:

    - "diagonal":    t_state == t_action at every step (current behavior).
    - "state_first": clean state first, then clean action ("plan-then-act").
    - "pyramid":     t_state advances ahead of t_action by a fixed offset.

  Both streams still share the same number of Euler sub-steps; the schedule
  only changes which (t_state, t_action) pair the network is queried at.

Loss path is identical to E2E except for the t per stream:

    z_t    = target_ln(encoder(obs))
    target = target_ln(encoder(goal_obs)).detach()
    s_t    = (1 - t_state)  * s_noise + t_state  * target
    a_t    = (1 - t_action) * a_noise + t_action * act
    v_s, v_a, _ = net(s_t, a_t, t_state, t_action, z_t, optimality)
    state_loss  = mean((v_s - target_dot)^2) / obs_dim
    action_loss = mean((v_a - act_dot)^2)    / act_dim
    loss = w_s * state_loss + w_a * action_loss

``optimization.joint_play_scheme`` selects how the two losses are assigned
over the decoupled (t_state, t_action) square:

- "legacy":       both losses on every row (the path above, unchanged).
- "noisier_all":  every row contributes only the loss of its noisier stream
                  (lower t) — predict the noisier stream from the strictly
                  cleaner one. The diagonal partitions the square into a
                  continuous IDM-like half (action loss, cleaner state) and
                  an FDM-like half (state loss, cleaner actions).
- "noisier_play": the rule above for play-source rows only; expert rows keep
                  both losses, so the expert policy objective is identical
                  to legacy and play adds continuous-spectrum dynamics
                  supervision on top.

Author: Zilai Zeng
"""

from __future__ import annotations

import os
from copy import deepcopy

import loguru
import numpy as np
import torch
import torch.nn as nn

from mip.agent_lbmdit_joint_e2e import LBMDiTJointE2EAgent
from mip.config import Config
from mip.interpolant import Interpolant
from mip.losses import get_norm
from mip.network_utils import get_encoder
from mip.networks.lbmdit_joint import LBMDiTJoint  # for NULL_IDX / EXPERT_IDX (same constants)
from mip.networks.lbmdit_joint_ddt import LBMDiTJointDDT
from mip.torch_utils import report_parameters


class LBMDiTJointDDTAgent(LBMDiTJointE2EAgent):
    """DDT-trunk joint agent with decoupled state/action time scheduling."""

    def __init__(self, config: Config):
        # Skip LBMDiTJointE2EAgent.__init__ (it instantiates LBMDiTJoint);
        # we re-do the rebuild path with LBMDiTJointDDT instead. Mirrors the
        # parent's "don't call super().__init__" pattern.
        self.config = config
        device = config.optimization.device

        # Ablation invariant: replacing x_state with a learnable token is
        # coherent only when state_loss is disabled (the state head would
        # otherwise be asked to predict per-sample velocity from a constant
        # input under a non-zero loss).
        if (
            getattr(config.optimization, "joint_replace_x_state", False)
            and config.optimization.joint_state_loss_weight != 0.0
        ):
            raise ValueError(
                "optimization.joint_replace_x_state=True requires "
                "optimization.joint_state_loss_weight == 0.0; got "
                f"{config.optimization.joint_state_loss_weight}"
            )

        # --- Encoder (instantiate first, then optionally warm-start) ---
        self.encoder = get_encoder(config.network, config.task).to(device)
        idm_path = config.optimization.idm_checkpoint_path
        if idm_path is not None:
            loguru.logger.info(
                f"Warm-starting encoder from {idm_path} (weights only; rest of "
                f"the IDM checkpoint is ignored in the DDT agent)"
            )
            state_dict = torch.load(
                idm_path, map_location=device, weights_only=False,
            )
            encoder_sd = state_dict["encoder"]
            has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
            if has_goal_dropout:
                inner_sd = {
                    k.removeprefix("encoder."): v
                    for k, v in encoder_sd.items()
                    if k.startswith("encoder.")
                }
                self.encoder.load_state_dict(inner_sd)
            else:
                self.encoder.load_state_dict(encoder_sd)
        else:
            loguru.logger.info(
                "No idm_checkpoint_path — encoder will train from scratch"
            )
        self.encoder.requires_grad_(True)

        # --- Target LayerNorm (replaces offline goal stats) ---
        obs_dim = config.network.encoder_out_dim or config.network.emb_dim
        if getattr(config.optimization, "joint_input_ln", True):
            self.target_ln = nn.LayerNorm(
                obs_dim,
                elementwise_affine=config.optimization.joint_target_ln_affine,
            ).to(device)
        else:
            # Raw encoder output as the AdaLN condition (no per-sample LN).
            # Identity is a drop-in with no params, so it is constructed here
            # before the optimizer/EMA below (no orphaned LN params) and every
            # ``self.target_ln(...)`` call site (train + eval condition) passes
            # through. See OptimizationConfig.joint_input_ln — used by the
            # frozen-target ablation to drop the condition/target
            # normalization-space mismatch.
            self.target_ln = nn.Identity().to(device)
            loguru.logger.info(
                "joint_input_ln=False: input encoder output used RAW as the "
                "AdaLN condition (target_ln = Identity)"
            )

        # --- Encoder + target_ln EMA (matches lbmdit / TrainingAgent
        # convention: at eval, both trunk and encoder are smoothed). EMA
        # tracking is gated by ema_rate < 1; if ema_rate >= 1, eval falls
        # through to the live encoder via the use_ema branch in sample().
        self.encoder_ema = deepcopy(self.encoder).requires_grad_(False)
        self.encoder_ema.eval()
        self.target_ln_ema = deepcopy(self.target_ln).requires_grad_(False)
        self.target_ln_ema.eval()

        # Goal-stats helpers from the parent are unused here; null them.
        self._goal_mean = None
        self._goal_var = None
        self._norm_eps = 1e-5

        # --- Joint trunk (DDT variant) + EMA ---
        # Trunk construction is delegated to ``_build_net`` so subclasses can
        # swap the architecture (e.g. ``LBMDiTJointPTAgent`` uses a single-
        # stack trunk) without duplicating the rest of ``__init__``.
        self.net = self._build_net(config, obs_dim, device)
        report_parameters(self.net, model_name=type(self.net).__name__)

        self.net_ema = deepcopy(self.net).requires_grad_(False)
        self.net_ema.eval()

        # --- Interpolant ---
        self.interpolant = Interpolant(config.optimization.interp_type)

        # --- Optimizer (trunk + encoder + LN) ---
        # The encoder + target_ln (the representation) can be optimized at a
        # reduced LR relative to the trunk via ``joint_encoder_lr_scale`` < 1,
        # slowing the representation so the trunk adapts to it rather than the
        # latent collapsing to ease the denoising objective (JEDI / JEPA /
        # TD-MPC2 anti-collapse lever). scale == 1.0 keeps the original single
        # param group for exact checkpoint/optimizer-state compatibility.
        enc_lr_scale = float(config.optimization.joint_encoder_lr_scale)
        base_lr = config.optimization.lr
        wd = config.optimization.weight_decay
        if enc_lr_scale == 1.0:
            params = (
                list(self.net.parameters())
                + list(self.encoder.parameters())
                + list(self.target_ln.parameters())
            )
            self.optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)
        else:
            # Trunk group first so ``get_last_lr()[0]`` logs the trunk LR.
            param_groups = [
                {"params": list(self.net.parameters()), "lr": base_lr},
                {
                    "params": (
                        list(self.encoder.parameters())
                        + list(self.target_ln.parameters())
                    ),
                    "lr": base_lr * enc_lr_scale,
                },
            ]
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
            loguru.logger.info(
                f"Encoder + target_ln LR scaled to {enc_lr_scale}x trunk LR "
                f"({base_lr * enc_lr_scale:g} vs {base_lr:g})"
            )

        # Cache scalars
        self._w_state = config.optimization.joint_state_loss_weight
        self._w_action = config.optimization.joint_action_loss_weight
        self._cfg_dropout_prob = config.optimization.joint_cfg_dropout_prob
        self._cfg_scale = config.optimization.joint_cfg_scale
        self._sample_mode = config.optimization.joint_sample_mode
        self._num_steps = config.optimization.joint_num_steps
        self._decouple_t = config.optimization.joint_decouple_t
        self._play_avoid_both_noise = bool(
            getattr(config.optimization, "joint_play_avoid_both_noise", False)
        )
        self._play_both_noise_tau = float(
            getattr(config.optimization, "joint_play_both_noise_tau", 0.5)
        )
        self._t_schedule = config.optimization.joint_t_schedule
        self._pyramid_offset = config.optimization.joint_pyramid_offset
        self._t_eps = config.optimization.joint_t_eps
        self._use_ema_target = bool(config.optimization.joint_use_ema_target)
        self._state_loss_to_encoder = bool(
            config.optimization.joint_state_loss_to_encoder
        )
        # --- Loss-assignment scheme over the decoupled (t_state, t_action)
        # square. See OptimizationConfig.joint_play_scheme.
        self._play_scheme = getattr(
            config.optimization, "joint_play_scheme", "legacy",
        )
        if self._play_scheme not in ("legacy", "noisier_all", "noisier_play"):
            raise ValueError(
                "joint_play_scheme must be 'legacy', 'noisier_all', or "
                f"'noisier_play'; got {self._play_scheme!r}"
            )
        if self._play_scheme != "legacy":
            if not self._decouple_t:
                raise ValueError(
                    f"joint_play_scheme={self._play_scheme!r} requires "
                    "joint_decouple_t=True (the noisier-stream rule needs "
                    "independent t_state / t_action; under a shared t every "
                    "row ties)."
                )
            if self._state_loss_to_encoder:
                loguru.logger.warning(
                    f"joint_play_scheme={self._play_scheme!r} with "
                    "joint_state_loss_to_encoder=True: active state-loss "
                    "rows will shape the encoder through the live condition "
                    "path (the self-referential channel that drove the "
                    "s2e=True collapse under legacy). Deliberate-ablation "
                    "setting; use False for the clean scheme comparison."
                )
            if self._play_avoid_both_noise:
                loguru.logger.info(
                    "joint_play_avoid_both_noise is inert under "
                    f"joint_play_scheme={self._play_scheme!r}: the noisier-"
                    "stream rule already assigns the both-noise corner a "
                    "single loss."
                )
        # Per-stream SD3-style time shift. 1.0 = identity (no shift).
        self._shift_state = float(config.optimization.joint_t_shift_state)
        self._shift_action = float(config.optimization.joint_t_shift_action)
        self._t_dist = config.optimization.joint_t_dist
        self._t_dist_mu = float(config.optimization.joint_t_dist_mu)
        self._t_dist_sigma = float(config.optimization.joint_t_dist_sigma)
        # Per-stream overrides (used only when joint_decouple_t=True). Each
        # falls back to the shared value when its config field is None, so
        # existing configs reproduce the shared-distribution behavior exactly.
        def _coalesce(stream_val, shared_val):
            return shared_val if stream_val is None else stream_val
        self._t_dist_state = _coalesce(
            getattr(config.optimization, "joint_t_dist_state", None), self._t_dist
        )
        self._t_dist_mu_state = float(_coalesce(
            getattr(config.optimization, "joint_t_dist_mu_state", None),
            self._t_dist_mu,
        ))
        self._t_dist_sigma_state = float(_coalesce(
            getattr(config.optimization, "joint_t_dist_sigma_state", None),
            self._t_dist_sigma,
        ))
        self._t_dist_action = _coalesce(
            getattr(config.optimization, "joint_t_dist_action", None), self._t_dist
        )
        self._t_dist_mu_action = float(_coalesce(
            getattr(config.optimization, "joint_t_dist_mu_action", None),
            self._t_dist_mu,
        ))
        self._t_dist_sigma_action = float(_coalesce(
            getattr(config.optimization, "joint_t_dist_sigma_action", None),
            self._t_dist_sigma,
        ))
        _valid_dists = ("uniform", "logit_normal", "beta", "reverse_beta")
        for name, val in (
            ("joint_t_dist", self._t_dist),
            ("joint_t_dist_state", self._t_dist_state),
            ("joint_t_dist_action", self._t_dist_action),
        ):
            if val not in _valid_dists:
                raise ValueError(
                    f"{name} must be one of {_valid_dists}; got {val!r}"
                )
        # State-head parameterization: "velocity" (raw v_state output) or
        # "x1" (UNITE-style — treat the raw output as the x1 estimate of the
        # LN'd target and derive v analytically; no LN is applied to the
        # prediction itself). See OptimizationConfig.joint_state_param.
        self._state_param = getattr(
            config.optimization, "joint_state_param", "velocity",
        )
        if self._state_param not in ("velocity", "x1"):
            raise ValueError(
                f"joint_state_param must be 'velocity' or 'x1'; "
                f"got {self._state_param!r}"
            )
        # Action-head parameterization (analog of state, no LN). See
        # OptimizationConfig.joint_action_param.
        self._action_param = getattr(
            config.optimization, "joint_action_param", "velocity",
        )
        if self._action_param not in ("velocity", "x1"):
            raise ValueError(
                f"joint_action_param must be 'velocity' or 'x1'; "
                f"got {self._action_param!r}"
            )
        self._x1_pred_eps = float(
            getattr(config.optimization, "joint_x1_pred_eps", 5e-2)
        )


    def _build_net(self, config: Config, obs_dim: int, device) -> nn.Module:
        """Build the joint trunk. Override in subclasses to swap architecture.

        The returned module must expose the joint forward signature
        ``forward(x_state, x_action, s, t, condition, optimality_idx)
        -> (v_state, v_action, None)`` so the inherited ``update`` /
        ``sample`` / ``sample_joint`` paths work unchanged.
        """
        enc_hidden = config.network.joint_ddt_d_model_enc or config.network.emb_dim
        dec_hidden = config.network.joint_ddt_d_model_dec or 2 * enc_hidden
        return LBMDiTJointDDT(
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
            # ``joint_cond_compose`` was added after the original DDT runs;
            # fall back to "add" (the only mode those runs supported) when
            # the saved hydra config predates the field.
            cond_compose=getattr(config.network, "joint_cond_compose", "add"),
            replace_x_state=getattr(
                config.optimization, "joint_replace_x_state", False,
            ),
        ).to(device)

    @staticmethod
    def _apply_t_shift(t, alpha: float):
        """SD3-style time shift t' = a*t / (1 + (a-1)*t).

        Identity when alpha == 1. Works on both torch tensors and numpy
        arrays (only uses element-wise arithmetic).
        """
        if alpha == 1.0:
            return t
        return alpha * t / (1.0 + (alpha - 1.0) * t)

    def _sample_base_t(
        self,
        shape: tuple,
        device: torch.device,
        lo: float,
        hi: float,
        dist: str | None = None,
        mu: float | None = None,
        sigma: float | None = None,
    ) -> torch.Tensor:
        """Sample base flow time before any per-stream shift is applied.

        ``"uniform"`` draws Uniform[lo, hi]. ``"logit_normal"`` draws
        sigmoid(N(mu, sigma)) and clamps to [lo, hi] (matches SD3/UNITE).
        ``"beta"`` mirrors ``flow_beta_loss`` (PI-0 schedule under mip's
        t=0 noise / t=1 data convention): u ~ Beta(1.5, 1.0); t = 0.999*(1-u).
        ``"reverse_beta"`` reproduces the original (pre-fix) flow_beta:
        t ~ Beta(1.5, 1.0) directly, mass at t≈1 (data end). For A/B only.
        lo/hi are ignored on the beta branches — caps are intrinsic.

        ``dist`` / ``mu`` / ``sigma`` default to the shared ``self._t_dist`` /
        ``_t_dist_mu`` / ``_t_dist_sigma`` so existing callers are unchanged;
        pass per-stream values to give the state and action streams different
        base distributions (mu/sigma only matter on the logit_normal branch).
        """
        dist = self._t_dist if dist is None else dist
        mu = self._t_dist_mu if mu is None else mu
        sigma = self._t_dist_sigma if sigma is None else sigma
        if dist == "uniform":
            return torch.empty(shape, device=device).uniform_(lo, hi)
        if dist == "beta":
            u = torch.distributions.Beta(1.5, 1.0).sample(shape).to(device)
            return 0.999 * (1.0 - u)
        if dist == "reverse_beta":
            return torch.distributions.Beta(1.5, 1.0).sample(shape).to(device)
        # logit_normal
        z = torch.randn(shape, device=device) * sigma + mu
        t = torch.sigmoid(z)
        return t.clamp(lo, hi)

    # ------------------------------- training -------------------------------

    def _encode_condition_target(self, obs, goal_obs):
        """Return ``(z_t, target)`` for one update step.

        ``z_t`` is the AdaLN condition embedding on the LIVE gradient path —
        the encoder's only gradient channel, and what the
        ``joint_state_loss_to_encoder`` routing in ``update`` keys off.
        ``target`` is the detached FM state-flow regression target.

        Default (joint_pt / E2E behaviour): both come from the trainable
        encoder + learnable ``target_ln``; the target optionally uses the EMA
        encoder + EMA LN (self-distillation) and is stop-grad either way.
        Subclasses override this to denoise toward a different state
        representation (see ``LBMDiTJointPTFrozenTargetAgent``).
        """
        config = self.config.optimization
        z_t = self.target_ln(self.encoder(obs, None))
        if self._use_ema_target and config.ema_rate < 1:
            with torch.no_grad():
                target = self.target_ln_ema(self.encoder_ema(goal_obs, None))
        else:
            z_goal_ln = self.target_ln(self.encoder(goal_obs, None))
            target = z_goal_ln.detach()
        return z_t, target

    def update(
        self,
        act: torch.Tensor,
        obs,
        goal_obs,
        delta_t: torch.Tensor,
        optimality: torch.Tensor | None = None,
    ) -> dict:
        """Joint flow-matching update with decoupled t_state / t_action."""
        config = self.config.optimization
        device = act.device
        B = act.shape[0]

        # 1. Condition embedding (live grad path) + detached FM state target.
        #    Factored into a hook so subclasses can swap the target
        #    representation (e.g. a frozen external target encoder) without
        #    duplicating the rest of update(). Default path: live encoder + LN
        #    for the condition; EMA-or-stop-grad target_ln of the encoder for
        #    the detached target (self-distillation when joint_use_ema_target).
        z_t, target = self._encode_condition_target(obs, goal_obs)

        # 2. Optimality labels with CFG dropout. ``optimality is None`` means
        # the dataset carries no labels (unlabeled fallback) — distinct from a
        # real all-play batch; the play corner-avoidance keys off real labels.
        labels_provided = optimality is not None
        if optimality is None:
            optimality = torch.full(
                (B,), LBMDiTJoint.NULL_IDX, dtype=torch.long, device=device,
            )
        else:
            optimality = optimality.to(device=device, dtype=torch.long)
        # Data-source play mask, captured pre-dropout (CFG dropout below
        # relabels some expert samples to NULL; corner-avoidance keys off the
        # data source, so dropped-expert keeps the full t-distribution).
        is_play_data = optimality == LBMDiTJoint.NULL_IDX
        if self.net.training and self._cfg_dropout_prob > 0:
            drop_mask = torch.rand(B, device=device) < self._cfg_dropout_prob
            optimality = torch.where(
                drop_mask,
                torch.full_like(optimality, LBMDiTJoint.NULL_IDX),
                optimality,
            )

        # 3. Per-stream flow time. Each stream:
        #    (a) draws a base t from joint_t_dist (uniform or logit_normal),
        #    (b) applies the per-stream SD3-style shift if alpha != 1.
        #    When ``joint_decouple_t`` is False, both streams share *one*
        #    base t and *one* shift (state's), recovering a single-t
        #    baseline on the new trunk for A/B comparison.
        eps = self._t_eps
        lo, hi = eps, 1.0 - eps
        if self._decouple_t:
            t_state_base = self._sample_base_t(
                (B,), device, lo, hi,
                self._t_dist_state, self._t_dist_mu_state, self._t_dist_sigma_state,
            )
            t_action_base = self._sample_base_t(
                (B,), device, lo, hi,
                self._t_dist_action, self._t_dist_mu_action,
                self._t_dist_sigma_action,
            )
            # Corner-avoidance (legacy scheme only — the noisier-stream rule
            # already assigns the both-noise corner a single loss). Apply
            # whenever optimality labels are real (mixed OR pure-play); skip
            # only the unlabeled fallback (optimality was None), where every
            # sample defaults to NULL and might actually be expert.
            if (
                self._play_scheme == "legacy"
                and self._play_avoid_both_noise
                and labels_provided
            ):
                # Play data: drop the both-near-noise corner (unconditional
                # joint generation); lift one random stream into [tau, hi] so
                # the sample lands in an IDM/FDM-like regime instead.
                tau = self._play_both_noise_tau
                fix = is_play_data & (t_state_base < tau) & (t_action_base < tau)
                pick_state = torch.rand(B, device=device) < 0.5
                new_t = torch.empty(B, device=device).uniform_(tau, hi)
                t_state_base = torch.where(fix & pick_state, new_t, t_state_base)
                t_action_base = torch.where(fix & ~pick_state, new_t, t_action_base)
            t_state = self._apply_t_shift(t_state_base, self._shift_state)
            t_action = self._apply_t_shift(t_action_base, self._shift_action)
        else:
            shared_base = self._sample_base_t((B,), device, lo, hi)
            shared = self._apply_t_shift(shared_base, self._shift_state)
            t_state = shared
            t_action = shared

        # 4. Per-stream noise + interpolation, each with its own t.
        s_noise = torch.randn_like(target)
        a_noise = torch.randn_like(act)

        s_t = self.interpolant.calc_It(t_state, s_noise, target)
        s_t_dot = self.interpolant.calc_It_dot(t_state, s_noise, target)
        a_t = self.interpolant.calc_It(t_action, a_noise, act)
        a_t_dot = self.interpolant.calc_It_dot(t_action, a_noise, act)

        # 5. Joint forward — DDT trunk takes the two times directly.
        #    If state_loss is configured to NOT flow into the encoder, run
        #    two forward passes: one with z_t.detach() to produce the state
        #    head output (encoder receives no state_loss grad), one with
        #    live z_t to produce the action head output (encoder still
        #    receives action_loss grad). Otherwise a single forward feeds
        #    both heads. ``s_head`` and ``a_head`` are raw head outputs;
        #    their meaning depends on the per-stream parameterization
        #    (velocity directly, or x1-estimate to be converted in step 6).
        if self._state_loss_to_encoder:
            s_head, a_head, _ = self.net(
                x_state=s_t,
                x_action=a_t,
                s=t_state,
                t=t_action,
                condition=z_t,
                optimality_idx=optimality,
            )
        else:
            s_head, _, _ = self.net(
                x_state=s_t,
                x_action=a_t,
                s=t_state,
                t=t_action,
                condition=z_t.detach(),
                optimality_idx=optimality,
            )
            _, a_head, _ = self.net(
                x_state=s_t,
                x_action=a_t,
                s=t_state,
                t=t_action,
                condition=z_t,
                optimality_idx=optimality,
            )

        # 6. Per-stream parameterization.
        #    State:
        #      "velocity": ``s_head`` is v_state directly (baseline); v_gt is
        #                  ``s_t_dot`` (= target - s_noise), no clamp.
        #      "x1":       ``s_head`` is an x1-estimate of ``target``; derive
        #                  v_state = (s_head - s_t) / (1 - t_state) with matched
        #                  clamp on v_gt. No LN on the prediction.
        #    Action:
        #      "velocity": ``a_head`` is v_action directly; v_gt = a_t_dot.
        #      "x1":       ``a_head`` is an x1-estimate of ``act``; derive
        #                  v_action = (a_head - a_t) / (1 - t_action), matched
        #                  clamp on v_gt.
        if self._state_param == "x1":
            denom_s = (1.0 - t_state).clamp_min(self._x1_pred_eps).view(-1, 1, 1)
            v_state = (s_head - s_t) / denom_s
            v_state_gt = (target - s_t) / denom_s
        else:
            v_state = s_head
            v_state_gt = s_t_dot

        if self._action_param == "x1":
            denom_a = (1.0 - t_action).clamp_min(self._x1_pred_eps).view(-1, 1, 1)
            v_action = (a_head - a_t) / denom_a
            v_action_gt = (act - a_t) / denom_a
        else:
            v_action = a_head
            v_action_gt = a_t_dot

        # 7. Per-stream losses (per-element MSE so weights are interpretable).
        #    "legacy": both losses over all rows.
        #    "noisier_all" / "noisier_play": a rule-governed row contributes
        #    only the loss of its noisier stream — predict the noisier stream
        #    from the strictly cleaner one. Ties (t_state == t_action,
        #    measure-zero under decoupled continuous draws) go to the action
        #    loss. "noisier_play" applies the rule to play-source rows only;
        #    expert rows keep both losses (policy objective identical to
        #    legacy). Each loss is a masked mean over its active rows, so its
        #    scale stays comparable to legacy. Encoder routing composes with
        #    the masks: under s2e=False a rule-governed row on the state-
        #    noisier side has zero action loss, hence contributes no encoder
        #    gradient; under s2e=True (warned ablation) its masked state
        #    loss reaches the encoder through the single live forward.
        if self._play_scheme == "legacy":
            state_loss_unscaled = torch.mean(
                get_norm(v_state - v_state_gt, config.norm_type)
            ) / float(getattr(self.net, "state_dim", self.net.obs_dim))
            action_loss_unscaled = torch.mean(
                get_norm(v_action - v_action_gt, config.norm_type)
            ) / float(self.net.act_dim)
        else:
            state_rows = get_norm(
                v_state - v_state_gt, config.norm_type
            ).mean(dim=1)  # (B,)
            action_rows = get_norm(
                v_action - v_action_gt, config.norm_type
            ).mean(dim=1)  # (B,)
            # Compare post-shift times — the rule is about actual noise
            # level, and the per-stream SD3 shifts change it.
            state_noisier = t_state < t_action
            if self._play_scheme == "noisier_all":
                rule_rows = torch.ones_like(state_noisier)
            else:  # noisier_play
                # Keyed off the pre-dropout data source, like the corner-
                # avoidance: the unlabeled fallback (optimality was None)
                # might actually be expert, so it routes as expert.
                rule_rows = (
                    is_play_data if labels_provided
                    else torch.zeros_like(is_play_data)
                )
            state_mask = (~rule_rows | state_noisier).float()
            action_mask = (~rule_rows | ~state_noisier).float()
            state_loss_unscaled = (
                (state_rows * state_mask).sum()
                / state_mask.sum().clamp_min(1.0)
            ) / float(getattr(self.net, "state_dim", self.net.obs_dim))
            action_loss_unscaled = (
                (action_rows * action_mask).sum()
                / action_mask.sum().clamp_min(1.0)
            ) / float(self.net.act_dim)
        loss = (
            self._w_state * state_loss_unscaled
            + self._w_action * action_loss_unscaled
        )

        loss.backward()

        params = (
            list(self.net.parameters())
            + list(self.encoder.parameters())
            + list(self.target_ln.parameters())
        )
        if config.grad_clip_norm:
            grad_norm = nn.utils.clip_grad_norm_(params, config.grad_clip_norm)
        else:
            grad_norm = torch.tensor(0.0, device=device)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if config.ema_rate < 1:
            self._ema_update()

        with torch.no_grad():
            target_std = target.std(dim=(0, 1)).mean()
            target_abs_mean = target.abs().mean()

        del delta_t
        info = {
            "dp_loss": loss.detach(),
            "state_loss": state_loss_unscaled.detach(),
            "action_loss": action_loss_unscaled.detach(),
            "grad_norm": grad_norm.detach(),
            "target_std": target_std.detach(),
            "target_abs_mean": target_abs_mean.detach(),
            "t_state_mean": t_state.mean().detach(),
            "t_action_mean": t_action.mean().detach(),
        }
        if self._play_scheme != "legacy":
            # Dose verification: fraction of rows contributing each loss.
            # noisier_all -> the two sum to 1 (~0.5 each under matched t
            # dists); noisier_play -> expert_frac + play_frac * (~0.5) each.
            info["state_mask_frac"] = state_mask.mean().detach()
            info["action_mask_frac"] = action_mask.mean().detach()
        return info

    # ------------------------------- sampling ------------------------------

    def _build_schedule(
        self, num_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t_state_grid, t_action_grid)``, each shape (num_steps + 1,).

        Both grids start at the noise endpoint (eps) and end at the data
        endpoint (1 - eps). The choice of trajectory between them depends
        on ``joint_t_schedule``. After the trajectory is chosen, each
        stream's grid is warped by its own SD3-style time shift so the
        Euler walk lands at exactly the t-distribution the trunk was
        trained on. With shift == 1.0 (default), this is a no-op.

        Convention reminder: mip's interpolant is ``(1 - t) * noise + t *
        data``, so ``t = 0`` is noise and ``t = 1`` is data. The Euler
        integration walks from t = 0 -> t = 1.
        """
        eps = self._t_eps
        lo, hi = eps, 1.0 - eps
        steps = num_steps

        if self._t_schedule == "diagonal":
            grid = np.linspace(lo, hi, steps + 1)
            t_state, t_action = grid, grid

        elif self._t_schedule == "action_only":
            # State stream pinned at lo throughout: ds = 0 every step, so
            # x_state never integrates and stays at its initial draw. Only
            # x_action walks lo -> hi. Use joint_sample_mode="stochastic" so
            # x_state inits to randn — what the trunk saw at training when
            # t_state ≈ eps (zero init is OOD at the noise endpoint).
            t_state = np.full(steps + 1, lo)
            t_action = np.linspace(lo, hi, steps + 1)
            assert t_state.shape == (steps + 1,)
            assert t_action.shape == (steps + 1,)

        elif self._t_schedule == "state_first":
            half = steps // 2 if steps >= 2 else 1
            t_state = np.concatenate([
                np.linspace(lo, hi, half + 1),     # state ramps 0 -> 1 first
                np.full(steps - half, hi),         # then holds at clean
            ])
            t_action = np.concatenate([
                np.full(half + 1, lo),             # action holds at noise first
                np.linspace(lo, hi, steps - half + 1)[1:],  # then ramps 0 -> 1
            ])
            assert t_state.shape == (steps + 1,)
            assert t_action.shape == (steps + 1,)

        elif self._t_schedule == "pyramid":
            # Lead-lag schedule. Both streams start at t=lo (matching the
            # pure-noise init of x_state and x_action) and end at t=hi
            # (matching clean data). State leads action by a fraction
            # ``offset`` of the total step budget: action is held at lo
            # for ``delay`` iterations while state ramps; then both ramp
            # together; then state holds at hi while action finishes.
            #   offset = 0   -> diagonal (full overlap, no lead)
            #   offset ~ 0.5 -> state_first (no overlap; state finishes
            #                   exactly when action starts)
            #   offset close to 1 -> hard state-first with extra holds
            offset = float(self._pyramid_offset)
            delay = int(round(offset * steps))
            delay = max(0, min(delay, steps - 1))   # keep both ramps non-trivial
            n_s = steps - delay                      # state's active sub-steps

            t_state = np.concatenate([
                np.linspace(lo, hi, n_s + 1),        # state ramps lo -> hi
                np.full(delay, hi),                  # then holds at clean
            ])
            t_action = np.concatenate([
                np.full(delay, lo),                  # action holds at noise
                np.linspace(lo, hi, steps - delay + 1),  # then ramps lo -> hi
            ])
            assert t_state.shape == (steps + 1,)
            assert t_action.shape == (steps + 1,)

        else:
            raise ValueError(f"Unknown joint_t_schedule: {self._t_schedule!r}")

        # Per-stream SD3-style shift. Identity when alpha == 1.
        t_state = self._apply_t_shift(t_state, self._shift_state)
        t_action = self._apply_t_shift(t_action, self._shift_action)
        return t_state, t_action

    def _head_to_velocity(
        self,
        head_cond: torch.Tensor,
        head_un: torch.Tensor | None,
        x_t: torch.Tensor,
        t_scalar: float,
        cfg_scale: float,
        ln_module: nn.Module | None,
        use_x1_param: bool,
    ) -> torch.Tensor:
        """Convert head output(s) into a velocity for one Euler step.

        Under ``use_x1_param=True``: optionally LN each branch via
        ``ln_module`` ("norm_first" CFG; pass ``ln_module=None`` to skip LN),
        CFG-mix, then derive ``v = (x_pred - x_t) / max(1 - t, eps)``.
        Under ``use_x1_param=False``: CFG-mix the raw outputs directly
        (treat them as velocity).

        ``head_un`` is unused when ``cfg_scale <= 0``.
        """
        if not use_x1_param:
            if cfg_scale > 0:
                return (1 + cfg_scale) * head_cond - cfg_scale * head_un
            return head_cond
        if ln_module is not None:
            head_cond = ln_module(head_cond)
            if cfg_scale > 0:
                head_un = ln_module(head_un)
        if cfg_scale > 0:
            x_pred = (1 + cfg_scale) * head_cond - cfg_scale * head_un
        else:
            x_pred = head_cond
        denom = max(1.0 - t_scalar, self._x1_pred_eps)
        return (x_pred - x_t) / denom

    @torch.no_grad()
    def sample(
        self,
        act_0: torch.Tensor,
        obs,
        num_steps: int = -1,
        use_ema: bool = True,
    ) -> torch.Tensor:
        net = self.net_ema if use_ema else self.net
        encoder, target_ln = self._eval_encoder_modules(use_ema)
        device = act_0.device
        B = act_0.shape[0]
        cfg_scale = self._cfg_scale
        steps = self._num_steps if num_steps < 1 else int(num_steps)

        z_t = target_ln(encoder(obs, None))
        obs_dim = z_t.shape[-1]
        # State-stream dim may be decoupled from the condition/obs dim (a
        # foreign target encoder of a different dim); init x_state at the
        # trunk's state dim, falling back to obs_dim for legacy trunks.
        state_dim = getattr(net, "state_dim", obs_dim)

        if self._sample_mode == "stochastic":
            x_state = torch.randn(B, 1, state_dim, device=device)
            x_action = act_0
        else:
            x_state = torch.zeros(B, 1, state_dim, device=device)
            x_action = torch.zeros_like(act_0)

        expert_idx = torch.full(
            (B,), LBMDiTJoint.EXPERT_IDX, device=device, dtype=torch.long,
        )
        null_idx = torch.full(
            (B,), LBMDiTJoint.NULL_IDX, device=device, dtype=torch.long,
        )

        t_state_grid, t_action_grid = self._build_schedule(steps)

        for i in range(steps):
            ts_now = float(t_state_grid[i])
            ta_now = float(t_action_grid[i])
            ds = float(t_state_grid[i + 1] - ts_now)
            da = float(t_action_grid[i + 1] - ta_now)

            t_state_b = torch.full((B,), ts_now, device=device)
            t_action_b = torch.full((B,), ta_now, device=device)

            s_head_cond, a_head_cond, _ = net(
                x_state, x_action, t_state_b, t_action_b, z_t, expert_idx,
            )
            if cfg_scale > 0:
                s_head_un, a_head_un, _ = net(
                    x_state, x_action, t_state_b, t_action_b, z_t, null_idx,
                )
            else:
                s_head_un, a_head_un = None, None

            # State stream: parameterization + CFG (no LN on the prediction).
            v_s = self._head_to_velocity(
                s_head_cond, s_head_un, x_state, ts_now, cfg_scale,
                ln_module=None,
                use_x1_param=(self._state_param == "x1"),
            )
            # Action stream: no LN; parameterization + CFG.
            v_a = self._head_to_velocity(
                a_head_cond, a_head_un, x_action, ta_now, cfg_scale,
                ln_module=None,
                use_x1_param=(self._action_param == "x1"),
            )

            x_state = x_state + v_s * ds
            x_action = x_action + v_a * da

        return x_action

    @torch.no_grad()
    def sample_joint(
        self,
        act_0: torch.Tensor,
        obs,
        num_steps: int = -1,
        use_ema: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as ``sample`` but also returns the predicted next-state in
        LN'd space (no inversion to raw encoder space).
        """
        net = self.net_ema if use_ema else self.net
        encoder, target_ln = self._eval_encoder_modules(use_ema)
        device = act_0.device
        B = act_0.shape[0]
        cfg_scale = self._cfg_scale
        steps = self._num_steps if num_steps < 1 else int(num_steps)

        z_t = target_ln(encoder(obs, None))
        obs_dim = z_t.shape[-1]
        # State-stream dim may be decoupled from the condition/obs dim (a
        # foreign target encoder of a different dim); init x_state at the
        # trunk's state dim, falling back to obs_dim for legacy trunks.
        state_dim = getattr(net, "state_dim", obs_dim)

        if self._sample_mode == "stochastic":
            x_state = torch.randn(B, 1, state_dim, device=device)
            x_action = act_0
        else:
            x_state = torch.zeros(B, 1, state_dim, device=device)
            x_action = torch.zeros_like(act_0)

        expert_idx = torch.full(
            (B,), LBMDiTJoint.EXPERT_IDX, device=device, dtype=torch.long,
        )
        null_idx = torch.full(
            (B,), LBMDiTJoint.NULL_IDX, device=device, dtype=torch.long,
        )

        t_state_grid, t_action_grid = self._build_schedule(steps)

        for i in range(steps):
            ts_now = float(t_state_grid[i])
            ta_now = float(t_action_grid[i])
            ds = float(t_state_grid[i + 1] - ts_now)
            da = float(t_action_grid[i + 1] - ta_now)

            t_state_b = torch.full((B,), ts_now, device=device)
            t_action_b = torch.full((B,), ta_now, device=device)

            s_head_cond, a_head_cond, _ = net(
                x_state, x_action, t_state_b, t_action_b, z_t, expert_idx,
            )
            if cfg_scale > 0:
                s_head_un, a_head_un, _ = net(
                    x_state, x_action, t_state_b, t_action_b, z_t, null_idx,
                )
            else:
                s_head_un, a_head_un = None, None

            # State stream: parameterization + CFG (no LN on the prediction).
            v_s = self._head_to_velocity(
                s_head_cond, s_head_un, x_state, ts_now, cfg_scale,
                ln_module=None,
                use_x1_param=(self._state_param == "x1"),
            )
            # Action stream: no LN; parameterization + CFG.
            v_a = self._head_to_velocity(
                a_head_cond, a_head_un, x_action, ta_now, cfg_scale,
                ln_module=None,
                use_x1_param=(self._action_param == "x1"),
            )

            x_state = x_state + v_s * ds
            x_action = x_action + v_a * da

        return x_action, x_state
