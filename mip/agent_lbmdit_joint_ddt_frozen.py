"""LBMDiTJointDDTFrozenAgent: DDT-trunk joint flow matching with a frozen
(or optionally fine-tuned) IDM-pretrained obs encoder.

Combines two patterns:
  - Frozen-encoder pattern from ``LBMDiTJointAgent`` (offline goal-stats
    z-scoring of the FM target, no learnable target LayerNorm, no encoder EMA).
  - DDT trunk + decoupled-time machinery from ``LBMDiTJointDDTAgent``
    (per-token AdaLN, encoder/decoder width split, configurable
    (t_state, t_action) sampling at training time and Euler schedule at
    inference time).

When the encoder is frozen (``optimization.joint_freeze_encoder=True``,
default for this agent), the encoder is forwarded under ``no_grad`` and its
output is the FM condition / target. The state-flow target is z-scored by
``goal_stats_path`` (loaded once, in the parent's ``__init__``), exactly as
in ``LBMDiTJointAgent``. When the flag is ``False``, encoder params join the
optimizer and gradients flow back through the condition path, but there is
still no ``target_ln`` and no encoder EMA — that's the E2E variant's job.

Author: Zilai Zeng
"""

from __future__ import annotations

from copy import deepcopy

import loguru
import numpy as np
import torch
import torch.nn as nn

from mip.agent_lbmdit_joint import LBMDiTJointAgent
from mip.config import Config
from mip.interpolant import Interpolant
from mip.losses import get_norm
from mip.network_utils import get_encoder
from mip.networks.lbmdit_joint import LBMDiTJoint  # for NULL_IDX / EXPERT_IDX
from mip.networks.lbmdit_joint_ddt import LBMDiTJointDDT
from mip.torch_utils import report_parameters


class LBMDiTJointDDTFrozenAgent(LBMDiTJointAgent):
    """DDT-trunk joint agent with frozen-encoder + goal-stats normalization.

    Inherits the encoder loading / freezing path, ``_normalize_goal`` /
    ``_denormalize_goal``, ``_ema_update`` (trunk-only), ``save`` / ``load``,
    and ``eval`` / ``train`` from ``LBMDiTJointAgent``. Overrides only what
    the DDT trunk and decoupled-time API require.
    """

    def __init__(self, config: Config):
        # Skip LBMDiTJointAgent.__init__ — it instantiates LBMDiTJoint. We
        # re-do the encoder + goal-stats path here, then build LBMDiTJointDDT.
        # Mirrors the "don't call super().__init__" pattern used elsewhere
        # (LBMDiTJointE2EAgent, LBMDiTJointDDTAgent).
        self.config = config
        device = config.optimization.device

        # --- Encoder: instantiate, then load weights from IDM checkpoint.
        # Logic copied from LBMDiTJointAgent.__init__ so this agent is
        # standalone and not coupled to whatever the parent's __init__
        # happens to do today. ---
        self.encoder = get_encoder(config.network, config.task).to(device)

        idm_path = config.optimization.idm_checkpoint_path
        if idm_path is None:
            raise ValueError(
                "idm_checkpoint_path must be set for LBMDiTJointDDTFrozenAgent "
                "so the encoder is initialized from a pretrained IDM."
            )
        loguru.logger.info(f"Loading pretrained encoder from {idm_path}")
        state_dict = torch.load(idm_path, map_location=device, weights_only=False)
        encoder_sd = state_dict["encoder"]

        has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
        if has_goal_dropout:
            inner_sd = {
                k.removeprefix("encoder."): v
                for k, v in encoder_sd.items()
                if k.startswith("encoder.")
            }
            self.encoder.load_state_dict(inner_sd)
            loguru.logger.info(
                "Loaded inner encoder from GoalDropoutEncoder checkpoint"
            )
        else:
            self.encoder.load_state_dict(encoder_sd)
            loguru.logger.info("Loaded encoder weights from IDM checkpoint")

        if config.optimization.joint_freeze_encoder:
            self.encoder.requires_grad_(False)
            loguru.logger.info("Encoder frozen for DDT joint training")
        else:
            loguru.logger.info("Encoder will be fine-tuned during DDT joint training")

        # --- Goal normalization stats (mirrors LBMDiTJointAgent) ---
        self._norm_eps = 1e-5
        stats_path = config.optimization.goal_stats_path
        if stats_path is not None:
            loguru.logger.info(f"Loading goal normalization stats from {stats_path}")
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

        # --- Joint trunk (DDT variant) + EMA ---
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

        # --- Optimizer (joint trunk + optionally encoder) ---
        params = list(self.net.parameters())
        if not config.optimization.joint_freeze_encoder:
            params += list(self.encoder.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

        # --- Scalar caches: parent fields + DDT-specific ---
        self._w_state = config.optimization.joint_state_loss_weight
        self._w_action = config.optimization.joint_action_loss_weight
        self._cfg_dropout_prob = config.optimization.joint_cfg_dropout_prob
        self._cfg_scale = config.optimization.joint_cfg_scale
        self._sample_mode = config.optimization.joint_sample_mode
        self._num_steps = config.optimization.joint_num_steps

        # DDT time scheduling
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

    # --------------------------- time helpers ---------------------------
    # These mirror the static / instance helpers on LBMDiTJointDDTAgent;
    # duplicated here so this agent does not depend on the E2E DDT class.

    @staticmethod
    def _apply_t_shift(t, alpha: float):
        """SD3-style time shift t' = a*t / (1 + (a-1)*t). Identity at a=1."""
        if alpha == 1.0:
            return t
        return alpha * t / (1.0 + (alpha - 1.0) * t)

    def _sample_base_t(
        self, shape: tuple, device: torch.device, lo: float, hi: float,
    ) -> torch.Tensor:
        if self._t_dist == "uniform":
            return torch.empty(shape, device=device).uniform_(lo, hi)
        z = torch.randn(shape, device=device) * self._t_dist_sigma + self._t_dist_mu
        t = torch.sigmoid(z)
        return t.clamp(lo, hi)

    def _build_schedule(
        self, num_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t_state_grid, t_action_grid)``, each shape ``(num_steps + 1,)``.

        Mirrors ``LBMDiTJointDDTAgent._build_schedule`` — see that docstring
        for trajectory semantics. Both grids walk lo -> hi (noise -> data),
        per-stream SD3 shift applied at the end.
        """
        eps = self._t_eps
        lo, hi = eps, 1.0 - eps
        steps = num_steps

        if self._t_schedule == "diagonal":
            grid = np.linspace(lo, hi, steps + 1)
            t_state, t_action = grid, grid

        elif self._t_schedule == "state_first":
            half = steps // 2 if steps >= 2 else 1
            t_state = np.concatenate([
                np.linspace(lo, hi, half + 1),
                np.full(steps - half, hi),
            ])
            t_action = np.concatenate([
                np.full(half + 1, lo),
                np.linspace(lo, hi, steps - half + 1)[1:],
            ])
            assert t_state.shape == (steps + 1,)
            assert t_action.shape == (steps + 1,)

        elif self._t_schedule == "pyramid":
            offset = float(self._pyramid_offset)
            delay = int(round(offset * steps))
            delay = max(0, min(delay, steps - 1))
            n_s = steps - delay

            t_state = np.concatenate([
                np.linspace(lo, hi, n_s + 1),
                np.full(delay, hi),
            ])
            t_action = np.concatenate([
                np.full(delay, lo),
                np.linspace(lo, hi, steps - delay + 1),
            ])
            assert t_state.shape == (steps + 1,)
            assert t_action.shape == (steps + 1,)

        else:
            raise ValueError(f"Unknown joint_t_schedule: {self._t_schedule!r}")

        t_state = self._apply_t_shift(t_state, self._shift_state)
        t_action = self._apply_t_shift(t_action, self._shift_action)
        return t_state, t_action

    # ------------------------------- training -------------------------------

    def update(
        self,
        act: torch.Tensor,
        obs,
        goal_obs,
        delta_t: torch.Tensor,
        optimality: torch.Tensor | None = None,
    ) -> dict:
        """Joint flow-matching update with decoupled t_state / t_action.

        Frozen-encoder variant: encoder is run under ``no_grad`` when
        ``joint_freeze_encoder`` is True; the FM target is z-scored by the
        offline goal-stats. No ``target_ln``, no encoder EMA.
        """
        config = self.config.optimization
        device = act.device
        B = act.shape[0]

        # 1. Encode obs + goal. Frozen path uses no_grad; fine-tune path
        #    keeps gradients on the condition side (matches parent semantics).
        if config.joint_freeze_encoder:
            with torch.no_grad():
                z_t = self.encoder(obs, None)              # (B, To, emb_dim)
                z_goal_raw = self.encoder(goal_obs, None)  # (B, 1, emb_dim)
        else:
            z_t = self.encoder(obs, None)
            z_goal_raw = self.encoder(goal_obs, None)

        # 2. State-flow target normalization. When the encoder is fine-tuned,
        #    detach so the FM target is a pure regression target (encoder
        #    receives signal only via the condition path z_t).
        target = self._normalize_goal(z_goal_raw)
        if not config.joint_freeze_encoder:
            target = target.detach()

        # 3. Optimality + CFG dropout.
        if optimality is None:
            optimality = torch.full(
                (B,), LBMDiTJoint.NULL_IDX, dtype=torch.long, device=device,
            )
        else:
            optimality = optimality.to(device=device, dtype=torch.long)
        if self.net.training and self._cfg_dropout_prob > 0:
            drop_mask = torch.rand(B, device=device) < self._cfg_dropout_prob
            optimality = torch.where(
                drop_mask,
                torch.full_like(optimality, LBMDiTJoint.NULL_IDX),
                optimality,
            )

        # 4. Per-stream flow time. When ``joint_decouple_t`` is False, both
        #    streams share one base t and one shift (state's), recovering a
        #    single-t baseline on the DDT trunk for A/B comparison.
        eps = self._t_eps
        lo, hi = eps, 1.0 - eps
        if self._decouple_t:
            t_state_base = self._sample_base_t((B,), device, lo, hi)
            t_action_base = self._sample_base_t((B,), device, lo, hi)
            t_state = self._apply_t_shift(t_state_base, self._shift_state)
            t_action = self._apply_t_shift(t_action_base, self._shift_action)
        else:
            shared_base = self._sample_base_t((B,), device, lo, hi)
            shared = self._apply_t_shift(shared_base, self._shift_state)
            t_state = shared
            t_action = shared

        # 5. Per-stream noise + interpolation, each with its own t.
        s_noise = torch.randn_like(target)
        a_noise = torch.randn_like(act)

        s_t = self.interpolant.calc_It(t_state, s_noise, target)
        s_t_dot = self.interpolant.calc_It_dot(t_state, s_noise, target)
        a_t = self.interpolant.calc_It(t_action, a_noise, act)
        a_t_dot = self.interpolant.calc_It_dot(t_action, a_noise, act)

        # 6. Joint forward — DDT trunk takes the two times directly.
        v_state, v_action, _ = self.net(
            x_state=s_t,
            x_action=a_t,
            s=t_state,
            t=t_action,
            condition=z_t,
            optimality_idx=optimality,
        )

        # 7. Per-stream losses (per-element so weights are interpretable
        #    when obs_dim != act_dim).
        state_loss_unscaled = torch.mean(
            get_norm(v_state - s_t_dot, config.norm_type)
        ) / float(self.net.obs_dim)
        action_loss_unscaled = torch.mean(
            get_norm(v_action - a_t_dot, config.norm_type)
        ) / float(self.net.act_dim)
        loss = (
            self._w_state * state_loss_unscaled
            + self._w_action * action_loss_unscaled
        )

        loss.backward()

        params = list(self.net.parameters())
        if not config.joint_freeze_encoder:
            params += list(self.encoder.parameters())
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

        del delta_t  # carried for interface compat
        return {
            "dp_loss": loss.detach(),
            "state_loss": state_loss_unscaled.detach(),
            "action_loss": action_loss_unscaled.detach(),
            "grad_norm": grad_norm.detach(),
            "target_std": target_std.detach(),
            "target_abs_mean": target_abs_mean.detach(),
            "t_state_mean": t_state.mean().detach(),
            "t_action_mean": t_action.mean().detach(),
        }

    # ------------------------------- sampling ------------------------------

    @torch.no_grad()
    def sample(
        self,
        act_0: torch.Tensor,
        obs,
        num_steps: int = -1,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Joint Euler ODE sampling with the configured (t_state, t_action)
        schedule. Returns the action chunk only.
        """
        net = self.net_ema if use_ema else self.net
        device = act_0.device
        B = act_0.shape[0]
        cfg_scale = self._cfg_scale
        steps = self._num_steps if num_steps < 1 else int(num_steps)

        # Frozen-encoder eval: live encoder, no LN.
        z_t = self.encoder(obs, None)  # (B, To, obs_dim)
        obs_dim = z_t.shape[-1]

        if self._sample_mode == "stochastic":
            x_state = torch.randn(B, 1, obs_dim, device=device)
            x_action = act_0
        else:
            x_state = torch.zeros(B, 1, obs_dim, device=device)
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

            v_s_cond, v_a_cond, _ = net(
                x_state, x_action, t_state_b, t_action_b, z_t, expert_idx,
            )
            if cfg_scale > 0:
                v_s_un, v_a_un, _ = net(
                    x_state, x_action, t_state_b, t_action_b, z_t, null_idx,
                )
                v_s = (1 + cfg_scale) * v_s_cond - cfg_scale * v_s_un
                v_a = (1 + cfg_scale) * v_a_cond - cfg_scale * v_a_un
            else:
                v_s, v_a = v_s_cond, v_a_cond

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
        raw encoder space (denormalized via goal stats if loaded).
        """
        net = self.net_ema if use_ema else self.net
        device = act_0.device
        B = act_0.shape[0]
        cfg_scale = self._cfg_scale
        steps = self._num_steps if num_steps < 1 else int(num_steps)

        z_t = self.encoder(obs, None)
        obs_dim = z_t.shape[-1]

        if self._sample_mode == "stochastic":
            x_state = torch.randn(B, 1, obs_dim, device=device)
            x_action = act_0
        else:
            x_state = torch.zeros(B, 1, obs_dim, device=device)
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

            v_s_cond, v_a_cond, _ = net(
                x_state, x_action, t_state_b, t_action_b, z_t, expert_idx,
            )
            if cfg_scale > 0:
                v_s_un, v_a_un, _ = net(
                    x_state, x_action, t_state_b, t_action_b, z_t, null_idx,
                )
                v_s = (1 + cfg_scale) * v_s_cond - cfg_scale * v_s_un
                v_a = (1 + cfg_scale) * v_a_cond - cfg_scale * v_a_un
            else:
                v_s, v_a = v_s_cond, v_a_cond

            x_state = x_state + v_s * ds
            x_action = x_action + v_a * da

        z_goal_pred = self._denormalize_goal(x_state)
        return x_action, z_goal_pred
