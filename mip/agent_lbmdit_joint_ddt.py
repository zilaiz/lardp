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
        self.target_ln = nn.LayerNorm(
            obs_dim, elementwise_affine=config.optimization.joint_target_ln_affine,
        ).to(device)

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

        # --- Optimizer (trunk + encoder + LN) ---
        params = (
            list(self.net.parameters())
            + list(self.encoder.parameters())
            + list(self.target_ln.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

        # Cache scalars
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
        self._use_ema_target = bool(config.optimization.joint_use_ema_target)
        self._state_loss_to_encoder = bool(
            config.optimization.joint_state_loss_to_encoder
        )
        # Per-stream SD3-style time shift. 1.0 = identity (no shift).
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
        self, shape: tuple, device: torch.device, lo: float, hi: float,
    ) -> torch.Tensor:
        """Sample base flow time before any per-stream shift is applied.

        ``"uniform"`` draws Uniform[lo, hi]. ``"logit_normal"`` draws
        sigmoid(N(mu, sigma)) and clamps to [lo, hi] (matches SD3/UNITE).
        """
        if self._t_dist == "uniform":
            return torch.empty(shape, device=device).uniform_(lo, hi)
        # logit_normal
        z = torch.randn(shape, device=device) * self._t_dist_sigma + self._t_dist_mu
        t = torch.sigmoid(z)
        return t.clamp(lo, hi)

    # ------------------------------- training -------------------------------

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

        # 1. Condition path always uses the live encoder + LN. Target path
        #    optionally uses the EMA encoder + EMA LN to stabilize the
        #    regression target as the live encoder evolves (self-distillation;
        #    fix for moving-target state_loss creep). Falls back to live
        #    stop-grad when joint_use_ema_target is False or ema_rate >= 1.
        z_t = self.target_ln(self.encoder(obs, None))
        if self._use_ema_target and config.ema_rate < 1:
            with torch.no_grad():
                target = self.target_ln_ema(self.encoder_ema(goal_obs, None))
        else:
            z_goal_ln = self.target_ln(self.encoder(goal_obs, None))
            target = z_goal_ln.detach()

        # 2. Optimality labels with CFG dropout.
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

        # 3. Per-stream flow time. Each stream:
        #    (a) draws a base t from joint_t_dist (uniform or logit_normal),
        #    (b) applies the per-stream SD3-style shift if alpha != 1.
        #    When ``joint_decouple_t`` is False, both streams share *one*
        #    base t and *one* shift (state's), recovering a single-t
        #    baseline on the new trunk for A/B comparison.
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

        # 4. Per-stream noise + interpolation, each with its own t.
        s_noise = torch.randn_like(target)
        a_noise = torch.randn_like(act)

        s_t = self.interpolant.calc_It(t_state, s_noise, target)
        s_t_dot = self.interpolant.calc_It_dot(t_state, s_noise, target)
        a_t = self.interpolant.calc_It(t_action, a_noise, act)
        a_t_dot = self.interpolant.calc_It_dot(t_action, a_noise, act)

        # 5. Joint forward — DDT trunk takes the two times directly.
        #    If state_loss is configured to NOT flow into the encoder, run
        #    two forward passes: one with z_t.detach() to produce v_state
        #    (encoder receives no state_loss grad), one with live z_t to
        #    produce v_action (encoder still receives action_loss grad).
        #    Otherwise a single forward feeds both heads (baseline path).
        if self._state_loss_to_encoder:
            v_state, v_action, _ = self.net(
                x_state=s_t,
                x_action=a_t,
                s=t_state,
                t=t_action,
                condition=z_t,
                optimality_idx=optimality,
            )
        else:
            v_state, _, _ = self.net(
                x_state=s_t,
                x_action=a_t,
                s=t_state,
                t=t_action,
                condition=z_t.detach(),
                optimality_idx=optimality,
            )
            _, v_action, _ = self.net(
                x_state=s_t,
                x_action=a_t,
                s=t_state,
                t=t_action,
                condition=z_t,
                optimality_idx=optimality,
            )

        # 6. Per-stream losses (per-element MSE so weights are interpretable).
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

        return x_action, x_state
