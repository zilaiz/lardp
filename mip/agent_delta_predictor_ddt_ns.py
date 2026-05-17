"""DeltaPredictorDDTNSAgent: DDT-NS trunk that predicts the goal *delta*.

Same architecture and noise-shift recipe as :class:`GoalPredictorDDTNSAgent`,
but the prediction target is the goal delta

    delta = z_goal − z_last_obs

instead of the absolute goal embedding. The trunk learns flow-matching in
*z-scored delta space* using stats loaded from ``optimization.delta_stats_path``.

Inference / action-reg path: the predicted (denormalized) delta is added
back to the most recent obs embedding to build a "fake absolute goal"
slot, which is then fed to the (frozen) delta-cond IDM exactly as the
goal predictor would feed a real goal:

    g_fake = z_last_obs + delta_hat
    obs_emb_for_idm = cat([z_t, g_fake], dim=1)

The delta-cond IDM's internal ``_summarize`` recovers the delta as
``g_fake − z_last_obs = delta_hat`` — so this slot reaches the action
trunk's AdaLN in exactly the form ``LBMDiTIDMv2Delta`` was trained on.

Pairs with ``LBMDiTIDMv2Delta`` (the delta-cond IDM trunk) and reuses
the goal predictor's training/sampling interface. The ``goal_obs``
argument is kept in the ``update(...)`` signature for compatibility with
the existing GP training loop; internally it's used only to compute the
delta target.
"""

from __future__ import annotations

import loguru
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from mip.agent_goal_predictor_ddt_ns import (
    GoalPredictorDDTNSAgent,
    GoalPredictorDDTNSEncoderWrapper,
    _apply_t_shift,
    _sample_base_t,
)
from mip.config import Config
from mip.losses import get_norm
from mip.torch_utils import at_least_ndim


class DeltaPredictorDDTNSEncoderWrapper(GoalPredictorDDTNSEncoderWrapper):
    """Wrapper encoder for the delta predictor.

    Same goal-ODE walk as the parent, but the (denormalized) ODE output is
    interpreted as a *delta*. The "goal slot" appended to the obs is
    ``z_last_obs + delta_hat`` so the delta-cond IDM's internal subtraction
    recovers ``delta_hat`` in its AdaLN cond vec.
    """

    def forward(self, obs, mask=None):
        z_t = self.encoder(obs, mask)  # (B, To, emb_dim)
        B, _, emb_dim = z_t.shape
        device = z_t.device

        if self.sample_mode == "stochastic":
            x_s = torch.randn(B, 1, emb_dim, device=device)
        else:
            x_s = torch.zeros(B, 1, emb_dim, device=device)

        eps = self.t_eps
        t_grid = _apply_t_shift(
            np.linspace(eps, 1.0 - eps, self.goal_num_steps + 1),
            self.t_shift,
        )
        for i in range(self.goal_num_steps):
            s_val = float(t_grid[i])
            t_val = float(t_grid[i + 1])
            s = torch.full((B,), s_val, device=device)
            v = self.goal_flow_map.get_velocity(s, x_s, z_t)
            x_s = x_s + v * (t_val - s_val)

        delta_hat = self._denormalize(x_s)        # (B, 1, emb_dim) raw delta
        g_fake = z_t[:, -1:].detach() + delta_hat  # (B, 1, emb_dim)
        return torch.cat([z_t, g_fake], dim=1)     # (B, To+1, emb_dim)


class DeltaPredictorDDTNSAgent(GoalPredictorDDTNSAgent):
    """DDT-NS predictor in delta-embedding space.

    Inherits the network construction, EMA, optimizer, save/load, eval/train,
    and ``_refine_to_one`` from :class:`GoalPredictorDDTNSAgent`. Overrides:

      * ``__init__``: loads ``delta_stats_path`` instead of ``goal_stats_path``
        for z-score normalization. Replaces the wrapper encoder with the
        delta variant. Logs an explicit warning if the loaded IDM is *not*
        the delta-cond trunk.
      * ``update``: target = delta. Action-reg path builds the fake-goal
        slot before feeding the IDM.
      * ``_cfg_sample``: same conversion at sampling time.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        device = config.optimization.device

        # --- Replace goal-stats with delta-stats (we already loaded the IDM
        # + base wrapper in the parent ``__init__``; this swap only changes
        # the normalization buffers and rebuilds the wrapper encoders).
        stats_path = config.optimization.delta_stats_path
        if stats_path is None:
            # Fall back to goal_stats only with an explicit warning. Better to
            # leave normalization disabled than to silently mix conventions.
            if config.optimization.goal_stats_path is not None:
                loguru.logger.warning(
                    "delta_stats_path is unset but goal_stats_path is set — "
                    "ignoring goal stats. Use scripts/compute_delta_stats.py "
                    "and pass --output_path to optimization.delta_stats_path."
                )
            loguru.logger.warning(
                "No delta_stats_path set — running without z-score normalization "
                "in delta space."
            )
            self._goal_mean = None
            self._goal_var = None
        else:
            loguru.logger.info(
                f"Loading delta normalization stats from {stats_path}"
            )
            stats = torch.load(stats_path, map_location=device, weights_only=False)
            self._goal_mean = stats["mean"].to(device)  # (emb_dim,)
            self._goal_var = stats["var"].to(device)    # (emb_dim,)
            loguru.logger.info(
                f"Delta stats loaded: mean norm={self._goal_mean.norm():.4f}, "
                f"var mean={self._goal_var.mean():.4f}"
            )

        # Sanity: warn if the IDM trunk doesn't look like the delta-cond one.
        idm_net_name = type(self.flow_map.net).__name__
        if "Delta" not in idm_net_name:
            loguru.logger.warning(
                f"DeltaPredictorDDTNSAgent paired with {idm_net_name}, not a "
                f"delta-cond IDM. The action trunk will read `g_fake - z_last_obs"
                f" = delta_hat` as if it were an absolute goal — actions will "
                f"be off-distribution. Use network=lbmidm_v2_delta (or another "
                f"delta-cond trunk)."
            )

        # Rebuild wrappers so their _denormalize uses the *delta* stats.
        self.wrapper_encoder = DeltaPredictorDDTNSEncoderWrapper(
            self._inner_encoder,
            self.goal_flow_map,
            config.optimization.goal_flow_num_steps,
            self._goal_mean,
            self._goal_var,
            self._norm_eps,
            sample_mode=self._goal_sample_mode,
            t_shift=self._t_shift,
            t_eps=self._t_eps,
        )
        self.wrapper_encoder_ema = DeltaPredictorDDTNSEncoderWrapper(
            self._inner_encoder,
            self.goal_flow_map_ema,
            config.optimization.goal_flow_num_steps,
            self._goal_mean,
            self._goal_var,
            self._norm_eps,
            sample_mode=self._goal_sample_mode,
            t_shift=self._t_shift,
            t_eps=self._t_eps,
        )

    # ------------------------------- training -------------------------------

    def update(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict | TensorDict,
        goal_obs: torch.Tensor | dict | TensorDict,
        delta_t: torch.Tensor,
    ) -> dict:
        """State-flow + action-reg update with delta target and noise-shifted t.

        Mirrors :meth:`GoalPredictorDDTNSAgent.update` but the flow-matching
        target is ``z_goal − z_last_obs`` (z-scored with delta stats), and
        the action-reg path adds ``z_last_obs`` back before feeding the IDM.
        """
        config = self.config.optimization

        # 1. Encode obs + goal under no_grad (encoder is frozen).
        with torch.no_grad():
            z_t = self._inner_encoder(obs, None)              # (B, To, emb_dim)
            z_goal_raw = self._inner_encoder(goal_obs, None)  # (B, 1, emb_dim)

        z_last_obs = z_t[:, -1:].detach()                     # (B, 1, emb_dim)
        delta_raw = z_goal_raw - z_last_obs                   # (B, 1, emb_dim) raw delta
        delta_norm = self._normalize(delta_raw)               # z-scored delta target

        B = delta_norm.shape[0]
        device = delta_norm.device

        # 2. Noise-shifted t sampling (same as goal-predictor variant).
        eps = self._t_eps
        lo, hi = eps, 1.0 - eps
        t_base = _sample_base_t(
            (B,), device, lo, hi, self._t_dist, self._t_dist_mu, self._t_dist_sigma,
        )
        t_flow = _apply_t_shift(t_base, self._t_shift)

        # 3. State flow loss on the delta target.
        x0 = torch.randn_like(delta_norm)
        x1 = delta_norm
        x_t = self.goal_interpolant.calc_It(t_flow, x0, x1)
        x_t_dot = self.goal_interpolant.calc_It_dot(t_flow, x0, x1)

        v_pred, _, _ = self.goal_dit(x_t, t_flow, t_flow, z_t, align_depth=None)
        state_flow_loss = self._goal_loss_scale * torch.mean(
            get_norm(v_pred - x_t_dot, config.norm_type)
        )

        # 4. Action-reg loss via K-step Euler refinement, then fake-goal feed.
        x_pred_clean = self._refine_to_one(
            x_t=x_t,
            t_flow=t_flow,
            v_pred_first=v_pred,
            z_t=z_t,
            K=self._action_reg_num_steps,
        )
        delta_hat = self._denormalize(x_pred_clean)          # raw delta
        g_fake = z_last_obs + delta_hat                       # (B, 1, emb_dim)
        obs_emb = torch.cat([z_t, g_fake], dim=1)             # (B, To+1, emb_dim)

        t_act = torch.empty_like(delta_t).uniform_(0, 1)
        act_0 = torch.randn_like(act)
        act_1 = act
        act_t = self.interpolant.calc_It(t_act, act_0, act_1)
        act_t_dot = self.interpolant.calc_It_dot(t_act, act_0, act_1)

        def _compute_action_losses() -> tuple[torch.Tensor, torch.Tensor]:
            b_t = self.flow_map.get_velocity(t_act, act_t, obs_emb)
            per_elem_loss = get_norm(b_t - act_t_dot, config.norm_type)
            unweighted = per_elem_loss.mean()
            if config.action_reg_t_weighting == "linear":
                weights = t_flow.unsqueeze(-1)
                weighted = (per_elem_loss * weights).mean() / t_flow.mean().clamp(min=1e-6)
            elif config.action_reg_t_weighting == "none":
                weighted = unweighted
            else:
                raise ValueError(
                    f"Unknown action_reg_t_weighting: {config.action_reg_t_weighting}"
                )
            return unweighted, weighted

        if self._action_reg_scale > 0:
            action_reg_loss_unscaled, weighted_unscaled = _compute_action_losses()
            action_reg_loss = self._action_reg_scale * weighted_unscaled
        else:
            with torch.no_grad():
                action_reg_loss_unscaled, _ = _compute_action_losses()
            action_reg_loss = torch.zeros((), device=state_flow_loss.device)

        # 5. Total loss + optimizer.
        loss = state_flow_loss + action_reg_loss
        loss.backward()

        if config.grad_clip_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.goal_dit.parameters(), config.grad_clip_norm
            )
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if config.ema_rate < 1:
            self._ema_update()

        return {
            "dp_loss": loss.detach(),
            "state_flow_loss": state_flow_loss.detach(),
            "action_reg_loss": action_reg_loss.detach(),
            "action_reg_loss_unscaled": action_reg_loss_unscaled.detach(),
            "grad_norm": grad_norm.detach(),
            "t_flow_mean": t_flow.mean().detach(),
            # Useful diagnostic — magnitude of the delta target tells you
            # how concentrated the target distribution is.
            "delta_target_norm_mean": delta_raw.norm(dim=-1).mean().detach(),
        }

    # ------------------------------- sampling -------------------------------

    def _cfg_sample(
        self,
        config,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        use_ema: bool,
    ) -> torch.Tensor:
        """CFG-blended action sampling with delta-space ODE + fake-goal feed."""
        goal_fm = self.goal_flow_map_ema if use_ema else self.goal_flow_map

        z_t = self._inner_encoder(obs, None)         # (B, To, emb_dim)
        B, _, emb_dim = z_t.shape
        device = z_t.device
        z_last_obs = z_t[:, -1:]                     # (B, 1, emb_dim)

        # Delta ODE on the shift-warped grid.
        goal_num_steps = config.goal_flow_num_steps
        if self._goal_sample_mode == "stochastic":
            x_s = torch.randn(B, 1, emb_dim, device=device)
        else:
            x_s = torch.zeros(B, 1, emb_dim, device=device)

        eps = self._t_eps
        goal_t_grid = _apply_t_shift(
            np.linspace(eps, 1.0 - eps, goal_num_steps + 1), self._t_shift,
        )
        for i in range(goal_num_steps):
            s_val = float(goal_t_grid[i])
            t_val = float(goal_t_grid[i + 1])
            s = torch.full((B,), s_val, device=device)
            v = goal_fm.get_velocity(s, x_s, z_t)
            x_s = x_s + v * (t_val - s_val)

        delta_hat = self._denormalize(x_s)           # raw delta
        g_fake = z_last_obs + delta_hat              # fake absolute goal

        cond_emb = torch.cat([z_t, g_fake], dim=1)
        # Uncond: feed the IDM the encoder's learned uncond_emb directly.
        # This is the same convention as the goal-predictor CFG path — the
        # IDM internally computes (uncond_emb - z_last_obs) as the "uncond
        # delta," which matches what the IDM saw during goal_dropout training.
        uncond = self._uncond_emb.expand(B, 1, -1)
        uncond_emb = torch.cat([z_t, uncond], dim=1)

        # Action ODE (uniform schedule — IDM trained without noise-shift).
        num_steps = config.num_steps
        act_t_schedule = np.linspace(0, 1, num_steps + 1)
        act_s = (
            torch.randn_like(act_0, device=device)
            if config.sample_mode == "stochastic"
            else torch.zeros_like(act_0, device=device)
        )
        for i in range(num_steps):
            s_val = act_t_schedule[i]
            t_val = act_t_schedule[i + 1]
            s = torch.full((B,), s_val, device=device)
            t = torch.full((B,), t_val, device=device)

            v_uncond = self.flow_map.get_velocity(s, act_s, uncond_emb)
            v_cond = self.flow_map.get_velocity(s, act_s, cond_emb)
            v_cfg = v_uncond + config.cfg_scale * (v_cond - v_uncond)

            s_expanded = at_least_ndim(s, act_s.dim())
            t_expanded = at_least_ndim(t, act_s.dim())
            act_s = act_s + v_cfg * (t_expanded - s_expanded)

        return act_s
