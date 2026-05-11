"""GoalPredictorDDTNSAgent: DDT trunk + noise-shift goal predictor.

Combines two upgrades over the legacy ``GoalPredictorDDTAgent``:

  * **DDT trunk** (``GoalPredictorDiTDDTNS``): encoder/decoder width split,
    single-t embedding, decoupled ``timestep_emb_dim``, AdaLN-Zero output.
    Same v3 functionality as ``GoalPredictorDiTAgent``: state-flow loss,
    action-reg loss with K-step Euler refinement, CFG sampling, encoder
    mode toggling for CropRandomizer.
  * **Noise-shift** (SD3-style): training draws ``t_flow`` from a configurable
    base distribution (``goal_t_dist``: uniform | logit_normal) clamped to
    ``[goal_t_eps, 1 - goal_t_eps]`` and warps it through the SD3 shift
    ``t' = a*t / (1 + (a-1)*t)`` with ``a = goal_t_shift`` (1.0 = identity).
    Inference walks the same shift-warped grid so the Euler steps land at
    the t-distribution the trunk was trained on.

This agent is implemented as a subclass of ``GoalPredictorDiTAgent`` that
overrides only the network construction in ``__init__`` and the t-sampling
in ``update`` / ``_cfg_sample`` / the encoder wrapper. All other v3 logic
(``_refine_to_one``, action-reg loss, EMA, save/load, eval/train) is
inherited unchanged.

Author: Zilai Zeng
"""

from __future__ import annotations

from copy import deepcopy

import loguru
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from mip.agent_goal_predictor_dit import (
    GoalPredictorDiTAgent,
    GoalPredictorDiTEncoderWrapper,
)
from mip.config import Config
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_norm
from mip.network_utils import get_encoder, get_network
from mip.networks.goal_predictor_dit import GoalPredictorDiTDDTNS
from mip.torch_utils import at_least_ndim, report_parameters


# -------------------------- noise-shift helpers --------------------------

def _apply_t_shift(t, alpha: float):
    """SD3-style time shift t' = a*t / (1 + (a-1)*t). Identity at a=1.

    Works on torch tensors and numpy arrays alike (only element-wise math).
    """
    if alpha == 1.0:
        return t
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def _sample_base_t(
    shape: tuple,
    device: torch.device,
    lo: float,
    hi: float,
    t_dist: str,
    mu: float,
    sigma: float,
) -> torch.Tensor:
    """Sample base flow time before per-stream shift is applied."""
    if t_dist == "uniform":
        return torch.empty(shape, device=device).uniform_(lo, hi)
    if t_dist == "logit_normal":
        z = torch.randn(shape, device=device) * sigma + mu
        return torch.sigmoid(z).clamp(lo, hi)
    raise ValueError(
        f"goal_t_dist must be 'uniform' or 'logit_normal'; got {t_dist!r}"
    )


# ---------------------------- wrapper encoder ----------------------------

class GoalPredictorDDTNSEncoderWrapper(GoalPredictorDiTEncoderWrapper):
    """Encoder wrapper whose goal-ODE walk uses the shift-warped t-grid.

    Identical to ``GoalPredictorDiTEncoderWrapper`` except the inference t
    schedule is ``apply_shift(linspace(eps, 1-eps, K+1))`` instead of
    ``linspace(0, 1, K+1)``. Reduces to the parent's behavior when
    ``t_shift == 1.0`` and ``t_eps == 0.0``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        goal_flow_map: FlowMap,
        goal_num_steps: int,
        goal_mean: torch.Tensor | None,
        goal_var: torch.Tensor | None,
        norm_eps: float = 1e-5,
        sample_mode: str = "stochastic",
        t_shift: float = 1.0,
        t_eps: float = 0.0,
    ):
        super().__init__(
            encoder=encoder,
            goal_flow_map=goal_flow_map,
            goal_num_steps=goal_num_steps,
            goal_mean=goal_mean,
            goal_var=goal_var,
            norm_eps=norm_eps,
            sample_mode=sample_mode,
        )
        self.t_shift = float(t_shift)
        self.t_eps = float(t_eps)

    def forward(self, obs, mask=None):
        z_t = self.encoder(obs, mask)  # (B, To, emb_dim)
        B, _, emb_dim = z_t.shape
        device = z_t.device

        if self.sample_mode == "stochastic":
            g_s = torch.randn(B, 1, emb_dim, device=device)
        else:
            g_s = torch.zeros(B, 1, emb_dim, device=device)

        eps = self.t_eps
        t_grid = _apply_t_shift(
            np.linspace(eps, 1.0 - eps, self.goal_num_steps + 1), self.t_shift,
        )

        for i in range(self.goal_num_steps):
            s_val = float(t_grid[i])
            t_val = float(t_grid[i + 1])
            s = torch.full((B,), s_val, device=device)
            v = self.goal_flow_map.get_velocity(s, g_s, z_t)
            g_s = g_s + v * (t_val - s_val)

        g_hat = self._denormalize(g_s)  # (B, 1, emb_dim) raw space
        return torch.cat([z_t, g_hat], dim=1)


# -------------------------------- agent ----------------------------------

class GoalPredictorDDTNSAgent(GoalPredictorDiTAgent):
    """DDT-trunk + noise-shift goal predictor.

    Inherits ``_normalize``, ``_denormalize``, ``_refine_to_one``,
    ``_ema_update``, ``sample``, ``save``, ``load``, ``eval``, ``train``
    from :class:`GoalPredictorDiTAgent`. Overrides only ``__init__``,
    ``update``, and ``_cfg_sample``.
    """

    def __init__(self, config: Config):
        # Skip GoalPredictorDiTAgent.__init__ — it instantiates GoalPredictorDiT.
        # We re-do the IDM-loading + goal-stats path here, then build the DDT-NS
        # trunk. Mirrors the "don't call super().__init__" pattern used in the
        # other DDT/E2E agents.
        self.config = config
        device = config.optimization.device

        # --- IDM (encoder + flow_map) ---
        net = get_network(config.network, config.task)
        self.flow_map = FlowMap(net).to(device)
        self.encoder = get_encoder(config.network, config.task).to(device)

        idm_path = config.optimization.idm_checkpoint_path
        if idm_path is None:
            raise ValueError(
                "idm_checkpoint_path must be set for GoalPredictorDDTNSAgent"
            )
        loguru.logger.info(f"Loading pretrained IDM from {idm_path}")
        state_dict = torch.load(idm_path, map_location=device, weights_only=False)
        self.flow_map.load_state_dict(state_dict["flow_map"])

        encoder_sd = state_dict["encoder"]
        if "uncond_emb" in encoder_sd:
            enc_out_dim = encoder_sd["uncond_emb"].shape[0]
            loguru.logger.info(
                f"Inferred enc_out_dim={enc_out_dim} from IDM uncond_emb"
            )
        else:
            enc_out_dim = config.network.encoder_out_dim or config.network.emb_dim
            loguru.logger.info(
                f"No uncond_emb in checkpoint; using config enc_out_dim={enc_out_dim}"
            )

        has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
        if has_goal_dropout:
            from mip.encoders import GoalDropoutEncoder

            self.encoder = GoalDropoutEncoder(
                self.encoder, enc_out_dim, config.task.obs_steps
            ).to(device)
            loguru.logger.info(
                "IDM checkpoint has GoalDropoutEncoder; wrapping encoder"
            )
        self.encoder.load_state_dict(encoder_sd)
        loguru.logger.info("Pretrained IDM loaded successfully")

        # Freeze IDM (weights). Encoder mode is toggled by eval()/train().
        self.encoder.requires_grad_(False)
        self.flow_map.requires_grad_(False)
        self.flow_map.eval()

        # Inner encoder + uncond_emb (for CFG)
        from mip.encoders import GoalDropoutEncoder

        if isinstance(self.encoder, GoalDropoutEncoder):
            self._inner_encoder = self.encoder.encoder
            self._uncond_emb = self.encoder.uncond_emb
        else:
            self._inner_encoder = self.encoder
            self._uncond_emb = None

        # v2 IDM To_obs sanity check
        idm_net = self.flow_map.net
        if hasattr(idm_net, "To_obs") and config.task.obs_steps != idm_net.To_obs:
            raise ValueError(
                f"task.obs_steps ({config.task.obs_steps}) must match the IDM's "
                f"To_obs ({idm_net.To_obs}); the v2 IDM's obs_summarizer expects "
                f"exactly To_obs frames."
            )

        # --- Goal stats ---
        self._norm_eps = 1e-5
        stats_path = config.optimization.goal_stats_path
        if stats_path is not None:
            loguru.logger.info(f"Loading goal normalization stats from {stats_path}")
            stats = torch.load(stats_path, map_location=device, weights_only=False)
            self._goal_mean = stats["mean"].to(device)
            self._goal_var = stats["var"].to(device)
            loguru.logger.info(
                f"Goal stats loaded: mean norm={self._goal_mean.norm():.4f}, "
                f"var mean={self._goal_var.mean():.4f}"
            )
        else:
            loguru.logger.warning(
                "No goal_stats_path set — skipping z-score normalization"
            )
            self._goal_mean = None
            self._goal_var = None

        # --- DDT-NS trunk ---
        self._goal_loss_scale = config.optimization.goal_flow_loss_scale / enc_out_dim
        self._action_reg_scale = config.optimization.action_reg_weight / config.task.act_dim
        self._action_reg_num_steps = config.optimization.action_reg_num_steps

        d_model_enc = config.network.goal_ddt_d_model_enc or enc_out_dim
        d_model_dec = config.network.goal_ddt_d_model_dec or d_model_enc

        self.goal_dit = GoalPredictorDiTDDTNS(
            act_dim=enc_out_dim,
            Ta=1,
            obs_dim=enc_out_dim,
            To=config.task.obs_steps,
            d_model_enc=d_model_enc,
            d_model_dec=d_model_dec,
            n_heads_enc=config.network.goal_ddt_n_heads_enc,
            n_heads_dec=config.network.goal_ddt_n_heads_dec,
            enc_depth=config.network.goal_ddt_enc_depth,
            dec_depth=config.network.goal_ddt_dec_depth,
            dropout=config.network.goal_ddt_dropout,
            timestep_emb_type=config.network.timestep_emb_type,
            timestep_emb_dim=config.network.goal_ddt_timestep_emb_dim,
        ).to(device)
        report_parameters(self.goal_dit, model_name="Goal Predictor DDT-NS")

        self.goal_flow_map = FlowMap(self.goal_dit).to(device)
        self.goal_dit_ema = deepcopy(self.goal_dit).requires_grad_(False)
        self.goal_flow_map_ema = FlowMap(self.goal_dit_ema).to(device)

        # Goal-ODE source mode
        self._goal_sample_mode = (
            config.optimization.goal_sample_mode
            if config.optimization.goal_sample_mode is not None
            else config.optimization.sample_mode
        )
        loguru.logger.info(
            f"Goal sample_mode: {self._goal_sample_mode} "
            f"(IDM action sample_mode: {config.optimization.sample_mode})"
        )

        # --- Noise-shift scalars ---
        self._t_shift = float(config.optimization.goal_t_shift)
        self._t_dist = config.optimization.goal_t_dist
        self._t_dist_mu = float(config.optimization.goal_t_dist_mu)
        self._t_dist_sigma = float(config.optimization.goal_t_dist_sigma)
        self._t_eps = float(config.optimization.goal_t_eps)
        if self._t_dist not in ("uniform", "logit_normal"):
            raise ValueError(
                f"goal_t_dist must be 'uniform' or 'logit_normal'; "
                f"got {self._t_dist!r}"
            )
        loguru.logger.info(
            f"Noise-shift: t_shift={self._t_shift}, t_dist={self._t_dist}, "
            f"mu={self._t_dist_mu}, sigma={self._t_dist_sigma}, eps={self._t_eps}"
        )

        # --- Encoder wrappers (live + EMA), shift-aware ---
        self.wrapper_encoder = GoalPredictorDDTNSEncoderWrapper(
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
        self.wrapper_encoder_ema = GoalPredictorDDTNSEncoderWrapper(
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

        # --- Interpolants ---
        self.interpolant = Interpolant(config.optimization.interp_type)
        self.goal_interpolant = Interpolant(config.optimization.interp_type)

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            self.goal_dit.parameters(),
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

    # ------------------------------- training -------------------------------

    def update(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict | TensorDict,
        goal_obs: torch.Tensor | dict | TensorDict,
        delta_t: torch.Tensor,
    ) -> dict:
        """Joint state-flow + (optional) action-reg update with noise-shifted t.

        Mirrors :meth:`GoalPredictorDiTAgent.update` but draws ``t_flow`` from
        the configured base distribution and warps it through the SD3 shift.
        """
        config = self.config.optimization

        # 1. Encode obs + goal under no_grad (encoder is frozen).
        with torch.no_grad():
            z_t = self._inner_encoder(obs, None)              # (B, To, emb_dim)
            z_goal_raw = self._inner_encoder(goal_obs, None)  # (B, 1, emb_dim)

        z_goal = self._normalize(z_goal_raw)
        B = z_goal.shape[0]
        device = z_goal.device

        # 2. Noise-shifted t sampling.
        eps = self._t_eps
        lo, hi = eps, 1.0 - eps
        t_base = _sample_base_t(
            (B,), device, lo, hi, self._t_dist, self._t_dist_mu, self._t_dist_sigma,
        )
        t_flow = _apply_t_shift(t_base, self._t_shift)

        # 3. State flow loss (velocity matching in normalized goal space).
        x0 = torch.randn_like(z_goal)
        x1 = z_goal
        x_t = self.goal_interpolant.calc_It(t_flow, x0, x1)
        x_t_dot = self.goal_interpolant.calc_It_dot(t_flow, x0, x1)

        v_pred, _, _ = self.goal_dit(x_t, t_flow, t_flow, z_t, align_depth=None)

        state_flow_loss = self._goal_loss_scale * torch.mean(
            get_norm(v_pred - x_t_dot, config.norm_type)
        )

        # 4. Action-reg loss via K-step Euler refinement (inherited).
        x_pred_clean = self._refine_to_one(
            x_t=x_t,
            t_flow=t_flow,
            v_pred_first=v_pred,
            z_t=z_t,
            K=self._action_reg_num_steps,
        )
        g_hat = self._denormalize(x_pred_clean)
        obs_emb = torch.cat([z_t, g_hat], dim=1)  # (B, To+1, emb_dim)

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
        }

    # ------------------------------- sampling -------------------------------

    def _cfg_sample(
        self,
        config,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        use_ema: bool,
    ) -> torch.Tensor:
        """CFG-blended action sampling with shift-warped goal-ODE schedule."""
        goal_fm = self.goal_flow_map_ema if use_ema else self.goal_flow_map

        z_t = self._inner_encoder(obs, None)  # (B, To, emb_dim)
        B, _, emb_dim = z_t.shape
        device = z_t.device

        # Goal ODE on the shift-warped grid.
        goal_num_steps = config.goal_flow_num_steps
        if self._goal_sample_mode == "stochastic":
            g_s = torch.randn(B, 1, emb_dim, device=device)
        else:
            g_s = torch.zeros(B, 1, emb_dim, device=device)

        eps = self._t_eps
        goal_t_grid = _apply_t_shift(
            np.linspace(eps, 1.0 - eps, goal_num_steps + 1), self._t_shift,
        )

        for i in range(goal_num_steps):
            s_val = float(goal_t_grid[i])
            t_val = float(goal_t_grid[i + 1])
            s = torch.full((B,), s_val, device=device)
            v = goal_fm.get_velocity(s, g_s, z_t)
            g_s = g_s + v * (t_val - s_val)

        g_hat = self._denormalize(g_s)

        cond_emb = torch.cat([z_t, g_hat], dim=1)
        uncond = self._uncond_emb.expand(B, 1, -1)
        uncond_emb = torch.cat([z_t, uncond], dim=1)

        # Action ODE (uniform schedule — IDM was trained without noise-shift).
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
