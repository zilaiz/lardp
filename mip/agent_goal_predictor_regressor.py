"""Goal Predictor Regressor Agent: deterministic regression head trained
through a frozen IDM.

Same overall plumbing as ``GoalPredictorDiTAgent`` (frozen IDM loaded from
checkpoint, GoalDropoutEncoder autodetect, goal stats z-score, EMA, CFG
support) — just with the flow-matching trunk swapped out for a deterministic
``GoalPredictorRegressor``. Two losses:

  1. L2 loss on the (normalized) expert goal embedding.
  2. Action flow-matching loss through the frozen IDM, with the regressor's
     deterministic ``g_hat`` slotted into the goal token.

No interpolant for goals, no Euler refinement, no ``t_flow`` — the regressor
is a single deterministic forward.
"""

from __future__ import annotations

from copy import deepcopy

import loguru
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from mip.config import Config
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_norm
from mip.network_utils import get_encoder, get_network
from mip.networks.goal_predictor_regressor import GoalPredictorRegressor
from mip.samplers import ode_sampler
from mip.torch_utils import at_least_ndim, report_parameters


class GoalPredictorRegressorEncoderWrapper(BaseEncoder):
    """Wraps frozen encoder + deterministic regressor into a single encoder
    interface for ``ode_sampler``.

    forward(obs):
        z_t = encoder(obs)
        g_hat_norm = regressor(z_t)
        g_hat = denormalize(g_hat_norm)
        return concat([z_t, g_hat], dim=1)
    """

    def __init__(
        self,
        encoder: nn.Module,
        regressor: GoalPredictorRegressor,
        goal_mean: torch.Tensor | None,
        goal_var: torch.Tensor | None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = encoder
        self.regressor = regressor
        self.register_buffer("goal_mean", goal_mean)
        self.register_buffer("goal_var", goal_var)
        self.norm_eps = norm_eps

    def _denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        if self.goal_mean is None:
            return z_norm
        return z_norm * torch.sqrt(self.goal_var + self.norm_eps) + self.goal_mean

    def forward(self, obs, mask=None):
        z_t = self.encoder(obs, mask)         # (B, To, emb_dim)
        g_hat_norm = self.regressor(z_t)      # (B, 1, emb_dim) in normalized space
        g_hat = self._denormalize(g_hat_norm)  # (B, 1, emb_dim) in raw encoder space
        return torch.cat([z_t, g_hat], dim=1)  # (B, To+1, emb_dim)


class GoalPredictorRegressorAgent:
    """Deterministic-regressor goal predictor through a frozen pretrained IDM."""

    def __init__(self, config: Config):
        self.config = config
        device = config.optimization.device

        # --- Instantiate IDM architecture (encoder + flow_map) ---
        net = get_network(config.network, config.task)
        self.flow_map = FlowMap(net).to(device)
        self.encoder = get_encoder(config.network, config.task).to(device)

        # --- Load pretrained IDM checkpoint ---
        idm_path = config.optimization.idm_checkpoint_path
        if idm_path is None:
            raise ValueError(
                "idm_checkpoint_path must be set for GoalPredictorRegressorAgent"
            )
        loguru.logger.info(f"Loading pretrained IDM from {idm_path}")
        state_dict = torch.load(idm_path, map_location=device, weights_only=False)
        self.flow_map.load_state_dict(state_dict["flow_map"])

        # Infer enc_out_dim from IDM checkpoint (handles GoalDropoutEncoder's uncond_emb).
        encoder_sd = state_dict["encoder"]
        if "uncond_emb" in encoder_sd:
            enc_out_dim = encoder_sd["uncond_emb"].shape[0]
            loguru.logger.info(f"Inferred enc_out_dim={enc_out_dim} from IDM uncond_emb")
        else:
            enc_out_dim = config.network.encoder_out_dim or config.network.emb_dim
            loguru.logger.info(
                f"No uncond_emb in checkpoint, using config enc_out_dim={enc_out_dim}"
            )

        # Handle IDM checkpoints with GoalDropoutEncoder wrapper.
        has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
        if has_goal_dropout:
            from mip.encoders import GoalDropoutEncoder

            self.encoder = GoalDropoutEncoder(
                self.encoder, enc_out_dim, config.task.obs_steps
            ).to(device)
            loguru.logger.info("IDM checkpoint has GoalDropoutEncoder, wrapping encoder")
        self.encoder.load_state_dict(encoder_sd)
        loguru.logger.info("Pretrained IDM loaded successfully")

        # --- Freeze IDM ---
        # Note: requires_grad_(False) freezes the WEIGHTS. The .train()/.eval()
        # mode flag is independent and controls augmentation behavior inside
        # MultiImageObsEncoder's CropRandomizer (random crop in train mode,
        # center crop in eval mode). We keep the encoder in train mode so that
        # random crops continue to fire during regressor training — matching how
        # the IDM was trained — and only switch to eval mode for inference (see
        # the eval() method below). flow_map has no augmentors, so eval() is fine.
        self.encoder.requires_grad_(False)
        self.flow_map.requires_grad_(False)
        self.flow_map.eval()

        # --- Get inner encoder (bypass GoalDropoutEncoder if present) ---
        from mip.encoders import GoalDropoutEncoder

        if isinstance(self.encoder, GoalDropoutEncoder):
            self._inner_encoder = self.encoder.encoder
            self._uncond_emb = self.encoder.uncond_emb
        else:
            self._inner_encoder = self.encoder
            self._uncond_emb = None

        # If loaded a v2 IDM, the regressor must feed it To_obs obs frames so
        # the IDM's internal obs_summarizer doesn't shape-mismatch.
        idm_net = self.flow_map.net
        if hasattr(idm_net, "To_obs") and config.task.obs_steps != idm_net.To_obs:
            raise ValueError(
                f"task.obs_steps ({config.task.obs_steps}) must match the IDM's "
                f"To_obs ({idm_net.To_obs}); the v2 IDM's obs_summarizer expects "
                f"exactly To_obs frames."
            )

        # --- Load precomputed goal normalization stats ---
        self._norm_eps = 1e-5
        stats_path = config.optimization.goal_stats_path
        if stats_path is not None:
            loguru.logger.info(f"Loading goal normalization stats from {stats_path}")
            stats = torch.load(stats_path, map_location=device, weights_only=False)
            self._goal_mean = stats["mean"].to(device)  # (emb_dim,)
            self._goal_var = stats["var"].to(device)    # (emb_dim,)
            loguru.logger.info(
                f"Goal stats loaded: mean norm={self._goal_mean.norm():.4f}, "
                f"var mean={self._goal_var.mean():.4f}"
            )
        else:
            loguru.logger.warning(
                "No goal_stats_path set — regressor will train in raw encoder space"
            )
            self._goal_mean = None
            self._goal_var = None

        # --- Loss scales (per-dim normalization keeps weights comparable across runs) ---
        self._goal_l2_scale = config.optimization.goal_l2_weight / enc_out_dim
        self._action_reg_scale = (
            config.optimization.action_reg_weight / config.task.act_dim
        )
        self._goal_l2_target_space = config.optimization.goal_l2_target_space
        if self._goal_l2_target_space not in ("normalized", "raw"):
            raise ValueError(
                f"goal_l2_target_space must be 'normalized' or 'raw', "
                f"got {self._goal_l2_target_space}"
            )
        if self._goal_l2_target_space == "normalized" and self._goal_mean is None:
            loguru.logger.warning(
                "goal_l2_target_space='normalized' but no goal stats — falling back to 'raw'"
            )
            self._goal_l2_target_space = "raw"

        # --- Create deterministic regressor ---
        d_model = config.network.regressor_d_model or enc_out_dim
        self.regressor = GoalPredictorRegressor(
            act_dim=enc_out_dim,
            Ta=1,
            obs_dim=enc_out_dim,
            To=config.task.obs_steps,
            d_model=d_model,
            n_heads=config.network.regressor_n_heads,
            depth=config.network.regressor_depth,
            dropout=config.network.regressor_dropout,
            mlp_ratio=config.network.regressor_mlp_ratio,
        ).to(device)
        report_parameters(self.regressor, model_name="Goal Predictor Regressor")

        # --- EMA twin ---
        self.regressor_ema = deepcopy(self.regressor).requires_grad_(False)

        # --- Encoder wrappers for sampling (main + ema) ---
        self.wrapper_encoder = GoalPredictorRegressorEncoderWrapper(
            self._inner_encoder,
            self.regressor,
            self._goal_mean,
            self._goal_var,
            self._norm_eps,
        )
        self.wrapper_encoder_ema = GoalPredictorRegressorEncoderWrapper(
            self._inner_encoder,
            self.regressor_ema,
            self._goal_mean,
            self._goal_var,
            self._norm_eps,
        )

        # --- Interpolant for action flow loss ---
        self.interpolant = Interpolant(config.optimization.interp_type)

        # --- Optimizer (only regressor params) ---
        self.optimizer = torch.optim.AdamW(
            self.regressor.parameters(),
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        if self._goal_mean is None:
            return z
        return (z - self._goal_mean) / torch.sqrt(self._goal_var + self._norm_eps)

    def _denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        if self._goal_mean is None:
            return z_norm
        return z_norm * torch.sqrt(self._goal_var + self._norm_eps) + self._goal_mean

    def update(
        self,
        act: torch.Tensor,
        obs,
        goal_obs,
        delta_t: torch.Tensor,
    ) -> dict:
        """Training step.

        Args:
            act: (B, horizon, act_dim) expert actions.
            obs: current observations (B, To, ...).
            goal_obs: goal observations (B, 1, ...).
            delta_t: (B,) — unused (kept for interface compat with the DiT agent).
        """
        del delta_t  # not used here; the regressor has no time variable

        config = self.config.optimization

        # 1. Encode current obs and goal obs with frozen encoder.
        with torch.no_grad():
            z_t = self._inner_encoder(obs, None)            # (B, To, emb_dim)
            z_goal_raw = self._inner_encoder(goal_obs, None)  # (B, 1, emb_dim)
        z_goal_norm = self._normalize(z_goal_raw)

        # 2. Deterministic regressor forward (in normalized space).
        g_hat_norm = self.regressor(z_t)                    # (B, 1, emb_dim)

        # 3. L2 loss against the expert goal embedding.
        # `goal_l2_loss_unscaled` is the per-element MSE (comparable across runs).
        # The scale `goal_l2_weight / enc_out_dim` keeps the loss-on-loss-magnitude
        # invariant to embedding dimension — matches the FM agent's convention.
        if self._goal_l2_target_space == "normalized":
            l2_diff = g_hat_norm - z_goal_norm
        else:
            g_hat_raw = self._denormalize(g_hat_norm)
            l2_diff = g_hat_raw - z_goal_raw
        goal_l2_loss_unscaled = l2_diff.pow(2).mean()
        goal_l2_loss = self._goal_l2_scale * l2_diff.pow(2).sum(dim=-1).mean()
        # ↑ `_goal_l2_scale * sum_over_dim(...) == goal_l2_weight * mean_per_elem`
        # so the unscaled form (per-element MSE) and the scaled form differ only
        # by the user-set goal_l2_weight, regardless of enc_out_dim.

        # 4. Action flow matching loss through frozen IDM.
        g_hat = self._denormalize(g_hat_norm)               # raw encoder space
        obs_emb = torch.cat([z_t, g_hat], dim=1)            # (B, To+1, emb_dim)

        B = act.shape[0]
        t_act = torch.empty(B, device=act.device).uniform_(0, 1)
        act_0 = torch.randn_like(act)
        act_1 = act
        act_t = self.interpolant.calc_It(t_act, act_0, act_1)
        act_t_dot = self.interpolant.calc_It_dot(t_act, act_0, act_1)

        def _compute_action_loss():
            b_t = self.flow_map.get_velocity(t_act, act_t, obs_emb)
            return torch.mean(get_norm(b_t - act_t_dot, config.norm_type))

        if self._action_reg_scale > 0:
            action_reg_loss_unscaled = _compute_action_loss()
            action_reg_loss = self._action_reg_scale * action_reg_loss_unscaled
        else:
            with torch.no_grad():
                action_reg_loss_unscaled = _compute_action_loss()
            action_reg_loss = torch.zeros((), device=goal_l2_loss.device)

        # 5. Total loss + backward.
        loss = goal_l2_loss + action_reg_loss
        loss.backward()

        if config.grad_clip_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.regressor.parameters(), config.grad_clip_norm
            )
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if config.ema_rate < 1:
            self._ema_update()

        # Diagnostic: predicted vs target embedding norms (helpful to catch collapse).
        with torch.no_grad():
            g_hat_norm_l2 = g_hat_norm.detach().pow(2).sum(dim=-1).sqrt().mean()
            z_goal_norm_l2 = z_goal_norm.detach().pow(2).sum(dim=-1).sqrt().mean()

        return {
            "dp_loss": loss.detach(),
            "goal_l2_loss": goal_l2_loss.detach(),
            "goal_l2_loss_unscaled": goal_l2_loss_unscaled.detach(),
            "action_reg_loss": action_reg_loss.detach(),
            "action_reg_loss_unscaled": action_reg_loss_unscaled.detach(),
            "g_hat_norm_l2": g_hat_norm_l2,
            "z_goal_norm_l2": z_goal_norm_l2,
            "grad_norm": grad_norm.detach(),
        }

    def _ema_update(self):
        ema_rate = self.config.optimization.ema_rate
        with torch.no_grad():
            for p, p_ema in zip(
                self.regressor.parameters(),
                self.regressor_ema.parameters(),
                strict=False,
            ):
                p_ema.data.mul_(ema_rate).add_(p.data, alpha=1.0 - ema_rate)

    def sample(
        self,
        act_0: torch.Tensor,
        obs,
        num_steps: int = -1,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Encode obs -> regressor goal -> run IDM action sampler.

        Mirrors ``GoalPredictorDiTAgent.sample`` exactly except the goal-ODE
        is replaced by a single regressor forward inside the wrapper encoder.
        """
        if num_steps >= 1:
            config = deepcopy(self.config.optimization)
            config.num_steps = int(num_steps)
        else:
            config = self.config.optimization

        with torch.no_grad():
            if config.cfg_scale != 1.0 and self._uncond_emb is not None:
                act = self._cfg_sample(config, act_0, obs, use_ema)
            else:
                wrapper = (
                    self.wrapper_encoder_ema if use_ema else self.wrapper_encoder
                )
                act = ode_sampler(config, self.flow_map, wrapper, act_0, obs)
        return act

    def _cfg_sample(
        self,
        config,
        act_0: torch.Tensor,
        obs,
        use_ema: bool,
    ) -> torch.Tensor:
        """CFG sampling: blend uncond and cond IDM velocities.

        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        """
        regressor = self.regressor_ema if use_ema else self.regressor

        z_t = self._inner_encoder(obs, None)                 # (B, To, emb_dim)
        B = z_t.shape[0]
        device = z_t.device

        g_hat_norm = regressor(z_t)                          # (B, 1, emb_dim)
        g_hat = self._denormalize(g_hat_norm)                # (B, 1, emb_dim)

        cond_emb = torch.cat([z_t, g_hat], dim=1)            # (B, To+1, emb_dim)
        uncond = self._uncond_emb.expand(B, 1, -1)
        uncond_emb = torch.cat([z_t, uncond], dim=1)         # (B, To+1, emb_dim)

        num_steps = config.num_steps
        t_schedule = np.linspace(0, 1, num_steps + 1)
        if config.sample_mode == "stochastic":
            act_s = torch.randn_like(act_0, device=device)
        else:
            act_s = torch.zeros_like(act_0, device=device)

        for i in range(num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((B,), s_val, device=device)
            t = torch.full((B,), t_val, device=device)

            v_uncond = self.flow_map.get_velocity(s, act_s, uncond_emb)
            v_cond = self.flow_map.get_velocity(s, act_s, cond_emb)
            v_cfg = v_uncond + config.cfg_scale * (v_cond - v_uncond)

            s_expanded = at_least_ndim(s, act_s.dim())
            t_expanded = at_least_ndim(t, act_s.dim())
            act_s = act_s + v_cfg * (t_expanded - s_expanded)

        return act_s

    def save(self, path: str, training_state: dict = None):
        checkpoint = {
            "regressor": self.regressor.state_dict(),
            "regressor_ema": self.regressor_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "idm_checkpoint_path": self.config.optimization.idm_checkpoint_path,
        }
        if training_state is not None:
            checkpoint["training_state"] = training_state
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False):
        """Load regressor checkpoint.

        Note: the frozen IDM is loaded from idm_checkpoint_path during __init__.
        """
        state_dict = torch.load(
            path, map_location=self.config.optimization.device, weights_only=False
        )
        self.regressor.load_state_dict(state_dict["regressor"])
        self.regressor_ema.load_state_dict(state_dict["regressor_ema"])

        if load_optimizer and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            loguru.logger.info("Loaded optimizer state")

        training_state = state_dict.get("training_state", None)
        if training_state:
            loguru.logger.info(
                f"Loaded training state from step "
                f"{training_state.get('n_gradient_step', 'unknown')}"
            )
        return training_state

    def eval(self):
        """Set regressor and (frozen) encoder to eval mode for inference."""
        self.regressor.eval()
        self.regressor_ema.eval()
        self.encoder.eval()

    def train(self):
        """Set regressor and (frozen) encoder to train mode."""
        self.regressor.train()
        self.encoder.train()
