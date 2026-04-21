"""Goal Predictor DiT Agent: trains a DiT-based flow matching goal predictor through a frozen IDM.

The frozen IDM (encoder + flow_map) is loaded from a pretrained checkpoint.
A DiT generates goal embeddings via flow matching in the frozen encoder's
embedding space. Training uses two losses:
  1. State flow loss: velocity matching in standardized goal embedding space
  2. Action regularization loss: intermediate DiT representations fed as goal
     embeddings into the frozen IDM to predict expert actions

Goal embeddings are standardized using precomputed per-element z-score stats
(following RAE). At inference, generated goals are denormalized back to raw
encoder space before feeding into the frozen IDM.
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
from mip.networks.goal_predictor_dit import GoalPredictorDiT
from mip.samplers import ode_sampler
from mip.torch_utils import at_least_ndim, report_parameters


class GoalPredictorDiTEncoderWrapper(BaseEncoder):
    """Wraps frozen encoder + goal DiT flow map into a single encoder interface.

    At inference, this wrapper:
    1. Encodes obs with frozen encoder -> z_t
    2. Runs goal DiT ODE to generate goal embedding in normalized space
    3. Denormalizes goal embedding back to raw encoder space
    4. Concatenates [z_t, g_hat] for the IDM
    """

    def __init__(
        self,
        encoder: nn.Module,
        goal_flow_map: FlowMap,
        goal_num_steps: int,
        goal_mean: torch.Tensor | None,
        goal_var: torch.Tensor | None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = encoder
        self.goal_flow_map = goal_flow_map
        self.goal_num_steps = goal_num_steps
        self.register_buffer("goal_mean", goal_mean)
        self.register_buffer("goal_var", goal_var)
        self.norm_eps = norm_eps

    def _denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        if self.goal_mean is None:
            return z_norm
        return z_norm * torch.sqrt(self.goal_var + self.norm_eps) + self.goal_mean

    def forward(self, obs, mask=None):
        z_t = self.encoder(obs, mask)  # (B, To, emb_dim)
        B, To, emb_dim = z_t.shape
        device = z_t.device

        # ODE integration for goal generation
        g_s = torch.randn(B, 1, emb_dim, device=device)
        t_schedule = np.linspace(0, 1, self.goal_num_steps + 1)

        for i in range(self.goal_num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((B,), s_val, device=device)
            v = self.goal_flow_map.get_velocity(s, g_s, z_t)
            g_s = g_s + v * (t_val - s_val)

        g_hat = self._denormalize(g_s)  # (B, 1, emb_dim) in raw space
        return torch.cat([z_t, g_hat], dim=1)  # (B, To+1, emb_dim)


class GoalPredictorDiTAgent:
    """Agent that trains a DiT-based goal predictor through a frozen pretrained IDM."""

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
            raise ValueError("idm_checkpoint_path must be set for GoalPredictorDiTAgent")
        loguru.logger.info(f"Loading pretrained IDM from {idm_path}")
        state_dict = torch.load(idm_path, map_location=device, weights_only=False)
        self.flow_map.load_state_dict(state_dict["flow_map"])

        # Infer enc_out_dim from IDM checkpoint
        encoder_sd = state_dict["encoder"]
        if "uncond_emb" in encoder_sd:
            enc_out_dim = encoder_sd["uncond_emb"].shape[0]
            loguru.logger.info(f"Inferred enc_out_dim={enc_out_dim} from IDM uncond_emb")
        else:
            enc_out_dim = config.network.encoder_out_dim or config.network.emb_dim
            loguru.logger.info(f"No uncond_emb in checkpoint, using config enc_out_dim={enc_out_dim}")

        # Handle IDM checkpoints with GoalDropoutEncoder wrapper
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
        self.encoder.requires_grad_(False)
        self.flow_map.requires_grad_(False)
        self.encoder.eval()
        self.flow_map.eval()

        # --- Get inner encoder (bypass GoalDropoutEncoder if present) ---
        from mip.encoders import GoalDropoutEncoder

        if isinstance(self.encoder, GoalDropoutEncoder):
            self._inner_encoder = self.encoder.encoder
            self._uncond_emb = self.encoder.uncond_emb
        else:
            self._inner_encoder = self.encoder
            self._uncond_emb = None

        # --- Load precomputed goal normalization stats ---
        self._norm_eps = 1e-5
        stats_path = config.optimization.goal_stats_path
        if stats_path is not None:
            loguru.logger.info(f"Loading goal normalization stats from {stats_path}")
            stats = torch.load(stats_path, map_location=device, weights_only=False)
            self._goal_mean = stats["mean"].to(device)  # (emb_dim,)
            self._goal_var = stats["var"].to(device)  # (emb_dim,)
            loguru.logger.info(
                f"Goal stats loaded: mean norm={self._goal_mean.norm():.4f}, "
                f"var mean={self._goal_var.mean():.4f}"
            )
        else:
            loguru.logger.warning("No goal_stats_path set — skipping z-score normalization")
            self._goal_mean = None
            self._goal_var = None

        # --- Create Goal Predictor DiT ---
        self._goal_loss_scale = config.optimization.goal_flow_loss_scale / enc_out_dim
        self._action_reg_scale = config.optimization.action_reg_weight / config.task.act_dim
        d_model = config.network.goal_dit_d_model or enc_out_dim

        self.goal_dit = GoalPredictorDiT(
            act_dim=enc_out_dim,
            Ta=1,
            obs_dim=enc_out_dim,
            To=config.task.obs_steps,
            d_model=d_model,
            n_heads=config.network.goal_dit_n_heads,
            depth=config.network.goal_dit_depth,
            dropout=config.network.goal_dit_dropout,
            timestep_emb_type=config.network.timestep_emb_type,
            projector_dim=config.network.goal_dit_projector_dim,
        ).to(device)
        report_parameters(self.goal_dit, model_name="Goal Predictor DiT")

        self.goal_flow_map = FlowMap(self.goal_dit).to(device)

        # --- EMA for goal DiT ---
        self.goal_dit_ema = deepcopy(self.goal_dit).requires_grad_(False)
        self.goal_flow_map_ema = FlowMap(self.goal_dit_ema).to(device)

        # --- Encoder wrappers for sampling (main + ema) ---
        self.wrapper_encoder = GoalPredictorDiTEncoderWrapper(
            self._inner_encoder,
            self.goal_flow_map,
            config.optimization.goal_flow_num_steps,
            self._goal_mean,
            self._goal_var,
            self._norm_eps,
        )
        self.wrapper_encoder_ema = GoalPredictorDiTEncoderWrapper(
            self._inner_encoder,
            self.goal_flow_map_ema,
            config.optimization.goal_flow_num_steps,
            self._goal_mean,
            self._goal_var,
            self._norm_eps,
        )

        # --- Interpolants ---
        self.interpolant = Interpolant(config.optimization.interp_type)
        self.goal_interpolant = Interpolant(config.optimization.interp_type)

        # --- Optimizer (only goal DiT params) ---
        self.optimizer = torch.optim.AdamW(
            self.goal_dit.parameters(),
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Z-score normalize goal embeddings."""
        if self._goal_mean is None:
            return z
        return (z - self._goal_mean) / torch.sqrt(self._goal_var + self._norm_eps)

    def _denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Invert z-score normalization."""
        if self._goal_mean is None:
            return z_norm
        return z_norm * torch.sqrt(self._goal_var + self._norm_eps) + self._goal_mean

    def update(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict | TensorDict,
        goal_obs: torch.Tensor | dict | TensorDict,
        delta_t: torch.Tensor,
    ) -> dict:
        """Training step.

        Args:
            act: (B, horizon, act_dim) expert actions
            obs: current observations (B, To, ...)
            goal_obs: goal observations (B, 1, ...) for state flow target
            delta_t: (B,) time step differences (unused, kept for interface compat)
        """
        config = self.config.optimization
        align_depth = self.config.network.goal_align_depth

        # 1. Encode current obs and goal obs with frozen encoder
        with torch.no_grad():
            z_t = self._inner_encoder(obs, None)  # (B, To, emb_dim)
            z_goal_raw = self._inner_encoder(goal_obs, None)  # (B, 1, emb_dim)

        # 2. Normalize goal target
        z_goal = self._normalize(z_goal_raw)  # (B, 1, emb_dim) standardized

        # 3. State flow loss (velocity matching in normalized goal space)
        B = z_goal.shape[0]
        t_flow = torch.empty(B, device=z_goal.device).uniform_(0, 1)
        x0 = torch.randn_like(z_goal)  # noise
        x1 = z_goal  # target

        x_t = self.goal_interpolant.calc_It(t_flow, x0, x1)
        x_t_dot = self.goal_interpolant.calc_It_dot(t_flow, x0, x1)

        # Forward through goal DiT with intermediate extraction
        v_pred, _, zs_tilde = self.goal_dit(x_t, t_flow, t_flow, z_t, align_depth=align_depth)

        state_flow_loss = self._goal_loss_scale * torch.mean(
            get_norm(v_pred - x_t_dot, config.norm_type)
        )

        # 4. Action regularization loss through frozen IDM
        if self._action_reg_scale > 0:
            g_inter = zs_tilde[0]  # (B, 1, d_model) projected intermediate state
            obs_emb = torch.cat([z_t, g_inter], dim=1)  # (B, To+1, emb_dim)

            t_act = torch.empty_like(delta_t).uniform_(0, 1)
            act_0 = torch.randn_like(act)
            act_1 = act

            act_t = self.interpolant.calc_It(t_act, act_0, act_1)
            act_t_dot = self.interpolant.calc_It_dot(t_act, act_0, act_1)
            b_t = self.flow_map.get_velocity(t_act, act_t, obs_emb)

            action_reg_loss_unscaled = torch.mean(
                get_norm(b_t - act_t_dot, config.norm_type)
            )
            action_reg_loss = self._action_reg_scale * action_reg_loss_unscaled
        else:
            action_reg_loss_unscaled = torch.zeros((), device=state_flow_loss.device)
            action_reg_loss = torch.zeros((), device=state_flow_loss.device)

        # 5. Total loss and backward
        loss = state_flow_loss + action_reg_loss
        loss.backward()

        # Gradient clipping
        if config.grad_clip_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.goal_dit.parameters(), config.grad_clip_norm
            )
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # EMA update
        if config.ema_rate < 1:
            self._ema_update()

        return {
            "dp_loss": loss.detach(),
            "state_flow_loss": state_flow_loss.detach(),
            "action_reg_loss": action_reg_loss.detach(),
            "action_reg_loss_unscaled": action_reg_loss_unscaled.detach(),
            "grad_norm": grad_norm.detach(),
        }

    def _ema_update(self):
        """Update EMA of goal DiT."""
        ema_rate = self.config.optimization.ema_rate
        with torch.no_grad():
            for p, p_ema in zip(
                self.goal_dit.parameters(),
                self.goal_dit_ema.parameters(),
                strict=False,
            ):
                p_ema.data.mul_(ema_rate).add_(p.data, alpha=1.0 - ema_rate)

    def sample(
        self,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        num_steps: int = -1,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Sample actions: encode obs -> generate goal via DiT ODE -> run IDM sampler.

        Uses CFG when cfg_scale != 1.0 and an unconditional embedding is available.

        Args:
            act_0: (B, Ta, act_dim) initial action noise
            obs: current observations (B, To, ...)
            num_steps: ODE solver steps for IDM (goal steps from config)
            use_ema: whether to use EMA goal DiT
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
                wrapper = self.wrapper_encoder_ema if use_ema else self.wrapper_encoder
                act = ode_sampler(config, self.flow_map, wrapper, act_0, obs)
        return act

    def _cfg_sample(
        self,
        config,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        use_ema: bool,
    ) -> torch.Tensor:
        """CFG sampling: blend unconditional and conditional IDM velocities.

        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        """
        goal_fm = self.goal_flow_map_ema if use_ema else self.goal_flow_map

        # Encode current obs
        z_t = self._inner_encoder(obs, None)  # (B, To, emb_dim)
        B, To, emb_dim = z_t.shape
        device = z_t.device

        # Generate goal via DiT ODE
        goal_num_steps = config.goal_flow_num_steps
        g_s = torch.randn(B, 1, emb_dim, device=device)
        t_schedule = np.linspace(0, 1, goal_num_steps + 1)

        for i in range(goal_num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((B,), s_val, device=device)
            v = goal_fm.get_velocity(s, g_s, z_t)
            g_s = g_s + v * (t_val - s_val)

        g_hat = self._denormalize(g_s)  # (B, 1, emb_dim)

        # Conditional: obs + predicted goal
        cond_emb = torch.cat([z_t, g_hat], dim=1)  # (B, To+1, emb_dim)

        # Unconditional: obs + learned uncond token
        uncond = self._uncond_emb.expand(B, 1, -1)
        uncond_emb = torch.cat([z_t, uncond], dim=1)  # (B, To+1, emb_dim)

        # ODE integration with CFG
        num_steps = config.num_steps
        act_t_schedule = np.linspace(0, 1, num_steps + 1)

        act_s = torch.randn_like(act_0, device=device) if config.sample_mode == "stochastic" else torch.zeros_like(act_0, device=device)

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

    def save(self, path: str, training_state: dict = None):
        """Save goal predictor DiT checkpoint."""
        checkpoint = {
            "goal_dit": self.goal_dit.state_dict(),
            "goal_dit_ema": self.goal_dit_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "idm_checkpoint_path": self.config.optimization.idm_checkpoint_path,
        }
        if training_state is not None:
            checkpoint["training_state"] = training_state
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False):
        """Load goal predictor DiT checkpoint.

        Note: the frozen IDM is loaded from idm_checkpoint_path during __init__.
        """
        state_dict = torch.load(
            path, map_location=self.config.optimization.device, weights_only=False
        )
        self.goal_dit.load_state_dict(state_dict["goal_dit"])
        self.goal_dit_ema.load_state_dict(state_dict["goal_dit_ema"])

        if load_optimizer and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            loguru.logger.info("Loaded optimizer state")

        training_state = state_dict.get("training_state", None)
        if training_state:
            loguru.logger.info(
                f"Loaded training state from step {training_state.get('n_gradient_step', 'unknown')}"
            )
        return training_state

    def eval(self):
        """Set goal DiT to eval mode. IDM is always in eval."""
        self.goal_dit.eval()
        self.goal_dit_ema.eval()

    def train(self):
        """Set goal DiT to train mode. IDM stays frozen/eval."""
        self.goal_dit.train()
