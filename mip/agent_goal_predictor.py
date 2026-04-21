"""Goal Predictor Agent: trains a goal predictor MLP through a frozen IDM.

The frozen IDM (encoder + flow_map) is loaded from a pretrained checkpoint.
A lightweight MLP predicts goal embeddings from encoded observations.
The MLP is optimized by backpropagating action-matching loss through the
frozen IDM, with an optional state-matching loss.
"""

from __future__ import annotations

from copy import deepcopy

import loguru
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mip.config import Config
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_norm
from mip.network_utils import get_encoder, get_network
from mip.samplers import ode_sampler
from mip.torch_utils import at_least_ndim, report_parameters


class GoalPredictorMLP(nn.Module):
    """MLP that predicts a goal embedding from encoded current observations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        """Args:
            input_dim: To * emb_dim (flattened encoded obs)
            output_dim: emb_dim (single goal frame embedding)
            hidden_dims: list of hidden layer dimensions
            dropout: dropout rate
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Args:
            z_t: (B, To, emb_dim) encoded current observations

        Returns:
            g_hat: (B, 1, emb_dim) predicted goal embedding
        """
        B = z_t.shape[0]
        z_flat = z_t.reshape(B, -1)  # (B, To * emb_dim)
        g = self.net(z_flat)  # (B, emb_dim)
        return g.unsqueeze(1)  # (B, 1, emb_dim)


class GoalPredictorEncoderWrapper(BaseEncoder):
    """Wraps frozen encoder + goal predictor into a single encoder interface.

    This allows reuse of the standard ode_sampler which calls encoder(obs, None).
    Expects the inner encoder (not GoalDropoutEncoder) to be passed in.
    """

    def __init__(self, encoder: nn.Module, goal_predictor: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.goal_predictor = goal_predictor

    def forward(self, obs, mask=None):
        z_t = self.encoder(obs, mask)  # (B, To, emb_dim)
        g_hat = self.goal_predictor(z_t)  # (B, 1, emb_dim)
        return torch.cat([z_t, g_hat], dim=1)  # (B, To+1, emb_dim)


class GoalPredictorAgent:
    """Agent that trains a goal predictor MLP through a frozen pretrained IDM."""

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
            raise ValueError("idm_checkpoint_path must be set for GoalPredictorAgent")
        loguru.logger.info(f"Loading pretrained IDM from {idm_path}")
        state_dict = torch.load(idm_path, map_location=device, weights_only=False)
        self.flow_map.load_state_dict(state_dict["flow_map"])

        # Handle IDM checkpoints with GoalDropoutEncoder wrapper
        encoder_sd = state_dict["encoder"]
        has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
        if has_goal_dropout:
            from mip.encoders import GoalDropoutEncoder

            enc_out_dim = config.network.encoder_out_dim or config.network.emb_dim
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

        # --- Create goal predictor MLP ---
        enc_out_dim = config.network.encoder_out_dim or config.network.emb_dim
        To = config.task.obs_steps
        input_dim = To * enc_out_dim
        output_dim = enc_out_dim
        hidden_dims = config.network.goal_predictor_hidden_dims or [512, 512]
        dropout = config.network.goal_predictor_dropout

        self.goal_predictor = GoalPredictorMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
        report_parameters(self.goal_predictor, model_name="Goal Predictor MLP")

        # --- EMA for goal predictor only ---
        self.goal_predictor_ema = deepcopy(self.goal_predictor).requires_grad_(False)

        # --- Encoder wrappers for sampling (main + ema) ---
        self.wrapper_encoder = GoalPredictorEncoderWrapper(
            self._inner_encoder, self.goal_predictor
        )
        self.wrapper_encoder_ema = GoalPredictorEncoderWrapper(
            self._inner_encoder, self.goal_predictor_ema
        )

        # --- Optimizer (only goal predictor params) ---
        self.optimizer = torch.optim.AdamW(
            self.goal_predictor.parameters(),
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

        # --- Interpolant for flow matching ---
        self.interpolant = Interpolant(config.optimization.interp_type)

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
            goal_obs: goal observations (B, 1, ...) for state-matching loss
            delta_t: (B,) time step differences
        """
        config = self.config.optimization

        # 1. Encode current obs (inner encoder, bypassing GoalDropoutEncoder)
        z_t = self._inner_encoder(obs, None)  # (B, To, emb_dim)

        # 2. Predict goal embedding
        g_hat = self.goal_predictor(z_t)  # (B, 1, emb_dim)

        # 3. Concatenate to form IDM conditioning
        obs_emb = torch.cat([z_t, g_hat], dim=1)  # (B, To+1, emb_dim)

        # 4. Action-matching loss through frozen IDM (flow matching velocity loss)
        t = torch.empty_like(delta_t).uniform_(0, 1)
        act_0 = torch.empty_like(act).normal_(0, 1)
        act_1 = act

        act_t = self.interpolant.calc_It(t, act_0, act_1)
        act_t_dot = self.interpolant.calc_It_dot(t, act_0, act_1)
        b_t = self.flow_map.get_velocity(t, act_t, obs_emb)

        action_loss = config.loss_scale * torch.mean(
            get_norm(b_t - act_t_dot, config.norm_type)
        )

        # 5. State-matching loss (always computed for diagnostics)
        with torch.no_grad():
            z_goal = self._inner_encoder(goal_obs, None)  # (B, 1, emb_dim)
        unscaled_state_loss = F.mse_loss(g_hat, z_goal)
        state_loss = config.state_matching_weight * unscaled_state_loss

        # 6. Total loss and backward
        loss = action_loss + state_loss
        loss.backward()

        # Gradient clipping
        if config.grad_clip_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.goal_predictor.parameters(), config.grad_clip_norm
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
            "action_loss": action_loss.detach(),
            "state_loss": state_loss.detach(),
            "unscaled_state_loss": unscaled_state_loss.detach(),
            "grad_norm": grad_norm.detach(),
        }

    def _ema_update(self):
        """Update EMA of goal predictor."""
        ema_rate = self.config.optimization.ema_rate
        with torch.no_grad():
            for p, p_ema in zip(
                self.goal_predictor.parameters(),
                self.goal_predictor_ema.parameters(),
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
        """Sample actions: encode obs -> predict goal -> concat -> run IDM sampler.

        Uses CFG when cfg_scale != 1.0 and an unconditional embedding is available.

        Args:
            act_0: (B, Ta, act_dim) initial action noise
            obs: current observations (B, To, ...)
            num_steps: ODE solver steps
            use_ema: whether to use EMA goal predictor
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
        goal_pred = self.goal_predictor_ema if use_ema else self.goal_predictor

        # Encode current obs
        z_t = self._inner_encoder(obs, None)  # (B, To, emb_dim)

        # Conditional: obs + predicted goal
        g_hat = goal_pred(z_t)  # (B, 1, emb_dim)
        cond_emb = torch.cat([z_t, g_hat], dim=1)  # (B, To+1, emb_dim)

        # Unconditional: obs + learned uncond token
        uncond = self._uncond_emb.expand(z_t.shape[0], 1, -1)  # (B, 1, emb_dim)
        uncond_emb = torch.cat([z_t, uncond], dim=1)  # (B, To+1, emb_dim)

        # ODE integration with CFG
        num_steps = config.num_steps
        t_schedule = np.linspace(0, 1, num_steps + 1)

        if config.sample_mode == "stochastic":
            act_s = torch.randn_like(act_0, device=act_0.device)
        else:
            act_s = torch.zeros_like(act_0, device=act_0.device)

        bs = act_0.shape[0]
        for i in range(num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((bs,), s_val, device=act_0.device)
            t = torch.full((bs,), t_val, device=act_0.device)

            v_uncond = self.flow_map.get_velocity(s, act_s, uncond_emb)
            v_cond = self.flow_map.get_velocity(s, act_s, cond_emb)
            v_cfg = v_uncond + config.cfg_scale * (v_cond - v_uncond)

            s_expanded = at_least_ndim(s, act_s.dim())
            t_expanded = at_least_ndim(t, act_s.dim())
            act_s = act_s + v_cfg * (t_expanded - s_expanded)

        return act_s

    def save(self, path: str, training_state: dict = None):
        """Save goal predictor checkpoint."""
        checkpoint = {
            "goal_predictor": self.goal_predictor.state_dict(),
            "goal_predictor_ema": self.goal_predictor_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "idm_checkpoint_path": self.config.optimization.idm_checkpoint_path,
        }
        if training_state is not None:
            checkpoint["training_state"] = training_state
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False):
        """Load goal predictor checkpoint.

        Note: the frozen IDM is loaded from idm_checkpoint_path during __init__.
        """
        state_dict = torch.load(
            path, map_location=self.config.optimization.device, weights_only=False
        )
        self.goal_predictor.load_state_dict(state_dict["goal_predictor"])
        self.goal_predictor_ema.load_state_dict(state_dict["goal_predictor_ema"])

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
        """Set goal predictor to eval mode. IDM is always in eval."""
        self.goal_predictor.eval()
        self.goal_predictor_ema.eval()

    def train(self):
        """Set goal predictor to train mode. IDM stays frozen/eval."""
        self.goal_predictor.train()
