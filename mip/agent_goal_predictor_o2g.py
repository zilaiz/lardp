"""Goal Predictor O2G Agent: trains an MLP-based velocity field that flows
from the last raw obs encoding directly to the raw goal embedding.

Training is pure flow matching:
  x0 = z_t[:, -1:, :]   (last raw obs frame from the frozen IDM encoder)
  x1 = z_goal_raw       (raw goal frame from the same encoder)
linear interpolant, MSE loss. No latent normalization, no goal-stats, no
action regularization, no FDM auxiliary loss.

Both x0 and x1 pass through the same frozen encoder, so they share per-dim
variance and mean — the flow target ut = x1 - x0 is well-calibrated by
construction without any z-score step.

obs_summarizer is loaded from the frozen IDM and used only as an OPTIONAL
conditioning input when ``o2g_cond_mode == "obs_summary"``.

The frozen IDM is also reused at inference: g_hat is fed to its action
flow map as the goal token in [z_t, g_hat].
"""

from __future__ import annotations

from copy import deepcopy

import loguru
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from mip.config import Config
from mip.encoders import BaseEncoder, GoalDropoutEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_norm
from mip.network_utils import get_encoder, get_network
from mip.networks.goal_predictor_o2g import GoalPredictorO2G
from mip.samplers import ode_sampler
from mip.torch_utils import at_least_ndim, report_parameters


class GoalPredictorO2GEncoderWrapper(BaseEncoder):
    """Wraps frozen encoder + (optional) obs_summarizer + goal flow map into
    a single encoder interface that the IDM ode_sampler can call.

    At inference:
      1. Encode obs with the frozen IDM encoder           -> z_t
      2. Build x0 = z_t[:, -1:, :]                        -> g_s (last raw frame)
      3. Build cond from z_t / obs_summary / None         (per cond_mode)
      4. Integrate Euler ODE through goal_flow_map        -> g_hat
      5. Return [z_t, g_hat] for the frozen IDM
    """

    def __init__(
        self,
        encoder: nn.Module,
        obs_summarizer: nn.Module | None,
        goal_flow_map: FlowMap,
        goal_num_steps: int,
        cond_mode: str = "none",
        To_obs: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.obs_summarizer = obs_summarizer
        self.goal_flow_map = goal_flow_map
        self.goal_num_steps = goal_num_steps
        self.cond_mode = cond_mode
        self.To_obs = To_obs
        if cond_mode == "obs_summary" and obs_summarizer is None:
            raise ValueError(
                "cond_mode='obs_summary' requires an obs_summarizer; got None."
            )

    def _make_x0(self, z_t: torch.Tensor) -> torch.Tensor:
        # z_t: (B, To_obs, obs_dim) -> last frame as (B, 1, obs_dim)
        return z_t[:, -1:, :]

    def _make_cond(self, z_t: torch.Tensor):
        if self.cond_mode == "none":
            return None
        if self.cond_mode == "obs_summary":
            # (B, 1, obs_dim) — produced by the frozen IDM obs_summarizer
            return self.obs_summarizer(z_t.flatten(1)).unsqueeze(1)
        raise ValueError(f"Unknown cond_mode '{self.cond_mode}'")

    def forward(self, obs, mask=None):
        z_t = self.encoder(obs, mask)  # (B, To_obs, obs_dim)
        B = z_t.shape[0]
        device = z_t.device

        g_s = self._make_x0(z_t)
        cond = self._make_cond(z_t)

        t_schedule = np.linspace(0, 1, self.goal_num_steps + 1)
        for i in range(self.goal_num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((B,), s_val, device=device)
            v = self.goal_flow_map.get_velocity(s, g_s, cond)
            g_s = g_s + v * (t_val - s_val)

        g_hat = g_s  # raw encoder space; no denormalize
        return torch.cat([z_t, g_hat], dim=1)  # (B, To_obs + 1, obs_dim)


class GoalPredictorO2GAgent:
    """Agent that trains a residual-MLP goal predictor flowing
    z_t[:, -1, :] -> z_goal_raw through a frozen pretrained IDM.

    Loss: state flow MSE; optional VITA-style consistency loss (full ODE
    integration -> action FM regression through the frozen IDM).
    """

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
                "idm_checkpoint_path must be set for GoalPredictorO2GAgent"
            )
        loguru.logger.info(f"Loading pretrained IDM from {idm_path}")
        state_dict = torch.load(idm_path, map_location=device, weights_only=False)
        self.flow_map.load_state_dict(state_dict["flow_map"])

        encoder_sd = state_dict["encoder"]
        if "uncond_emb" in encoder_sd:
            enc_out_dim = encoder_sd["uncond_emb"].shape[0]
            loguru.logger.info(f"Inferred enc_out_dim={enc_out_dim} from IDM uncond_emb")
        else:
            enc_out_dim = config.network.encoder_out_dim or config.network.emb_dim
            loguru.logger.info(
                f"No uncond_emb in checkpoint, using config enc_out_dim={enc_out_dim}"
            )

        has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
        if has_goal_dropout:
            self.encoder = GoalDropoutEncoder(
                self.encoder, enc_out_dim, config.task.obs_steps
            ).to(device)
            loguru.logger.info(
                "IDM checkpoint has GoalDropoutEncoder, wrapping encoder"
            )
        self.encoder.load_state_dict(encoder_sd)
        loguru.logger.info("Pretrained IDM loaded successfully")

        # --- Freeze IDM (weights). Keep encoder in train mode for CropRandomizer. ---
        self.encoder.requires_grad_(False)
        self.flow_map.requires_grad_(False)
        self.flow_map.eval()

        # --- Get inner encoder (bypass GoalDropoutEncoder if present) ---
        if isinstance(self.encoder, GoalDropoutEncoder):
            self._inner_encoder = self.encoder.encoder
            self._uncond_emb = self.encoder.uncond_emb
        else:
            self._inner_encoder = self.encoder
            self._uncond_emb = None

        # --- Reuse frozen obs_summarizer from the IDM (used as optional cond) ---
        idm_net = self.flow_map.net
        if hasattr(idm_net, "obs_summarizer") and hasattr(idm_net, "To_obs"):
            self._obs_summarizer = idm_net.obs_summarizer
            self._To_obs = idm_net.To_obs
            loguru.logger.info(
                f"Reusing frozen IDM obs_summarizer; To_obs={self._To_obs}, "
                f"obs_dim={enc_out_dim}"
            )
        else:
            self._obs_summarizer = None
            self._To_obs = config.task.obs_steps
            loguru.logger.info(
                f"IDM has no obs_summarizer; cond_mode='obs_summary' will be unavailable. "
                f"To_obs={self._To_obs}, obs_dim={enc_out_dim}"
            )

        # IDM expects obs to have exactly To_obs frames; mismatched task.obs_steps
        # would crash later inside the IDM's obs_summarizer (shape mismatch).
        # Fail loudly here instead of mid-training.
        if config.task.obs_steps != self._To_obs:
            raise ValueError(
                f"task.obs_steps ({config.task.obs_steps}) must match the IDM's "
                f"To_obs ({self._To_obs}). Re-train with matching obs_steps or "
                f"switch IDM checkpoints."
            )

        # --- Loss scaling ---
        # State flow MSE: divided by enc_out_dim so the magnitude is
        # interpretable as a per-element MSE and comparable across encoder
        # widths (mirrors the consistency-loss convention below).
        self._goal_loss_scale = (
            config.optimization.goal_flow_loss_scale / enc_out_dim
        )
        # Consistency loss (full-ODE goal integration -> frozen IDM action FM).
        # Disabled by default (weight=0). Kept divided by act_dim to mirror the
        # v1 action-reg convention.
        self._consistency_weight = config.optimization.consistency_weight
        self._consistency_loss_scale = (
            self._consistency_weight / config.task.act_dim
        )
        self._consistency_num_steps = (
            config.optimization.consistency_num_steps
            or config.optimization.goal_flow_num_steps
        )

        # --- Build the velocity network ---
        hidden_dim = config.network.o2g_hidden_dim or (2 * enc_out_dim)
        timestep_emb_dim = config.network.o2g_timestep_emb_dim or enc_out_dim
        cond_mode = config.network.o2g_cond_mode
        if cond_mode == "obs_summary" and self._obs_summarizer is None:
            raise ValueError(
                "o2g_cond_mode='obs_summary' requires an IDM with obs_summarizer "
                "(e.g., lbmidm_v2). Loaded IDM has none."
            )

        self.goal_net = GoalPredictorO2G(
            act_dim=enc_out_dim,
            Ta=1,
            obs_dim=enc_out_dim,
            To=self._To_obs,
            hidden_dim=hidden_dim,
            num_layers=config.network.o2g_num_layers,
            mlp_ratio=config.network.o2g_mlp_ratio,
            dropout=config.network.o2g_dropout,
            timestep_emb_dim=timestep_emb_dim,
            cond_mode=cond_mode,
            timestep_emb_type=config.network.timestep_emb_type,
        ).to(device)
        report_parameters(self.goal_net, model_name="Goal Predictor O2G")
        self._cond_mode = cond_mode

        self.goal_flow_map = FlowMap(self.goal_net).to(device)

        # --- EMA for goal net ---
        self.goal_net_ema = deepcopy(self.goal_net).requires_grad_(False)
        self.goal_flow_map_ema = FlowMap(self.goal_net_ema).to(device)

        # --- Encoder wrappers (main + ema) ---
        self.wrapper_encoder = GoalPredictorO2GEncoderWrapper(
            self._inner_encoder,
            self._obs_summarizer,
            self.goal_flow_map,
            config.optimization.goal_flow_num_steps,
            cond_mode=cond_mode,
            To_obs=self._To_obs,
        )
        self.wrapper_encoder_ema = GoalPredictorO2GEncoderWrapper(
            self._inner_encoder,
            self._obs_summarizer,
            self.goal_flow_map_ema,
            config.optimization.goal_flow_num_steps,
            cond_mode=cond_mode,
            To_obs=self._To_obs,
        )

        # --- Interpolants (separate instances for goal and action paths) ---
        self.goal_interpolant = Interpolant(config.optimization.interp_type)
        self.action_interpolant = Interpolant(config.optimization.interp_type)

        # --- Optimizer (only the goal net) ---
        self.optimizer = torch.optim.AdamW(
            self.goal_net.parameters(),
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

    # --- Utilities -----------------------------------------------------------

    def _make_x0(self, z_t: torch.Tensor) -> torch.Tensor:
        # Last raw obs frame as the flow source.
        # z_t: (B, To_obs, obs_dim) -> (B, 1, obs_dim)
        return z_t[:, -1:, :]

    def _make_cond(self, z_t: torch.Tensor):
        if self._cond_mode == "none":
            return None
        if self._cond_mode == "obs_summary":
            # Frozen IDM obs_summarizer; produced fresh from z_t (independent of x0).
            return self._obs_summarizer(z_t.flatten(1)).unsqueeze(1)
        raise ValueError(f"Unknown cond_mode '{self._cond_mode}'")

    # --- Training step -------------------------------------------------------

    def update(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict | TensorDict,
        goal_obs: torch.Tensor | dict | TensorDict,
        delta_t: torch.Tensor,
    ) -> dict:
        """One training step.

        Always computes the state flow MSE. When consistency_weight > 0,
        additionally runs a full-ODE goal integration and regresses the frozen
        IDM's action velocity against ground-truth (VITA-style "FLC" with the
        action FM as comparator rather than direct latent MSE).

        `delta_t` is accepted for API compatibility but unused here.
        """
        del delta_t  # unused
        config = self.config.optimization

        with torch.no_grad():
            z_t = self._inner_encoder(obs, None)             # (B, To_obs, obs_dim)
            z_goal = self._inner_encoder(goal_obs, None)     # (B, 1, obs_dim)
            x0 = self._make_x0(z_t)                          # (B, 1, obs_dim) — last raw frame
        x1 = z_goal                                          # raw, no z-score

        B = x1.shape[0]
        device = x1.device
        cond = self._make_cond(z_t)

        # --- State flow loss (always on) ---
        t_flow = torch.empty(B, device=device).uniform_(0, 1)
        xt = self.goal_interpolant.calc_It(t_flow, x0, x1)
        ut = self.goal_interpolant.calc_It_dot(t_flow, x0, x1)
        v_pred, _ = self.goal_net(xt, t_flow, t_flow, cond)
        state_flow_loss = self._goal_loss_scale * torch.mean(
            get_norm(v_pred - ut, config.norm_type)
        )

        # --- Consistency loss (optional) ---
        if self._consistency_weight > 0:
            consistency_loss_unscaled = self._consistency_loss(
                z_t=z_t, x0=x0, cond=cond, act=act, config=config
            )
            consistency_loss = self._consistency_loss_scale * consistency_loss_unscaled
        else:
            consistency_loss_unscaled = torch.zeros((), device=device)
            consistency_loss = torch.zeros((), device=device)

        loss = state_flow_loss + consistency_loss
        loss.backward()

        if config.grad_clip_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.goal_net.parameters(), config.grad_clip_norm
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
            "consistency_loss": consistency_loss.detach(),
            "consistency_loss_unscaled": consistency_loss_unscaled.detach(),
            "grad_norm": grad_norm.detach(),
        }

    def _integrate_goal_with_grad(
        self, x0: torch.Tensor, cond: torch.Tensor | None, num_steps: int,
    ) -> torch.Tensor:
        """Euler-integrate the goal flow from t=0 to t=1 with autograd enabled.

        Mirrors the wrapper's inference loop but keeps the graph attached so
        gradients flow back through every solver step into goal_net.
        """
        B = x0.shape[0]
        device = x0.device
        g_s = x0
        t_sched = np.linspace(0, 1, num_steps + 1)
        for i in range(num_steps):
            s_val = t_sched[i]
            t_val = t_sched[i + 1]
            s = torch.full((B,), s_val, device=device)
            v = self.goal_flow_map.get_velocity(s, g_s, cond)
            g_s = g_s + v * (t_val - s_val)
        return g_s

    def _consistency_loss(
        self,
        z_t: torch.Tensor,
        x0: torch.Tensor,
        cond: torch.Tensor | None,
        act: torch.Tensor,
        config,
    ) -> torch.Tensor:
        """Full-ODE goal integration -> action FM regression through frozen IDM."""
        B = act.shape[0]
        device = act.device

        g_hat = self._integrate_goal_with_grad(
            x0=x0, cond=cond, num_steps=self._consistency_num_steps,
        )                                                    # (B, 1, obs_dim)
        obs_emb = torch.cat([z_t, g_hat], dim=1)             # (B, To_obs+1, obs_dim)

        t_act = torch.empty(B, device=device).uniform_(0, 1)
        act_0 = torch.randn_like(act)
        act_1 = act
        act_t = self.action_interpolant.calc_It(t_act, act_0, act_1)
        act_t_dot = self.action_interpolant.calc_It_dot(t_act, act_0, act_1)

        b_t = self.flow_map.get_velocity(t_act, act_t, obs_emb)
        per_elem = get_norm(b_t - act_t_dot, config.norm_type)
        return per_elem.mean()

    def _ema_update(self):
        ema_rate = self.config.optimization.ema_rate
        with torch.no_grad():
            for p, p_ema in zip(
                self.goal_net.parameters(),
                self.goal_net_ema.parameters(),
                strict=False,
            ):
                p_ema.data.mul_(ema_rate).add_(p.data, alpha=1.0 - ema_rate)

    # --- Inference -----------------------------------------------------------

    def sample(
        self,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        num_steps: int = -1,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Sample actions: encode obs -> generate goal via O2G ODE -> run IDM sampler.

        Uses CFG when cfg_scale != 1.0 and an unconditional embedding is available.
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
        goal_fm = self.goal_flow_map_ema if use_ema else self.goal_flow_map

        z_t = self._inner_encoder(obs, None)              # (B, To_obs, obs_dim)
        B, To, emb_dim = z_t.shape
        device = z_t.device

        g_s = self._make_x0(z_t)
        cond = self._make_cond(z_t)

        goal_num_steps = config.goal_flow_num_steps
        t_schedule = np.linspace(0, 1, goal_num_steps + 1)
        for i in range(goal_num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((B,), s_val, device=device)
            v = goal_fm.get_velocity(s, g_s, cond)
            g_s = g_s + v * (t_val - s_val)

        g_hat = g_s
        cond_emb = torch.cat([z_t, g_hat], dim=1)         # (B, To+1, emb_dim)
        uncond = self._uncond_emb.expand(B, 1, -1)
        uncond_emb = torch.cat([z_t, uncond], dim=1)

        num_steps = config.num_steps
        act_t_schedule = np.linspace(0, 1, num_steps + 1)
        if config.sample_mode == "stochastic":
            act_s = torch.randn_like(act_0, device=device)
        else:
            act_s = torch.zeros_like(act_0, device=device)

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

    # --- Save / load ---------------------------------------------------------

    def save(self, path: str, training_state: dict = None):
        checkpoint = {
            "goal_net": self.goal_net.state_dict(),
            "goal_net_ema": self.goal_net_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "idm_checkpoint_path": self.config.optimization.idm_checkpoint_path,
        }
        if training_state is not None:
            checkpoint["training_state"] = training_state
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False):
        state_dict = torch.load(
            path, map_location=self.config.optimization.device, weights_only=False
        )
        self.goal_net.load_state_dict(state_dict["goal_net"])
        self.goal_net_ema.load_state_dict(state_dict["goal_net_ema"])
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
        self.goal_net.eval()
        self.goal_net_ema.eval()
        self.encoder.eval()

    def train(self):
        self.goal_net.train()
        self.encoder.train()
