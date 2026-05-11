"""LBMDiTJointAgent: single-stage joint flow matching over (next_state, action_chunk).

The agent owns:
  - A frozen, IDM-pretrained encoder (loaded from ``idm_checkpoint_path``).
    Used to encode both the current obs window (-> AdaLN cond) and the goal
    obs frame (-> next-state target) in the IDM's representation space.
  - An ``LBMDiTJoint`` trunk that flow-matches the joint variable
    (z_goal, action_chunk) under AdaLN conditioning on (time, encoded obs,
    optimality).
  - An EMA copy of the joint trunk used at inference.

Training step (per batch):
  1. Encode obs and goal_obs with the frozen encoder (no grad).
  2. (Optionally) z-score the goal embedding using precomputed stats so the
     state-flow target lives in roughly N(0, I) — same trick as
     ``GoalPredictorDiTAgent``.
  3. CFG dropout: with prob ``joint_cfg_dropout_prob``, override the
     optimality label with the null/play index (slot 1).
  4. Sample shared flow time t ~ U(0,1), sample noise per stream, interpolate.
  5. Forward through the joint trunk -> (v_state, v_action).
  6. Loss = w_state * L_state + w_action * L_action where each per-stream
     loss is `mean(get_norm(...)) / dim` — the dim-divide cancels the linear
     scaling that ``get_norm`` introduces by summing over the last dim, so
     state and action losses are on a comparable per-element scale and the
     weights remain interpretable when ``obs_dim != act_dim``.

Inference (sample):
  Joint Euler ODE from t=0 to t=1. With ``joint_cfg_scale`` (= w) > 0, each
  step does two trunk forward passes (expert and null/play) and blends:
      v = (1+w) * v(opt=expert) - w * v(opt=null)
  applied identically to v_state and v_action.

Optimality labels:
  - Slot 0 = expert
  - Slot 1 = null/play (also doubles as the unconditional reference for CFG)
  Caller passes ``optimality`` per-sample in ``update``. If absent, all samples
  default to the null/play slot — the safer default, since it doesn't assert
  expertise on data we know nothing about.

Author: Zilai Zeng
"""

from __future__ import annotations

import os
from copy import deepcopy

import loguru
import numpy as np
import torch
import torch.nn as nn

from mip.config import Config
from mip.interpolant import Interpolant
from mip.losses import get_norm
from mip.network_utils import get_encoder
from mip.networks.lbmdit_joint import LBMDiTJoint
from mip.torch_utils import report_parameters


class LBMDiTJointAgent:
    """Trains and samples from a joint (next_state, action) flow matching DiT."""

    def __init__(self, config: Config):
        self.config = config
        device = config.optimization.device

        # --- Encoder: instantiate, then load weights from IDM checkpoint ---
        self.encoder = get_encoder(config.network, config.task).to(device)

        idm_path = config.optimization.idm_checkpoint_path
        if idm_path is None:
            raise ValueError(
                "idm_checkpoint_path must be set for LBMDiTJointAgent so the "
                "encoder is initialized from a pretrained IDM."
            )
        loguru.logger.info(f"Loading pretrained encoder from {idm_path}")
        state_dict = torch.load(idm_path, map_location=device, weights_only=False)
        encoder_sd = state_dict["encoder"]

        # Handle GoalDropoutEncoder-wrapped checkpoints — only the inner
        # encoder is needed for joint training. Strip the "encoder." prefix.
        has_goal_dropout = any(k.startswith("encoder.") for k in encoder_sd)
        if has_goal_dropout:
            inner_sd = {
                k.removeprefix("encoder."): v
                for k, v in encoder_sd.items()
                if k.startswith("encoder.")
            }
            self.encoder.load_state_dict(inner_sd)
            loguru.logger.info("Loaded inner encoder from GoalDropoutEncoder checkpoint")
        else:
            self.encoder.load_state_dict(encoder_sd)
            loguru.logger.info("Loaded encoder weights from IDM checkpoint")

        # Freeze (default) or fine-tune the encoder
        if config.optimization.joint_freeze_encoder:
            self.encoder.requires_grad_(False)
            loguru.logger.info("Encoder frozen for joint training")
        else:
            loguru.logger.info("Encoder will be fine-tuned during joint training")

        # --- Goal normalization stats (mirrors GoalPredictorDiTAgent) ---
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

        # --- Joint trunk + EMA ---
        self.net = LBMDiTJoint(
            act_dim=config.task.act_dim,
            Ta=config.task.horizon,
            obs_dim=(config.network.encoder_out_dim or config.network.emb_dim),
            To=config.task.obs_steps,
            d_model=config.network.emb_dim,
            n_heads=config.network.n_heads,
            depth=config.network.num_layers,
            dropout=config.network.dropout,
            timestep_emb_type=config.network.timestep_emb_type,
            timestep_emb_dim=config.network.timestep_emb_dim,
            opt_emb_dim=config.network.joint_opt_emb_dim,
        ).to(device)
        report_parameters(self.net, model_name="LBMDiTJoint")

        self.net_ema = deepcopy(self.net).requires_grad_(False)
        self.net_ema.eval()

        # --- Interpolant (shared by both streams) ---
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

        # Cache scalar weights / scales for the inner update closure.
        self._w_state = config.optimization.joint_state_loss_weight
        self._w_action = config.optimization.joint_action_loss_weight
        self._cfg_dropout_prob = config.optimization.joint_cfg_dropout_prob
        self._cfg_scale = config.optimization.joint_cfg_scale
        self._sample_mode = config.optimization.joint_sample_mode
        self._num_steps = config.optimization.joint_num_steps

    # ------------------------- normalization helpers -------------------------

    def _normalize_goal(self, z: torch.Tensor) -> torch.Tensor:
        if self._goal_mean is None:
            return z
        return (z - self._goal_mean) / torch.sqrt(self._goal_var + self._norm_eps)

    def _denormalize_goal(self, z_norm: torch.Tensor) -> torch.Tensor:
        if self._goal_mean is None:
            return z_norm
        return z_norm * torch.sqrt(self._goal_var + self._norm_eps) + self._goal_mean

    # ------------------------------- training -------------------------------

    def update(
        self,
        act: torch.Tensor,
        obs,
        goal_obs,
        delta_t: torch.Tensor,
        optimality: torch.Tensor | None = None,
    ) -> dict:
        """One joint flow matching update step.

        Args:
            act:         (B, Ta, act_dim) clean expert/play actions (normalized).
            obs:         current obs (B, To, ...).
            goal_obs:    goal obs (B, 1, ...) — encoded -> next-state target.
            delta_t:     (B,) shape donor (kept for parent-style interface).
            optimality:  (B,) long in {0=expert, 1=null/play}; None -> all null/play.
        """
        config = self.config.optimization
        device = act.device
        B = act.shape[0]

        # 1. Encode obs + goal under no_grad if encoder is frozen
        if config.joint_freeze_encoder:
            with torch.no_grad():
                z_t = self.encoder(obs, None)              # (B, To, emb_dim)
                z_goal_raw = self.encoder(goal_obs, None)  # (B, 1, emb_dim)
        else:
            z_t = self.encoder(obs, None)
            z_goal_raw = self.encoder(goal_obs, None)

        # 2. Optional normalization for the state-flow target only
        z_goal = self._normalize_goal(z_goal_raw)

        # 3. Optimality labels (default to null/play if not provided) + CFG dropout
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

        # 4. Shared flow time, noise per stream, interpolate
        t = torch.empty(B, device=device).uniform_(0, 1)

        s_noise = torch.randn_like(z_goal)
        a_noise = torch.randn_like(act)

        s_t = self.interpolant.calc_It(t, s_noise, z_goal)
        s_t_dot = self.interpolant.calc_It_dot(t, s_noise, z_goal)
        a_t = self.interpolant.calc_It(t, a_noise, act)
        a_t_dot = self.interpolant.calc_It_dot(t, a_noise, act)

        # 5. Joint forward
        v_state, v_action, _ = self.net(
            x_state=s_t,
            x_action=a_t,
            s=t,
            t=t,
            condition=z_t,
            optimality_idx=optimality,
        )

        # 6. Per-stream losses (per-element MSE so weights are interpretable
        # across obs_dim != act_dim), weighted sum.
        state_loss_unscaled = torch.mean(
            get_norm(v_state - s_t_dot, config.norm_type)
        ) / float(self.net.obs_dim)
        action_loss_unscaled = torch.mean(
            get_norm(v_action - a_t_dot, config.norm_type)
        ) / float(self.net.act_dim)
        loss = self._w_state * state_loss_unscaled + self._w_action * action_loss_unscaled

        loss.backward()

        # Grad clip
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

        del delta_t  # carried for interface compat with other agents
        return {
            "dp_loss": loss.detach(),
            "state_loss": state_loss_unscaled.detach(),
            "action_loss": action_loss_unscaled.detach(),
            "grad_norm": grad_norm.detach(),
        }

    def _ema_update(self):
        ema_rate = self.config.optimization.ema_rate
        with torch.no_grad():
            for p, p_ema in zip(
                self.net.parameters(),
                self.net_ema.parameters(),
                strict=False,
            ):
                p_ema.data.mul_(ema_rate).add_(p.data, alpha=1.0 - ema_rate)

    # ------------------------------- sampling -------------------------------

    @torch.no_grad()
    def sample(
        self,
        act_0: torch.Tensor,
        obs,
        num_steps: int = -1,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Joint Euler ODE sampling. Returns the action chunk only.

        Args:
            act_0:     (B, Ta, act_dim) initial action noise — also donates the
                       (B, Ta) shape for the rest of the action stream.
            obs:       current observations (B, To, ...).
            num_steps: ODE steps. -1 -> use ``joint_num_steps`` from config.
            use_ema:   sample from the EMA trunk.
        """
        net = self.net_ema if use_ema else self.net
        device = act_0.device
        B = act_0.shape[0]
        cfg_scale = self._cfg_scale
        steps = self._num_steps if num_steps < 1 else int(num_steps)

        # Encode obs (encoder is frozen and in eval / train mode externally)
        z_t = self.encoder(obs, None)  # (B, To, obs_dim)
        obs_dim = z_t.shape[-1]

        # Initial noise per stream. The action stream uses the externally
        # supplied `act_0` (already drawn ~ N(0, I) by the eval loop), the
        # state stream is sampled internally — same convention as
        # GoalPredictorDiTAgent.
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

        t_schedule = np.linspace(0, 1, steps + 1)
        for i in range(steps):
            s_val = float(t_schedule[i])
            t_val = float(t_schedule[i + 1])
            dt = t_val - s_val
            t_b = torch.full((B,), s_val, device=device)

            v_s_cond, v_a_cond, _ = net(
                x_state=x_state,
                x_action=x_action,
                s=t_b,
                t=t_b,
                condition=z_t,
                optimality_idx=expert_idx,
            )

            if cfg_scale > 0:
                v_s_un, v_a_un, _ = net(
                    x_state=x_state,
                    x_action=x_action,
                    s=t_b,
                    t=t_b,
                    condition=z_t,
                    optimality_idx=null_idx,
                )
                v_s = (1 + cfg_scale) * v_s_cond - cfg_scale * v_s_un
                v_a = (1 + cfg_scale) * v_a_cond - cfg_scale * v_a_un
            else:
                v_s = v_s_cond
                v_a = v_a_cond

            x_state = x_state + v_s * dt
            x_action = x_action + v_a * dt

        # x_state at t=1 is the predicted next-state in normalized space; left
        # auxiliary for now and not returned. Callers that want it can call
        # sample_joint() instead.
        return x_action

    @torch.no_grad()
    def sample_joint(
        self,
        act_0: torch.Tensor,
        obs,
        num_steps: int = -1,
        use_ema: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as ``sample`` but also returns the predicted next-state in raw
        encoder space (denormalized if goal stats are loaded).
        """
        # Re-run the loop since `sample` discards x_state. Cheap relative to
        # action execution and only used for diagnostics, so duplicating the
        # loop is acceptable here.
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

        t_schedule = np.linspace(0, 1, steps + 1)
        for i in range(steps):
            s_val = float(t_schedule[i])
            t_val = float(t_schedule[i + 1])
            dt = t_val - s_val
            t_b = torch.full((B,), s_val, device=device)

            v_s_cond, v_a_cond, _ = net(
                x_state, x_action, t_b, t_b, z_t, expert_idx,
            )
            if cfg_scale > 0:
                v_s_un, v_a_un, _ = net(
                    x_state, x_action, t_b, t_b, z_t, null_idx,
                )
                v_s = (1 + cfg_scale) * v_s_cond - cfg_scale * v_s_un
                v_a = (1 + cfg_scale) * v_a_cond - cfg_scale * v_a_un
            else:
                v_s, v_a = v_s_cond, v_a_cond

            x_state = x_state + v_s * dt
            x_action = x_action + v_a * dt

        z_goal_pred = self._denormalize_goal(x_state)
        return x_action, z_goal_pred

    # ------------------------------- modes / IO -------------------------------

    def eval(self):
        self.net.eval()
        self.net_ema.eval()
        self.encoder.eval()

    def train(self):
        self.net.train()
        # Encoder stays in train mode so its CropRandomizer continues to fire
        # during joint training, matching how the IDM was trained. Weights are
        # frozen via requires_grad_ regardless.
        self.encoder.train()

    def save(self, path: str | os.PathLike, training_state: dict | None = None):
        checkpoint = {
            "net": self.net.state_dict(),
            "net_ema": self.net_ema.state_dict(),
            "encoder": self.encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "idm_checkpoint_path": self.config.optimization.idm_checkpoint_path,
        }
        if training_state is not None:
            checkpoint["training_state"] = training_state
        torch.save(checkpoint, path)

    def load(self, path: str | os.PathLike, load_optimizer: bool = False):
        state_dict = torch.load(
            path,
            map_location=self.config.optimization.device,
            weights_only=False,
        )
        self.net.load_state_dict(state_dict["net"])
        self.net_ema.load_state_dict(state_dict["net_ema"])
        if "encoder" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder"])
        if load_optimizer and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            loguru.logger.info("Loaded optimizer state")
        return state_dict.get("training_state", None)
