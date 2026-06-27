"""LBMDiTJointE2EAgent: joint flow matching with end-to-end trainable encoder.

Differences from ``LBMDiTJointAgent``:

- The encoder is **trainable** alongside the joint trunk (no frozen branch).
- A ``LayerNorm`` (``self.target_ln``) sits at the encoder's output and is
  treated as the encoder's final layer. It is applied to *both* the AdaLN
  condition (current obs) and the FM target (next-state), so the trunk sees
  one consistent LN'd embedding space.
- The FM target is stop-grad'd **after** LN (``LN(z).detach()``). This means
  encoder body **and** LN params are symmetrically blocked from FM-target
  gradient. Both learn end-to-end through the AdaLN-condition path only
  (action loss + state loss flowing through the trunk into the LN'd z_t).
- ``idm_checkpoint_path`` becomes optional: if provided, encoder weights are
  warm-started from the IDM checkpoint; otherwise the encoder is randomly
  initialized.

Loss path:

    z_body = encoder(obs)
    z_t    = target_ln(z_body)                       # condition, LN'd, full grad path
    target = target_ln(encoder(goal_obs)).detach()   # FM target, LN'd, no grad

    x0 ~ N(0, I)
    xt     = (1-t) * x0 + t * target
    xt_dot = target - x0                             # linear interpolant

    v_state, v_action, _ = net(xt, x_action_t, t, t, z_t, optimality)
    state_loss  = mean((v_state  - xt_dot)^2) / obs_dim
    action_loss = mean((v_action - act_dot)^2) / act_dim
    loss = w_state * state_loss + w_action * action_loss

EMA covers only the trunk for now (per the "stop-grad rather than EMA" call).
The encoder + LN are not EMA-tracked.

Author: Zilai Zeng
"""

from __future__ import annotations

import os
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
from mip.networks.lbmdit_joint import LBMDiTJoint
from mip.torch_utils import report_parameters


class LBMDiTJointE2EAgent(LBMDiTJointAgent):
    """E2E variant of LBMDiTJointAgent: trainable encoder + target LayerNorm."""

    def __init__(self, config: Config):
        # NOTE: we deliberately don't call super().__init__ — that path errors
        # when idm_checkpoint_path is None and pulls in the offline goal-stats
        # logic we no longer need. We rebuild the agent state explicitly.
        # `self.config` and other attrs that the parent's helper methods read
        # are populated below.
        self.config = config
        device = config.optimization.device

        # --- Encoder (instantiate first, then optionally warm-start) ---
        self.encoder = get_encoder(config.network, config.task).to(device)
        idm_path = config.optimization.idm_checkpoint_path
        if idm_path is not None:
            loguru.logger.info(
                f"Warm-starting encoder from {idm_path} (weights only; rest of "
                f"the IDM checkpoint is ignored in the E2E agent)"
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

        # E2E always trainable; ignore joint_freeze_encoder for this variant.
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

        # Goal-stats helpers from the parent are unused here; null them so a
        # stray call surfaces immediately rather than silently no-op'ing.
        self._goal_mean = None
        self._goal_var = None
        self._norm_eps = 1e-5

        # --- Joint trunk + EMA ---
        self.net = LBMDiTJoint(
            act_dim=config.task.act_dim,
            Ta=config.task.horizon,
            obs_dim=obs_dim,
            To=config.task.obs_steps,
            d_model=config.network.emb_dim,
            n_heads=config.network.n_heads,
            depth=config.network.num_layers,
            dropout=config.network.dropout,
            timestep_emb_type=config.network.timestep_emb_type,
            timestep_emb_dim=config.network.timestep_emb_dim,
            opt_emb_dim=config.network.joint_opt_emb_dim,
        ).to(device)
        report_parameters(self.net, model_name="LBMDiTJoint (e2e)")

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
        self._use_ema_target = bool(config.optimization.joint_use_ema_target)

    # ------------------------------- training -------------------------------

    def update(
        self,
        act: torch.Tensor,
        obs,
        goal_obs,
        delta_t: torch.Tensor,
        optimality: torch.Tensor | None = None,
    ) -> dict:
        """Joint flow-matching update with E2E encoder + LN'd target."""
        config = self.config.optimization
        device = act.device
        B = act.shape[0]

        # 1. Condition path always uses the live encoder + LN — that's how
        #    the encoder receives gradient signal. The FM target side is
        #    optionally computed from the EMA encoder + EMA LN, which
        #    stabilizes the regression target as the live encoder evolves
        #    (self-distillation pattern, BYOL/DINO/REPA-E). When
        #    ``joint_use_ema_target`` is False or ``ema_rate >= 1``, falls
        #    back to the live-encoder-with-stop-grad path.
        z_t = self.target_ln(self.encoder(obs, None))           # (B, To, obs_dim), grad
        if self._use_ema_target and config.ema_rate < 1:
            with torch.no_grad():
                target = self.target_ln_ema(self.encoder_ema(goal_obs, None))
        else:
            z_goal_ln = self.target_ln(self.encoder(goal_obs, None))
            target = z_goal_ln.detach()

        # 3. Optimality labels (default null/play; CFG dropout)
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

        # 4. Shared flow time, per-stream noise + interpolation
        t = torch.empty(B, device=device).uniform_(0, 1)

        s_noise = torch.randn_like(target)
        a_noise = torch.randn_like(act)

        s_t = self.interpolant.calc_It(t, s_noise, target)
        s_t_dot = self.interpolant.calc_It_dot(t, s_noise, target)
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

        # 6. Per-stream losses (per-element MSE so weights are interpretable)
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

        # Grad clip across all trainable params
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

        # EMA the trunk, encoder, and target_ln together.
        if config.ema_rate < 1:
            self._ema_update()

        # Diagnostic: target distribution stats — if std collapses toward 0 the
        # representation is collapsing.
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
        }

    # ------------------------------- ema -----------------------------------

    def _ema_update(self) -> None:
        """Extend the parent's trunk EMA with encoder + target_ln EMA.

        Matches the lbmdit / TrainingAgent convention where the encoder is
        also EMA-tracked so that eval-time (use_ema=True) sees a smoothed
        encoder consistent with the trunk EMA.
        """
        super()._ema_update()
        rate = self.config.optimization.ema_rate
        with torch.no_grad():
            # Encoder parameters. Skip params shared by identity between the live
            # encoder and its EMA copy (``p is p_ema``): a frozen-ViT encoder
            # shares ONE backbone with encoder_ema (FrozenVisionBackbone
            # __deepcopy__ returns self), and the in-place ``mul_(rate).add_(p,
            # 1-rate)`` would alias — reading p AFTER scaling it — slowly
            # decaying those (meant-to-be-frozen) weights. EMA of a param against
            # itself is a no-op anyway, so skipping is both correct and a no-op
            # for the normal (fully-copied) case.
            for p, p_ema in zip(
                self.encoder.parameters(),
                self.encoder_ema.parameters(),
                strict=False,
            ):
                if p is p_ema:
                    continue
                p_ema.data.mul_(rate).add_(p.data, alpha=1.0 - rate)
            # Encoder buffers (BN running stats etc.) — copy not EMA, since
            # they track distributional statistics rather than learned params.
            for b, b_ema in zip(
                self.encoder.buffers(),
                self.encoder_ema.buffers(),
                strict=False,
            ):
                if b is b_ema:
                    continue
                b_ema.data.copy_(b.data)
            # Target LayerNorm parameters (gamma/beta when affine=True).
            for p, p_ema in zip(
                self.target_ln.parameters(),
                self.target_ln_ema.parameters(),
                strict=False,
            ):
                p_ema.data.mul_(rate).add_(p.data, alpha=1.0 - rate)

    # ------------------------------- sampling ------------------------------
    # Override sample / sample_joint so the inherited eval path sees the LN'd
    # condition. The state-stream ODE then runs entirely in LN'd space (xt
    # interpolates two LN-distributed endpoints; state head was trained to
    # predict velocity in that space). Action stream is unaffected.

    def _eval_encoder_modules(self, use_ema: bool) -> tuple[nn.Module, nn.Module]:
        """Return ``(encoder, target_ln)`` to use at sampling time.

        Returns the EMA copies when ``use_ema=True`` AND ``ema_rate < 1``
        (EMA is actively tracked). Falls through to the live modules
        otherwise — matches lbmdit / TrainingAgent behavior.
        """
        if use_ema and self.config.optimization.ema_rate < 1:
            return self.encoder_ema, self.target_ln_ema
        return self.encoder, self.target_ln

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

        # LN'd condition — matches what the trunk saw at training time.
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

        return x_action, x_state

    # --------------------------------- IO ---------------------------------

    def save(self, path: str | os.PathLike, training_state: dict | None = None):
        checkpoint = {
            "net": self.net.state_dict(),
            "net_ema": self.net_ema.state_dict(),
            "encoder": self.encoder.state_dict(),
            "encoder_ema": self.encoder_ema.state_dict(),
            "target_ln": self.target_ln.state_dict(),
            "target_ln_ema": self.target_ln_ema.state_dict(),
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
        if "target_ln" in state_dict:
            self.target_ln.load_state_dict(state_dict["target_ln"])
        # Backward-compat: older checkpoints (pre-EMA-encoder) won't have
        # these keys. Initialize EMA copies from the live weights so eval
        # works without re-warming the EMA from scratch.
        if "encoder_ema" in state_dict:
            self.encoder_ema.load_state_dict(state_dict["encoder_ema"])
        else:
            self.encoder_ema.load_state_dict(self.encoder.state_dict())
            loguru.logger.info(
                "encoder_ema not in checkpoint; initialized from live encoder."
            )
        if "target_ln_ema" in state_dict:
            self.target_ln_ema.load_state_dict(state_dict["target_ln_ema"])
        else:
            self.target_ln_ema.load_state_dict(self.target_ln.state_dict())
        if load_optimizer and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            loguru.logger.info("Loaded optimizer state")
        return state_dict.get("training_state", None)

    def eval(self):
        super().eval()
        self.target_ln.eval()
        # EMA copies are always eval — they're never trained.
        self.encoder_ema.eval()
        self.target_ln_ema.eval()

    def train(self):
        super().train()
        self.target_ln.train()
        # EMA copies stay in eval mode even when the agent is training.
        self.encoder_ema.eval()
        self.target_ln_ema.eval()
