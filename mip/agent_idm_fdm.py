"""IDM training agent with auxiliary forward-dynamics (FDM) loss.

Trains an IDM (LBMDiTIDMv2) jointly with a forward-dynamics auxiliary that
predicts the goal embedding from (summarized obs, projected clean actions).
The FDM loss backprops into the encoder, shaping it to be forward-predictable
in addition to the IDM's controllability signal — essentially the ICM/SPR
recipe applied to encoder pretraining for downstream goal-predictor use.

Key differences from ``TrainingAgent``:

- The update step is rewritten inline (does not go through ``loss_fn``) so
  that the encoded obs+goal stack can be reused for both the IDM flow loss
  and the FDM auxiliary, and so the FDM's stop-grad target can be cleanly
  taken from the same encoder pass.
- Expects the underlying network to expose ``forward_predict(condition,
  clean_action) -> predicted_goal`` (e.g., LBMDiTIDMv2). Without that method
  this agent will raise.
- The FDM target is the *un-normalized* encoder output of the goal frame,
  detached (stop-gradient) to prevent trivial collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mip.agent import TrainingAgent
from mip.config import Config
from mip.losses import get_norm


class IDMFDMAgent(TrainingAgent):
    """IDM training with auxiliary FDM head for encoder shaping.

    If the encoder is wrapped in ``GoalDropoutEncoder`` (CFG path), goal
    dropout is applied only to the IDM input — the FDM auxiliary always
    targets the *real* (pre-dropout) goal embedding so encoder shaping is
    not contaminated by uncond_emb targets.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        if not hasattr(self.flow_map.net, "forward_predict"):
            raise TypeError(
                f"IDMFDMAgent requires a network with a forward_predict method "
                f"(e.g., LBMDiTIDMv2); got {type(self.flow_map.net).__name__}."
            )

    def _create_update_impl(self):
        # Resolve once (encoder identity is fixed for the agent's lifetime).
        encoder_is_wrapped = (
            hasattr(self.encoder, "apply_goal_dropout")
            and hasattr(self.encoder, "encoder")
            and hasattr(self.encoder, "uncond_emb")
        )
        ortho_reg_enabled = self.config.optimization.ortho_reg_weight > 0

        def update_impl(data: TensorDict):
            act = data["act"]
            obs = data["obs"]
            delta_t = data["delta_t"]
            cfg = self.config.optimization

            # --- Encode obs+goal stack once (B, To+1, emb_dim) ---
            if encoder_is_wrapped:
                # Run inner encoder to get raw (pre-dropout) embeddings, then
                # apply dropout only to the IDM-input copy. FDM target and
                # ortho regularizer always use raw (real goal) embeddings.
                raw_encoded = self.encoder.encoder(obs, None)
                if raw_encoded.shape[1] == self.encoder.obs_steps:
                    uncond = self.encoder.uncond_emb.expand(
                        raw_encoded.shape[0], 1, -1,
                    )
                    raw_encoded = torch.cat([raw_encoded, uncond], dim=1)
                encoded = self.encoder.apply_goal_dropout(raw_encoded)
            else:
                raw_encoded = self.encoder(obs, None)
                encoded = raw_encoded
            target_z_goal = raw_encoded[:, -1].detach()

            # --- IDM (action flow matching) loss ---
            t = torch.empty_like(delta_t).uniform_(0, 1)
            act_0 = torch.empty_like(act).normal_(0, 1)
            act_t = self.interpolant.calc_It(t, act_0, act)
            act_t_dot = self.interpolant.calc_It_dot(t, act_0, act)
            b_t = self.flow_map.get_velocity(t, act_t, encoded)
            idm_loss = cfg.loss_scale * torch.mean(
                get_norm(b_t - act_t_dot, cfg.norm_type)
            )

            # --- FDM (forward-dynamics) auxiliary loss ---
            # Target: real (pre-dropout) goal embedding, stop-grad.
            predicted_goal = self.flow_map.net.forward_predict(encoded, act)
            fdm_loss_unscaled = F.mse_loss(predicted_goal, target_z_goal)
            fdm_loss = cfg.fdm_loss_scale * fdm_loss_unscaled

            # --- Ortho regularizer: hinge on cos_sim(last_obs, goal) ---
            # Penalizes per-sample cos_sim above `ortho_reg_threshold`; below
            # the threshold the gradient is zero, giving FDM full control of
            # the encoder. This stable equilibrium avoids the runaway
            # orthogonality / FDM divergence observed with no-threshold form.
            # Both inputs are raw encoder outputs (no MLP in between).
            # `ortho_loss_unscaled` is the raw mean cos_sim (the diagnostic
            # metric, comparable across runs); `ortho_loss` is the gradient-
            # bearing hinge term scaled by `ortho_reg_weight`.
            if ortho_reg_enabled:
                last_obs_raw = raw_encoded[:, -2]
                goal_raw = raw_encoded[:, -1]
                cos_sim_per_sample = F.cosine_similarity(
                    last_obs_raw, goal_raw, dim=-1,
                )
                ortho_loss_unscaled = cos_sim_per_sample.mean()
                hinge = F.relu(
                    cos_sim_per_sample - cfg.ortho_reg_threshold
                ).mean()
                ortho_loss = cfg.ortho_reg_weight * hinge
            else:
                ortho_loss_unscaled = torch.zeros((), device=encoded.device)
                ortho_loss = torch.zeros((), device=encoded.device)

            loss = idm_loss + fdm_loss + ortho_loss
            loss.backward()

            # --- Grad clip + step + zero_grad ---
            params = list(self.encoder.parameters()) + list(
                self.flow_map.parameters()
            )
            if cfg.grad_clip_norm:
                grad_norm = nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            else:
                grad_norm = torch.tensor(0.0, device=loss.device)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if cfg.ema_rate < 1:
                self._ema_update_impl()

            return TensorDict(
                {
                    "loss": loss.detach(),
                    "idm_loss": idm_loss.detach(),
                    "fdm_loss": fdm_loss.detach(),
                    "fdm_loss_unscaled": fdm_loss_unscaled.detach(),
                    "ortho_loss": ortho_loss.detach(),
                    "ortho_loss_unscaled": ortho_loss_unscaled.detach(),
                    "grad_norm": grad_norm.detach(),
                },
                batch_size=(),
            )

        return update_impl

    def update(
        self,
        act: torch.Tensor,
        obs,
        delta_t: torch.Tensor,
    ):
        """Run one IDM+FDM update step.

        Args:
            act: (B, Ta, act_dim) clean expert actions (normalized).
            obs: (B, To+1, ...) — obs frames stacked with goal frame at the end.
            delta_t: (B,) shape donor (kept for parent interface compat).
        """
        if self.use_cudagraphs:
            if not hasattr(self, "_expected_batch_size"):
                self._expected_batch_size = act.shape[0]
            elif act.shape[0] != self._expected_batch_size:
                raise ValueError(
                    f"CUDA graphs require static batch sizes. "
                    f"Expected {self._expected_batch_size}, got {act.shape[0]}."
                )

        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        data = TensorDict(
            {"act": act, "obs": obs, "delta_t": delta_t},
            batch_size=act.shape[0],
        )
        result = self._compiled_update(data)

        return {
            "dp_loss": result["loss"],
            "idm_loss": result["idm_loss"],
            "fdm_loss": result["fdm_loss"],
            "fdm_loss_unscaled": result["fdm_loss_unscaled"],
            "ortho_loss": result["ortho_loss"],
            "ortho_loss_unscaled": result["ortho_loss_unscaled"],
            "grad_norm": result["grad_norm"],
        }
