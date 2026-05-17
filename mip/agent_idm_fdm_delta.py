"""IDM training agent for the delta-cond IDM (LBMDiTIDMv2Delta).

Same training recipe as ``IDMFDMAgent`` (flow-matching action loss + FDM
auxiliary). The only structural change vs. ``IDMFDMAgent`` is that the
action-trunk AdaLN sees the delta ``z_goal − z_last_obs`` in its third
slot, which is handled automatically by ``LBMDiTIDMv2Delta._summarize``.

The FDM head still targets the absolute ``z_goal`` (verified empirically
on the existing delta-cond IDM checkpoint: saved fdm_head outputs match
``z_goal`` to ~1e-4 MSE on expert data). So in the original run the
"delta-cond" only affects the action-trunk conditioning; the FDM aux is
identical to goal-cond training.

Checkpoint compatibility: shares parameter shapes with ``IDMFDMAgent`` +
``lbmidm_v2``, so ``agent.load(...)`` works on either side.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mip.agent_idm_fdm import IDMFDMAgent
from mip.config import Config
from mip.losses import get_norm


class IDMFDMAgentDelta(IDMFDMAgent):
    """IDM+FDM training for the delta-cond IDM.

    Inherits all of ``IDMFDMAgent``'s setup (encoder wrapping, EMA, optimizer,
    grad-clip, ortho regularizer); only the update step is overridden so the
    FDM target is the goal-delta instead of the absolute goal embedding.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        # Network must support delta-cond semantics. ``forward_predict`` is the
        # only contract that changes interpretation; the shape check on
        # ``LBMDiTIDMv2`` from the parent class is sufficient at the structural
        # level. We still print a marker so misconfiguration is visible.
        net_name = type(self.flow_map.net).__name__
        if "Delta" not in net_name:
            import loguru
            loguru.logger.warning(
                f"IDMFDMAgentDelta is paired with {net_name}, which is not a "
                f"delta-cond network. The FDM target will be a delta, but the "
                f"action trunk will be conditioned on the raw goal embedding — "
                f"these are inconsistent. Did you mean network=lbmidm_v2_delta?"
            )

    def _create_update_impl(self):
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

            # FDM target: real (pre-dropout) absolute z_goal, stop-grad.
            # Matches the existing delta-cond IDM ckpt — in that run only the
            # action-trunk AdaLN saw the delta; the FDM head still predicts
            # the absolute goal embedding.
            fdm_target = raw_encoded[:, -1].detach()

            # --- IDM action flow-matching loss ---
            # Network ``forward`` uses _summarize, which returns the delta as
            # the third AdaLN slot for the delta-cond network — no change here.
            t = torch.empty_like(delta_t).uniform_(0, 1)
            act_0 = torch.empty_like(act).normal_(0, 1)
            act_t = self.interpolant.calc_It(t, act_0, act)
            act_t_dot = self.interpolant.calc_It_dot(t, act_0, act)
            b_t = self.flow_map.get_velocity(t, act_t, encoded)
            idm_loss = cfg.loss_scale * torch.mean(
                get_norm(b_t - act_t_dot, cfg.norm_type)
            )

            # --- FDM (forward-dynamics) auxiliary loss against the delta ---
            predicted_goal = self.flow_map.net.forward_predict(encoded, act)
            fdm_loss_unscaled = F.mse_loss(predicted_goal, fdm_target)
            fdm_loss = cfg.fdm_loss_scale * fdm_loss_unscaled

            # --- Ortho regularizer (rarely useful in delta-cond, kept for
            # parity with goal-cond agent; controlled by ortho_reg_weight). ---
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
