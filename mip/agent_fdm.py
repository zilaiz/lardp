"""Forward-dynamics agent for BYOL-style encoder pretraining.

Trains the obs encoder by predicting the next-state embedding z_{s'} from
(z_s, action_chunk) under a BYOL-style recipe:

  - Online: obs encoder f_theta + dynamics-predictor net (obs_summarizer +
    action_proj + dynamics_predictor MLP).
  - Target: f_xi = EMA(f_theta), stop-grad. No projector, no second
    predictor. The action-conditioned online MLP is the only asymmetric
    piece — that asymmetry, together with stop-grad on the target, is what
    keeps the representation from collapsing.
  - Loss: ||normalize(z_pred) - normalize(stop_grad(z_{s'}))||^2, i.e.
    -2 * cos_sim + 2 (BYOL form).

The pretrained ``encoder`` state-dict layout matches what
``compute_goal_stats.py`` and ``LBMDiTJointDDTFrozenAgent`` expect, so this
agent's checkpoint is a drop-in pretraining source for the downstream
joint-DDT-frozen pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mip.agent import TrainingAgent
from mip.config import Config


class FDMAgent(TrainingAgent):
    """BYOL-style forward-dynamics pretraining agent.

    Inherits encoder + flow_map + EMA + save/load from ``TrainingAgent``.
    ``self.flow_map.net`` is an ``FDMNet`` (online dynamics predictor);
    ``self.encoder_ema`` plays the role of the EMA target encoder.

    The inherited ``self.flow_map_ema`` is unused at training time but kept
    so the parent's ``save()`` / ``load()`` / ``_ema_update_impl()`` work
    unchanged.
    """

    def __init__(self, config: Config):
        # Precompute the sorted list of low_dim obs keys BEFORE super().__init__,
        # because the parent's __init__ calls self.__compile__() which calls our
        # overridden _create_update_impl() and that reads self._proprio_keys.
        # Sorted for deterministic concat order across runs.
        self._proprio_keys: list[str] = self._extract_lowdim_keys(config.task)

        super().__init__(config)

        if not hasattr(self.flow_map.net, "forward_online"):
            raise TypeError(
                f"FDMAgent requires a network with a forward_online method "
                f"(e.g., FDMNet); got {type(self.flow_map.net).__name__}."
            )

        # Fail fast if recon is requested but the network can't do it.
        if (
            config.optimization.proprio_recon_loss_scale > 0
            and self.flow_map.net.proprio_decoder is None
        ):
            raise ValueError(
                "optimization.proprio_recon_loss_scale > 0 but FDMNet was "
                "built with proprio_dim=0 — no proprio_decoder exists. "
                "Check that the task's shape_meta has low_dim obs keys."
            )

    @staticmethod
    def _extract_lowdim_keys(task_config) -> list[str]:
        """Sorted list of low_dim obs keys from ``task_config.shape_meta``."""
        if not hasattr(task_config, "shape_meta") or task_config.shape_meta is None:
            return []
        obs_meta = task_config.shape_meta.get("obs", {})
        return sorted(
            k for k, attr in obs_meta.items()
            if attr.get("type", "low_dim") == "low_dim"
        )

    def _create_update_impl(self):
        # Closure-time constants — torch.compile will dead-code-eliminate the
        # disabled branch instead of branching every step.
        cfg = self.config.optimization
        proprio_enabled = cfg.proprio_recon_loss_scale > 0
        proprio_keys = self._proprio_keys

        def update_impl(data: TensorDict):
            obs = data["obs"]
            goal_obs = data["goal_obs"]
            act = data["act"]
            cfg = self.config.optimization

            # --- Online: encode obs and predict z_{s'} ---
            z_obs = self.encoder(obs, None)  # (B, To_obs, D)
            z_pred = self.flow_map.net.forward_online(z_obs, act)  # (B, D)

            # --- Target: EMA encoder, stop-grad ---
            with torch.no_grad():
                z_goal = self.encoder_ema(goal_obs, None)  # (B, 1, D)
                z_goal = z_goal[:, 0]  # (B, D)

            # --- BYOL loss: L2-normalize then MSE  ==  2 - 2 * cos_sim ---
            pred_n = F.normalize(z_pred, dim=-1)
            target_n = F.normalize(z_goal, dim=-1)
            cos_sim_mean = (pred_n * target_n).sum(dim=-1).mean()
            byol_loss = 2.0 - 2.0 * cos_sim_mean

            # --- Optional proprio reconstruction aux loss ---
            # Non-collapsible supervised signal on the encoder. Target is
            # the per-frame low-dim obs (already in `obs`), so we're asking
            # the encoder to preserve information it already saw — but
            # crucially the loss is zero only if z_obs actually retains it.
            if proprio_enabled:
                proprio_target = torch.cat(
                    [obs[k] for k in proprio_keys], dim=-1,
                )  # (B, To_obs, proprio_dim)
                pred_proprio = self.flow_map.net.decode_proprio(z_obs)
                proprio_loss_unscaled = F.mse_loss(
                    pred_proprio, proprio_target,
                )
                proprio_loss = cfg.proprio_recon_loss_scale * proprio_loss_unscaled
                loss = byol_loss + proprio_loss
            else:
                proprio_loss_unscaled = torch.zeros((), device=byol_loss.device)
                proprio_loss = torch.zeros((), device=byol_loss.device)
                loss = byol_loss

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

            # EMA update — also drags ``flow_map_ema`` along; it's never read
            # but parent's _ema_update_impl iterates over the joined param list.
            if cfg.ema_rate < 1:
                self._ema_update_impl()

            # Collapse diagnostics: feature-dim std on the un-normalized
            # outputs. Healthy BYOL runs keep these well above zero; a
            # crash toward zero signals representational collapse.
            pred_std = z_pred.std(dim=0).mean()
            target_std = z_goal.std(dim=0).mean()

            return TensorDict(
                {
                    "loss": loss.detach(),
                    "byol_loss": byol_loss.detach(),
                    "proprio_loss": proprio_loss.detach(),
                    "proprio_loss_unscaled": proprio_loss_unscaled.detach(),
                    "cos_sim": cos_sim_mean.detach(),
                    "pred_std": pred_std.detach(),
                    "target_std": target_std.detach(),
                    "grad_norm": grad_norm.detach(),
                },
                batch_size=(),
            )

        return update_impl

    def update(
        self,
        act: torch.Tensor,
        obs,
        goal_obs,
    ):
        """Run one BYOL-FDM update step.

        Args:
            act: (B, Ta, act_dim) clean expert actions (normalized).
            obs: (B, To_obs, ...) — current obs frames (no goal).
            goal_obs: (B, 1, ...) — next-state frame s'.
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
            {"act": act, "obs": obs, "goal_obs": goal_obs},
            batch_size=act.shape[0],
        )
        result = self._compiled_update(data)

        return {
            "loss": result["loss"],
            "byol_loss": result["byol_loss"],
            "proprio_loss": result["proprio_loss"],
            "proprio_loss_unscaled": result["proprio_loss_unscaled"],
            "cos_sim": result["cos_sim"],
            "pred_std": result["pred_std"],
            "target_std": result["target_std"],
            "grad_norm": result["grad_norm"],
        }
