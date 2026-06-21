"""LBMDiTJointPTAgent: single-trunk variant of LBMDiTJointDDTAgent.

Identical training / sampling surface to ``LBMDiTJointDDTAgent`` —
decoupled state/action flow times, SD3 shifts, ``joint_t_schedule``,
optimality + CFG, ``joint_state_loss_to_encoder``, target LayerNorm,
encoder EMA — but the joint trunk is the single-stack
``LBMDiTJointPT`` (one width, one depth) instead of the DDT
encoder/decoder split.

All update / sample / save / load / EMA logic is inherited from
``LBMDiTJointDDTAgent``; this subclass only overrides ``_build_net``
to swap the trunk constructor.

Author: Zilai Zeng
Date: 2026-05-18
"""

from __future__ import annotations

import torch.nn as nn

from mip.agent_lbmdit_joint_ddt import LBMDiTJointDDTAgent
from mip.config import Config
from mip.networks.lbmdit_joint_pt import LBMDiTJointPT


class LBMDiTJointPTAgent(LBMDiTJointDDTAgent):
    """Single-trunk joint agent. See module docstring."""

    def _build_net(self, config: Config, obs_dim: int, device) -> nn.Module:
        return LBMDiTJointPT(
            act_dim=config.task.act_dim,
            Ta=config.task.horizon,
            obs_dim=obs_dim,
            To=config.task.obs_steps,
            d_model=config.network.emb_dim,
            depth=config.network.num_layers,
            n_heads=config.network.n_heads,
            dropout=config.network.dropout,
            timestep_emb_type=config.network.timestep_emb_type,
            timestep_emb_dim=config.network.timestep_emb_dim,
            opt_emb_dim=config.network.joint_opt_emb_dim,
            # Same backward-compat fallback as ``LBMDiTJointDDTAgent._build_net``:
            # older configs without ``joint_cond_compose`` default to "add".
            cond_compose=getattr(config.network, "joint_cond_compose", "add"),
            # Decouple the state/target dim from obs_dim when set (e.g. a
            # foreign target encoder of a different dim); None -> obs_dim.
            state_dim=getattr(config.network, "state_target_dim", None) or obs_dim,
        ).to(device)
