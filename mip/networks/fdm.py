"""Forward-dynamics network for BYOL-style encoder pretraining.

Online branch: pool the encoded obs frames into a single obs_summary, project
the clean action chunk to an action embedding, concat them, and run a
dynamics-predictor MLP to predict the next-state embedding z_{s'}.

Target branch: no module lives here. The agent encodes s' with the EMA target
encoder directly and stop-grads it; there is no projector and no second
predictor head (the asymmetry already comes from the dynamics predictor on
the online side conditioning on the action).

Shape conventions for ``obs_summarizer`` and ``action_proj`` match
``LBMDiTIDMv2`` so the obs encoder pretrained with this network can be
loaded by downstream goal-cond IDM / joint DDT agents without surprises.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mip.networks.base import BaseNetwork


class FDMNet(BaseNetwork):
    """Online-side modules for forward-dynamics self-prediction."""

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        To_obs: int,
        summarizer_hidden: int | None = None,
        action_proj_hidden: int | None = None,
        predictor_hidden: int | None = None,
        proprio_dim: int = 0,
        proprio_decoder_hidden: int | None = None,
    ):
        # BaseNetwork wants (emb_dim, n_layers) for legacy reasons; we don't
        # use them — pass obs_dim and 0 to keep the contract.
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim=obs_dim, n_layers=0)
        self.To_obs = To_obs
        self.proprio_dim = proprio_dim

        os_hidden = summarizer_hidden or (2 * obs_dim)
        self.obs_summarizer = nn.Sequential(
            nn.Linear(To_obs * obs_dim, os_hidden),
            nn.GELU(),
            nn.Linear(os_hidden, obs_dim),
        )

        ap_hidden = action_proj_hidden or (2 * obs_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(Ta * act_dim, ap_hidden),
            nn.GELU(),
            nn.Linear(ap_hidden, obs_dim),
        )

        ph = predictor_hidden or (2 * obs_dim)
        # LayerNorm in the middle stabilizes the dynamics MLP — important
        # because the loss is on L2-normalized vectors and we want the MLP
        # to be well-behaved before the normalize step.
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(2 * obs_dim, ph),
            nn.LayerNorm(ph),
            nn.GELU(),
            nn.Linear(ph, obs_dim),
        )

        # Optional proprioception decoder: predicts the per-frame low-dim
        # observation from each encoded frame. When enabled, the
        # agent-side recon loss provides a non-collapsible supervised
        # signal on the encoder. Conditionally instantiated so disabled
        # runs add zero params.
        if proprio_dim > 0:
            pd_hidden = proprio_decoder_hidden or (2 * obs_dim)
            self.proprio_decoder = nn.Sequential(
                nn.Linear(obs_dim, pd_hidden),
                nn.LayerNorm(pd_hidden),
                nn.GELU(),
                nn.Linear(pd_hidden, proprio_dim),
            )
        else:
            self.proprio_decoder = None

        print(
            f"number of FDMNet parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def forward_online(self, z_obs: Tensor, action: Tensor) -> Tensor:
        """Predict z_{s'} from encoded obs frames and the action chunk.

        Args:
            z_obs:  (B, To_obs, obs_dim) encoded current observation frames.
            action: (B, Ta, act_dim) clean expert action chunk (normalized).
        Returns:
            z_pred: (B, obs_dim) predicted next-state embedding.
        """
        obs_summary = self.obs_summarizer(z_obs[:, : self.To_obs].flatten(1))
        action_emb = self.action_proj(action.flatten(1))
        return self.dynamics_predictor(
            torch.cat([obs_summary, action_emb], dim=-1)
        )

    def decode_proprio(self, z_obs: Tensor) -> Tensor:
        """Predict per-frame proprioception from encoded latents.

        Args:
            z_obs: (B, To_obs, obs_dim) encoded observation frames.
        Returns:
            pred_proprio: (B, To_obs, proprio_dim) per-frame prediction.
        Raises:
            RuntimeError: if proprio_decoder was not constructed
                (proprio_dim == 0 at __init__).
        """
        if self.proprio_decoder is None:
            raise RuntimeError(
                "FDMNet was built with proprio_dim=0; proprio_decoder is None. "
                "Set proprio_dim > 0 at construction to enable proprio recon."
            )
        return self.proprio_decoder(z_obs)

    def forward(self, x=None, s=None, t=None, condition=None):
        raise NotImplementedError(
            "FDMNet has no action-flow forward pass; use forward_online."
        )
