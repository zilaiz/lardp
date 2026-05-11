"""Deterministic goal-embedding regressor.

A small pre-norm transformer that takes the To-frame obs embedding sequence
``z_t`` (B, To, obs_dim) and predicts a single goal embedding (B, 1, obs_dim)
via a learnable [GOAL] query token. Intended as a non-flow alternative to
``GoalPredictorDiT``: same conditioning interface (consume ``z_t``, emit a
single goal token), but trained with direct L2 on the expert goal embedding
plus an action-flow loss through a frozen IDM, instead of FM in the latent.

No time embedding, no flow-matching input, no AdaLN. This matches our
matched-parameter A/B against ``GoalPredictorDiT``.
"""

import torch
import torch.nn as nn
from torch import Tensor

from mip.networks.base import BaseNetwork


class _SelfAttnBlock(nn.Module):
    """Plain pre-norm transformer block (MHA + MLP), no AdaLN, no cross-attn."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, h: Tensor) -> Tensor:
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + attn_out
        h = h + self.mlp(self.norm2(h))
        return h


class GoalPredictorRegressor(BaseNetwork):
    """Deterministic regressor: z_t (B, To, obs_dim) -> g_hat (B, 1, obs_dim).

    Args:
        act_dim: goal embedding dimension (= encoder output dim).
            Kept named ``act_dim`` for ``BaseNetwork`` compatibility; semantically
            this is the *goal* dimension.
        Ta: number of goal tokens emitted (always 1 in the current pipeline).
        obs_dim: observation embedding dimension.
        To: number of obs frames.
        d_model: transformer hidden dim.
        n_heads: attention heads.
        depth: number of transformer blocks.
        dropout: dropout rate.
        mlp_ratio: MLP expansion factor inside each block.
    """

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 6,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
    ):
        super().__init__(act_dim, Ta, obs_dim, To, d_model, depth)

        self.d_model = d_model

        # Project obs embeddings into transformer hidden dim.
        self.input_proj = nn.Linear(obs_dim, d_model)

        # Learnable positional embeddings for the To obs tokens.
        self.pos_embedding = nn.Parameter(
            torch.empty(1, To, d_model).normal_(std=0.02),
        )

        # Learnable goal-query tokens (Ta of them; Ta==1 in our pipeline).
        self.goal_query = nn.Parameter(
            torch.empty(1, Ta, d_model).normal_(std=0.02),
        )

        self.blocks = nn.ModuleList([
            _SelfAttnBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, act_dim)

        print(
            f"number of GoalPredictorRegressor parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def forward(self, condition: Tensor) -> Tensor:
        """Predict the goal embedding from the obs sequence.

        Args:
            condition: (B, To, obs_dim) encoded current observations.

        Returns:
            g_hat: (B, Ta, act_dim) predicted goal embedding(s).
        """
        if condition.shape[1] != self.To:
            raise ValueError(
                f"GoalPredictorRegressor expects condition with To={self.To} frames, "
                f"got {condition.shape[1]}."
            )

        B = condition.shape[0]

        h_obs = self.input_proj(condition) + self.pos_embedding  # (B, To, d_model)
        q = self.goal_query.expand(B, -1, -1)                    # (B, Ta, d_model)
        h = torch.cat([h_obs, q], dim=1)                         # (B, To+Ta, d_model)

        for block in self.blocks:
            h = block(h)

        h_goal = self.norm_out(h[:, self.To :])                  # (B, Ta, d_model)
        g_hat = self.output_proj(h_goal)                         # (B, Ta, act_dim)
        return g_hat


def _self_test():
    emb_dim = 128
    To = 2
    Ta = 1
    B = 4

    model = GoalPredictorRegressor(
        act_dim=emb_dim,
        Ta=Ta,
        obs_dim=emb_dim,
        To=To,
        d_model=emb_dim,
        n_heads=4,
        depth=3,
        dropout=0.0,
    )

    z_t = torch.randn(B, To, emb_dim)
    g = model(z_t)
    assert g.shape == (B, Ta, emb_dim), g.shape

    target = torch.randn(B, Ta, emb_dim)
    loss = (g - target).pow(2).mean()
    loss.backward()
    grad_norms = [
        p.grad.detach().norm() for p in model.parameters() if p.grad is not None
    ]
    assert any(g_.item() > 0 for g_ in grad_norms), "no gradient flowed"

    print(
        f"GoalPredictorRegressor self-test passed: "
        f"{sum(p.numel() for p in model.parameters())} params"
    )


if __name__ == "__main__":
    _self_test()
