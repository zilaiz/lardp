"""LBMDiTJoint: single-stage DiT that flow-matches (next_state, action_chunk).

Token vocabulary:
    [state_token, action_token_0, ..., action_token_{Ta-1}]
    - state_token comes from `state_input_proj(x_state)` where x_state is a
      (B, 1, obs_dim) noisy next-state embedding (in encoder space).
    - action tokens come from `action_input_proj(x_action)` where x_action is
      a (B, Ta, act_dim) noisy action chunk.
A single learned positional embedding of shape (1, Ta+1, d_model) is added to
the concatenated token sequence. Two output heads (state / action) recover
per-stream velocities at the original dims.

AdaLN conditioning vector:
    cond = concat(time_features, obs_flat, optimality_emb)
    cond_dim = timestep_emb_dim + To*obs_dim + opt_emb_dim
where:
    - `time_features` is `time_mlp(time_embedder(t))` (single t embedding,
      decoupled `timestep_emb_dim` from `d_model`).
    - `obs_flat` is the flattened encoded current obs window (B, To*obs_dim).
    - `optimality_emb` is `nn.Embedding(2, opt_emb_dim)` with slot 0 = expert,
      slot 1 = null/play (doubles as the unconditional reference for CFG).

Author: Zilai Zeng
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork
from mip.networks.lbmdit import _TransformerBlock


class LBMDiTJoint(BaseNetwork):
    """Joint DiT trunk over (next_state, action_chunk).

    The action-side path mirrors LBMDiT (decoder-only DiT with AdaLN-Zero blocks,
    no RoPE). The state-side path adds a single token at position 0, sharing the
    transformer trunk and AdaLN cond with the action tokens. An optimality
    label (expert vs. null/play) is concatenated into the AdaLN cond, with slot
    1 doubling as the CFG unconditional reference — no separate null slot.

    Returns:
        v_state:  (B, 1, obs_dim)  — predicted velocity for next state
        v_action: (B, Ta, act_dim) — predicted velocity for action chunk
        scalar:   None             — placeholder for interface symmetry
    """

    EXPERT_IDX = 0
    NULL_IDX = 1  # also used by play-data conditioning at training time

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
        timestep_emb_dim: int | None = None,
        opt_emb_dim: int | None = None,
    ):
        super().__init__(act_dim, Ta, obs_dim, To, d_model, depth)

        self.d_model = d_model
        self.disable_time_embedding = disable_time_embedding
        self._timestep_emb_dim = (
            timestep_emb_dim if timestep_emb_dim is not None else d_model
        )
        self._opt_emb_dim = opt_emb_dim if opt_emb_dim is not None else obs_dim

        # --- Single t embedding (s ignored: callers always pass s == t) ---
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.time_embedder = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                self._timestep_emb_dim, **timestep_emb_params,
            )
        else:
            self.time_embedder = None

        self.time_mlp = nn.Sequential(
            nn.Linear(self._timestep_emb_dim, 2 * self._timestep_emb_dim),
            nn.GELU(),
            nn.Linear(2 * self._timestep_emb_dim, self._timestep_emb_dim),
            nn.GELU(),
        )

        # --- Optimality embedding (2 slots: expert + null/play) ---
        self.optimality_embedding = nn.Embedding(2, self._opt_emb_dim)

        # --- AdaLN cond dim ---
        cond_dim = self._timestep_emb_dim + obs_dim * To + self._opt_emb_dim

        # --- Heterogeneous input projections ---
        self.state_input_proj = nn.Linear(obs_dim, d_model)
        self.action_input_proj = nn.Linear(act_dim, d_model)

        # --- Shared positional embedding over (state, action) tokens ---
        self.pos_embedding = nn.Parameter(
            torch.empty(1, Ta + 1, d_model).normal_(std=0.02),
        )

        # --- Transformer trunk ---
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # --- Heterogeneous output heads ---
        self.state_output_proj = nn.Linear(d_model, obs_dim)
        self.action_output_proj = nn.Linear(d_model, act_dim)

        self._initialize_weights()

        print(
            f"number of LBMDiTJoint parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _build_cond(
        self,
        t: Tensor,
        condition: Tensor | None,
        optimality_idx: Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        # Time
        if self.time_embedder is not None:
            time_features = self.time_mlp(self.time_embedder(t))
        else:
            time_features = torch.zeros(
                batch_size, self._timestep_emb_dim, device=device,
            )

        # Obs flat
        if condition is not None:
            cond_flat = torch.flatten(condition, 1)
        else:
            cond_flat = torch.zeros(
                batch_size, self.obs_dim * self.To, device=device,
            )

        # Optimality (default to null if not provided)
        if optimality_idx is None:
            optimality_idx = torch.full(
                (batch_size,), self.NULL_IDX, device=device, dtype=torch.long,
            )
        opt_emb = self.optimality_embedding(optimality_idx)

        return torch.cat([time_features, cond_flat, opt_emb], dim=-1)

    def forward(
        self,
        x_state: Tensor,
        x_action: Tensor,
        s: Tensor,
        t: Tensor,
        condition: Tensor | None = None,
        optimality_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, None]:
        """Args:
            x_state:        (B, 1, obs_dim)  noisy next-state embedding
            x_action:       (B, Ta, act_dim) noisy action chunk
            s:              (B,) — unused (kept for FlowMap signature symmetry)
            t:              (B,) flow-matching time
            condition:      (B, To, obs_dim) encoded current obs, or None
            optimality_idx: (B,) long in {0=expert, 1=null/play}, or None (=null)

        Returns:
            v_state:  (B, 1, obs_dim)
            v_action: (B, Ta, act_dim)
            scalar:   None
        """
        del s  # callers always pass s == t in this codebase
        B = x_action.shape[0]
        device = x_action.device

        cond_vec = self._build_cond(t, condition, optimality_idx, B, device)

        # Build token sequence
        h_state = self.state_input_proj(x_state)              # (B, 1, d_model)
        h_action = self.action_input_proj(x_action)           # (B, Ta, d_model)
        h = torch.cat([h_state, h_action], dim=1)             # (B, Ta+1, d_model)
        h = h + self.pos_embedding[:, : h.shape[1], :]

        for block in self.blocks:
            h = block(h, cond_vec)

        v_state = self.state_output_proj(h[:, :1, :])         # (B, 1, obs_dim)
        v_action = self.action_output_proj(h[:, 1:, :])       # (B, Ta, act_dim)
        return v_state, v_action, None


def test_lbmdit_joint():
    """Smoke test for LBMDiTJoint."""
    print("=" * 50)
    print("Testing LBMDiTJoint")
    print("=" * 50)

    obs_dim = 64
    act_dim = 7
    Ta = 8
    To = 2
    B = 4

    model = LBMDiTJoint(
        act_dim=act_dim,
        Ta=Ta,
        obs_dim=obs_dim,
        To=To,
        d_model=128,
        n_heads=4,
        depth=4,
        dropout=0.0,
        timestep_emb_dim=64,
        opt_emb_dim=32,
    )

    x_state = torch.randn(B, 1, obs_dim)
    x_action = torch.randn(B, Ta, act_dim)
    t = torch.rand(B)
    cond = torch.randn(B, To, obs_dim)

    # Plain forward (default null optimality)
    v_state, v_action, scalar = model(x_state, x_action, t, t, cond)
    assert v_state.shape == (B, 1, obs_dim)
    assert v_action.shape == (B, Ta, act_dim)
    assert scalar is None
    print(f"v_state: {v_state.shape}, v_action: {v_action.shape}")

    # Expert label
    opt = torch.zeros(B, dtype=torch.long)
    v_state_e, v_action_e, _ = model(x_state, x_action, t, t, cond, opt)
    print(f"expert: v_state={v_state_e.shape}, v_action={v_action_e.shape}")

    # Mixed batch (half expert, half null/play)
    opt_mixed = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    v_state_m, v_action_m, _ = model(x_state, x_action, t, t, cond, opt_mixed)
    print(f"mixed: v_state={v_state_m.shape}, v_action={v_action_m.shape}")

    # Gradient flow
    model.train()
    v_state, v_action, _ = model(x_state, x_action, t, t, cond, opt)
    (v_state.sum() + v_action.sum()).backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert has_grad
    print("Gradients flow through both heads.")

    # No condition
    model.zero_grad()
    v_s_nc, v_a_nc, _ = model(x_state, x_action, t, t, None, None)
    assert v_s_nc.shape == (B, 1, obs_dim)
    print(f"no condition: v_state={v_s_nc.shape}, v_action={v_a_nc.shape}")

    print("=" * 50)
    print("LBMDiTJoint test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_lbmdit_joint()
