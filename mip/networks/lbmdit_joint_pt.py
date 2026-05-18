"""LBMDiTJointPT: single-trunk per-token-AdaLN joint DiT over (next_state, action_chunk).

Vanilla single-stack DiT (one width ``d_model``, one stack of ``depth``
blocks) wearing the LBMDiTJointDDT conditioning surface:
- Per-token AdaLN cond throughout (DFoT-style), so position 0 and
  positions 1..Ta can receive different time modulation.
- ``s`` is the state-token flow time, ``t`` is the action flow time;
  ``t`` may be scalar ``(B,)`` or per-action-step ``(B, Ta)``.
- Optimality embedding (expert / null) folded additively into the
  per-token cond, doubling as the CFG unconditional reference at slot 1.

Differences vs ``LBMDiTJointDDT``:
- No encoder/decoder width split; one uniform width.
- No bridge re-injection; the final AdaLN layer reads the same per-token
  cond the blocks see (rather than a separately constructed ``s_dec``
  that mixed encoder features with time).

Differences vs ``LBMDiTJoint``:
- Per-token cond rows instead of one global cond vector.
- Two separate times instead of a single shared ``t``.
- Optimality enters as an additive per-token signal rather than via
  concatenation into a flat cond vector.

The block + final-layer primitives (``_DDTBlock``, ``_DDTFinalLayer``) are
imported from ``lbmdit_joint_ddt`` so per-token modulation has one source
of truth.

Author: Zilai Zeng
Date: 2026-05-18
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork
from mip.networks.lbmdit_joint_ddt import _DDTBlock, _DDTFinalLayer


class LBMDiTJointPT(BaseNetwork):
    """Single-trunk joint DiT with per-token AdaLN and decoupled state/action time.

    Token vocabulary (Ta+1 tokens): ``[state_token, action_token_0, ...,
    action_token_{Ta-1}]``. All blocks run at width ``d_model``.

    Returns:
        v_state:  (B, 1, obs_dim)
        v_action: (B, Ta, act_dim)
        scalar:   None  (interface symmetry with FlowMap / joint trunks)
    """

    EXPERT_IDX = 0
    NULL_IDX = 1  # also used by play-data conditioning at training time

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        d_model: int = 256,
        depth: int = 8,
        n_heads: int = 8,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        timestep_emb_dim: int = 128,
        opt_emb_dim: int | None = None,
        cond_compose: str = "add",
    ):
        # BaseNetwork stores act_dim / Ta / obs_dim / To / emb_dim / n_layers
        # as attrs; we use d_model as ``emb_dim`` and depth as ``n_layers``.
        super().__init__(
            act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
            emb_dim=d_model, n_layers=depth,
        )

        if cond_compose not in ("add", "concat"):
            raise ValueError(
                f"cond_compose must be 'add' or 'concat'; got {cond_compose!r}"
            )

        self.d_model = d_model
        self.depth = depth
        self._timestep_emb_dim = timestep_emb_dim
        self._opt_emb_dim = opt_emb_dim if opt_emb_dim is not None else d_model
        self.cond_compose = cond_compose
        # Cond width fed to block / final-layer AdaLN modulation.
        self._cond_dim = 3 * d_model if cond_compose == "concat" else d_model

        # --- Time embedder + projection to d_model cond width ---
        timestep_emb_params = timestep_emb_params or {}
        self.time_embedder = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
            timestep_emb_dim, **timestep_emb_params,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_emb_dim, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, d_model),
        )

        # --- Obs context flat -> d_model (broadcast across tokens) ---
        self.obs_mlp = nn.Linear(To * obs_dim, d_model)

        # --- Optimality embedding (2 slots: expert + null/play) ---
        self.optimality_embedding = nn.Embedding(2, self._opt_emb_dim)
        self.opt_mlp = nn.Linear(self._opt_emb_dim, d_model)

        # --- Input projections (state and action tokens) ---
        self.state_input_proj = nn.Linear(obs_dim, d_model)
        self.action_input_proj = nn.Linear(act_dim, d_model)

        # --- Learned 1D pos embedding over (Ta+1) tokens ---
        self.pos_embedding = nn.Parameter(
            torch.empty(1, Ta + 1, d_model).normal_(std=0.02),
        )

        # --- Single stack of per-token-AdaLN blocks ---
        self.blocks = nn.ModuleList([
            _DDTBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                cond_dim=self._cond_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # --- Output heads (per-token AdaLN final, then linear) ---
        self.state_final = _DDTFinalLayer(d_model, obs_dim, cond_dim=self._cond_dim)
        self.action_final = _DDTFinalLayer(d_model, act_dim, cond_dim=self._cond_dim)

        self._initialize_weights()

        print(
            f"number of LBMDiTJointPT parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _initialize_weights(self):
        # AdaLN-Zero: zero the last Linear of every block's modulation
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        # Final-layer AdaLN modulation + output linear: zero
        for final in (self.state_final, self.action_final):
            nn.init.constant_(final.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final.linear.weight, 0)
            nn.init.constant_(final.linear.bias, 0)

    # ------------------------------- helpers -------------------------------

    def _per_token_time_features(
        self, t_state: Tensor, t_action: Tensor, B: int,
    ) -> Tensor:
        """Build per-token time features at d_model width ``(B, Ta+1, d_model)``.

        Args:
            t_state:  ``(B,)``              flow time of the state token.
            t_action: ``(B,)`` or ``(B, Ta)`` shared or per-step action flow
                time. ``(B,)`` is broadcast across the Ta action tokens.
        """
        if t_action.dim() == 1:
            t_action_b = t_action.unsqueeze(1).expand(B, self.Ta)  # (B, Ta)
        elif t_action.dim() == 2 and t_action.shape == (B, self.Ta):
            t_action_b = t_action
        else:
            raise ValueError(
                f"t_action shape must be (B,) or (B, Ta=({self.Ta})); "
                f"got {tuple(t_action.shape)}"
            )

        t_state_b = t_state.unsqueeze(1)                              # (B, 1)
        t_per_token = torch.cat([t_state_b, t_action_b], dim=1)       # (B, Ta+1)

        # PositionalEmbedding expects 1D input; flatten then reshape.
        t_flat = t_per_token.reshape(-1)                              # (B*(Ta+1),)
        t_emb_flat = self.time_embedder(t_flat)                       # (B*(Ta+1), te_dim)
        t_emb = t_emb_flat.view(B, self.Ta + 1, -1)                   # (B, Ta+1, te_dim)
        return self.time_mlp(t_emb)                                   # (B, Ta+1, d_model)

    def _build_cond(
        self,
        time_per_tok: Tensor,
        condition: Tensor | None,
        optimality_idx: Tensor | None,
        B: int,
        device: torch.device,
    ) -> Tensor:
        """Per-token cond ``(B, Ta+1, d_model)``.

        Additive composition (DFoT pattern of additive compositional cond):
            cond = time_per_token + obs_broadcast + opt_broadcast.
        """
        if condition is not None:
            obs_feat = self.obs_mlp(condition.flatten(1))             # (B, d_model)
        else:
            obs_feat = torch.zeros(B, self.d_model, device=device)
        obs_per_tok = obs_feat.unsqueeze(1).expand(B, self.Ta + 1, -1)

        if optimality_idx is None:
            optimality_idx = torch.full(
                (B,), self.NULL_IDX, device=device, dtype=torch.long,
            )
        opt_feat = self.opt_mlp(self.optimality_embedding(optimality_idx))
        opt_per_tok = opt_feat.unsqueeze(1).expand(B, self.Ta + 1, -1)

        if self.cond_compose == "concat":
            # (B, Ta+1, 3 * d_model) — each component keeps its own subspace.
            return torch.cat([time_per_tok, obs_per_tok, opt_per_tok], dim=-1)
        return time_per_tok + obs_per_tok + opt_per_tok

    # ------------------------------- forward -------------------------------

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
            x_state:        ``(B, 1, obs_dim)``    noisy next-state embedding.
            x_action:       ``(B, Ta, act_dim)``   noisy action chunk.
            s:              ``(B,)``               state-token flow time.
            t:              ``(B,)`` or ``(B, Ta)`` action flow time(s).
            condition:      ``(B, To, obs_dim)``   encoded current obs, or None.
            optimality_idx: ``(B,)`` long in {0=expert, 1=null/play}, or None
                (defaults to null).

        Returns:
            v_state:  ``(B, 1, obs_dim)``
            v_action: ``(B, Ta, act_dim)``
            scalar:   ``None``  (interface symmetry)
        """
        B = x_action.shape[0]
        device = x_action.device

        # --- 1. Per-token time features ---
        time_per_tok = self._per_token_time_features(s, t, B)  # (B, Ta+1, d_model)

        # --- 2. Per-token cond ---
        cond = self._build_cond(
            time_per_tok, condition, optimality_idx, B, device,
        )                                                       # (B, Ta+1, d_model)

        # --- 3. Tokenize x ---
        h = torch.cat([
            self.state_input_proj(x_state),     # (B, 1, d_model)
            self.action_input_proj(x_action),   # (B, Ta, d_model)
        ], dim=1)                                # (B, Ta+1, d_model)
        h = h + self.pos_embedding[:, : h.shape[1], :]

        # --- 4. Single stack of per-token-AdaLN blocks ---
        for blk in self.blocks:
            h = blk(h, cond)

        # --- 5. Output heads with per-token AdaLN final layer ---
        v_state = self.state_final(h[:, :1, :], cond[:, :1, :])     # (B, 1, obs_dim)
        v_action = self.action_final(h[:, 1:, :], cond[:, 1:, :])   # (B, Ta, act_dim)
        return v_state, v_action, None


# ----------------------------- smoke tests --------------------------------

def test_lbmdit_joint_pt():
    print("=" * 50)
    print("Testing LBMDiTJointPT")
    print("=" * 50)

    obs_dim = 64
    act_dim = 7
    Ta = 8
    To = 2
    B = 4

    model = LBMDiTJointPT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        d_model=128, depth=4, n_heads=4, timestep_emb_dim=64,
    )

    x_state = torch.randn(B, 1, obs_dim)
    x_action = torch.randn(B, Ta, act_dim)
    t_state = torch.rand(B)
    t_action = torch.rand(B)
    cond = torch.randn(B, To, obs_dim)

    # Independent scalar t_state / t_action
    v_state, v_action, scalar = model(x_state, x_action, t_state, t_action, cond)
    assert v_state.shape == (B, 1, obs_dim)
    assert v_action.shape == (B, Ta, act_dim)
    assert scalar is None
    print(f"[independent t]   v_state {v_state.shape}, v_action {v_action.shape}")

    # Per-action-step t
    t_action_per_step = torch.rand(B, Ta)
    v_s_p, v_a_p, _ = model(x_state, x_action, t_state, t_action_per_step, cond)
    assert v_s_p.shape == (B, 1, obs_dim)
    assert v_a_p.shape == (B, Ta, act_dim)
    print(f"[per-step t]      v_state {v_s_p.shape}, v_action {v_a_p.shape}")

    # Diagonal (t_state == t_action) with expert label
    t_diag = torch.rand(B)
    opt = torch.zeros(B, dtype=torch.long)
    v_s_d, v_a_d, _ = model(x_state, x_action, t_diag, t_diag, cond, opt)
    print(f"[diagonal expert] v_state {v_s_d.shape}, v_action {v_a_d.shape}")

    # Mixed batch (half expert, half null/play)
    opt_mixed = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    v_s_m, v_a_m, _ = model(x_state, x_action, t_state, t_action, cond, opt_mixed)
    print(f"[mixed]           v_state {v_s_m.shape}, v_action {v_a_m.shape}")

    # No condition
    v_s_nc, v_a_nc, _ = model(x_state, x_action, t_state, t_action, None, None)
    print(f"[no cond]         v_state {v_s_nc.shape}, v_action {v_a_nc.shape}")

    # Gradient flow
    model.train()
    v_state, v_action, _ = model(x_state, x_action, t_state, t_action, cond, opt)
    (v_state.sum() + v_action.sum()).backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert has_grad
    print("Gradients flow through both heads.")

    # AdaLN-Zero init: untrained outputs should be exactly zero
    model_init = LBMDiTJointPT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        d_model=64, depth=2, n_heads=4,
    )
    model_init.eval()
    with torch.no_grad():
        v_s, v_a, _ = model_init(x_state, x_action, t_state, t_action, cond)
    assert torch.allclose(v_s, torch.zeros_like(v_s)), \
        "state final should be zero at init"
    assert torch.allclose(v_a, torch.zeros_like(v_a)), \
        "action final should be zero at init"
    print("AdaLN-Zero init verified (final outputs are zero at init).")

    print("=" * 50)
    print("LBMDiTJointPT test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_lbmdit_joint_pt()
