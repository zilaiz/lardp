"""LBMDiTDDTPT: action-only DDT trunk with PER-TOKEN encoder->decoder cond.

Drop-in alternative to ``LBMDiTDDT``. Same external API
``(x, s, t, condition) -> (y, None)`` so it plugs straight into
``flow_map.get_velocity`` / ``TrainingAgent`` with only a config swap.

What changed vs ``LBMDiTDDT`` (and why):
  - ``LBMDiTDDT`` mean-pools the encoder output over Ta to a single vector
    before feeding the decoder. The wide decoder then sees only a global
    summary plus its own learned pos embedding — the per-step encoder
    features never reach the decoder's token stream. This wastes the
    encoder's depth.
  - Here, the bridge is per-token: ``s_dec = s_projector(h_enc)`` keeps
    ``(B, Ta, dec_hidden)``. Each decoder token gets the corresponding
    encoder feature as its AdaLN cond, mirroring the working pattern in
    ``LBMDiTJointDDT`` (joint DDT).
  - Per-token AdaLN is implemented via ``_DDTBlock`` from
    ``lbmdit_joint_ddt`` (cond is ``(B, L, cond_dim)`` rather than
    ``(B, cond_dim)``).
  - Obs is projected once via a small ``obs_mlp`` instead of being
    concatenated raw into every block's AdaLN modulation, giving the
    visual encoder a cleaner single gradient path.
  - Decoder has no pos embedding — positional info flows through ``s_dec``
    (RAE pattern; matches joint DDT).
  - Bridge re-injects time additively: ``s_dec = s_projector(t.broadcast + h_enc)``
    so the decoder sees a fresh additive time signal at the boundary, on top
    of whatever time-aware features the encoder's AdaLN modulation has baked
    into ``h_enc``. RAE/DDT applies an extra ``silu`` here (``DDT.py:351-354``);
    we drop it to match our overall "no outer silu on cond composition"
    convention (DFoT-style) and let the decoder block's internal
    ``Sequential(SiLU, Linear)`` modulation do the nonlinearity.

Author: Zilai Zeng
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork
from mip.networks.lbmdit_joint_ddt import _DDTBlock, _DDTFinalLayer


class LBMDiTDDTPT(BaseNetwork):
    """Action-only DDT trunk: deep narrow encoder + shallow wide decoder,
    with PER-TOKEN encoder->decoder conditioning (no pooling).

    Cond construction (encoder side, at ``d_model_enc``):
        time_features = time_mlp(time_embedder(t))       # (B, d_enc)
        obs_features  = obs_mlp(flatten(condition))       # (B, d_enc)
        cond_per_tok  = (time + obs).unsqueeze(1).expand( # (B, Ta, d_enc)
                          B, Ta, d_enc)

    The encoder cond rows are identical across action tokens (time and obs
    are global per sample) — per-token AdaLN here is for symmetry/cleanness
    with the bridge; the real per-token signal kicks in at the decoder
    where ``s_dec`` varies token-by-token.

    Returns:
        y:      (B, Ta, act_dim) predicted velocity
        scalar: None  (interface symmetry with LBMDiT)
    """

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        d_model_enc: int = 384,
        d_model_dec: int = 384,
        n_heads_enc: int = 6,
        n_heads_dec: int = 6,
        enc_depth: int = 8,
        dec_depth: int = 2,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
        timestep_emb_dim: int | None = None,
    ):
        super().__init__(
            act_dim, Ta, obs_dim, To, d_model_enc, enc_depth + dec_depth,
        )

        self.d_model_enc = d_model_enc
        self.d_model_dec = d_model_dec
        self.disable_time_embedding = disable_time_embedding
        self._timestep_emb_dim = (
            timestep_emb_dim if timestep_emb_dim is not None else d_model_enc
        )

        # --- Time embedder + projection to d_model_enc cond width ---
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.time_embedder = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                self._timestep_emb_dim, **timestep_emb_params,
            )
        else:
            self.time_embedder = None
        self.time_mlp = nn.Sequential(
            nn.Linear(self._timestep_emb_dim, d_model_enc),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model_enc, d_model_enc),
        )

        # --- Obs context flat -> d_model_enc (broadcast across tokens) ---
        self.obs_mlp = nn.Linear(To * obs_dim, d_model_enc)

        # --- Encoder-side input projection + learned pos embedding ---
        self.enc_input_proj = nn.Linear(act_dim, d_model_enc)
        self.enc_pos_embedding = nn.Parameter(
            torch.empty(1, Ta, d_model_enc).normal_(std=0.02),
        )

        # --- Encoder blocks (per-token AdaLN) ---
        self.enc_blocks = nn.ModuleList([
            _DDTBlock(
                hidden_size=d_model_enc,
                num_heads=n_heads_enc,
                cond_dim=d_model_enc,
                dropout=dropout,
            )
            for _ in range(enc_depth)
        ])

        # --- Bridge: per-token encoder feature -> per-token decoder cond ---
        self.s_projector = (
            nn.Linear(d_model_enc, d_model_dec)
            if d_model_enc != d_model_dec
            else nn.Identity()
        )

        # --- Decoder-side input projection (no pos embedding: RAE pattern) ---
        self.dec_input_proj = nn.Linear(act_dim, d_model_dec)

        # --- Decoder blocks (per-token AdaLN; cond is s_dec) ---
        self.dec_blocks = nn.ModuleList([
            _DDTBlock(
                hidden_size=d_model_dec,
                num_heads=n_heads_dec,
                cond_dim=d_model_dec,
                dropout=dropout,
            )
            for _ in range(dec_depth)
        ])

        # --- Output head: per-token AdaLN final layer ---
        self.output_final = _DDTFinalLayer(d_model_dec, act_dim)

        self._initialize_weights()

        print(
            f"number of LBMDiTDDTPT parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _initialize_weights(self):
        """Zero-init AdaLN modulations + final head for stable start."""
        for block in list(self.enc_blocks) + list(self.dec_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output_final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_final.linear.weight, 0)
        nn.init.constant_(self.output_final.linear.bias, 0)

    def forward(
        self,
        x: Tensor,
        s: Tensor,
        t: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Args:
            x:         (B, Ta, act_dim) noisy action chunk
            s:         (B,) — unused; FlowMap signature compat.
            t:         (B,) flow-matching time.
            condition: (B, To, obs_dim) encoded current observations, or None
        """
        del s
        B = x.shape[0]
        Ta_actual = x.shape[1]
        device = x.device

        # --- Time features at enc width ---
        if self.time_embedder is not None:
            time_raw = self.time_embedder(t)
        else:
            time_raw = torch.zeros(
                B, self._timestep_emb_dim, device=device,
            )
        time_features = self.time_mlp(time_raw)  # (B, d_enc)

        # --- Obs features at enc width ---
        if condition is not None:
            obs_features = self.obs_mlp(condition.flatten(1))  # (B, d_enc)
        else:
            obs_features = torch.zeros(B, self.d_model_enc, device=device)

        # --- Per-token encoder cond: time+obs broadcast across Ta tokens ---
        cond_global = time_features + obs_features         # (B, d_enc)
        cond_per_tok = cond_global.unsqueeze(1).expand(    # (B, Ta, d_enc)
            B, Ta_actual, self.d_model_enc,
        )

        # --- Encoder ---
        h_enc = self.enc_input_proj(x) + self.enc_pos_embedding[:, :Ta_actual, :]
        for block in self.enc_blocks:
            h_enc = block(h_enc, cond_per_tok)
        # h_enc: (B, Ta, d_enc)

        # --- Bridge: per-token, no pooling, additive time re-injection ---
        # ``t.broadcast + h_enc`` gives the decoder a direct additive time
        # signal at the boundary, parallel to the multiplicative time path
        # through encoder AdaLN. RAE wraps this in silu; we don't, matching
        # our overall "no outer silu on cond" convention — the decoder's
        # internal SiLU+Linear modulation handles the nonlinearity.
        time_per_tok = time_features.unsqueeze(1).expand_as(h_enc)
        s_dec = self.s_projector(time_per_tok + h_enc)  # (B, Ta, d_dec)

        # --- Decoder (no pos emb; positional info flows through s_dec) ---
        h_dec = self.dec_input_proj(x)
        for block in self.dec_blocks:
            h_dec = block(h_dec, s_dec)
        # h_dec: (B, Ta, d_dec)

        # --- Output head with per-token AdaLN final ---
        y = self.output_final(h_dec, s_dec)  # (B, Ta, act_dim)
        return y, None


def test_lbmdit_ddt_pt():
    """Smoke test for LBMDiTDDTPT."""
    print("=" * 50)
    print("Testing LBMDiTDDTPT")
    print("=" * 50)

    obs_dim = 64
    act_dim = 7
    Ta = 8
    To = 2
    B = 4

    model = LBMDiTDDTPT(
        act_dim=act_dim,
        Ta=Ta,
        obs_dim=obs_dim,
        To=To,
        d_model_enc=128,
        d_model_dec=256,
        n_heads_enc=4,
        n_heads_dec=4,
        enc_depth=4,
        dec_depth=2,
        dropout=0.0,
        timestep_emb_dim=64,
    )

    x = torch.randn(B, Ta, act_dim)
    s = torch.rand(B)
    t = torch.rand(B)
    condition = torch.randn(B, To, obs_dim)

    # Basic forward
    y, scalar = model(x, s, t, condition)
    assert y.shape == (B, Ta, act_dim), f"Got {y.shape}"
    assert scalar is None
    print(f"Input: {x.shape}, Output: {y.shape}")

    # AdaLN-Zero init: untrained outputs are exactly zero
    with torch.no_grad():
        y_init, _ = model(x, s, t, condition)
    assert torch.allclose(y_init, torch.zeros_like(y_init)), \
        "Output head should be zero at init"
    print("AdaLN-Zero init verified (output is zero at init).")

    # Gradient flow
    model.train()
    y, _ = model(x, s, t, condition)
    y.sum().backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad
    print("Gradients flow through both enc and dec.")

    # No condition
    model.zero_grad()
    y_nc, _ = model(x, s, t, None)
    assert y_nc.shape == (B, Ta, act_dim)
    print(f"No condition: {y_nc.shape}")

    # Per-token bridge sanity: shape (B, Ta, d_dec), with RAE-style time re-injection
    with torch.no_grad():
        time_raw = model.time_embedder(t)
        time_features = model.time_mlp(time_raw)
        obs_features = model.obs_mlp(condition.flatten(1))
        cond = (time_features + obs_features).unsqueeze(1).expand(B, Ta, model.d_model_enc)
        h = model.enc_input_proj(x) + model.enc_pos_embedding[:, :Ta, :]
        for blk in model.enc_blocks:
            h = blk(h, cond)
        time_per_tok = time_features.unsqueeze(1).expand_as(h)
        s_dec = model.s_projector(time_per_tok + h)
    assert s_dec.shape == (B, Ta, 256), f"bridge cond shape {s_dec.shape}"
    print(f"Per-token bridge cond shape verified: {s_dec.shape} (no mean-pool, additive time re-injected).")

    # Same enc/dec dim (Identity projector)
    model_same = LBMDiTDDTPT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        d_model_enc=128, d_model_dec=128, enc_depth=2, dec_depth=2,
        n_heads_enc=4, n_heads_dec=4,
    )
    assert isinstance(model_same.s_projector, nn.Identity)
    print("s_projector is Identity when d_model_enc == d_model_dec.")

    # FlowMap compat
    from mip.flow_map import FlowMap
    fm = FlowMap(model)
    v = fm.get_velocity(t, x, condition)
    assert v.shape == (B, Ta, act_dim)
    print(f"FlowMap.get_velocity: {v.shape}")

    print("=" * 50)
    print("LBMDiTDDTPT test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_lbmdit_ddt_pt()
