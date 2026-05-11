"""LBMDiTDDT: action-only DDT-style policy trunk.

Counterpart to ``GoalPredictorDiTDDTNS`` but for *action* prediction
instead of goal embedding prediction. Drops the joint-trunk machinery
(no state token, no per-token AdaLN, no decoupled t) — at this scale
they are wasted complexity for a plain BC policy.

Architecture (RAE-style encoder/decoder width split with global cond):
    h_enc = enc_input_proj(noisy_action_chunk) + pos_emb
    enc_cond = concat(time_features, obs_flat)
    for blk in enc_blocks: h_enc = blk(h_enc, enc_cond)        # (B, Ta, enc_h)
    s_out  = squeeze(s_projector(h_enc).mean(dim=1)) ? — NO, see note below
    dec_cond = concat(time_proj(time_features), s_proj(h_enc_pooled))
    h_dec = dec_input_proj(noisy_action_chunk)
    for blk in dec_blocks: h_dec = blk(h_dec, dec_cond)         # (B, Ta, dec_h)
    y = output_proj(h_dec)                                       # (B, Ta, act_dim)

Pooling note: the goal-predictor DDT-NS sibling has Ta=1, so
``s_projector(h_enc).squeeze(1)`` collapses cleanly. With Ta>1 (action
chunks) we need to pool across the action sequence to produce a *single*
global cond vector for the decoder. We use **mean pooling over time**;
this matches the standard "summarize encoder into one cond" pattern from
RAE/DDT and keeps the decoder cond shape global ``(B, 2*d_model_dec)``,
which is the shape the AdaLN modulation in ``_TransformerBlock`` expects.

The network is **noise-shift agnostic** — it just receives a t value.
The shift is applied in the loss (training) and sampler (inference).

Author: Zilai Zeng
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork
from mip.networks.lbmdit import _TransformerBlock


class LBMDiTDDT(BaseNetwork):
    """Action-only DDT trunk: deep narrow encoder + shallow wide decoder.

    Cond is GLOBAL per-sample (``(B, cond_dim)``) — not per-token. Single
    flow time. Returns 2-tuple ``(y, None)`` matching the LBMDiT API so it
    is drop-in for ``flow_map.get_velocity`` and the existing loss/sampler
    paths used by ``TrainingAgent``.
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

        # --- Single t embedding ---
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.time_embedder = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                self._timestep_emb_dim, **timestep_emb_params,
            )
        else:
            self.time_embedder = None

        # --- Time MLP: timestep_emb_dim -> d_model_enc ---
        self.time_mlp = nn.Sequential(
            nn.Linear(self._timestep_emb_dim, 2 * d_model_enc),
            nn.GELU(),
            nn.Linear(2 * d_model_enc, d_model_enc),
            nn.GELU(),
        )

        # --- Encoder ---
        enc_cond_dim = d_model_enc + obs_dim * To
        self.enc_input_proj = nn.Linear(act_dim, d_model_enc)
        self.enc_pos_embedding = nn.Parameter(
            torch.empty(1, Ta, d_model_enc).normal_(std=0.02),
        )
        self.enc_blocks = nn.ModuleList([
            _TransformerBlock(
                hidden_size=d_model_enc,
                num_heads=n_heads_enc,
                cond_dim=enc_cond_dim,
                dropout=dropout,
            )
            for _ in range(enc_depth)
        ])

        # --- Bridge: (mean-pooled) encoder output -> decoder cond ---
        self.s_projector = (
            nn.Linear(d_model_enc, d_model_dec)
            if d_model_enc != d_model_dec
            else nn.Identity()
        )
        self.time_proj = (
            nn.Linear(d_model_enc, d_model_dec)
            if d_model_enc != d_model_dec
            else nn.Identity()
        )

        # --- Decoder ---
        dec_cond_dim = d_model_dec + d_model_dec  # concat(time_proj, s_proj_pooled)
        self.dec_input_proj = nn.Linear(act_dim, d_model_dec)
        # Decoder positional embedding: REQUIRED here because the bridge cond
        # is global per-sample (not per-token like the joint DDT), so the
        # decoder has no other positional handle. Without this, the decoder's
        # self-attention is permutation-equivariant and cannot tell which
        # timestep it is predicting velocity for.
        self.dec_pos_embedding = nn.Parameter(
            torch.empty(1, Ta, d_model_dec).normal_(std=0.02),
        )
        self.dec_blocks = nn.ModuleList([
            _TransformerBlock(
                hidden_size=d_model_dec,
                num_heads=n_heads_dec,
                cond_dim=dec_cond_dim,
                dropout=dropout,
            )
            for _ in range(dec_depth)
        ])

        # --- Output projection (zero-init for AdaLN-Zero on the head) ---
        self.output_proj = nn.Linear(d_model_dec, act_dim)

        self._initialize_weights()

        print(
            f"number of LBMDiTDDT parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _initialize_weights(self):
        """Zero-init AdaLN modulations + output head for stable start."""
        for block in list(self.enc_blocks) + list(self.dec_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        x: Tensor,
        s: Tensor,
        t: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Args:
            x:         (b, Ta, act_dim) noisy action chunk
            s:         (b,) — unused; kept for FlowMap signature compat.
            t:         (b,) flow-matching time (already noise-shifted by
                       the loss/sampler if shift != 1.0).
            condition: (b, To, obs_dim) encoded current observations

        Returns:
            y:      (b, Ta, act_dim) predicted velocity
            scalar: None
        """
        del s
        batch_size = x.shape[0]
        device = x.device

        # --- Time conditioning (single t) ---
        if self.time_embedder is not None:
            time_raw = self.time_embedder(t)
        else:
            time_raw = torch.zeros(
                batch_size, self._timestep_emb_dim, device=device,
            )
        time_features = self.time_mlp(time_raw)  # (b, d_model_enc)

        # --- Encoder cond (time + flat obs) ---
        if condition is not None:
            cond_flat = torch.flatten(condition, 1)
        else:
            cond_flat = torch.zeros(
                batch_size, self.obs_dim * self.To, device=device,
            )
        enc_cond = torch.cat([time_features, cond_flat], dim=-1)

        # --- Encoder over the noisy action chunk ---
        h_enc = self.enc_input_proj(x)
        h_enc = h_enc + self.enc_pos_embedding[:, : x.shape[1], :]
        for block in self.enc_blocks:
            h_enc = block(h_enc, enc_cond)
        # h_enc: (b, Ta, d_model_enc)

        # --- Bridge: mean-pool over Ta then project to dec width ---
        # Pooling over the action sequence summarizes the encoder into a
        # single per-sample vector — required because the decoder uses
        # GLOBAL AdaLN cond (one cond row per sample, not per-token).
        h_enc_pooled = h_enc.mean(dim=1)             # (b, d_model_enc)
        s_out = self.s_projector(h_enc_pooled)       # (b, d_model_dec)
        t_dec = self.time_proj(time_features)        # (b, d_model_dec)
        dec_cond = torch.cat([t_dec, s_out], dim=-1) # (b, 2*d_model_dec)

        # --- Decoder over the noisy action chunk (re-tokenized at dec width) ---
        h_dec = self.dec_input_proj(x)
        h_dec = h_dec + self.dec_pos_embedding[:, : x.shape[1], :]
        for block in self.dec_blocks:
            h_dec = block(h_dec, dec_cond)
        # h_dec: (b, Ta, d_model_dec)

        y = self.output_proj(h_dec)
        return y, None


def test_lbmdit_ddt():
    """Smoke test for LBMDiTDDT."""
    print("=" * 50)
    print("Testing LBMDiTDDT")
    print("=" * 50)

    obs_dim = 64
    act_dim = 7
    Ta = 8
    To = 2
    B = 4

    model = LBMDiTDDT(
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

    # Same enc/dec dim (Identity projector)
    model_same = LBMDiTDDT(
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
    print("LBMDiTDDT test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_lbmdit_ddt()
