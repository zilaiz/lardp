"""LBMDiTJointDDT: per-token AdaLN joint DiT with encoder/decoder width split.

Combines two design ideas:
  - DFoT-style per-token AdaLN-Zero throughout (no global cond). Each token
    carries its own (shift, scale, gate) modulation built from a per-token
    cond row. Reference: ``diffusion-forcing-transformer/algorithms/dfot/
    backbones/dit/dit_blocks.py:DiTBlock``.
  - RAE/DDT-style encoder/decoder width split: a deep narrow encoder over
    the noisy ``[state_token, action_tokens]`` sequence produces per-token
    features that, after a linear projection, serve as the per-token cond
    for a shallow wide decoder operating on the same noisy input
    re-tokenized at the wider width. Reference: ``RAE/src/stage2/models/
    DDT.py:DiTwDDTHead``.

Per-token cond construction (encoder side, at ``enc_hidden``):
    cond[b, k, :] = time_proj(time_emb(t[b, k]))   # per-token time
                  + obs_proj(obs_flat[b])           # broadcast over tokens
                  + opt_proj(opt_emb[b])            # broadcast over tokens
where ``t[b, 0] = t_state[b]`` and ``t[b, 1:] = t_action[b]`` (or
``t_action[b, :]`` if per-action-step). Obs and optimality are global per
sample and replicated across the ``Ta+1`` tokens.

Forward sketch:
    cond_enc = build_cond_enc(t_state, t_action, obs, opt)        # (B, Ta+1, enc_h)
    h_enc    = pos_emb + cat(state_proj_enc(x_s), action_proj_enc(x_a))
    for blk in encoder_blocks: h_enc = blk(h_enc, cond_enc)
    s_dec    = s_projector(h_enc)                                 # (B, Ta+1, dec_h)
    h_dec    = cat(state_proj_dec(x_s), action_proj_dec(x_a))     # no pos emb on dec side
    for blk in decoder_blocks: h_dec = blk(h_dec, s_dec)
    v_state  = state_final(h_dec[:, :1], s_dec[:, :1])
    v_action = action_final(h_dec[:, 1:], s_dec[:, 1:])

Notes vs. ``LBMDiTJoint``:
  - Re-uses the ``s, t`` two-time API; ``s`` is now the state-token flow
    time and ``t`` is the action flow time. ``t`` may be scalar (B,) or
    per-action-step (B, Ta).
  - Single uniform width is replaced by (enc_hidden, dec_hidden); the
    s_projector bridges them.
  - All cond is per-token; no concat-into-flat-cond global vector.

Author: Zilai Zeng
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN modulation: x * (1 + scale) + shift, broadcasting on last dims."""
    return x * (1 + scale) + shift


class _DDTBlock(nn.Module):
    """DiT block with per-token AdaLN-Zero modulation.

    cond is a per-token sequence ``(B, L, cond_dim)``. Each token receives
    its own (shift, scale, gate) for both attention and MLP sub-layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Args:
            x:    (B, L, hidden_size)
            cond: (B, L, cond_dim)  — per-token conditioning row
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        attn_in = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + gate_msa * attn_out

        mlp_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(mlp_in)
        return x


class _DDTFinalLayer(nn.Module):
    """Final AdaLN layer with per-token cond, then linear to output dim.

    Used at the head of each output stream (state, action) so the
    last modulation is also per-token.
    """

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Args:
            x:    (B, L, hidden_size)
            cond: (B, L, hidden_size)
        """
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class LBMDiTJointDDT(BaseNetwork):
    """Joint DDT-style trunk over (next_state, action_chunk).

    Token vocabulary (Ta+1 tokens): ``[state_token, action_token_0, ...,
    action_token_{Ta-1}]``. The encoder runs at ``enc_hidden`` width, the
    decoder at ``dec_hidden``. Both run on the same noisy input,
    re-tokenized per side.

    Returns:
        v_state:  (B, 1, obs_dim)
        v_action: (B, Ta, act_dim)
        scalar:   None  (interface symmetry)
    """

    EXPERT_IDX = 0
    NULL_IDX = 1  # also used by play-data conditioning at training time

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        enc_hidden: int = 256,
        enc_depth: int = 8,
        enc_n_heads: int = 8,
        dec_hidden: int = 512,
        dec_depth: int = 2,
        dec_n_heads: int = 8,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        timestep_emb_dim: int = 128,
        opt_emb_dim: int | None = None,
    ):
        # BaseNetwork stores act_dim/Ta/obs_dim/To/emb_dim/n_layers as attrs;
        # we use enc_hidden as ``emb_dim`` and the total depth as ``n_layers``.
        super().__init__(
            act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
            emb_dim=enc_hidden, n_layers=enc_depth + dec_depth,
        )

        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self._timestep_emb_dim = timestep_emb_dim
        self._opt_emb_dim = opt_emb_dim if opt_emb_dim is not None else enc_hidden

        # --- Time embedder + projection to enc_hidden cond width ---
        timestep_emb_params = timestep_emb_params or {}
        self.time_embedder = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
            timestep_emb_dim, **timestep_emb_params,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_emb_dim, enc_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(enc_hidden, enc_hidden),
        )

        # --- Obs context flat -> enc_hidden (broadcast across tokens) ---
        self.obs_mlp = nn.Linear(To * obs_dim, enc_hidden)

        # --- Optimality embedding (2 slots: expert + null/play) ---
        self.optimality_embedding = nn.Embedding(2, self._opt_emb_dim)
        self.opt_mlp = nn.Linear(self._opt_emb_dim, enc_hidden)

        # --- Encoder-side input projections ---
        self.state_input_proj_enc = nn.Linear(obs_dim, enc_hidden)
        self.action_input_proj_enc = nn.Linear(act_dim, enc_hidden)

        # --- Learned 1D pos embedding (encoder side only, RAE pattern) ---
        self.pos_embedding = nn.Parameter(
            torch.empty(1, Ta + 1, enc_hidden).normal_(std=0.02),
        )

        # --- Encoder blocks ---
        self.encoder_blocks = nn.ModuleList([
            _DDTBlock(
                hidden_size=enc_hidden,
                num_heads=enc_n_heads,
                cond_dim=enc_hidden,
                dropout=dropout,
            )
            for _ in range(enc_depth)
        ])

        # --- Encoder->decoder bridge: project per-token features to dec_hidden ---
        self.s_projector = (
            nn.Linear(enc_hidden, dec_hidden)
            if enc_hidden != dec_hidden
            else nn.Identity()
        )

        # --- Decoder-side input projections (same noisy x re-tokenized) ---
        self.state_input_proj_dec = nn.Linear(obs_dim, dec_hidden)
        self.action_input_proj_dec = nn.Linear(act_dim, dec_hidden)

        # --- Decoder blocks ---
        self.decoder_blocks = nn.ModuleList([
            _DDTBlock(
                hidden_size=dec_hidden,
                num_heads=dec_n_heads,
                cond_dim=dec_hidden,
                dropout=dropout,
            )
            for _ in range(dec_depth)
        ])

        # --- Output heads (per-token AdaLN final, then linear) ---
        self.state_final = _DDTFinalLayer(dec_hidden, obs_dim)
        self.action_final = _DDTFinalLayer(dec_hidden, act_dim)

        self._initialize_weights()

        print(
            f"number of LBMDiTJointDDT parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _initialize_weights(self):
        # AdaLN-Zero: zero the last Linear of every block's modulation
        for blk in self.encoder_blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        for blk in self.decoder_blocks:
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
        """Build per-token time features at enc_hidden width (B, Ta+1, enc_hidden).

        Args:
            t_state:  (B,)              flow time of the state token
            t_action: (B,) or (B, Ta)   shared or per-step action flow time
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

        # The PositionalEmbedding expects 1D input; flatten then reshape.
        t_flat = t_per_token.reshape(-1)                              # (B*(Ta+1),)
        t_emb_flat = self.time_embedder(t_flat)                       # (B*(Ta+1), te_dim)
        t_emb = t_emb_flat.view(B, self.Ta + 1, -1)                   # (B, Ta+1, te_dim)
        return self.time_mlp(t_emb)                                   # (B, Ta+1, enc_hidden)

    def _build_cond_enc(
        self,
        t_state: Tensor,
        t_action: Tensor,
        condition: Tensor | None,
        optimality_idx: Tensor | None,
        B: int,
        device: torch.device,
    ) -> Tensor:
        """Per-token encoder cond ``(B, Ta+1, enc_hidden)``.

        Components are summed (DFoT pattern of additive composition):
            cond = time_per_token + obs_broadcast + opt_broadcast.
        """
        time_per_tok = self._per_token_time_features(t_state, t_action, B)

        if condition is not None:
            obs_feat = self.obs_mlp(condition.flatten(1))             # (B, enc_hidden)
        else:
            obs_feat = torch.zeros(B, self.enc_hidden, device=device)
        obs_per_tok = obs_feat.unsqueeze(1).expand(B, self.Ta + 1, -1)

        if optimality_idx is None:
            optimality_idx = torch.full(
                (B,), self.NULL_IDX, device=device, dtype=torch.long,
            )
        opt_feat = self.opt_mlp(self.optimality_embedding(optimality_idx))
        opt_per_tok = opt_feat.unsqueeze(1).expand(B, self.Ta + 1, -1)

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
            x_state:        (B, 1, obs_dim)    noisy next-state embedding
            x_action:       (B, Ta, act_dim)   noisy action chunk
            s:              (B,)               state-token flow time
            t:              (B,) or (B, Ta)    action flow time(s)
            condition:      (B, To, obs_dim)   encoded current obs, or None
            optimality_idx: (B,) long {0=expert, 1=null/play}, or None (=null)

        Returns:
            v_state:  (B, 1, obs_dim)
            v_action: (B, Ta, act_dim)
            scalar:   None  (interface symmetry)
        """
        B = x_action.shape[0]
        device = x_action.device

        # --- 1. Per-token cond at encoder width ---
        cond_enc = self._build_cond_enc(s, t, condition, optimality_idx, B, device)

        # --- 2. Encoder: tokenize x at enc_hidden, run encoder blocks ---
        h_enc = torch.cat([
            self.state_input_proj_enc(x_state),     # (B, 1, enc_hidden)
            self.action_input_proj_enc(x_action),   # (B, Ta, enc_hidden)
        ], dim=1)                                   # (B, Ta+1, enc_hidden)
        h_enc = h_enc + self.pos_embedding[:, : h_enc.shape[1], :]

        for blk in self.encoder_blocks:
            h_enc = blk(h_enc, cond_enc)

        # --- 3. Bridge: encoder output becomes per-token decoder cond ---
        s_dec = self.s_projector(h_enc)             # (B, Ta+1, dec_hidden)

        # --- 4. Decoder: re-tokenize x at dec_hidden, run decoder blocks ---
        # No pos emb on decoder side — RAE pattern; positional information
        # reaches the decoder via the per-token cond from the encoder.
        h_dec = torch.cat([
            self.state_input_proj_dec(x_state),     # (B, 1, dec_hidden)
            self.action_input_proj_dec(x_action),   # (B, Ta, dec_hidden)
        ], dim=1)                                   # (B, Ta+1, dec_hidden)

        for blk in self.decoder_blocks:
            h_dec = blk(h_dec, s_dec)

        # --- 5. Output heads with per-token AdaLN final layer ---
        v_state = self.state_final(h_dec[:, :1, :], s_dec[:, :1, :])     # (B, 1, obs_dim)
        v_action = self.action_final(h_dec[:, 1:, :], s_dec[:, 1:, :])   # (B, Ta, act_dim)
        return v_state, v_action, None


# ----------------------------- smoke tests --------------------------------

def test_lbmdit_joint_ddt():
    print("=" * 50)
    print("Testing LBMDiTJointDDT")
    print("=" * 50)

    obs_dim = 64
    act_dim = 7
    Ta = 8
    To = 2
    B = 4

    model = LBMDiTJointDDT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        enc_hidden=128, enc_depth=4, enc_n_heads=4,
        dec_hidden=256, dec_depth=2, dec_n_heads=4,
        timestep_emb_dim=64,
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

    # Mixed batch
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
    model_init = LBMDiTJointDDT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        enc_hidden=64, enc_depth=2, dec_hidden=128, dec_depth=2,
    )
    model_init.eval()
    with torch.no_grad():
        v_s, v_a, _ = model_init(x_state, x_action, t_state, t_action, cond)
    assert torch.allclose(v_s, torch.zeros_like(v_s)), \
        "state final should be zero at init"
    assert torch.allclose(v_a, torch.zeros_like(v_a)), \
        "action final should be zero at init"
    print("AdaLN-Zero init verified (final outputs are zero at init).")

    # Width split sanity: enc_hidden=dec_hidden => s_projector is Identity
    model_eq = LBMDiTJointDDT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        enc_hidden=128, enc_depth=2, dec_hidden=128, dec_depth=2,
    )
    assert isinstance(model_eq.s_projector, nn.Identity)
    print("s_projector is Identity when enc_hidden == dec_hidden.")

    print("=" * 50)
    print("LBMDiTJointDDT test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_lbmdit_joint_ddt()
