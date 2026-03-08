"""DiT network with cross-attention decoder for condistill.

Uses cross-attention between decoder layers and the final encoder layer output,
with AdaLN modulation from timestep + mean-pooled encoder output on the residual paths.

Author: Chaoyi Pan
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


def build_mlp(hidden_size, projector_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, hidden_size),
    )


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _SelfAttnEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        src2 = self.norm1(src)
        q = k = _with_pos_embed(src2, pos)
        src2, _ = self.self_attn(q, k, value=src2, need_weights=False)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    """AdaLN modulation: shift + scale from a conditioning vector."""

    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    """Zero-initialized gating modulation."""

    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTXAttnDecoder(nn.Module):
    """Decoder block with cross-attention to encoder output and AdaLN.

    Each block does:
    1. AdaLN-modulated self-attention (modulated by timestep + mean-pooled encoder)
    2. Cross-attention to final encoder output (plain residual)
    3. AdaLN-modulated FFN (modulated by timestep + mean-pooled encoder)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # AdaLN modulation from timestep for self-attention path
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        # AdaLN modulation from timestep for FFN path
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, memory):
        """Args:
        x:      (Ta, B, d_model) - decoder tokens
        t:      (B, d_model)     - timestep embedding
        memory: (S, B, d_model)  - final encoder layer output
        """
        # Combine timestep with mean-pooled encoder output for AdaLN
        c = t + torch.mean(memory, dim=0)  # (B, d_model)

        # 1. AdaLN-modulated self-attention
        x2 = self.attn_mod1(self.norm1(x), c)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), c) + x

        # 2. Cross-attention to encoder output (plain residual, no gating)
        x2 = self.norm2(x)
        x2, _ = self.cross_attn(query=x2, key=memory, value=memory, need_weights=False)
        x = self.dropout2(x2) + x

        # 3. AdaLN-modulated FFN
        x2 = self.mlp_mod1(self.norm3(x), c)
        x2 = self.linear2(self.dropout3(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout4(x2), c)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, memory):
        """Args:
        x: (Ta, B, d_model)
        t: (B, d_model) - timestep embedding
        memory: (S, B, d_model) - final encoder layer output
        """
        c = t + torch.mean(memory, dim=0)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale[None]) + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class SudeepDiTOGCondistillXAttn(BaseNetwork):
    """DiT with cross-attention decoder for condistill.

    Encoder: self-attention over observation tokens (same as original).
    Decoder: each block has self-attention + cross-attention to FINAL encoder
             output, with AdaLN modulation from timestep + mean-pooled encoder.
    """

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
    ):
        emb_dim = d_model
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, depth)

        self.act_dim = act_dim
        self.Ta = Ta
        self.obs_dim = obs_dim
        self.To = To
        self.d_model = d_model
        self.disable_time_embedding = disable_time_embedding

        # Time embeddings
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params
            )
        else:
            self.map_s = None
            self.map_t = None

        # Input projection (action tokens)
        self.x_proj = nn.Sequential(
            nn.Linear(act_dim, act_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(act_dim, d_model),
        )

        # Positional encoding for encoder
        self.enc_pos = _PositionalEncoding(d_model)

        # Learned positional embedding for decoder
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(Ta, 1, d_model), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # Encoder blocks (self-attention only, same as original)
        encoder_module = _SelfAttnEncoder(
            d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_module) for _ in range(depth // 2)]
        )
        for layer in self.encoder:
            layer.reset_parameters()

        # Decoder blocks (self-attn + cross-attn + AdaLN from timestep)
        decoder_module = _DiTXAttnDecoder(
            d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.decoder = nn.ModuleList(
            [copy.deepcopy(decoder_module) for _ in range(depth)]
        )
        for layer in self.decoder:
            layer.reset_parameters()

        # REPA projector
        self.projector = build_mlp(d_model, d_model * 2)

        # Output layer (zero-initialized for AdaLN-Zero)
        self.final_layer = _FinalLayer(d_model, act_dim)
        self.final_layer.reset_parameters()

        print(
            f"number of DiT-XAttn parameters: {sum(p.numel() for p in self.parameters()):e}"
        )

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        align_depth: int | None = None,
    ):
        """Args:
            x:          (b, Ta, act_dim)
            s:          (b, ) - source time
            t:          (b, ) - target time
            condition:  (b, To+extra, obs_dim) or None
            align_depth: decoder layer index (1-based) at which to extract projections
        Returns:
            y:          (b, Ta, act_dim)
            scalar:     None
            zs_tilde:   list of projected hidden states or None
        """
        batch_size, Ta, act_dim = x.shape
        device = x.device

        # Time embedding
        if self.map_s is not None and self.map_t is not None:
            s_emb = self.map_s(s)
            t_emb = self.map_t(t)
            time_emb = torch.cat([s_emb, t_emb], dim=-1)  # (b, d_model)
        else:
            time_emb = torch.zeros(batch_size, self.d_model, device=device)

        # Encode observation tokens
        if condition is not None:
            obs_tokens = condition.transpose(0, 1)  # (S, b, d_model)
        else:
            obs_tokens = torch.zeros(1, batch_size, self.d_model, device=device)

        # Run encoder, keep only final output
        pos = self.enc_pos(obs_tokens)
        enc_out = obs_tokens
        for layer in self.encoder:
            enc_out = layer(enc_out, pos)
        # enc_out: (S, b, d_model) - final encoder layer output

        # Project action tokens
        x_tokens = self.x_proj(x).transpose(0, 1)  # (Ta, b, d_model)
        x_tokens = x_tokens + self.dec_pos[:Ta]

        # Decode with cross-attention to final encoder output
        y_tokens = x_tokens
        zs_tilde = None
        for i, layer in enumerate(self.decoder):
            y_tokens = layer(y_tokens, time_emb, enc_out)
            if (i + 1) == align_depth:
                if self.training:
                    zs_tilde = [self.projector(y_tokens).transpose(0, 1)]
                else:
                    zs_tilde = [y_tokens.transpose(0, 1)]

        # Final output
        y = self.final_layer(y_tokens, time_emb, enc_out)  # (b, Ta, act_dim)

        scalar = None
        return y, scalar, zs_tilde


def test_sudeepdit_xattn():
    print("=" * 50)
    print("Testing SudeepDiTOGCondistillXAttn")
    print("=" * 50)

    act_dim = 2
    Ta = 4
    d_model = 128
    obs_dim = d_model
    To = 2
    batch_size = 4

    model = SudeepDiTOGCondistillXAttn(
        act_dim=act_dim,
        Ta=Ta,
        obs_dim=obs_dim,
        To=To,
        d_model=d_model,
        n_heads=4,
        depth=2,
        dropout=0.1,
    )

    x = torch.randn(batch_size, Ta, act_dim)
    s = torch.randn(batch_size)
    t = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, obs_dim)

    # Basic forward
    y, scalar, zs_tilde = model(x, s, t, condition)
    print(f"Input: {x.shape}, Output: {y.shape}, scalar: {scalar}, zs_tilde: {zs_tilde}")

    # With align_depth=1
    y, _, zs = model(x, s, t, condition, align_depth=1)
    print(f"align_depth=1: y={y.shape}, zs[0]={zs[0].shape}")

    # Variable-length condition (condistill: obs + extra_cond tokens)
    cond_long = torch.randn(batch_size, 11, obs_dim)
    y_long, _, zs_long = model(x, s, t, cond_long, align_depth=1)
    print(f"Variable-length cond (11 tokens): y={y_long.shape}, zs[0]={zs_long[0].shape}")

    # No condition
    y_nc, _, _ = model(x, s, t, None)
    print(f"No condition: {y_nc.shape}")

    # Time invariance with disable_time_embedding
    model_nt = SudeepDiTOGCondistillXAttn(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        d_model=d_model, n_heads=4, depth=2, disable_time_embedding=True,
    )
    y1, _, _ = model_nt(x, s, t, condition)
    y2, _, _ = model_nt(x, torch.randn(batch_size), torch.randn(batch_size), condition)
    print(f"Time invariant: {torch.allclose(y1, y2, atol=1e-6)}")

    print("=" * 50)
    print("Test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_sudeepdit_xattn()
