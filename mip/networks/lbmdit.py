"""LBMDiT network adapted from Multi-Task DiT Policy by Bryson Jones.

Decoder-only DiT with AdaLN-Zero (6-param modulation with gating),
optional RoPE, and single-vector conditioning.

Reference: https://github.com/facebookresearch/DiT

Author: Zilai Zeng
Date: 2026-04-05
"""

import torch
import torch.nn as nn
from torch import Tensor

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork


class _RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes position by rotating Q and K vectors so that the dot product
    naturally captures relative positions.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("_sin_cached", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        cos = self._cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self._sin_cached[:, :, :seq_len, :].to(q.dtype)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class _RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout_p = dropout

        self.rope = _RotaryPositionalEmbedding(
            head_dim=self.head_dim, max_seq_len=max_seq_len, base=rope_base,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.hidden_size)
        return self.out_proj(attn_out)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class _TransformerBlock(nn.Module):
    """DiT transformer block with AdaLN-Zero (6-param modulation with gating).

    Supports both standard nn.MultiheadAttention and RoPE attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_dim: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.use_rope = use_rope

        if use_rope:
            self.attn = _RoPEAttention(
                hidden_size=hidden_size, num_heads=num_heads, dropout=dropout,
                max_seq_len=max_seq_len, rope_base=rope_base,
            )
        else:
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

        # AdaLN-Zero: 6 modulation params (shift, scale, gate) x (attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Args:
            x: (b, T, hidden_size)
            cond: (b, cond_dim)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=1)
        )

        # Attention with AdaLN-Zero
        attn_in = _modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        if self.use_rope:
            attn_out = self.attn(attn_in)
        else:
            attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with AdaLN-Zero
        mlp_in = _modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)

        return x


class LBMDiT(BaseNetwork):
    """Decoder-only DiT with AdaLN-Zero modulation and optional RoPE.

    Adapted from the Multi-Task DiT Policy (Bryson Jones) to the MIP
    framework interface (x, s, t, condition) -> (action, scalar).
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
        use_rope: bool = False,
        use_positional_encoding: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__(act_dim, Ta, obs_dim, To, d_model, depth)

        self.d_model = d_model
        self.disable_time_embedding = disable_time_embedding
        self.use_rope = use_rope

        # --- Time embeddings for s and t ---
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params,
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params,
            )
        else:
            self.map_s = None
            self.map_t = None

        # --- Time MLP (project raw timestep embedding to conditioning dim) ---
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )

        # Conditioning dimension fed to AdaLN = time + obs concatenated
        cond_dim = d_model + obs_dim * To  # time_features + flattened obs

        # --- Input projection ---
        self.input_proj = nn.Linear(act_dim, d_model)

        # --- Positional embedding (independent of RoPE, can coexist) ---
        if use_positional_encoding:
            self.pos_embedding = nn.Parameter(
                torch.empty(1, Ta, d_model).normal_(std=0.02),
            )
        else:
            self.pos_embedding = None

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                cond_dim=cond_dim,
                dropout=dropout,
                use_rope=use_rope,
                max_seq_len=Ta,
                rope_base=rope_base,
            )
            for _ in range(depth)
        ])

        # --- Output projection ---
        self.output_proj = nn.Linear(d_model, act_dim)

        # --- AdaLN-Zero initialization ---
        self._initialize_weights()

        print(
            f"number of LBMDiT parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _initialize_weights(self):
        """Zero-init the final linear in each adaLN_modulation for stable training."""
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: Tensor,
        s: Tensor,
        t: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Args:
            x:         (b, Ta, act_dim) noisy action sequence
            s:         (b,) source time parameter
            t:         (b,) target time parameter
            condition: (b, To, obs_dim) or None

        Returns:
            y:      (b, Ta, act_dim) predicted action / velocity
            scalar: None (placeholder for interface compatibility)
        """
        batch_size = x.shape[0]
        device = x.device

        # --- Time conditioning ---
        if self.map_s is not None and self.map_t is not None:
            s_emb = self.map_s(s)  # (b, d_model//2)
            t_emb = self.map_t(t)  # (b, d_model//2)
            time_raw = torch.cat([s_emb, t_emb], dim=-1)  # (b, d_model)
        else:
            time_raw = torch.zeros(batch_size, self.d_model, device=device)

        time_features = self.time_mlp(time_raw)  # (b, d_model)

        # --- Observation conditioning ---
        if condition is not None:
            cond_flat = torch.flatten(condition, 1)  # (b, To * obs_dim)
        else:
            cond_flat = torch.zeros(batch_size, self.obs_dim * self.To, device=device)

        # Concatenate time and obs features -> full conditioning vector
        cond_vec = torch.cat([time_features, cond_flat], dim=-1)  # (b, d_model + To*obs_dim)

        # --- Input projection ---
        h = self.input_proj(x)  # (b, Ta, d_model)

        if self.pos_embedding is not None:
            h = h + self.pos_embedding[:, : x.shape[1], :]

        # --- Transformer blocks ---
        for block in self.blocks:
            h = block(h, cond_vec)

        # --- Output projection ---
        y = self.output_proj(h)  # (b, Ta, act_dim)

        return y, None


def test_lbmdit():
    """Test LBMDiT network."""
    print("=" * 50)
    print("Testing LBMDiT")
    print("=" * 50)

    act_dim = 2
    Ta = 4
    obs_dim = 3
    To = 2
    batch_size = 4

    # Test with RoPE
    model = LBMDiT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        d_model=128, n_heads=4, depth=2, dropout=0.1,
        timestep_emb_type="positional", use_rope=True,
    )

    x = torch.randn(batch_size, Ta, act_dim)
    s = torch.randn(batch_size)
    t = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, obs_dim)

    y, scalar = model(x, s, t, condition)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Scalar output: {scalar}")

    # Test without condition
    y_nc, _ = model(x, s, t, None)
    print(f"Output without condition shape: {y_nc.shape}")

    # Test without RoPE
    print("\nTesting without RoPE (learned positional embeddings):")
    model_no_rope = LBMDiT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        d_model=128, n_heads=4, depth=2, use_rope=False,
    )
    y_nr, _ = model_no_rope(x, s, t, condition)
    print(f"Output shape (no RoPE): {y_nr.shape}")

    # Test with disabled time embedding
    print("\nTesting with disable_time_embedding=True:")
    model_no_time = LBMDiT(
        act_dim=act_dim, Ta=Ta, obs_dim=obs_dim, To=To,
        d_model=128, n_heads=4, depth=2, disable_time_embedding=True,
    )
    y1, _ = model_no_time(x, s, t, condition)
    y2, _ = model_no_time(x, torch.randn(batch_size), torch.randn(batch_size), condition)
    print(f"Time invariant: {torch.allclose(y1, y2, atol=1e-6)}")

    print("=" * 50)
    print("LBMDiT test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_lbmdit()
