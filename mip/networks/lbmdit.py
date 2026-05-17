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


class LBMDiTIDM(LBMDiT):
    """LBMDiT for inverse dynamics: obs (To) + goal (1) = To+1 conditioning frames."""

    pass


class LBMDiTIDMv2(BaseNetwork):
    """IDM with obs summarizer, decoupled timestep_emb_dim, and an FDM auxiliary head.

    Differences from LBMDiTIDM:
    - Single time embedding (no dual map_s / map_t).
    - timestep_emb_dim is independent of d_model.
    - To observation frames are first summarized by an MLP into a single
      vector of size obs_dim (= encoder output dim) before being concatenated
      with the goal embedding for AdaLN conditioning.
    - Adds a forward-dynamics (FDM) head: given (obs summary, projected clean
      action), predict the goal embedding. Used for encoder shaping at training.

    Conditioning vector for AdaLN:
        cond = concat(time_emb, obs_summary, goal_emb)
        cond_dim = timestep_emb_dim + 2 * obs_dim
    """

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        To_obs: int,
        d_model: int = 256,
        timestep_emb_dim: int = 128,
        n_heads: int = 8,
        depth: int = 8,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
        obs_summarizer_hidden: int | None = None,
        action_proj_hidden: int | None = None,
        fdm_hidden: int | None = None,
    ):
        super().__init__(act_dim, Ta, obs_dim, To, d_model, depth)

        assert To == To_obs + 1, (
            f"To ({To}) must equal To_obs ({To_obs}) + 1 (single goal frame)"
        )

        self.d_model = d_model
        self.timestep_emb_dim = timestep_emb_dim
        self.disable_time_embedding = disable_time_embedding
        self.To_obs = To_obs

        # --- Time embedding (single t, projected to timestep_emb_dim) ---
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.time_embedder = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                timestep_emb_dim, **timestep_emb_params,
            )
        else:
            self.time_embedder = None

        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_emb_dim, 2 * timestep_emb_dim),
            nn.GELU(),
            nn.Linear(2 * timestep_emb_dim, timestep_emb_dim),
            nn.GELU(),
        )

        # --- Obs summarizer: (To_obs, obs_dim) -> (obs_dim) ---
        os_hidden = obs_summarizer_hidden or (2 * obs_dim)
        self.obs_summarizer = nn.Sequential(
            nn.Linear(To_obs * obs_dim, os_hidden),
            nn.GELU(),
            nn.Linear(os_hidden, obs_dim),
        )

        # --- AdaLN conditioning dim ---
        cond_dim = timestep_emb_dim + 2 * obs_dim  # time + obs_summary + goal

        # --- Action trunk (DiT-style) ---
        self.input_proj = nn.Linear(act_dim, d_model)
        self.pos_embedding = nn.Parameter(
            torch.empty(1, Ta, d_model).normal_(std=0.02),
        )
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.output_proj = nn.Linear(d_model, act_dim)

        # --- FDM head: (obs_summary, action_proj(action_chunk)) -> goal_pred ---
        ap_hidden = action_proj_hidden or (2 * obs_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(Ta * act_dim, ap_hidden),
            nn.GELU(),
            nn.Linear(ap_hidden, obs_dim),
        )
        fh = fdm_hidden or (2 * obs_dim)
        self.fdm_head = nn.Sequential(
            nn.Linear(2 * obs_dim, fh),
            nn.GELU(),
            nn.Linear(fh, obs_dim),
        )

        self._initialize_weights()

        print(
            f"number of LBMDiTIDMv2 parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _summarize(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Split condition into (obs_summary, goal_emb).

        Args:
            condition: (B, To, obs_dim) where the first To_obs frames are obs
                and the last frame is the goal.
        Returns:
            obs_summary: (B, obs_dim) — pooled obs context.
            goal_emb:    (B, obs_dim) — goal frame embedding.
        """
        obs_part = condition[:, : self.To_obs]                  # (B, To_obs, obs_dim)
        obs_summary = self.obs_summarizer(obs_part.flatten(1))  # (B, obs_dim)
        goal_emb = condition[:, -1]                              # (B, obs_dim)
        return obs_summary, goal_emb

    def forward(
        self,
        x: Tensor,
        s: Tensor,
        t: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Standard IDM forward.

        Args:
            x:         (B, Ta, act_dim) noisy action sequence.
            s:         (B,) — unused (kept for FlowMap compatibility).
            t:         (B,) flow-matching time.
            condition: (B, To, obs_dim) encoded obs+goal stack, or None.
        Returns:
            y:      (B, Ta, act_dim) predicted velocity.
            scalar: None.
        """
        B = x.shape[0]
        device = x.device

        # Time
        if self.time_embedder is not None:
            t_raw = self.time_embedder(t)
        else:
            t_raw = torch.zeros(B, self.timestep_emb_dim, device=device)
        t_emb = self.time_mlp(t_raw)  # (B, timestep_emb_dim)

        # Obs + goal
        if condition is None:
            condition = torch.zeros(
                B, self.To_obs + 1, self.obs_dim, device=device,
            )
        obs_summary, goal_emb = self._summarize(condition)

        cond_vec = torch.cat([t_emb, obs_summary, goal_emb], dim=-1)

        # Action trunk
        h = self.input_proj(x) + self.pos_embedding[:, : x.shape[1], :]
        for block in self.blocks:
            h = block(h, cond_vec)
        y = self.output_proj(h)
        return y, None

    def forward_with_summary(
        self,
        x: Tensor,
        s: Tensor,
        t: Tensor,
        obs_summary: Tensor,
        goal_emb: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Action-trunk forward with a precomputed (obs_summary, goal_emb)
        pair, bypassing ``self.obs_summarizer`` and ``self._summarize``.

        Used by retrieval-based goal predictors that already hold both halves
        of the AdaLN conditioning in the same space the IDM saw at training,
        and would otherwise round-trip through the encoder + summarizer
        unnecessarily.

        Args:
            x:           (B, Ta, act_dim) noisy action sequence.
            s:           (B,) — unused (kept for FlowMap compatibility).
            t:           (B,) flow-matching time.
            obs_summary: (B, obs_dim) precomputed obs summary.
            goal_emb:    (B, obs_dim) precomputed goal embedding.
        Returns:
            y:      (B, Ta, act_dim) predicted velocity.
            scalar: None.
        """
        del s
        B = x.shape[0]
        device = x.device

        if self.time_embedder is not None:
            t_raw = self.time_embedder(t)
        else:
            t_raw = torch.zeros(B, self.timestep_emb_dim, device=device)
        t_emb = self.time_mlp(t_raw)  # (B, timestep_emb_dim)

        cond_vec = torch.cat([t_emb, obs_summary, goal_emb], dim=-1)

        h = self.input_proj(x) + self.pos_embedding[:, : x.shape[1], :]
        for block in self.blocks:
            h = block(h, cond_vec)
        y = self.output_proj(h)
        return y, None

    def forward_predict(
        self,
        condition: Tensor,
        clean_action: Tensor,
    ) -> Tensor:
        """FDM head: predict the goal embedding from obs + clean action chunk.

        The goal slot of `condition` is *not* used; only the first To_obs
        frames are read via the obs summarizer.

        Args:
            condition:    (B, To, obs_dim) encoded stack — only first To_obs read.
            clean_action: (B, Ta, act_dim) clean expert actions (normalized).
        Returns:
            predicted_goal: (B, obs_dim).
        """
        obs_part = condition[:, : self.To_obs]
        obs_summary = self.obs_summarizer(obs_part.flatten(1))      # (B, obs_dim)
        action_emb = self.action_proj(clean_action.flatten(1))      # (B, obs_dim)
        return self.fdm_head(torch.cat([obs_summary, action_emb], dim=-1))


class LBMDiTIDMv2Delta(LBMDiTIDMv2):
    """Delta-cond variant of LBMDiTIDMv2.

    Architecture is identical to ``LBMDiTIDMv2`` (same parameter names and
    shapes), so checkpoints are interchangeable at the ``load_state_dict``
    level. The only behavioral difference is the third slot of the AdaLN
    conditioning vector:

        v2       cond_vec = concat(time_emb, obs_summary, goal_emb)
        v2-delta cond_vec = concat(time_emb, obs_summary, goal_emb - last_obs_emb)

    The FDM head's output is interpreted as the predicted *delta*
    ``z_goal − z_last_obs`` rather than the absolute goal embedding. The
    network itself doesn't enforce this — it's a contract with the agent,
    which must compute the FDM target accordingly.

    ``forward_with_summary`` callers (e.g., retrieval pipelines) must pass
    the *delta* embedding as the ``goal_emb`` argument; the kwarg name is
    kept for interface compatibility.
    """

    def _summarize(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Return (obs_summary, goal_delta).

        ``goal_delta = z_goal − z_last_obs`` is the displacement from the
        most recent observation embedding to the goal embedding.
        """
        obs_summary, goal_emb = super()._summarize(condition)
        last_obs_emb = condition[:, self.To_obs - 1]  # (B, obs_dim)
        goal_delta = goal_emb - last_obs_emb
        return obs_summary, goal_delta


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
