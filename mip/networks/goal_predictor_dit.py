"""Goal Predictor DiT: LBMDiT-based flow matching model for goal embedding generation.

Generates goal embeddings via flow matching in the frozen encoder's embedding space.
Based on LBMDiT architecture with added align_depth support for intermediate
hidden state extraction (used for action regularization through frozen IDM).

Author: Zilai Zeng
"""

import torch
import torch.nn as nn
from torch import Tensor

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork
from mip.networks.lbmdit import _TransformerBlock


def _build_projector(hidden_size: int, projector_dim: int, output_dim: int) -> nn.Sequential:
    """MLP projector for intermediate hidden state extraction.

    Maps from transformer hidden dim to the encoder output dim so that
    projected states can be concatenated with encoder outputs.
    """
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, output_dim),
    )


class GoalPredictorDiT(BaseNetwork):
    """LBMDiT-based DiT for goal embedding generation via flow matching.

    Operates in the frozen encoder's embedding space:
    - Input: noisy goal embedding (B, 1, emb_dim)
    - Condition: encoded current obs z_t (B, To, emb_dim) via AdaLN
    - Output: velocity field (B, 1, emb_dim)

    Supports align_depth for extracting intermediate hidden states
    (projected via MLP) for action regularization through frozen IDM.
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
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
        projector_dim: int | None = None,
    ):
        """Args:
        act_dim: goal embedding dimension (= emb_dim of frozen encoder)
        Ta: number of goal tokens (typically 1)
        obs_dim: observation embedding dimension (= emb_dim of frozen encoder)
        To: number of observation steps
        d_model: hidden dimension of transformer blocks
        n_heads: number of attention heads
        depth: number of transformer blocks
        dropout: dropout rate
        timestep_emb_type: type of timestep embedding
        timestep_emb_params: additional params for timestep embedding
        disable_time_embedding: if True, zero out time embeddings
        projector_dim: hidden dim of projector MLP (None = 2 * d_model)
        """
        super().__init__(act_dim, Ta, obs_dim, To, d_model, depth)

        self.d_model = d_model
        self.disable_time_embedding = disable_time_embedding

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

        # --- Time MLP ---
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )

        # Conditioning dimension: time_features + flattened obs
        cond_dim = d_model + obs_dim * To

        # --- Input projection ---
        self.input_proj = nn.Linear(act_dim, d_model)

        # --- Positional embedding ---
        self.pos_embedding = nn.Parameter(
            torch.empty(1, Ta, d_model).normal_(std=0.02),
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # --- Output projection ---
        self.output_proj = nn.Linear(d_model, act_dim)

        # --- Projector for intermediate state extraction ---
        # Maps d_model -> act_dim so projected states match encoder output dim
        proj_dim = projector_dim or (2 * d_model)
        self.projector = _build_projector(d_model, proj_dim, act_dim)

        # --- AdaLN-Zero initialization ---
        self._initialize_weights()

        print(
            f"number of GoalPredictorDiT parameters: "
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
        align_depth: int | list[int] | None = None,
    ) -> tuple[Tensor, Tensor | None, list[Tensor] | None]:
        """Args:
            x:           (b, Ta, act_dim) noisy goal embedding(s)
            s:           (b,) source time parameter
            t:           (b,) target time parameter
            condition:   (b, To, obs_dim) encoded current observations
            align_depth: layer index(es) (1-based) to extract intermediate states

        Returns:
            y:        (b, Ta, act_dim) predicted velocity
            scalar:   None
            zs_tilde: list of projected hidden states at align_depth, or None
        """
        batch_size = x.shape[0]
        device = x.device

        # --- Time conditioning ---
        if self.map_s is not None and self.map_t is not None:
            s_emb = self.map_s(s)
            t_emb = self.map_t(t)
            time_raw = torch.cat([s_emb, t_emb], dim=-1)
        else:
            time_raw = torch.zeros(batch_size, self.d_model, device=device)

        time_features = self.time_mlp(time_raw)

        # --- Observation conditioning ---
        if condition is not None:
            cond_flat = torch.flatten(condition, 1)
        else:
            cond_flat = torch.zeros(
                batch_size, self.obs_dim * self.To, device=device
            )

        cond_vec = torch.cat([time_features, cond_flat], dim=-1)

        # --- Input projection ---
        h = self.input_proj(x)
        h = h + self.pos_embedding[:, : x.shape[1], :]

        # --- Transformer blocks with optional intermediate extraction ---
        if align_depth is None:
            align_depths = set()
        elif isinstance(align_depth, int):
            align_depths = {align_depth}
        else:
            align_depths = set(align_depth)

        zs_tilde = [] if align_depths else None

        for i, block in enumerate(self.blocks):
            h = block(h, cond_vec)
            if (i + 1) in align_depths:
                zs_tilde.append(self.projector(h))

        # --- Output projection ---
        y = self.output_proj(h)

        return y, None, zs_tilde


class GoalPredictorDiTDDT(BaseNetwork):
    """Encoder-decoder DiT with DDT head for goal embedding generation.

    Faithful to RAE's DiTwDDTHead design: both encoder and decoder receive
    the same noisy goal input (embedded to different hidden dims). The encoder
    builds a rich representation conditioned on time+obs, which then conditions
    a lightweight decoder via AdaLN modulation.

    Architecture:
        Encoder: noisy_goal → enc_proj → [Blocks x enc_depth, AdaLN(time+obs)]
        Bridge:  concat(time_proj(time), s_proj(enc_output))
        Decoder: noisy_goal → dec_proj → [Blocks x dec_depth, AdaLN(bridge_cond)]
        Output:  dec_output_proj → velocity
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
    ):
        super().__init__(act_dim, Ta, obs_dim, To, d_model_enc, enc_depth + dec_depth)

        self.d_model_enc = d_model_enc
        self.d_model_dec = d_model_dec
        self.disable_time_embedding = disable_time_embedding

        # --- Time embeddings for s and t ---
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model_enc // 2, **timestep_emb_params,
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model_enc // 2, **timestep_emb_params,
            )
        else:
            self.map_s = None
            self.map_t = None

        # --- Time MLP ---
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model_enc, 2 * d_model_enc),
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

        # --- Bridge: encoder output -> decoder conditioning ---
        self.s_projector = (
            nn.Linear(d_model_enc, d_model_dec)
            if d_model_enc != d_model_dec
            else nn.Identity()
        )
        # Project time features to decoder dim for concat with encoder output
        self.time_proj = (
            nn.Linear(d_model_enc, d_model_dec)
            if d_model_enc != d_model_dec
            else nn.Identity()
        )

        # --- Decoder ---
        dec_cond_dim = d_model_dec + d_model_dec  # concat(time_proj, s_proj)
        self.dec_input_proj = nn.Linear(act_dim, d_model_dec)
        self.dec_blocks = nn.ModuleList([
            _TransformerBlock(
                hidden_size=d_model_dec,
                num_heads=n_heads_dec,
                cond_dim=dec_cond_dim,
                dropout=dropout,
            )
            for _ in range(dec_depth)
        ])

        # --- Output projection ---
        self.output_proj = nn.Linear(d_model_dec, act_dim)

        # --- AdaLN-Zero initialization ---
        self._initialize_weights()

        print(
            f"number of GoalPredictorDiTDDT parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _initialize_weights(self):
        """Zero-init the final linear in each adaLN_modulation for stable training."""
        for block in list(self.enc_blocks) + list(self.dec_blocks):
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
            x:         (b, Ta, act_dim) noisy goal embedding(s)
            s:         (b,) source time parameter
            t:         (b,) target time parameter
            condition: (b, To, obs_dim) encoded current observations

        Returns:
            y:      (b, Ta, act_dim) predicted velocity
            scalar: None
        """
        batch_size = x.shape[0]
        device = x.device

        # --- Time conditioning ---
        if self.map_s is not None and self.map_t is not None:
            s_emb = self.map_s(s)
            t_emb = self.map_t(t)
            time_raw = torch.cat([s_emb, t_emb], dim=-1)
        else:
            time_raw = torch.zeros(batch_size, self.d_model_enc, device=device)

        time_features = self.time_mlp(time_raw)  # (b, d_model_enc)

        # --- Encoder conditioning (time + obs) ---
        if condition is not None:
            cond_flat = torch.flatten(condition, 1)
        else:
            cond_flat = torch.zeros(
                batch_size, self.obs_dim * self.To, device=device
            )
        enc_cond = torch.cat([time_features, cond_flat], dim=-1)

        # --- Encoder (processes noisy goal, conditioned on time+obs) ---
        h_enc = self.enc_input_proj(x)
        h_enc = h_enc + self.enc_pos_embedding[:, : x.shape[1], :]
        for block in self.enc_blocks:
            h_enc = block(h_enc, enc_cond)

        # --- Bridge: project encoder output, concat with time for decoder cond ---
        s_out = self.s_projector(h_enc).squeeze(1)  # (b, d_model_dec)
        t_dec = self.time_proj(time_features)  # (b, d_model_dec)
        dec_cond = torch.cat([t_dec, s_out], dim=-1)  # (b, 2*d_model_dec)

        # --- Decoder (processes noisy goal, conditioned on time + encoder output) ---
        h_dec = self.dec_input_proj(x)  # (b, Ta, d_model_dec)
        for block in self.dec_blocks:
            h_dec = block(h_dec, dec_cond)

        # --- Output ---
        y = self.output_proj(h_dec)

        return y, None


def test_goal_predictor_dit():
    """Test GoalPredictorDiT network."""
    print("=" * 50)
    print("Testing GoalPredictorDiT")
    print("=" * 50)

    emb_dim = 128
    To = 2
    batch_size = 4

    model = GoalPredictorDiT(
        act_dim=emb_dim,
        Ta=1,
        obs_dim=emb_dim,
        To=To,
        d_model=emb_dim,
        n_heads=4,
        depth=4,
        dropout=0.1,
    )

    x = torch.randn(batch_size, 1, emb_dim)
    s = torch.rand(batch_size)
    t = torch.rand(batch_size)
    condition = torch.randn(batch_size, To, emb_dim)

    # Basic forward
    y, scalar, zs_tilde = model(x, s, t, condition)
    print(f"Input: {x.shape}, Output: {y.shape}, scalar: {scalar}, zs_tilde: {zs_tilde}")
    assert y.shape == (batch_size, 1, emb_dim)
    assert zs_tilde is None

    # With align_depth
    y, _, zs = model(x, s, t, condition, align_depth=2)
    print(f"align_depth=2: y={y.shape}, zs[0]={zs[0].shape}")
    assert zs[0].shape == (batch_size, 1, emb_dim)

    # Gradient flow through align_depth
    model.train()
    y, _, zs = model(x, s, t, condition, align_depth=2)
    loss = zs[0].sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Gradient flows through projector: {has_grad}")
    assert has_grad

    # No condition
    model.zero_grad()
    y_nc, _, _ = model(x, s, t, None)
    print(f"No condition: {y_nc.shape}")

    # Test d_model != act_dim (projector must output act_dim, not d_model)
    model2 = GoalPredictorDiT(
        act_dim=emb_dim,
        Ta=1,
        obs_dim=emb_dim,
        To=To,
        d_model=emb_dim * 2,  # different from act_dim
        n_heads=4,
        depth=4,
        dropout=0.1,
    )
    model2.train()
    y2, _, zs2 = model2(x, s, t, condition, align_depth=2)
    assert y2.shape == (batch_size, 1, emb_dim), f"Expected (4,1,{emb_dim}), got {y2.shape}"
    assert zs2[0].shape == (batch_size, 1, emb_dim), (
        f"Projector output should be act_dim={emb_dim}, got {zs2[0].shape}"
    )
    print(f"d_model!=act_dim: y={y2.shape}, zs[0]={zs2[0].shape} (projector outputs act_dim)")

    print("=" * 50)
    print("GoalPredictorDiT test completed!")
    print("=" * 50)


def test_goal_predictor_dit_ddt():
    """Test GoalPredictorDiTDDT encoder-decoder network."""
    print("=" * 50)
    print("Testing GoalPredictorDiTDDT")
    print("=" * 50)

    emb_dim = 128
    To = 2
    batch_size = 4

    # Test with different enc/dec hidden dims
    d_model_enc = 96
    d_model_dec = 192
    model = GoalPredictorDiTDDT(
        act_dim=emb_dim,
        Ta=1,
        obs_dim=emb_dim,
        To=To,
        d_model_enc=d_model_enc,
        d_model_dec=d_model_dec,
        n_heads_enc=4,
        n_heads_dec=4,
        enc_depth=4,
        dec_depth=2,
        dropout=0.1,
    )

    x = torch.randn(batch_size, 1, emb_dim)
    s = torch.rand(batch_size)
    t = torch.rand(batch_size)
    condition = torch.randn(batch_size, To, emb_dim)

    # Basic forward
    y, scalar = model(x, s, t, condition)
    print(f"Input: {x.shape}, Output: {y.shape}, scalar: {scalar}")
    assert y.shape == (batch_size, 1, emb_dim)
    assert scalar is None

    # Gradient flow
    model.train()
    y, _ = model(x, s, t, condition)
    loss = y.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Gradient flows: {has_grad}")
    assert has_grad

    # No condition
    model.zero_grad()
    y_nc, _ = model(x, s, t, None)
    print(f"No condition: {y_nc.shape}")
    assert y_nc.shape == (batch_size, 1, emb_dim)

    # Same enc/dec dim (Identity projector)
    model_same = GoalPredictorDiTDDT(
        act_dim=emb_dim,
        Ta=1,
        obs_dim=emb_dim,
        To=To,
        d_model_enc=128,
        d_model_dec=128,
        n_heads_enc=4,
        n_heads_dec=4,
        enc_depth=2,
        dec_depth=2,
    )
    y_same, _ = model_same(x, s, t, condition)
    print(f"Same enc/dec dim: {y_same.shape}")
    assert y_same.shape == (batch_size, 1, emb_dim)

    # FlowMap compatibility
    from mip.flow_map import FlowMap

    fm = FlowMap(model)
    v = fm.get_velocity(t, x, condition)
    print(f"FlowMap.get_velocity: {v.shape}")
    assert v.shape == (batch_size, 1, emb_dim)

    print("=" * 50)
    print("GoalPredictorDiTDDT test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_goal_predictor_dit()
    test_goal_predictor_dit_ddt()
