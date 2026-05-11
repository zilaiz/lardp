"""Goal Predictor O2G: adaLN-Zero MLP velocity network for raw-obs -> goal flow.

Operates on a single (B, 1, obs_dim) token. The flow source x0 is the last
raw obs frame from the frozen IDM encoder (z_t[:, -1, :]), the target x1 is
the raw goal embedding, and the network learns the velocity
v(xt, t) ≈ x1 - x0 (constant in t under the linear interpolant). Conditioning
is optional (default: none, VITA-style; or "obs_summary" — the frozen IDM's
obs_summarizer applied to z_t — added into the time embedding before the
modulator).

Architecture (adaLN-Zero, matching VITA/A2A SimpleFlowNet):
    sinusoidal(t) -> Linear -> Mish -> Linear  -> t_emb (B, hidden_dim)
    [t_emb += cond_proj(cond)            if cond_mode != "none"]
    h = Linear(obs_dim -> hidden) (xt squeezed)
    repeat num_layers times:
        γ, scale, shift = Linear(SiLU(t_emb))            # zero-init
        h_norm = LN(h, affine=False) * (1+scale) + shift
        h = h + γ · MLP(h_norm)                          # γ-gated residual
    h = LN(h) -> Linear(hidden -> obs_dim) -> v

Init: xavier_uniform on every Linear; time_mlp Linears overridden to
N(0, 0.02²); modulator's final Linear in every block re-zeroed after
apply() so γ=scale=shift=0 ⇒ each block is identity at init ⇒ trunk is a
clean pass-through and v_init = out_proj(LN(input_proj(x))).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork


class _AdaLNZeroBlock(nn.Module):
    """DiT-style adaLN-Zero residual MLP block for a single token.

    forward:
        h_norm = LN(h) · (1 + scale) + shift
        h      = h + γ · MLP(h_norm)

    The modulator's final Linear is zero-init'd in the parent network so
    γ=scale=shift=0 at init, making each block exactly identity. Combined
    with `LayerNorm(elementwise_affine=False)`, modulation is the only
    affine transformation in the block.
    """

    def __init__(self, hidden_dim: int, mod_dim: int, mlp_ratio: int, dropout: float):
        super().__init__()
        inner = hidden_dim * mlp_ratio

        # LN with NO learnable affine — modulation provides γ/β.
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        # MLP block (timm Mlp shape): Linear → GELU(tanh) → Drop → Linear → Drop.
        drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(approximate="tanh"),
            drop1,
            nn.Linear(inner, hidden_dim),
            drop2,
        )

        # adaLN-Zero modulator: SiLU then Linear(mod_dim, 3*hidden_dim).
        # Final Linear weight & bias zero-init'd in GoalPredictorO2G._init_weights.
        self.modulator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, 3 * hidden_dim),
        )

        self.hidden_dim = hidden_dim

    def forward(self, h: Tensor, t_emb: Tensor) -> Tensor:
        B = h.shape[0]
        gamma, scale, shift = (
            self.modulator(t_emb).view(B, 3, self.hidden_dim).unbind(1)
        )

        h_norm = self.norm(h)
        h_norm = h_norm.mul(scale.add(1)).add_(shift)
        h = h + self.mlp(h_norm).mul_(gamma)
        return h


class GoalPredictorO2G(BaseNetwork):
    """adaLN-Zero MLP velocity network for last-obs -> goal flow matching.

    Forward signature is FlowMap-compatible:
        forward(x, s, t, condition=None) -> (velocity, scalar=None)

    Args:
        act_dim:    goal embedding dimension (= obs_dim of frozen encoder).
        Ta:         must be 1 — the goal is a single token.
        obs_dim:    observation embedding dimension (== act_dim here).
        To:         number of observation steps (kept for BaseNetwork interface).
        hidden_dim: trunk width.
        num_layers: number of adaLN-Zero blocks.
        mlp_ratio:  expansion factor inside each block's MLP.
        dropout:    dropout in each block's MLP.
        timestep_emb_dim: dim of the sinusoidal/positional time embedding.
        cond_mode:  "none" | "obs_summary".
                    - "none"        : only time conditions modulation.
                    - "obs_summary" : also adds a (B, obs_dim) summary token,
                                       projected to hidden_dim and SUMMED into
                                       t_emb before the modulator.
        timestep_emb_type: positional / sinusoidal / fourier (see embeddings.py).
        timestep_emb_params: kwargs for the embedding constructor.
    """

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        hidden_dim: int,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        timestep_emb_dim: int = 128,
        cond_mode: str = "none",
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
    ):
        super().__init__(act_dim, Ta, obs_dim, To, hidden_dim, num_layers)

        if Ta != 1:
            raise ValueError(f"GoalPredictorO2G expects Ta=1 (single goal token), got Ta={Ta}")
        if cond_mode not in ("none", "obs_summary"):
            raise ValueError(f"Unknown cond_mode '{cond_mode}'")
        if act_dim != obs_dim:
            raise ValueError(
                f"GoalPredictorO2G expects act_dim == obs_dim "
                f"(goal lives in encoder space), got act_dim={act_dim}, obs_dim={obs_dim}"
            )

        self.hidden_dim = hidden_dim
        self.cond_mode = cond_mode
        self.timestep_emb_dim = timestep_emb_dim

        # --- Time embedding: sinusoidal -> 4× expansion -> hidden_dim ---
        timestep_emb_params = timestep_emb_params or {}
        self.time_embedder = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
            timestep_emb_dim, **timestep_emb_params,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_emb_dim, 4 * timestep_emb_dim),
            nn.Mish(),
            nn.Linear(4 * timestep_emb_dim, hidden_dim),
        )

        # --- Optional cond projection (additive into t_emb at hidden_dim) ---
        if cond_mode == "obs_summary":
            self.cond_proj = nn.Linear(obs_dim, hidden_dim)
        else:
            self.cond_proj = None

        # --- Trunk ---
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            _AdaLNZeroBlock(
                hidden_dim=hidden_dim,
                mod_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, obs_dim)

        self._init_weights()

        print(
            f"number of GoalPredictorO2G parameters: "
            f"{sum(p.numel() for p in self.parameters()):e}"
        )

    def _init_weights(self):
        # (a) Global xavier_uniform + zero bias on every Linear.
        def basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(basic_init)

        # (b) Time-MLP Linears overridden to N(0, 0.02²) so the time embedding
        # doesn't dominate the modulator's input at step 0.
        nn.init.normal_(self.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_mlp[2].weight, std=0.02)

        # (c) Modulator's final Linear zero-init'd in every block ⇒
        # γ=scale=shift=0 ⇒ block is identity at init. Step (a) wiped this;
        # restore it here. (VITA/A2A's published code omits this step, so
        # their "identity at init" claim is silently broken — fixed here.)
        for blk in self.blocks:
            nn.init.zeros_(blk.modulator[-1].weight)
            nn.init.zeros_(blk.modulator[-1].bias)

    def _build_t_emb(self, t: Tensor, condition: Tensor | None) -> Tensor:
        """Compute the (B, hidden_dim) modulation signal: time + optional cond."""
        t_emb = self.time_mlp(self.time_embedder(t))  # (B, hidden_dim)

        if self.cond_proj is None or condition is None:
            return t_emb

        # condition is (B, 1, obs_dim) or (B, obs_dim)
        if condition.dim() == 3:
            condition = condition.squeeze(1)
        return t_emb + self.cond_proj(condition)

    def forward(
        self,
        x: Tensor,
        s: Tensor,
        t: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Args:
            x:         (B, 1, obs_dim) noisy goal token at time t.
            s:         (B,) — unused (kept for FlowMap signature compatibility).
            t:         (B,) flow-matching time.
            condition: depends on cond_mode (see class docstring); None ok.
        Returns:
            v:      (B, 1, obs_dim) predicted velocity.
            scalar: None (no auxiliary scalar head).
        """
        del s  # unused — get_velocity passes t as both s and t

        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"x must be (B, 1, obs_dim); got {tuple(x.shape)}")

        t_emb = self._build_t_emb(t, condition)

        h = self.input_proj(x.squeeze(1))                        # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.out_norm(h)
        v = self.output_proj(h).unsqueeze(1)                     # (B, 1, obs_dim)
        return v, None


def test_goal_predictor_o2g():
    print("=" * 60)
    print("Testing GoalPredictorO2G (adaLN-Zero)")
    print("=" * 60)

    obs_dim, To, B = 256, 2, 4
    for cond_mode in ("none", "obs_summary"):
        net = GoalPredictorO2G(
            act_dim=obs_dim,
            Ta=1,
            obs_dim=obs_dim,
            To=To,
            hidden_dim=2 * obs_dim,
            num_layers=3,
            mlp_ratio=4,
            dropout=0.0,
            timestep_emb_dim=128,
            cond_mode=cond_mode,
        )
        x = torch.randn(B, 1, obs_dim)
        t = torch.rand(B)
        cond = None if cond_mode == "none" else torch.randn(B, obs_dim)

        v, scalar = net(x, t, t, cond)
        assert v.shape == (B, 1, obs_dim), v.shape
        assert scalar is None
        # adaLN-Zero ⇒ trunk is identity at init ⇒ v_init = out_proj(LN(input_proj(x))) ≠ 0
        assert not torch.allclose(v, torch.zeros_like(v)), \
            f"v should be non-zero at init (xavier output_proj) for cond_mode={cond_mode}"

        # Identity-at-init check: each block must pass h through unchanged.
        net.eval()
        with torch.no_grad():
            h = net.input_proj(x.squeeze(1))
            t_emb = net._build_t_emb(t, cond)
            for i, blk in enumerate(net.blocks):
                h_after = blk(h, t_emb)
                max_delta = (h_after - h).abs().max().item()
                assert torch.allclose(h_after, h, atol=1e-6), (
                    f"Block {i} is not identity at init "
                    f"(cond_mode={cond_mode}, max delta = {max_delta:.3e})"
                )
                h = h_after
        net.train()

        # Gradient flow check
        v, _ = net(x, t, t, cond)
        v.sum().backward()
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
        )
        assert any_grad, f"no gradient flow for cond_mode={cond_mode}"
        n_with_grad = sum(
            1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        n_total = sum(1 for _ in net.parameters())
        print(f"cond_mode={cond_mode}: v.shape={v.shape}  grads {n_with_grad}/{n_total} non-zero")

    print("=" * 60)
    print("GoalPredictorO2G test passed")
    print("=" * 60)


if __name__ == "__main__":
    test_goal_predictor_o2g()
