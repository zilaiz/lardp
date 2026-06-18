"""Routing tests for the noisier-stream loss schemes (``joint_play_scheme``
in {"noisier_all", "noisier_play"}) in ``LBMDiTJointDDTAgent.update()``.

The agent is assembled via ``object.__new__`` with tiny modules so the tests
exercise the real ``update()`` without hydra / image-encoder plumbing. Flow
times are forced by stubbing ``_sample_base_t`` (state drawn first). Verified:

* noisier_play on an all-expert batch == legacy exactly (losses and encoder
  grads) — the expert policy objective is unchanged
* noisier_all, all rows state-noisier: action loss exactly zero, action head
  untrained, encoder receives zero gradient (state loss is trunk-only)
* the same batch under the warned s2e=True ablation: the masked state loss
  DOES shape the encoder (single live forward)
* noisier_all, all rows action-noisier: state loss exactly zero, state head
  untrained, encoder trained via the action loss
* diagonal partition: masks complementary, ties (t_state == t_action) take
  the action loss
* noisier_play source routing: expert rows keep both losses, play rows get
  exactly the noisier stream's loss
* legacy runs carry no mask metrics
"""

from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn

from mip.agent_lbmdit_joint_ddt import LBMDiTJointDDTAgent
from mip.interpolant import Interpolant
from mip.networks.lbmdit_joint_pt import LBMDiTJointPT

B, TO, TA = 6, 2, 4
OBS_F, OBS_D, ACT_D = 5, 8, 3

# Mixed fixed times. state_noisier (t_s < t_a): rows 0, 3, 4. Row 2 is an
# exact tie -> action loss by convention.
T_STATE = [0.10, 0.90, 0.50, 0.70, 0.20, 0.60]
T_ACTION = [0.15, 0.30, 0.50, 0.80, 0.45, 0.55]


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(OBS_F, OBS_D)

    def forward(self, obs, mask=None):
        return self.proj(obs)


class _NullOptimizer:
    """Leaves grads intact through update() so the test can read them."""

    def step(self):
        pass

    def zero_grad(self):
        pass


def _make_agent(scheme: str, seed: int = 0):
    torch.manual_seed(seed)
    agent = object.__new__(LBMDiTJointDDTAgent)
    agent.config = SimpleNamespace(
        optimization=SimpleNamespace(
            ema_rate=1.0,  # disables _ema_update in update()
            norm_type="l2",
            grad_clip_norm=0.0,
        )
    )
    agent.encoder = _Encoder()
    agent.target_ln = nn.LayerNorm(OBS_D, elementwise_affine=False)
    agent.encoder_ema = deepcopy(agent.encoder).requires_grad_(False)
    agent.target_ln_ema = deepcopy(agent.target_ln)
    agent.net = LBMDiTJointPT(
        act_dim=ACT_D,
        Ta=TA,
        obs_dim=OBS_D,
        To=TO,
        d_model=16,
        depth=1,
        n_heads=2,
        timestep_emb_dim=8,
    )
    # AdaLN-Zero init makes head outputs (and their condition-path grads)
    # exactly zero; re-randomize so gradients actually flow.
    for p in agent.net.parameters():
        nn.init.normal_(p, std=0.02)
    agent.interpolant = Interpolant("linear")
    agent.optimizer = _NullOptimizer()

    agent._w_state = 1.0
    agent._w_action = 1.0
    agent._cfg_dropout_prob = 0.0
    agent._decouple_t = True
    agent._play_avoid_both_noise = False
    agent._play_both_noise_tau = 0.5
    agent._t_eps = 0.0
    agent._use_ema_target = False
    agent._state_loss_to_encoder = False
    agent._play_scheme = scheme
    agent._shift_state = 1.0
    agent._shift_action = 1.0
    agent._t_dist = "uniform"
    agent._t_dist_mu = 0.0
    agent._t_dist_sigma = 1.0
    agent._t_dist_state = "uniform"
    agent._t_dist_mu_state = 0.0
    agent._t_dist_sigma_state = 1.0
    agent._t_dist_action = "uniform"
    agent._t_dist_mu_action = 0.0
    agent._t_dist_sigma_action = 1.0
    agent._state_param = "x1"
    agent._action_param = "velocity"
    agent._x1_pred_eps = 5e-2
    return agent


def _fix_t(agent, t_state=T_STATE, t_action=T_ACTION):
    """Make _sample_base_t return fixed values (state drawn first)."""
    vals = iter(
        [
            torch.tensor(t_state, dtype=torch.float32),
            torch.tensor(t_action, dtype=torch.float32),
        ]
    )
    agent._sample_base_t = lambda shape, device, lo, hi, *a, **k: next(vals).to(device)


def _data():
    gen = torch.Generator().manual_seed(7)
    act = torch.randn(B, TA, ACT_D, generator=gen)
    obs = torch.randn(B, TO, OBS_F, generator=gen)
    goal = torch.randn(B, 1, OBS_F, generator=gen)
    return act, obs, goal


def _run(agent, optimality, t_state=T_STATE, t_action=T_ACTION):
    _fix_t(agent, t_state, t_action)
    act, obs, goal = _data()
    torch.manual_seed(123)  # identical interpolation noise across variants
    info = agent.update(act, obs, goal, torch.zeros(B), optimality)
    grads = {
        "encoder": [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in agent.encoder.parameters()
        ],
        "state_head": [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in agent.net.state_final.parameters()
        ],
        "action_head": [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in agent.net.action_final.parameters()
        ],
    }
    return info, grads


def _max_abs(grads):
    return max(g.abs().max().item() for g in grads)


EXPERT = torch.zeros(B, dtype=torch.long)


def test_noisier_play_all_expert_matches_legacy():
    # Expert rows are exempt from the rule, so an all-expert batch must
    # reproduce legacy exactly: same losses, same encoder grads. Both paths
    # consume the identical RNG stream (t is stubbed, corner-avoidance off).
    info_l, g_l = _run(_make_agent("legacy"), EXPERT)
    info_n, g_n = _run(_make_agent("noisier_play"), EXPERT)
    for key in ("state_loss", "action_loss", "dp_loss"):
        assert torch.isclose(info_l[key], info_n[key], atol=1e-6), key
    for a, b in zip(g_l["encoder"], g_n["encoder"], strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_noisier_all_state_side():
    # Every row state-noisier -> state loss only. The action loss is the
    # encoder's only gradient path (s2e=False), so the encoder must be
    # untouched by a pure state-side batch.
    info, grads = _run(
        _make_agent("noisier_all"), EXPERT,
        t_state=[0.1, 0.2, 0.3, 0.15, 0.25, 0.05],
        t_action=[0.6, 0.7, 0.8, 0.65, 0.75, 0.95],
    )
    assert info["action_loss"].item() == 0.0
    assert _max_abs(grads["action_head"]) == 0.0, (
        "state-side batch must not train the action head"
    )
    assert _max_abs(grads["encoder"]) == 0.0, (
        "state loss must stay trunk-only (no encoder gradient)"
    )
    assert info["state_loss"].item() > 0.0
    assert _max_abs(grads["state_head"]) > 0.0
    assert info["state_mask_frac"].item() == 1.0
    assert info["action_mask_frac"].item() == 0.0


def test_noisier_all_state_side_s2e_true_ablation():
    # Same pure state-side batch as above, but with the warned s2e=True
    # ablation: the single live forward lets the masked state loss shape
    # the encoder.
    agent = _make_agent("noisier_all")
    agent._state_loss_to_encoder = True
    info, grads = _run(
        agent, EXPERT,
        t_state=[0.1, 0.2, 0.3, 0.15, 0.25, 0.05],
        t_action=[0.6, 0.7, 0.8, 0.65, 0.75, 0.95],
    )
    assert info["action_loss"].item() == 0.0
    assert info["state_loss"].item() > 0.0
    assert _max_abs(grads["encoder"]) > 0.0, (
        "s2e=True ablation must route the state loss into the encoder"
    )


def test_noisier_all_action_side():
    # Every row action-noisier -> action loss only; encoder trained via it.
    info, grads = _run(
        _make_agent("noisier_all"), EXPERT,
        t_state=[0.6, 0.7, 0.8, 0.65, 0.75, 0.95],
        t_action=[0.1, 0.2, 0.3, 0.15, 0.25, 0.05],
    )
    assert info["state_loss"].item() == 0.0
    assert _max_abs(grads["state_head"]) == 0.0, (
        "action-side batch must not train the state head"
    )
    assert info["action_loss"].item() > 0.0
    assert _max_abs(grads["action_head"]) > 0.0
    assert _max_abs(grads["encoder"]) > 0.0, (
        "action loss must reach the encoder"
    )
    assert info["state_mask_frac"].item() == 0.0
    assert info["action_mask_frac"].item() == 1.0


def test_noisier_all_partition_and_ties():
    # Mixed times: rows 0/3/4 state-noisier, rows 1/5 action-noisier, row 2
    # an exact tie -> action. Masks must partition the batch 3/3; a
    # tie-to-state convention would read 4/2 instead.
    info, _ = _run(_make_agent("noisier_all"), EXPERT)
    assert torch.isclose(info["state_mask_frac"], torch.tensor(0.5))
    assert torch.isclose(info["action_mask_frac"], torch.tensor(0.5))
    assert info["state_loss"].item() > 0.0
    assert info["action_loss"].item() > 0.0


def test_noisier_play_source_routing():
    # Expert rows (0-2) keep both losses; play rows (3-5) follow the rule:
    # rows 3/4 state-noisier (state loss only), row 5 action-noisier.
    optimality = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    info, _ = _run(_make_agent("noisier_play"), optimality)
    assert torch.isclose(info["state_mask_frac"], torch.tensor(5.0 / 6.0))
    assert torch.isclose(info["action_mask_frac"], torch.tensor(4.0 / 6.0))


def test_noisier_play_unlabeled_fallback_routes_as_expert():
    # optimality=None (unlabeled dataset) defaults every row to NULL but
    # might actually be expert -> the rule must not engage (all rows keep
    # both losses, matching legacy).
    info, _ = _run(_make_agent("noisier_play"), None)
    assert info["state_mask_frac"].item() == 1.0
    assert info["action_mask_frac"].item() == 1.0


def test_legacy_has_no_mask_metrics():
    info, _ = _run(_make_agent("legacy"), EXPERT)
    assert "state_mask_frac" not in info
    assert "action_mask_frac" not in info


if __name__ == "__main__":
    test_noisier_play_all_expert_matches_legacy()
    test_noisier_all_state_side()
    test_noisier_all_state_side_s2e_true_ablation()
    test_noisier_all_action_side()
    test_noisier_all_partition_and_ties()
    test_noisier_play_source_routing()
    test_noisier_play_unlabeled_fallback_routes_as_expert()
    test_legacy_has_no_mask_metrics()
    print("All noisier-scheme routing tests passed.")
