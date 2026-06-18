"""Probe: are the joint_pt optimality embeddings trained, and do they change
behavior under expert observations?

Read-only. Builds the agent from a reconstructed config, loads the mixed-data
checkpoint, then for a batch of real expert observations samples the action
chunk TWICE with identical init noise — once conditioned on EXPERT (idx 0),
once on NULL/play (idx 1) — and compares:
  * do the two action chunks differ at all (optimality has causal effect)?
  * which is closer to the ground-truth expert action (expert-cond better)?
  * relative magnitude of the optimality conditioning vs the obs conditioning
    in the additive AdaLN cond (does opt even have signal to matter?).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

CKPT = (
    "logs/pusht_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_"
    "stduniform_atduniform_tsdiagonal_ed256_d256_L8_h16_0.8_s2efalse_"
    "schnoisier_play_spx1_apvelocity_pt_mixed/2026_06_13_02_47_47/models/model_latest.pt"
)
NUM_STEPS = 25
B = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_config():
    with initialize(version_base=None, config_path="../examples/configs"):
        cfg = compose(
            config_name="main",
            overrides=[
                "task=pusht_image_mixed",
                "network=lbmdit_joint_pt",
                "network.encoder_type=image",
                "network.emb_dim=256",
                "network.encoder_out_dim=256",
                "network.num_layers=8",
                "network.joint_cond_compose=add",
                "optimization.loss_type=joint_dit",
                "optimization.joint_state_param=x1",
                "optimization.joint_action_param=velocity",
                "optimization.joint_decouple_t=true",
                "optimization.joint_play_scheme=noisier_play",
                "optimization.joint_t_dist=uniform",
                "optimization.joint_target_ln_affine=true",
                "optimization.joint_use_ema_target=true",
                "optimization.joint_state_loss_to_encoder=false",
                "optimization.joint_sample_mode=stochastic",
                "optimization.joint_cfg_scale=0.0",
                "task.horizon=16",
                "task.act_steps=8",
                "task.val_dataset_percentage=0.8",
                # don't need rollouts for this probe; expert-only dataset
                "task.rollout_dataset_paths=null",
            ],
        )
    cfg.task.obs_dim = cfg.network.emb_dim
    return cfg


@torch.no_grad()
def sample_with_opt(agent, obs, act_0, opt_value, steps):
    """Faithful copy of LBMDiTJointPTAgent.sample diagonal loop, but with a
    fixed optimality index (so we can flip expert<->null on identical noise)."""
    net = agent.net_ema
    encoder, target_ln = agent._eval_encoder_modules(use_ema=True)
    device = act_0.device
    Bn = act_0.shape[0]
    z_t = target_ln(encoder(obs, None))
    obs_dim = z_t.shape[-1]
    x_state = torch.randn(Bn, 1, obs_dim, generator=None, device=device) if False else None
    # caller supplies init noise for determinism
    x_state = sample_with_opt._x_state
    x_action = act_0.clone()
    opt_idx = torch.full((Bn,), opt_value, device=device, dtype=torch.long)
    t_s_grid, t_a_grid = agent._build_schedule(steps)
    for i in range(steps):
        ts, ta = float(t_s_grid[i]), float(t_a_grid[i])
        ds, da = float(t_s_grid[i + 1] - ts), float(t_a_grid[i + 1] - ta)
        tsb = torch.full((Bn,), ts, device=device)
        tab = torch.full((Bn,), ta, device=device)
        s_head, a_head, _ = net(x_state, x_action, tsb, tab, z_t, opt_idx)
        v_s = agent._head_to_velocity(s_head, None, x_state, ts, 0.0, None,
                                      use_x1_param=(agent._state_param == "x1"))
        v_a = agent._head_to_velocity(a_head, None, x_action, ta, 0.0, None,
                                      use_x1_param=(agent._action_param == "x1"))
        x_state = x_state + v_s * ds
        x_action = x_action + v_a * da
    return x_action


def main():
    from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
    from mip.datasets.pusht_dataset import make_pusht_goal_dataset

    cfg = build_config()
    cfg.optimization.device = DEVICE
    agent = LBMDiTJointPTAgent(cfg)
    agent.load(CKPT, load_optimizer=False)
    agent.eval()

    # Expert observations + gt actions (expert-only goal dataset, train split).
    ds = make_pusht_goal_dataset(cfg.task, mode="train")
    if hasattr(ds, "datasets"):
        ds = ds.datasets[0]
    idxs = np.linspace(0, len(ds) - 1, B).astype(int)
    obs = {k: torch.stack([ds[i]["obs"][k] for i in idxs]).to(DEVICE) for k in ds[0]["obs"]}
    a_gt = torch.stack([ds[i]["action"] for i in idxs]).to(DEVICE)  # (B, H, 2) normalized

    # --- conditioning-magnitude check: ||opt|| vs ||obs|| in the additive cond ---
    net = agent.net_ema
    enc, tln = agent._eval_encoder_modules(use_ema=True)
    z_t = tln(enc(obs, None))
    obs_feat = net.obs_mlp(z_t.flatten(1))                       # (B, d_model)
    emb = net.optimality_embedding.weight                        # (2, opt_dim)
    opt_feat = net.opt_mlp(emb)                                  # (2, d_model)
    print("\n=== conditioning magnitudes (additive cond = time + obs + opt) ===")
    print(f"  ||obs_feat|| mean over batch: {obs_feat.norm(dim=-1).mean():.3f}")
    print(f"  ||opt_feat|| expert={opt_feat[0].norm():.3f}  null={opt_feat[1].norm():.3f}")
    print(f"  opt/obs norm ratio: expert={opt_feat[0].norm()/obs_feat.norm(dim=-1).mean():.3f}"
          f"  null={opt_feat[1].norm()/obs_feat.norm(dim=-1).mean():.3f}")

    # --- behavior probe: identical init noise, flip optimality ---
    g = torch.Generator(device=DEVICE).manual_seed(0)
    sample_with_opt._x_state = torch.randn(B, 1, z_t.shape[-1], generator=g, device=DEVICE)
    act_0 = torch.randn(B, cfg.task.horizon, cfg.task.act_dim, generator=g, device=DEVICE)

    a_exp = sample_with_opt(agent, obs, act_0, agent.net.EXPERT_IDX, NUM_STEPS)
    a_null = sample_with_opt(agent, obs, act_0, agent.net.NULL_IDX, NUM_STEPS)

    act_scale = a_gt.abs().mean().item()
    d_en = (a_exp - a_null).abs().mean().item()
    mse_exp = F.mse_loss(a_exp, a_gt).item()
    mse_null = F.mse_loss(a_null, a_gt).item()
    print("\n=== behavior under EXPERT obs: expert-cond vs null-cond (same noise) ===")
    print(f"  mean|a_gt| (action scale, normalized): {act_scale:.4f}")
    print(f"  mean|a_expert - a_null|:               {d_en:.4f}  "
          f"({100*d_en/act_scale:.1f}% of action scale)")
    print(f"  per-step max|a_expert - a_null|:       {(a_exp-a_null).abs().amax(dim=(0,2)).mean():.4f}")
    print(f"  MSE(a_expert, a_gt) = {mse_exp:.4f}")
    print(f"  MSE(a_null,   a_gt) = {mse_null:.4f}")
    print(f"  -> expert-cond {'CLOSER' if mse_exp < mse_null else 'NOT closer'} to expert gt"
          f" (ratio null/expert = {mse_null/mse_exp:.3f})")
    if d_en / act_scale < 0.02:
        print("\n  *** optimality has NEGLIGIBLE effect on the action (<2% of scale):"
              " the trunk effectively ignores it -> expert inference still sees play-contaminated behavior.")


if __name__ == "__main__":
    main()
