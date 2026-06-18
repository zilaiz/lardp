"""Decisive check: does mixed-data joint_pt diverge from expert-only specifically
OFF the expert manifold (where closed-loop rollouts live)?

On-manifold = held-out VAL expert observations.
Off-manifold = collected ROLLOUT observations (the states the policy actually
visits in closed loop; mostly non-expert).

For each obs set, with identical init noise, measure:
  * agreement between expert-only and mixed expert-conditioned actions
    (mean|a_expert_ckpt - a_mixed_ckpt|) -> do the two policies still agree?
  * within each checkpoint, expert-cond vs null-cond action divergence
    -> does the optimality conditioning still separate off-manifold?
  * state-stream prediction magnitude / disagreement.
If the two checkpoints agree on-manifold but diverge off-manifold, the play data
changed behavior precisely where closed-loop operates.
"""
import os
import numpy as np
import torch
from hydra import compose, initialize

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

EXPERT_CKPT = ("logs/pusht_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_"
    "stduniform_atduniform_tsdiagonal_ed256_d256_L8_h16_0.8_s2efalse_"
    "schnoisier_play_spx1_apvelocity_pt/2026_06_12_23_30_21/models/model_latest.pt")
MIXED_CKPT = ("logs/pusht_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_"
    "stduniform_atduniform_tsdiagonal_ed256_d256_L8_h16_0.8_s2efalse_"
    "schnoisier_play_spx1_apvelocity_pt_mixed/2026_06_13_02_47_47/models/model_latest.pt")
ROLLOUT = "data/pusht/image_rollouts.hdf5"
NUM_STEPS = 25
B = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_config():
    with initialize(version_base=None, config_path="../examples/configs"):
        cfg = compose(config_name="main", overrides=[
            "task=pusht_image_mixed", "network=lbmdit_joint_pt",
            "network.encoder_type=image", "network.emb_dim=256",
            "network.encoder_out_dim=256", "network.num_layers=8",
            "network.joint_cond_compose=add", "optimization.loss_type=joint_dit",
            "optimization.joint_state_param=x1", "optimization.joint_action_param=velocity",
            "optimization.joint_decouple_t=true", "optimization.joint_play_scheme=noisier_play",
            "optimization.joint_t_dist=uniform", "optimization.joint_target_ln_affine=true",
            "optimization.joint_use_ema_target=true", "optimization.joint_state_loss_to_encoder=false",
            "optimization.joint_sample_mode=stochastic", "optimization.joint_cfg_scale=0.0",
            "task.horizon=16", "task.act_steps=8", "task.val_dataset_percentage=0.8",
            "task.rollout_dataset_paths=null",
        ])
    cfg.task.obs_dim = cfg.network.emb_dim
    return cfg


@torch.no_grad()
def sample_with_opt(agent, z_t, x_state0, act_0, opt_value, steps):
    net = agent.net_ema; Bn = act_0.shape[0]; device = act_0.device
    x_state = x_state0.clone(); x_action = act_0.clone()
    opt_idx = torch.full((Bn,), opt_value, device=device, dtype=torch.long)
    ts_g, ta_g = agent._build_schedule(steps)
    for i in range(steps):
        ts, ta = float(ts_g[i]), float(ta_g[i]); ds, da = float(ts_g[i+1]-ts), float(ta_g[i+1]-ta)
        s_head, a_head, _ = net(x_state, x_action,
                                torch.full((Bn,),ts,device=device), torch.full((Bn,),ta,device=device),
                                z_t, opt_idx)
        v_s = agent._head_to_velocity(s_head, None, x_state, ts, 0.0, None, use_x1_param=(agent._state_param=="x1"))
        v_a = agent._head_to_velocity(a_head, None, x_action, ta, 0.0, None, use_x1_param=(agent._action_param=="x1"))
        x_state = x_state + v_s*ds; x_action = x_action + v_a*da
    return x_action, x_state


def load_agent(ckpt, cfg):
    from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
    a = LBMDiTJointPTAgent(cfg); a.load(ckpt, load_optimizer=False); a.eval(); return a


def val_obs(cfg):
    from mip.datasets.pusht_dataset import make_pusht_goal_dataset
    ds = make_pusht_goal_dataset(cfg.task, mode="val")
    idxs = np.linspace(0, len(ds)-1, B).astype(int)
    return {k: torch.stack([ds[i]["obs"][k] for i in idxs]).to(DEVICE) for k in ds[0]["obs"]}


def rollout_obs(cfg):
    from mip.datasets.pusht_dataset import PushTImageGoalDataset, load_pusht_rollout_replay_buffer
    rb = load_pusht_rollout_replay_buffer(ROLLOUT)
    ds = PushTImageGoalDataset(replay_buffer=rb, shape_meta=cfg.task.shape_meta,
                               n_obs_steps=cfg.task.obs_steps, horizon=cfg.task.horizon,
                               pad_before=cfg.task.obs_steps-1, pad_after=cfg.task.act_steps-1,
                               normalizer=None)
    idxs = np.linspace(0, len(ds)-1, B).astype(int)
    return {k: torch.stack([ds[i]["obs"][k] for i in idxs]).to(DEVICE) for k in ds[0]["obs"]}


@torch.no_grad()
def measure(ae, am, obs, noise):
    x0, a0 = noise
    out = {}
    for tag, ag in [("expert", ae), ("mixed", am)]:
        enc, tln = ag._eval_encoder_modules(True); z = tln(enc(obs, None))
        a_e,_ = sample_with_opt(ag, z, x0, a0, ag.net.EXPERT_IDX, NUM_STEPS)
        a_n,_ = sample_with_opt(ag, z, x0, a0, ag.net.NULL_IDX, NUM_STEPS)
        out[tag+"_a"] = a_e
        out[tag+"_expnull_div"] = (a_e-a_n).abs().mean().item()
    out["cross_ckpt_div"] = (out["expert_a"]-out["mixed_a"]).abs().mean().item()
    return out


def main():
    cfg = build_config(); cfg.optimization.device = DEVICE
    g = torch.Generator(device=DEVICE).manual_seed(0); D = cfg.network.encoder_out_dim
    noise = (torch.randn(B,1,D,generator=g,device=DEVICE),
             torch.randn(B,cfg.task.horizon,cfg.task.act_dim,generator=g,device=DEVICE))
    print("loading agents ..."); ae = load_agent(EXPERT_CKPT, cfg); am = load_agent(MIXED_CKPT, cfg)
    on = measure(ae, am, val_obs(cfg), noise)
    off = measure(ae, am, rollout_obs(cfg), noise)
    print(f"\n{'metric':34s} {'ON-manifold(val)':>18s} {'OFF-manifold(rollout)':>22s}")
    print(f"{'expert-only: expert-vs-null div':34s} {on['expert_expnull_div']:>18.4f} {off['expert_expnull_div']:>22.4f}")
    print(f"{'mixed:       expert-vs-null div':34s} {on['mixed_expnull_div']:>18.4f} {off['mixed_expnull_div']:>22.4f}")
    print(f"{'cross-ckpt expert-action div':34s} {on['cross_ckpt_div']:>18.4f} {off['cross_ckpt_div']:>22.4f}")
    print(f"\ncross-ckpt divergence off/on ratio: {off['cross_ckpt_div']/on['cross_ckpt_div']:.2f}x")
    print("(if >>1: the two policies agree on expert states but diverge where closed-loop operates)")


if __name__ == "__main__":
    main()
