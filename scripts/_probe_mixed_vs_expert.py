"""Localize WHY mixed-data joint_pt degrades closed-loop vs expert-only.

Loads both checkpoints and, on identical TRAIN-expert and VAL-expert (held-out)
observation batches, reports per checkpoint:
  * expert-cond action MSE on train vs val  -> generalization gap
  * null-cond action MSE on val + a_expert/a_null divergence -> conditioning effect
  * FM state-stream prediction MSE (expert-cond) vs the LN'd goal embedding
  * encoder representation health: effective rank + target_std on val obs
And across the two checkpoints on the same val obs:
  * linear CKA between encoder features -> how far play shifted the representation
All read-only, EMA weights, fixed noise for fair comparison.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

EXPERT_CKPT = (
    "logs/pusht_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_"
    "stduniform_atduniform_tsdiagonal_ed256_d256_L8_h16_0.8_s2efalse_"
    "schnoisier_play_spx1_apvelocity_pt/2026_06_12_23_30_21/models/model_latest.pt"
)
MIXED_CKPT = (
    "logs/pusht_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_"
    "stduniform_atduniform_tsdiagonal_ed256_d256_L8_h16_0.8_s2efalse_"
    "schnoisier_play_spx1_apvelocity_pt_mixed/2026_06_13_02_47_47/models/model_latest.pt"
)
NUM_STEPS = 25
B = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_config():
    with initialize(version_base=None, config_path="../examples/configs"):
        cfg = compose(config_name="main", overrides=[
            "task=pusht_image_mixed", "network=lbmdit_joint_pt",
            "network.encoder_type=image", "network.emb_dim=256",
            "network.encoder_out_dim=256", "network.num_layers=8",
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
            "task.horizon=16", "task.act_steps=8",
            "task.val_dataset_percentage=0.8", "task.rollout_dataset_paths=null",
        ])
    cfg.task.obs_dim = cfg.network.emb_dim
    return cfg


@torch.no_grad()
def sample_with_opt(agent, z_t, x_state0, act_0, opt_value, steps):
    net = agent.net_ema
    device = act_0.device
    Bn = act_0.shape[0]
    x_state = x_state0.clone()
    x_action = act_0.clone()
    opt_idx = torch.full((Bn,), opt_value, device=device, dtype=torch.long)
    t_s_grid, t_a_grid = agent._build_schedule(steps)
    for i in range(steps):
        ts, ta = float(t_s_grid[i]), float(t_a_grid[i])
        ds, da = float(t_s_grid[i+1]-ts), float(t_a_grid[i+1]-ta)
        tsb = torch.full((Bn,), ts, device=device); tab = torch.full((Bn,), ta, device=device)
        s_head, a_head, _ = net(x_state, x_action, tsb, tab, z_t, opt_idx)
        v_s = agent._head_to_velocity(s_head, None, x_state, ts, 0.0, None,
                                      use_x1_param=(agent._state_param == "x1"))
        v_a = agent._head_to_velocity(a_head, None, x_action, ta, 0.0, None,
                                      use_x1_param=(agent._action_param == "x1"))
        x_state = x_state + v_s*ds
        x_action = x_action + v_a*da
    return x_action, x_state


def eff_rank(X):
    """Stable rank + participation ratio of centered features X (N, D)."""
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    lam = s**2
    stable = (s.sum()**2) / (lam.sum())          # ~ participation ratio of singular values
    part = (lam.sum()**2) / (lam**2).sum()        # participation ratio of eigenvalues
    return float(stable.item()), float(part.item())


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
    num = (Xc.T @ Yc).norm()**2
    den = (Xc.T @ Xc).norm() * (Yc.T @ Yc).norm()
    return float((num/den).item())


def load_agent(ckpt, cfg):
    from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
    a = LBMDiTJointPTAgent(cfg)
    a.load(ckpt, load_optimizer=False)
    a.eval()
    return a


def get_batch(cfg, mode):
    from mip.datasets.pusht_dataset import make_pusht_goal_dataset
    ds = make_pusht_goal_dataset(cfg.task, mode=mode)
    if hasattr(ds, "datasets"):
        ds = ds.datasets[0]
    idxs = np.linspace(0, len(ds)-1, B).astype(int)
    obs = {k: torch.stack([ds[i]["obs"][k] for i in idxs]).to(DEVICE) for k in ds[0]["obs"]}
    goal = {k: torch.stack([ds[i]["goal_obs"][k] for i in idxs]).to(DEVICE) for k in ds[0]["goal_obs"]}
    a_gt = torch.stack([ds[i]["action"] for i in idxs]).to(DEVICE)
    return obs, goal, a_gt


@torch.no_grad()
def eval_ckpt(agent, batches, noise):
    out = {}
    x_state0, act_0 = noise
    for mode,(obs,goal,a_gt) in batches.items():
        enc, tln = agent._eval_encoder_modules(use_ema=True)
        z_t = tln(enc(obs, None))
        a_exp,_ = sample_with_opt(agent, z_t, x_state0, act_0, agent.net.EXPERT_IDX, NUM_STEPS)
        out[f"{mode}_act_mse_expert"] = F.mse_loss(a_exp, a_gt).item()
        if mode == "val":
            a_null,_ = sample_with_opt(agent, z_t, x_state0, act_0, agent.net.NULL_IDX, NUM_STEPS)
            out["val_act_mse_null"] = F.mse_loss(a_null, a_gt).item()
            out["val_a_exp_vs_null"] = (a_exp-a_null).abs().mean().item()
            # state-stream: predicted next-state embedding vs LN'd gt goal embedding
            tgt = tln(enc(goal, None))                      # (B,1,D) LN'd goal
            _, x_state_pred = sample_with_opt(agent, z_t, x_state0, act_0, agent.net.EXPERT_IDX, NUM_STEPS)
            out["val_state_mse_expert"] = F.mse_loss(x_state_pred, tgt).item()
            # representation health on val
            zf = z_t.reshape(-1, z_t.shape[-1])
            sr, pr = eff_rank(zf)
            out["val_z_stable_rank"] = sr
            out["val_z_part_ratio"] = pr
            out["val_target_std"] = z_t.std(dim=(0,1)).mean().item()
    return out


def main():
    cfg = build_config(); cfg.optimization.device = DEVICE
    batches = {"train": get_batch(cfg, "train"), "val": get_batch(cfg, "val")}
    g = torch.Generator(device=DEVICE).manual_seed(0)
    D = cfg.network.encoder_out_dim
    noise = (torch.randn(B,1,D,generator=g,device=DEVICE),
             torch.randn(B,cfg.task.horizon,cfg.task.act_dim,generator=g,device=DEVICE))

    print("\nloading expert-only ..."); ae = load_agent(EXPERT_CKPT, cfg)
    re = eval_ckpt(ae, batches, noise)
    print("loading mixed ...");        am = load_agent(MIXED_CKPT, cfg)
    rm = eval_ckpt(am, batches, noise)

    # cross-checkpoint representation shift on val obs
    obs_val = batches["val"][0]
    ze = ae._eval_encoder_modules(True); zm = am._eval_encoder_modules(True)
    fe = ze[1](ze[0](obs_val, None)).reshape(-1, D)
    fm = zm[1](zm[0](obs_val, None)).reshape(-1, D)
    cka = linear_cka(fe, fm)

    keys = ["train_act_mse_expert","val_act_mse_expert","val_act_mse_null",
            "val_a_exp_vs_null","val_state_mse_expert","val_z_stable_rank",
            "val_z_part_ratio","val_target_std"]
    print(f"\n{'metric':28s} {'EXPERT-ONLY':>14s} {'MIXED':>14s}")
    for k in keys:
        print(f"{k:28s} {re.get(k,float('nan')):>14.5f} {rm.get(k,float('nan')):>14.5f}")
    print(f"\ntrain->val action-MSE blowup:  expert x{re['val_act_mse_expert']/re['train_act_mse_expert']:.1f}"
          f"   mixed x{rm['val_act_mse_expert']/rm['train_act_mse_expert']:.1f}")
    print(f"linear CKA(expert-enc, mixed-enc) on val obs: {cka:.4f}  (1.0=identical representation)")


if __name__ == "__main__":
    main()
