"""Closed-loop CFG sweep on the MIXED latest checkpoint.

At inference, cfg_scale>0 does v = (1+w)*v_expert - w*v_null, and here the null
slot is the PLAY conditional -> CFG guides expert AWAY from play. Open-loop on
on-manifold val states this was flat (no contamination there); closed-loop is
where the rollout drifts off-manifold into the play-shaped region, so test it
where it could actually help. Same env seeds across cfg scales.
"""
import os
import numpy as np
import torch
from hydra import compose, initialize

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

MIXED_CKPT = ("logs/pusht_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_"
    "stduniform_atduniform_tsdiagonal_ed256_d256_L8_h16_0.8_s2efalse_"
    "schnoisier_play_spx1_apvelocity_pt_mixed/2026_06_13_02_47_47/models/model_latest.pt")
EVAL_EPISODES = 40
NFE = 25
CFG_SCALES = [0.0, 0.5, 1.0, 2.0, 4.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_config():
    with initialize(version_base=None, config_path="../examples/configs"):
        cfg = compose(config_name="main", overrides=[
            "task=pusht_image", "network=lbmdit_joint_pt",
            "network.encoder_type=image", "network.emb_dim=256",
            "network.encoder_out_dim=256", "network.num_layers=8",
            "network.joint_cond_compose=add", "optimization.loss_type=joint_dit",
            "optimization.joint_state_param=x1", "optimization.joint_action_param=velocity",
            "optimization.joint_decouple_t=true", "optimization.joint_play_scheme=noisier_play",
            "optimization.joint_t_dist=uniform", "optimization.joint_target_ln_affine=true",
            "optimization.joint_use_ema_target=true", "optimization.joint_state_loss_to_encoder=false",
            "optimization.joint_sample_mode=stochastic", "optimization.joint_cfg_scale=0.0",
            "task.horizon=16", "task.act_steps=8", "task.val_dataset_percentage=0.8",
        ])
    cfg.task.obs_dim = cfg.network.emb_dim
    cfg.optimization.device = DEVICE
    cfg.log.eval_episodes = EVAL_EPISODES
    cfg.log.save_video = False
    return cfg


def main():
    import loguru
    from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
    from mip.datasets.pusht_dataset import make_pusht_goal_dataset
    from mip.envs.pusht import make_vec_env
    from examples.train_pusht_lbmdit_joint_pt import evaluate

    cfg = build_config()
    envs = make_vec_env(cfg.task, seed=cfg.optimization.seed)
    envs.reset()
    dataset = make_pusht_goal_dataset(cfg.task, mode="train")

    agent = LBMDiTJointPTAgent(cfg)
    agent.load(MIXED_CKPT, load_optimizer=False)
    agent.eval()

    res = {}
    for w in CFG_SCALES:
        agent._cfg_scale = w
        loguru.logger.info(f"=== MIXED latest, cfg_scale={w}, NFE={NFE} ===")
        m = evaluate(cfg, envs, dataset, agent, None, num_steps=NFE)
        res[w] = m[f"mean_success_{NFE}"]

    print(f"\n=== MIXED latest closed-loop, NFE={NFE}, {EVAL_EPISODES} eps, same seeds ===")
    print(f"{'cfg_scale':>10s} {'mean_success_25':>16s}")
    for w in CFG_SCALES:
        print(f"{w:>10.1f} {res[w]:>16.4f}")
    print("\nref: expert-only latest (cfg=0) = 0.597 ; mixed latest (cfg=0) baseline above")


if __name__ == "__main__":
    main()
