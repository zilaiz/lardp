"""Closed-loop head-to-head: expert-only LATEST vs mixed LATEST checkpoints.

Runs the real PushT closed-loop evaluate() (coverage metric) on both with
identical env seeds, NFE=25 (primary) and 10. This is the ground-truth
comparison the open-loop probes can't give.
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
EVAL_EPISODES = 40   # multiple of num_envs(20) -> 2 rounds
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
    # Expert goal dataset -> normalizer (shared by both runs at train time).
    dataset = make_pusht_goal_dataset(cfg.task, mode="train")

    results = {}
    for tag, ck in [("EXPERT-ONLY-latest", EXPERT_CKPT), ("MIXED-latest", MIXED_CKPT)]:
        loguru.logger.info(f"=== closed-loop eval: {tag} ===")
        agent = LBMDiTJointPTAgent(cfg)
        agent.load(ck, load_optimizer=False)
        agent.eval()
        m = {}
        for nfe in [25, 10]:
            m.update(evaluate(cfg, envs, dataset, agent, None, num_steps=nfe))
        results[tag] = m

    print(f"\n{'metric':24s} {'EXPERT-ONLY-latest':>20s} {'MIXED-latest':>16s}")
    for nfe in [25, 10]:
        for stat in ["mean_success", "mean_reward"]:
            k = f"{stat}_{nfe}"
            e = results["EXPERT-ONLY-latest"].get(k, float("nan"))
            mx = results["MIXED-latest"].get(k, float("nan"))
            print(f"{k:24s} {e:>20.4f} {mx:>16.4f}")
    print(f"\n(eval_episodes={EVAL_EPISODES}, identical env seeds; mean_success = mean per-episode peak coverage)")


if __name__ == "__main__":
    main()
