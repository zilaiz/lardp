"""Closed-loop schedule comparison on the MIXED latest checkpoint.

diagonal    : t_state == t_action (joint state+action denoising; the trained default)
action_only : t_state pinned at eps (state stays at init noise; only action walks)

The state stream is where play contaminated most (+27% worse state pred). If the
(contaminated) joint state token hurts the action via attention in closed loop,
action_only should recover. Open-loop the state->action coupling was tiny (3%);
this tests it in closed loop. cfg=0, NFE=25, same env seeds.
"""
import os
import torch
from hydra import compose, initialize

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

MIXED_CKPT = ("logs/pusht_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_"
    "stduniform_atduniform_tsdiagonal_ed256_d256_L8_h16_0.8_s2efalse_"
    "schnoisier_play_spx1_apvelocity_pt_mixed/2026_06_13_02_47_47/models/model_latest.pt")
EVAL_EPISODES = 40
NFE = 25
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
    for sched in ["diagonal", "action_only"]:
        agent._t_schedule = sched
        loguru.logger.info(f"=== MIXED latest, schedule={sched}, cfg=0, NFE={NFE} ===")
        m = evaluate(cfg, envs, dataset, agent, None, num_steps=NFE)
        res[sched] = m[f"mean_success_{NFE}"]

    print(f"\n=== MIXED latest closed-loop, NFE={NFE}, {EVAL_EPISODES} eps, cfg=0, same seeds ===")
    print(f"{'schedule':>14s} {'mean_success_25':>16s}")
    for sched in ["diagonal", "action_only"]:
        print(f"{sched:>14s} {res[sched]:>16.4f}")
    print("\nref: expert-only latest (diagonal) = 0.597")


if __name__ == "__main__":
    main()
