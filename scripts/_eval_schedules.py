"""Roll out ckpt2 (s2e=true) under three inference t-schedules (diagonal,
state_first, action_only), 50 rollouts each, at 25 and 10 denoising steps.
Loads the run's exact resolved config from its wandb debug.log so the network/
task/sampling config matches training; builds env+dataset+agent once."""
import ast, os, sys
from pathlib import Path
os.environ["MUJOCO_GL"]="egl"
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np, loguru
from omegaconf import OmegaConf
from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
from mip.datasets.robomimic_dataset import make_idm_dataset
from mip.envs.robomimic.robomimic_env import make_vec_env
from mip.torch_utils import set_seed
from examples.train_robomimic_lbmdit_joint_e2e import eval as eval_fn

RUN=ROOT/"logs/can_ph_image_joint_dit_None_lbmdit_joint_pt_256_seed0_stduniform_atduniform_tsdiagonal_ys1.0_ya1.0_ed256_d256_L8_h10_0.9_s2etrue_schnoisier_all_elr1.0_esfnull_cs0.0_ccadd_spx1_apvelocity_sw1.0_panfalse_pt/2026_06_12_16_07_04"
CKPT="/tmp/ckpt2_frozen.pt"  # frozen step-80k snapshot (training still writing model_latest)

def load_cfg():
    dbg=list((RUN/"wandb/latest-run/logs").glob("debug.log"))[0].read_text()
    i=dbg.index("config: {'log'")+len("config: ")  # skip wandb's sweep_config: {}
    depth=0; j=i
    for j in range(i,len(dbg)):
        if dbg[j]=='{': depth+=1
        elif dbg[j]=='}':
            depth-=1
            if depth==0: break
    d=ast.literal_eval(dbg[i:j+1])
    d.pop("_wandb",None)
    return OmegaConf.create(d)

class DummyLogger:
    def video_init(self,*a,**k): pass

def main():
    cfg=load_cfg()
    cfg.mode="eval"; cfg.optimization.device="cuda"; cfg.log.save_video=False
    cfg.optimization.model_path=CKPT; cfg.optimization.auto_resume=False
    cfg.log.eval_episodes=50
    set_seed(cfg.optimization.seed)
    loguru.logger.info(f"dataset={list(cfg.task.dataset_paths)} act_dim={cfg.task.act_dim} "
                       f"net={cfg.network.network_type} emb={cfg.network.emb_dim} L={cfg.network.num_layers} "
                       f"sample_mode={cfg.optimization.joint_sample_mode} eval_eps={cfg.log.eval_episodes}")
    envs=make_vec_env(cfg.task, seed=cfg.optimization.seed)
    cfg.task.obs_dim=cfg.network.emb_dim
    dataset=make_idm_dataset(cfg.task)
    agent=LBMDiTJointPTAgent(cfg)
    agent.load(CKPT, load_optimizer=False)
    agent.eval()
    logger=DummyLogger()
    scheds=sys.argv[1:] or ["diagonal","state_first","action_only"]
    for sched in scheds:
        agent._t_schedule=sched
        set_seed(cfg.optimization.seed)  # same env reset sequence across schedules
        m=eval_fn(cfg, envs, dataset, agent, logger, 25)
        print(f">>> sched={sched:<12} steps=25  success={m['mean_success_25']:.3f}  "
              f"reward={m['mean_reward_25']:.2f}", flush=True)

if __name__=="__main__": main()
