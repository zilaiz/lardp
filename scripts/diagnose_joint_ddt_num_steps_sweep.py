"""num_steps sweep for an LBMDiTJointDDTAgent checkpoint.

Answers: "would more Euler steps narrow the gap to oracle_pinned?". For
each step count in --num_steps_list, runs four regimes on identical obs +
identical (act_0, x_state_init):

  diagonal      — default sampler (t_state == t_action 0->1)
  state_first   — clean state first, then clean action
  noise_pinned  — t_state pinned at lo, x_state frozen at randn (lower bound)
  oracle_pinned — t_state pinned at hi, x_state frozen at target_ln(enc(goal))
                  (upper bound — note this regime doesn't depend on num_steps
                  on the state side, so its num_steps trend reflects only the
                  action-stream Euler discretization)

Reports per (num_steps, regime): pos mean/max (mm), rot mean/max (deg),
gripper L1, gripper binary-flip %, and cos(x_state_final, oracle).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

LARDP_PATH = Path(__file__).resolve().parents[1]
if str(LARDP_PATH) not in sys.path:
    sys.path.append(str(LARDP_PATH))

from mip.agent_lbmdit_joint_ddt import LBMDiTJointDDTAgent
from mip.datasets.robomimic_dataset import make_idm_dataset
# Re-use the helpers from the schedule-sweep script — same conventions and
# physical-unit decoding (un-normalize, rot6d -> matrix, geodesic deg).
from scripts.diagnose_joint_ddt_schedule_sweep import (
    _abs_space_stats,
    _sample_with_pinning,
    _sample_with_schedule,
)


def diagnose(args):
    cfg = OmegaConf.load(args.config_path)
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)
    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)

    device = torch.device(cfg.optimization.device)
    print(f"\n== device: {device} ==")
    print(f"   ckpt:   {args.ckpt_path}")
    print(f"   config: {args.config_path}")

    print("\n[1/3] Building agent + loading checkpoint...")
    agent = LBMDiTJointDDTAgent(cfg)
    agent.load(args.ckpt_path, load_optimizer=False)
    agent.eval()

    # ---- dataset (heldout with re-derived training normalizer) ----
    norm_override = None
    if args.dataset_path is not None:
        print("\n[2a/3] Re-deriving normalizer from training dataset...")
        train_dataset = make_idm_dataset(cfg.task, mode="train")
        base_train = (
            train_dataset.datasets[0]
            if isinstance(train_dataset, torch.utils.data.ConcatDataset)
            else train_dataset
        )
        norm_override = base_train.normalizer
        del train_dataset, base_train
        OmegaConf.update(cfg, "task.dataset_paths", [args.dataset_path],
                         merge=False)
        OmegaConf.update(cfg, "task.dataset_path", None, merge=False)
        OmegaConf.update(cfg, "task.val_dataset_percentage", 0.0, merge=False)
        print(f"\n[2/3] Loading heldout dataset: {args.dataset_path}")
    else:
        print("\n[2/3] Loading training dataset...")

    dataset = make_idm_dataset(cfg.task, mode="train", normalizer=norm_override)
    base_ds = (
        dataset.datasets[0]
        if isinstance(dataset, torch.utils.data.ConcatDataset)
        else dataset
    )
    action_normalizer = base_ds.normalizer["action"]
    n = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    print(f"   dataset size={len(dataset)}, sampling {n} indices")

    sample_keys = list(cfg.task.shape_meta["obs"].keys())
    obs_batch = {k: [] for k in sample_keys}
    goal_batch = {k: [] for k in sample_keys}
    act_batch_normed = []
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
            goal_batch[k].append(s["goal_obs"][k])
        act_batch_normed.append(s["action"][: cfg.task.horizon])

    obs_torch = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in obs_batch.items()
    }
    goal_torch = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in goal_batch.items()
    }
    act_true = torch.from_numpy(
        np.stack([t.numpy() for t in act_batch_normed])
    ).to(device)

    obs_dim = cfg.network.encoder_out_dim or cfg.network.emb_dim
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    act_0 = torch.randn(
        (n, cfg.task.horizon, cfg.task.act_dim), generator=g,
    ).to(device)
    x_state_init = torch.randn(
        (n, 1, obs_dim), generator=g,
    ).to(device)

    with torch.no_grad():
        encoder, target_ln = agent._eval_encoder_modules(use_ema=True)
        s_oracle = target_ln(encoder(goal_torch, None))

    # ---- modes to sweep ----
    modes = [
        ("diagonal",      "schedule", "diagonal",     0.0),
        ("state_first",   "schedule", "state_first",  0.0),
        ("noise_pinned",  "pin",      "noise_pinned", 0.0),
        ("oracle_pinned", "pin",      "oracle_pinned", 0.0),
    ]

    print("\n[3/3] Sweeping num_steps...")
    step_list = [int(x) for x in args.num_steps_list]
    print(f"   step counts: {step_list}  | num_samples={n}")
    print()
    print(
        f"  {'num_steps':>10s}  {'mode':<16s}"
        f"{'pos mean':>11s}{'pos max':>11s}"
        f"{'rot mean':>11s}{'rot max':>11s}"
        f"{'grip L1':>11s}{'flip %':>9s}{'cos(s,oracle)':>16s}"
    )
    for ns in step_list:
        for name, kind, sched_or_pin, off in modes:
            if kind == "schedule":
                a, s = _sample_with_schedule(
                    agent,
                    obs=obs_torch, act_0=act_0, x_state_init=x_state_init,
                    num_steps=ns,
                    schedule=sched_or_pin, pyramid_offset=off,
                )
            else:
                a, s = _sample_with_pinning(
                    agent,
                    obs=obs_torch, goal_obs=goal_torch,
                    act_0=act_0, x_state_init=x_state_init,
                    num_steps=ns,
                    pin_mode=sched_or_pin,
                )
            st = _abs_space_stats(a, act_true, action_normalizer)
            cos = torch.nn.functional.cosine_similarity(
                s.flatten(1), s_oracle.flatten(1), dim=-1,
            ).mean().item()
            print(
                f"  {ns:>10d}  {name:<16s}"
                f"{st['pos_l2_mean_mm']:>9.2f}mm"
                f"{st['pos_l2_max_mm']:>9.2f}mm"
                f"{st['rot_geo_mean_deg']:>9.3f}°"
                f"{st['rot_geo_max_deg']:>9.3f}°"
                f"{st['gripper_l1_mean']:>11.4f}"
                f"{st['gripper_disagree_pct']:>8.2f}%"
                f"{cos:>16.4f}"
            )
        print()  # blank line between step counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--num_steps_list", type=int, nargs="+",
                        default=[5, 10, 25, 50, 100])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    diagnose(args)
