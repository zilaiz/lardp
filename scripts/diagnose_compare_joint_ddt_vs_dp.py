"""Side-by-side encoder + action-prediction diagnostic for two checkpoints.

Compares a joint-DDT-replace_x_state checkpoint (LBMDiTJointDDTAgent) against
a vanilla DP checkpoint (TrainingAgent with flow_beta / lbmdit) on the same
heldout dataset, using the same normalizer for both. Tells you whether the
deploy gap on the joint-DDT run is caused by the encoder collapsing onto a
narrower subspace than the working DP encoder uses.

Usage:
    python scripts/diagnose_compare_joint_ddt_vs_dp.py \\
        --ddt_ckpt   logs/<ddt>/<ts>/models/model_step_100000.pt \\
        --ddt_config outputs/<date>/<time>/.hydra/config.yaml \\
        --dp_ckpt    logs/<dp>/<ts>/models/model_step_90000.pt \\
        --dp_config  outputs/<date>/<time>/.hydra/config.yaml \\
        --dataset_path    data/franka_coffee_pod_cog/image_heldout.hdf5 \\
        --normalizer_path checkpoints/franka_coffee_pod_cog_e2e_normalizer.pkl \\
        --num_samples 32 \\
        --num_steps 25
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

LARDP_PATH = Path(__file__).resolve().parents[1]
if str(LARDP_PATH) not in sys.path:
    sys.path.append(str(LARDP_PATH))

from mip.agent import TrainingAgent
from mip.agent_lbmdit_joint_ddt import LBMDiTJointDDTAgent
from mip.agent_lbmdit_joint_pt import LBMDiTJointPTAgent
from mip.datasets.robomimic_dataset import make_idm_dataset


_JOINT_AGENT_BY_TYPE = {
    "lbmdit_joint_ddt": LBMDiTJointDDTAgent,
    "lbmdit_joint_pt":  LBMDiTJointPTAgent,
}


def _pick_joint_agent_cls(cfg):
    nt = cfg.network.network_type
    if nt not in _JOINT_AGENT_BY_TYPE:
        raise ValueError(
            f"Unknown joint network_type: {nt!r}. "
            f"Supported: {list(_JOINT_AGENT_BY_TYPE.keys())}"
        )
    return _JOINT_AGENT_BY_TYPE[nt]


def _fmt(t: torch.Tensor) -> str:
    return (f"shape={tuple(t.shape)} norm={t.norm().item():.3e} "
            f"std={t.std().item():.3e} abs_max={t.abs().max().item():.3e}")


def _effective_rank(x: torch.Tensor, eps: float = 1e-12) -> float:
    x_flat = x.reshape(-1, x.shape[-1]).float()
    x_flat = x_flat - x_flat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x_flat)
    p = s / (s.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))


def _prep_cfg(cfg, device_str: str | None = None):
    OmegaConf.update(cfg, "optimization.use_compile", False, merge=False)
    OmegaConf.update(cfg, "optimization.use_cudagraphs", False, merge=False)
    if device_str:
        OmegaConf.update(cfg, "optimization.device", device_str, merge=False)
    elif not torch.cuda.is_available():
        OmegaConf.update(cfg, "optimization.device", "cpu", merge=False)
    if cfg.task.obs_type == "image":
        OmegaConf.update(cfg, "task.obs_dim", cfg.network.emb_dim, merge=False)
    return cfg


def _build_dataset(cfg, dataset_path: str, normalizer):
    """Override task.dataset_paths to a single file, use the provided normalizer."""
    OmegaConf.update(cfg, "task.dataset_paths", [dataset_path], merge=False)
    OmegaConf.update(cfg, "task.dataset_path", None, merge=False)
    OmegaConf.update(cfg, "task.val_dataset_percentage", 0.0, merge=False)
    return make_idm_dataset(cfg.task, mode="train", normalizer=normalizer)


def _stack_batch(dataset, idxs, sample_keys, device):
    obs_batch = {k: [] for k in sample_keys}
    act_batch = []
    for i in idxs:
        s = dataset[int(i)]
        for k in sample_keys:
            obs_batch[k].append(s["obs"][k])
        act_batch.append(s["action"])
    obs = {
        k: torch.from_numpy(np.stack([t.numpy() for t in v])).to(device)
        for k, v in obs_batch.items()
    }
    act = torch.from_numpy(np.stack([t.numpy() for t in act_batch])).to(device)
    return obs, act


def _encoder_report(name, encoder, target_ln, obs, device):
    with torch.no_grad():
        z_raw = encoder(obs, None)
        if target_ln is not None:
            z_post = target_ln(z_raw)
            ln_tag = "target_ln(.)"
        else:
            z_post = z_raw
            ln_tag = "(no target_ln on this agent)"
    print(f"  [{name:>10s}] raw enc(obs):  {_fmt(z_raw):<70s} eff_rank={_effective_rank(z_raw):.2f}")
    if target_ln is not None:
        print(f"  [{name:>10s}] {ln_tag:<12s}: {_fmt(z_post):<70s} eff_rank={_effective_rank(z_post):.2f}")
    else:
        print(f"  [{name:>10s}] {ln_tag}")


def _action_report(name, pred, true, dim_names):
    err = pred - true
    mse = (err ** 2).mean(dim=(0, 1))
    mae = err.abs().mean(dim=(0, 1))
    pred_min = pred.amin(dim=(0, 1))
    pred_max = pred.amax(dim=(0, 1))
    pred_std = pred.std(dim=(0, 1))
    true_min = true.amin(dim=(0, 1))
    true_max = true.amax(dim=(0, 1))
    true_std = true.std(dim=(0, 1))
    print(f"  [{name:>10s}] overall MSE={mse.mean().item():.4f}  "
          f"MAE={mae.mean().item():.4f}  "
          f"RMSE={err.pow(2).mean().sqrt().item():.4f}")
    print(f"  [{name:>10s}] {'dim':<8s}{'MSE':>9s}{'MAE':>9s}"
          f"{'pred[min,max]':>22s}{'true[min,max]':>22s}"
          f"{'pred_std':>10s}{'true_std':>10s}")
    for i, dn in enumerate(dim_names):
        flag = ""
        if pred_min[i].item() < true_min[i].item() - 0.05:
            flag += " ↓"
        if pred_max[i].item() > true_max[i].item() + 0.05:
            flag += " ↑"
        print(f"  [{name:>10s}] {dn:<8s}{mse[i].item():>9.4f}"
              f"{mae[i].item():>9.4f}"
              f"   [{pred_min[i].item():+.3f},{pred_max[i].item():+.3f}]"
              f"   [{true_min[i].item():+.3f},{true_max[i].item():+.3f}]"
              f"   {pred_std[i].item():>7.3f}   {true_std[i].item():>7.3f}{flag}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n== device: {device} ==")

    # ---- Load normalizer (shared) ----
    with open(args.normalizer_path, "rb") as f:
        norm = pickle.load(f)
    print(f"normalizer: {args.normalizer_path}")
    print(f"  obs keys: {sorted(norm['obs'].keys())}")

    # ---- Load DDT agent (or any other joint agent — dispatched by config) ----
    print(f"\n[1/3] Building joint agent: {args.ddt_ckpt}")
    ddt_cfg = _prep_cfg(OmegaConf.load(args.ddt_config), str(device))
    # Apply cfg_scale override BEFORE construction so the agent's cached
    # ``_cfg_scale`` picks it up (read once in __init__).
    if args.ddt_cfg_scale is not None:
        OmegaConf.update(
            ddt_cfg, "optimization.joint_cfg_scale",
            float(args.ddt_cfg_scale), merge=False,
        )
        print(f"    joint_cfg_scale override: {args.ddt_cfg_scale}")
    JointCls = _pick_joint_agent_cls(ddt_cfg)
    print(f"    network_type={ddt_cfg.network.network_type} → {JointCls.__name__}")
    ddt = JointCls(ddt_cfg)
    ddt.load(args.ddt_ckpt, load_optimizer=False)
    ddt.eval()

    # ---- Load DP agent ----
    print(f"\n[2/3] Building DP agent:  {args.dp_ckpt}")
    dp_cfg = _prep_cfg(OmegaConf.load(args.dp_config), str(device))
    dp = TrainingAgent(dp_cfg)
    dp.load(args.dp_ckpt, load_optimizer=False)
    dp.eval()

    # ---- Build heldout dataset (use DDT's task config for obs schema) ----
    print(f"\n[3/3] Loading heldout dataset: {args.dataset_path}")
    dataset = _build_dataset(OmegaConf.load(args.ddt_config), args.dataset_path, norm)
    n = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    print(f"  dataset size={len(dataset)}, sampling {n} indices")

    sample_keys = list(ddt_cfg.task.shape_meta["obs"].keys())
    obs, act_true_normed = _stack_batch(dataset, idxs, sample_keys, device)
    act_true_normed = act_true_normed[:, : ddt_cfg.task.horizon]
    print(f"  obs shapes:        {[(k, tuple(v.shape)) for k, v in obs.items()]}")
    print(f"  act_true_normed:   {tuple(act_true_normed.shape)}")

    # ---- learnable_state_token (DDT only) ----
    print("\n== (1) learnable_state_token (DDT only) ==")
    for tag, net in [("net (live)", ddt.net), ("net_ema", ddt.net_ema)]:
        if getattr(net, "replace_x_state", False):
            t = net.learnable_state_token.detach()
            print(f"  {tag:14s}: {_fmt(t)} all_zero={bool((t == 0).all())}")

    # ---- Encoder output stats ----
    print("\n== (2) encoder output stats on heldout obs ==")
    _encoder_report("DDT live", ddt.encoder, ddt.target_ln, obs, device)
    _encoder_report("DDT EMA",  ddt.encoder_ema, ddt.target_ln_ema, obs, device)
    _encoder_report("DP live",  dp.encoder, None, obs, device)
    _encoder_report("DP EMA",   dp.encoder_ema, None, obs, device)

    # ---- Action prediction MSE ----
    print(f"\n== (3) predicted vs true (normalized) actions, num_steps={args.num_steps} ==")
    dim_names = ["pos_x", "pos_y", "pos_z",
                 "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5",
                 "gripper"]

    with torch.no_grad():
        # DDT sample
        act_0 = torch.randn(
            (n, ddt_cfg.task.horizon, ddt_cfg.task.act_dim), device=device,
        )
        ddt_pred = ddt.sample(
            act_0=act_0, obs=obs, num_steps=args.num_steps, use_ema=True,
        )
        # DP sample — same agent.sample signature
        dp_act_0 = torch.randn(
            (n, dp_cfg.task.horizon, dp_cfg.task.act_dim), device=device,
        )
        dp_pred = dp.sample(
            act_0=dp_act_0, obs=obs, num_steps=args.num_steps, use_ema=True,
        )
        # Ensure DP shape matches DDT (same horizon assumed)
        assert dp_pred.shape == ddt_pred.shape, (
            f"horizon mismatch: ddt {ddt_pred.shape} vs dp {dp_pred.shape}"
        )

    _action_report("DDT (rxs)", ddt_pred, act_true_normed, dim_names)
    print()
    _action_report("DP", dp_pred, act_true_normed, dim_names)

    # ---- Summary delta ----
    err_ddt = (ddt_pred - act_true_normed).pow(2).mean(dim=(0, 1))
    err_dp = (dp_pred - act_true_normed).pow(2).mean(dim=(0, 1))
    print("\n== (4) per-dim MSE ratio (DDT / DP) on heldout — >1 means DDT worse ==")
    print(f"  {'dim':<8s}{'DDT MSE':>10s}{'DP MSE':>10s}{'ratio':>10s}")
    for i, dn in enumerate(dim_names):
        r = err_ddt[i].item() / max(err_dp[i].item(), 1e-12)
        print(f"  {dn:<8s}{err_ddt[i].item():>10.4f}{err_dp[i].item():>10.4f}{r:>10.2f}")

    print("\n== done ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddt_ckpt", type=str, required=True)
    parser.add_argument("--ddt_config", type=str, required=True)
    parser.add_argument("--dp_ckpt", type=str, required=True)
    parser.add_argument("--dp_config", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Heldout dataset .hdf5")
    parser.add_argument("--normalizer_path", type=str, required=True,
                        help="normalizer.pkl that the models were trained against")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--ddt_cfg_scale", type=float, default=None,
                        help="Override optimization.joint_cfg_scale on the "
                             "joint-side agent at construction (sampling uses "
                             "the cached value). Has effect only when the run "
                             "was trained with optimality labels (expert vs "
                             "play) or with joint_cfg_dropout_prob>0.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
