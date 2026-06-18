"""Feed PLAY (rollout) observations into the mixed checkpoint under EXPERT
conditioning, and check whether the sampled action matches the action that was
actually taken in the rollout (i.e. expert-cond collapses to play behavior at
off-manifold states) or diverges to something else.

For each rollout obs we have the recorded rollout action chunk a_play. We
sample (same init noise) under expert and null conditioning and compare:
  MSE(a_expert, a_play)  vs  MSE(a_null, a_play)  vs  MSE(a_expert, a_null)
Reference: MSE(a_expert, a_expert_gt) on real expert obs (should be ~0).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, "scripts")
from _probe_mixed_vs_expert import (  # noqa: E402
    build_config, load_agent, sample_with_opt, MIXED_CKPT, EXPERT_CKPT, B, NUM_STEPS, DEVICE,
)

ROLLOUT = "data/pusht/image_rollouts.hdf5"


def main():
    from mip.datasets.pusht_dataset import (
        make_pusht_goal_dataset, PushTImageGoalDataset, load_pusht_rollout_replay_buffer,
    )
    cfg = build_config(); cfg.optimization.device = DEVICE

    # expert goal dataset -> normalizer (shared) + expert obs/gt for reference
    exp_ds = make_pusht_goal_dataset(cfg.task, mode="train")
    if hasattr(exp_ds, "datasets"):
        exp_ds = exp_ds.datasets[0]
    normalizer = exp_ds.normalizer

    # rollout (play) goal dataset, SAME expert normalizer so actions live in the
    # model's normalized output space. action = the recorded play action chunk.
    rb = load_pusht_rollout_replay_buffer(ROLLOUT)
    roll_ds = PushTImageGoalDataset(
        replay_buffer=rb, shape_meta=cfg.task.shape_meta, n_obs_steps=cfg.task.obs_steps,
        horizon=cfg.task.horizon, pad_before=cfg.task.obs_steps - 1,
        pad_after=cfg.task.act_steps - 1, normalizer=normalizer,
    )

    def batch(ds):
        idx = np.linspace(0, len(ds) - 1, B).astype(int)
        obs = {k: torch.stack([ds[i]["obs"][k] for i in idx]).to(DEVICE) for k in ds[0]["obs"]}
        act = torch.stack([ds[i]["action"] for i in idx]).to(DEVICE)
        return obs, act

    roll_obs, a_play = batch(roll_ds)
    exp_obs, a_expgt = batch(exp_ds)

    g = torch.Generator(device=DEVICE).manual_seed(0); D = cfg.network.encoder_out_dim
    x0 = torch.randn(B, 1, D, generator=g, device=DEVICE)
    a0 = torch.randn(B, cfg.task.horizon, cfg.task.act_dim, generator=g, device=DEVICE)

    agent = load_agent(MIXED_CKPT, cfg)
    enc, tln = agent._eval_encoder_modules(True)

    @torch.no_grad()
    def sample(obs, opt):
        z = tln(enc(obs, None))
        a, _ = sample_with_opt(agent, z, x0, a0, opt, NUM_STEPS)
        return a

    # On PLAY obs: expert-cond vs null-cond, both vs the recorded play action
    a_exp_roll = sample(roll_obs, agent.net.EXPERT_IDX)
    a_null_roll = sample(roll_obs, agent.net.NULL_IDX)
    # On EXPERT obs: expert-cond AND null(play)-cond, both vs expert gt action
    a_exp_exp = sample(exp_obs, agent.net.EXPERT_IDX)
    a_null_exp = sample(exp_obs, agent.net.NULL_IDX)

    print("\n=== 2x2: MIXED checkpoint, conditioning x obs-source (MSE to that source's gt) ===")
    print(f"{'':22s}{'expert-cond':>14s}{'null(play)-cond':>18s}{'expert-vs-null':>16s}")
    print(f"{'on EXPERT obs':22s}{F.mse_loss(a_exp_exp,a_expgt):>14.4f}"
          f"{F.mse_loss(a_null_exp,a_expgt):>18.4f}{F.mse_loss(a_exp_exp,a_null_exp):>16.4f}")
    print(f"{'on PLAY obs':22s}{F.mse_loss(a_exp_roll,a_play):>14.4f}"
          f"{F.mse_loss(a_null_roll,a_play):>18.4f}{F.mse_loss(a_exp_roll,a_null_roll):>16.4f}")
    print("  (expert-vs-null = how much the optimality label changes the action on that manifold)")

    scale = a_play.abs().mean().item()
    print(f"\naction scale (mean|a|, normalized): play={scale:.3f}  expert={a_expgt.abs().mean():.3f}\n")
    print("=== MIXED checkpoint, on PLAY (rollout) observations ===")
    print(f"  MSE(expert-cond, recorded play action) = {F.mse_loss(a_exp_roll, a_play):.4f}")
    print(f"  MSE(null-cond,   recorded play action) = {F.mse_loss(a_null_roll, a_play):.4f}")
    print(f"  MSE(expert-cond, null-cond)            = {F.mse_loss(a_exp_roll, a_null_roll):.4f}")
    print(f"  mean|expert-cond - play| = {(a_exp_roll-a_play).abs().mean():.4f} "
          f"({100*(a_exp_roll-a_play).abs().mean()/scale:.0f}% of action scale)")
    print(f"  mean|null-cond   - play| = {(a_null_roll-a_play).abs().mean():.4f} "
          f"({100*(a_null_roll-a_play).abs().mean()/scale:.0f}% of action scale)")
    print("\n=== reference: MIXED on EXPERT observations ===")
    print(f"  MSE(expert-cond, expert gt action)     = {F.mse_loss(a_exp_exp, a_expgt):.4f}")

    # Decisive: compare mixed expert-cond vs expert-ONLY expert-cond on the SAME
    # play obs. If mixed is pulled CLOSER to the play action than expert-only is,
    # that is the play contamination of the expert conditional off-manifold.
    eo = load_agent(EXPERT_CKPT, cfg)
    enc2, tln2 = eo._eval_encoder_modules(True)
    with torch.no_grad():
        z2 = tln2(enc2(roll_obs, None))
        a_eo_roll, _ = sample_with_opt(eo, z2, x0, a0, eo.net.EXPERT_IDX, NUM_STEPS)
    print("\n=== expert-cond on PLAY obs: MIXED vs EXPERT-ONLY ===")
    print(f"  MSE(mixed-expert-cond,       play action) = {F.mse_loss(a_exp_roll, a_play):.4f}")
    print(f"  MSE(expertonly-expert-cond,  play action) = {F.mse_loss(a_eo_roll, a_play):.4f}")
    print(f"  MSE(mixed-expert-cond, expertonly-expert-cond) = {F.mse_loss(a_exp_roll, a_eo_roll):.4f}")
    closer = "MIXED" if F.mse_loss(a_exp_roll, a_play) < F.mse_loss(a_eo_roll, a_play) else "EXPERT-ONLY"
    print(f"  -> {closer} expert-cond is closer to the play action on play obs")
    print("\ninterpretation:")
    print("  - mixed-expert-cond CLOSER to play than expert-only -> the play data pulled the")
    print("    mixed expert conditional toward play behavior off-manifold (contamination).")
    print("  - mixed ~= expert-only -> expert conditional is clean; degradation is elsewhere.")


if __name__ == "__main__":
    main()
