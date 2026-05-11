"""Open-loop IDM oracle-goal eval with closed-loop replay (square_ph_image, expert-only).

Procedure
---------
1. Roll out the BC policy in deterministic env seeds, log per-chunk obs and
   per-chunk action, accumulate reward → BC success rate.
2. For every BC chunk i, run the IDM open-loop on (BC's obs[i], oracle_goal),
   where oracle_goal = BC's chunk[i+1] last frame (1 env-step early proxy
   for the training-distribution goal at env_time i·8 + 9).
3. Reset env to the same seeds, replay the IDM-predicted action chunks
   (no IDM re-querying mid-episode), accumulate reward → IDM-replay success.

This isolates whether the IDM, given the *same observations BC sees and a
real future goal*, can produce action chunks that solve the task when
executed. If yes → IDM has the BC capacity, gap to BC is just goal quality
under the live goal predictor. If no → IDM action distribution itself is
limited.

Caveats
-------
- Open-loop on BC's obs trajectory: IDM doesn't see its own divergence.
- 1 env-step early oracle (i·8 + 8 vs ideal i·8 + 9). Minor.
- Each BC and IDM-replay run for `max_episode_steps`; SyncVectorEnv
  auto-resets a finished env, but binary success-by-summed-reward is
  preserved (matches the existing inline eval semantics).
"""

import os
import sys
import pickle

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "/oscar/data/csun45/zzeng28/repo/lardp")

import hydra
import loguru
import numpy as np
import torch

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.encoders import GoalDropoutEncoder
from mip.envs.robomimic.robomimic_env import make_vec_env
from mip.torch_utils import limit_threads, set_seed

# ---- knobs ----
DEVICE = "cuda"
N_ROUNDS = int(os.environ.get("N_ROUNDS", 25))  # 25 rounds × 4 envs = 100 episodes
NFE_BC = int(os.environ.get("NFE_BC", 9))      # BC action sampling steps
NFE_IDM = int(os.environ.get("NFE_IDM", 9))    # IDM action sampling steps
SEED_BASE = 1000              # seeds = SEED_BASE + round*num_envs + env_idx
CONFIG_DIR = "/oscar/data/csun45/zzeng28/repo/lardp/examples/configs"

# --- BC config (env-var overridable for parallel sweeps) ---
BC_CKPT = os.environ.get(
    "BC_CKPT",
    "/oscar/data/csun45/zzeng28/repo/lardp/logs/tool_hang_ph_image_flow_beta_None_lbmdit_256_seed0_0.2_ac16/2026_05_01_01_14_59/models/model_best.pt",
)
BC_TASK = os.environ.get("BC_TASK", "tool_hang_ph_image_delta")
BC_NETWORK = os.environ.get("BC_NETWORK", "lbmdit")
BC_LOSS = os.environ.get("BC_LOSS", "flow_beta")
BC_EXTRA = os.environ.get(
    "BC_EXTRA", "task.horizon=16,task.act_steps=8,task.val_dataset_percentage=0.2"
).split(",") if os.environ.get(
    "BC_EXTRA", "task.horizon=16,task.act_steps=8,task.val_dataset_percentage=0.2"
) else []

# --- IDM config ---
IDM_CKPT = os.environ.get(
    "IDM_CKPT",
    "/oscar/data/csun45/zzeng28/repo/lardp/logs/tool_hang_ph_image_flow_None_lbmidm_v2_256_seed0_idm_v2_fdm_aux_0.2_ac16/2026_04_30_23_48_48/models/model_step_200000.pt",
)
IDM_TASK = os.environ.get("IDM_TASK", "tool_hang_ph_image_idm")
IDM_NETWORK = os.environ.get("IDM_NETWORK", "lbmidm_v2")  # lbmidm | lbmidm_v2
IDM_LOSS = os.environ.get("IDM_LOSS", "flow")
IDM_EXTRA = os.environ.get(
    "IDM_EXTRA",
    "task.horizon=16,task.act_steps=8,task.val_dataset_percentage=0.2,network.dropout=0.1,optimization.fdm_loss_scale=1.0",
).split(",")
IDM_EXTRA = [x for x in IDM_EXTRA if x]  # drop empty

OUT_NPZ = os.environ.get("OUT_NPZ", "/tmp/oracle_eval_arrays.npz")


def _build_cfg(task_name, network_name, loss_type, extra_overrides=None):
    overrides = [
        f"task={task_name}",
        f"network={network_name}",
        "network.emb_dim=256",
        "network.encoder_type=image",
        f"optimization.loss_type={loss_type}",
        "optimization.batch_size=128",
        "optimization.lr=1e-4",
        "optimization.auto_resume=False",
        "log.wandb_mode=disabled",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = hydra.compose(config_name="main", overrides=overrides)
    return cfg


def _normalize_obs_dict(obs_np: dict, normalizer, device):
    """Normalize a dict of np arrays to a dict of torch tensors on device."""
    out = {}
    for k, v in obs_np.items():
        v = v.astype(np.float32)
        v = normalizer["obs"][k].normalize(v)
        out[k] = torch.tensor(v, device=device, dtype=torch.float32)
    return out


def _concat_obs_goal(obs_np: dict, goal_np: dict, normalizer, device):
    """Normalize both, concatenate goal to the end of the obs along time axis."""
    out = {}
    for k in obs_np:
        o = normalizer["obs"][k].normalize(obs_np[k].astype(np.float32))
        g = normalizer["obs"][k].normalize(goal_np[k].astype(np.float32))
        cat = np.concatenate([o, g], axis=1)  # (B, To+1, ...)
        out[k] = torch.tensor(cat, device=device, dtype=torch.float32)
    return out


def main():
    set_seed(0)
    limit_threads(1)
    torch.set_float32_matmul_precision("high")

    # ---- configs ----
    bc_cfg = _build_cfg(BC_TASK, BC_NETWORK, BC_LOSS, BC_EXTRA)
    bc_cfg.task.obs_dim = bc_cfg.network.emb_dim
    idm_cfg = _build_cfg(IDM_TASK, IDM_NETWORK, IDM_LOSS, IDM_EXTRA)
    idm_cfg.task.obs_dim = idm_cfg.network.emb_dim

    # sanity: both configs should agree on the env-side params we depend on
    assert bc_cfg.task.obs_steps == idm_cfg.task.obs_steps
    assert bc_cfg.task.act_steps == idm_cfg.task.act_steps
    assert bc_cfg.task.horizon == idm_cfg.task.horizon
    assert bc_cfg.task.act_dim == idm_cfg.task.act_dim
    assert bc_cfg.task.num_envs == idm_cfg.task.num_envs
    assert bc_cfg.task.max_episode_steps == idm_cfg.task.max_episode_steps

    To = bc_cfg.task.obs_steps
    horizon = bc_cfg.task.horizon
    act_steps = bc_cfg.task.act_steps
    act_dim = bc_cfg.task.act_dim
    num_envs = bc_cfg.task.num_envs
    max_steps = bc_cfg.task.max_episode_steps
    # Oracle-goal offset: training distribution puts the goal at last_obs + (horizon-1)
    # env-steps. Closed-loop chunks log obs every act_steps env-steps, so use the BC
    # chunk whose last frame env-time is closest to last_obs + (horizon-1).
    oracle_offset = max(1, round((horizon - 1) / act_steps))

    loguru.logger.info(
        f"To={To} horizon={horizon} act_steps={act_steps} act_dim={act_dim} "
        f"num_envs={num_envs} max_steps={max_steps} oracle_offset={oracle_offset}"
    )

    # ---- env ----
    envs = make_vec_env(idm_cfg.task, seed=0, save_rollouts=False)

    # ---- normalizers ----
    # Cache the BC normalizer (constructing the BC dataset just to read its
    # normalizer reloads ~15k images each run; the normalizer itself is tiny).
    bc_norm_cache = (
        f"/tmp/bc_norm_cache_{bc_cfg.task.env_name}_{bc_cfg.task.action_type}_"
        f"val{bc_cfg.task.val_dataset_percentage}.pkl"
    )
    if os.path.exists(bc_norm_cache):
        with open(bc_norm_cache, "rb") as f:
            bc_norm = pickle.load(f)
        loguru.logger.info(f"Loaded BC normalizer from cache {bc_norm_cache}")
    else:
        loguru.logger.info("Building BC dataset to extract its normalizer...")
        bc_dataset = make_dataset(bc_cfg.task)
        bc_norm = bc_dataset.normalizer
        with open(bc_norm_cache, "wb") as f:
            pickle.dump(bc_norm, f)
        loguru.logger.info(f"Cached BC normalizer to {bc_norm_cache}")

    idm_norm_path = os.path.join(os.path.dirname(IDM_CKPT), "normalizer.pkl")
    with open(idm_norm_path, "rb") as f:
        idm_norm = pickle.load(f)
    loguru.logger.info(f"Loaded IDM normalizer from {idm_norm_path}")

    # ---- BC agent ----
    loguru.logger.info(f"Loading BC agent from {BC_CKPT}")
    bc_agent = TrainingAgent(bc_cfg)
    bc_agent.load(BC_CKPT)
    bc_agent.eval()

    # ---- IDM agent (network-specific construction) ----
    loguru.logger.info(f"Loading IDM agent ({IDM_NETWORK}) from {IDM_CKPT}")
    if IDM_NETWORK == "lbmidm_v2":
        # IDMFDMAgent: no GoalDropoutEncoder wrapping, has FDM head inside the network.
        from mip.agent_idm_fdm import IDMFDMAgent
        idm_agent = IDMFDMAgent(idm_cfg)
        idm_agent.load(IDM_CKPT)
        idm_agent.eval()
    elif IDM_NETWORK == "lbmidm":
        # TrainingAgent + GoalDropoutEncoder (wrap before load).
        idm_agent = TrainingAgent(idm_cfg)
        enc_out_dim = idm_cfg.network.encoder_out_dim or idm_cfg.network.emb_dim
        idm_agent.encoder = GoalDropoutEncoder(
            idm_agent.encoder,
            enc_out_dim,
            idm_cfg.task.obs_steps,
            idm_cfg.optimization.goal_dropout_prob,
        ).to(DEVICE)
        idm_agent.encoder_ema = GoalDropoutEncoder(
            idm_agent.encoder_ema,
            enc_out_dim,
            idm_cfg.task.obs_steps,
            idm_cfg.optimization.goal_dropout_prob,
        ).to(DEVICE)
        idm_agent.optimizer.add_param_group(
            {"params": [idm_agent.encoder.uncond_emb]}
        )
        idm_agent.__compile__()
        idm_agent.load(IDM_CKPT)
        idm_agent.eval()
    else:
        raise ValueError(f"Unknown IDM_NETWORK: {IDM_NETWORK}")

    # ---- Run rounds ----
    # All BC episodes (for the BC success rate)
    bc_success = []         # 0/1 per episode, length = N_ROUNDS * num_envs (closed-loop)
    bc_replay_success = []  # 0/1 per episode (BC's actions replayed open-loop in env)
    bc_reward_log = []
    bc_replay_reward_log = []
    # Per-BC-success records (only populated when BC succeeded for that env)
    succ_records = []  # list of dicts: round, env_idx, idm_succ, mse_per_chunk (n_chunks,), bc_act_full (n_chunks, horizon, act_dim), idm_act_full
    # Verification: max-abs-diff between Phase-A and Phase-C reset obs, per round per key
    reset_verify_log = []  # list of dicts: {round, max_abs_diff_per_key}

    for round_idx in range(N_ROUNDS):
        seeds = [SEED_BASE + round_idx * num_envs + k for k in range(num_envs)]

        # === Phase A: BC rollout ===
        obs, _ = envs.reset(seed=seeds)
        bc_obs_chunks = []  # list of dicts of (num_envs, To, ...) np arrays
        bc_act_chunks_full = []  # list of (num_envs, horizon, act_dim) — full predicted horizon
        bc_act_chunks_exec = []  # list of (num_envs, act_steps, act_dim) — executed slice
        bc_reward = np.zeros(num_envs, dtype=np.float32)
        t = 0
        while t < max_steps:
            bc_obs_chunks.append({k: obs[k].copy() for k in obs})
            obs_t = _normalize_obs_dict(obs, bc_norm, DEVICE)
            act_0 = torch.randn(
                (num_envs, horizon, act_dim), device=DEVICE, dtype=torch.float32
            )
            act_normed = bc_agent.sample(
                act_0=act_0, obs=obs_t, num_steps=NFE_BC, use_ema=True
            )
            act_full = bc_norm["action"].unnormalize(
                act_normed.detach().cpu().numpy()
            )  # (num_envs, horizon, act_dim) unnormalized
            act_exec = act_full[:, To - 1 : To - 1 + act_steps, :]  # [1:9]
            bc_act_chunks_full.append(act_full.copy())
            bc_act_chunks_exec.append(act_exec.copy())
            obs, reward, term, trunc, info = envs.step(act_exec)
            bc_reward += reward
            t += act_steps
        bc_succ_mask = bc_reward > 0  # (num_envs,) bool
        bc_success.extend(bc_succ_mask.astype(int).tolist())
        bc_reward_log.extend(bc_reward.tolist())

        # === Phase A2: replay BC's logged actions in env from same seeds ===
        # Sanity check on env determinism. If env is fully deterministic, this
        # should reproduce Phase A's success exactly.
        obs, _ = envs.reset(seed=seeds)
        bc_replay_reward = np.zeros(num_envs, dtype=np.float32)
        for i in range(len(bc_act_chunks_exec)):
            obs, reward, term, trunc, info = envs.step(bc_act_chunks_exec[i])
            bc_replay_reward += reward
        bc_replay_succ_mask = bc_replay_reward > 0
        bc_replay_success.extend(bc_replay_succ_mask.astype(int).tolist())
        bc_replay_reward_log.extend(bc_replay_reward.tolist())

        # If no BC successes this round, skip IDM Phase B/C entirely (saves compute).
        if not bc_succ_mask.any():
            running_n = (round_idx + 1) * num_envs
            loguru.logger.info(
                f"[round {round_idx + 1}/{N_ROUNDS}] BC={bc_succ_mask.astype(int).tolist()} "
                f"BC-replay={bc_replay_succ_mask.astype(int).tolist()} "
                f"(no BC successes — skipping IDM) | running BC={np.mean(bc_success):.3f} "
                f"({sum(bc_success)}/{running_n}), BC-replay={np.mean(bc_replay_success):.3f}"
            )
            continue

        # === Phase B: IDM open-loop on BC's logged obs trajectory (all envs, batched) ===
        idm_act_chunks = []
        idm_act_chunks_full = []
        n_chunks = len(bc_obs_chunks)
        for i in range(n_chunks):
            obs_chunk = bc_obs_chunks[i]
            # Oracle goal = BC's chunk (i + oracle_offset) last frame (clipped at end).
            j = min(i + oracle_offset, n_chunks - 1)
            goal_chunk = {
                k: bc_obs_chunks[j][k][:, -1:, ...] for k in bc_obs_chunks[j]
            }
            obs_t = _concat_obs_goal(obs_chunk, goal_chunk, idm_norm, DEVICE)
            act_0 = torch.randn(
                (num_envs, horizon, act_dim), device=DEVICE, dtype=torch.float32
            )
            act_normed = idm_agent.sample(
                act_0=act_0, obs=obs_t, num_steps=NFE_IDM, use_ema=True
            )
            act_full = idm_norm["action"].unnormalize(
                act_normed.detach().cpu().numpy()
            )
            act_exec = act_full[:, To - 1 : To - 1 + act_steps, :]
            idm_act_chunks_full.append(act_full.copy())
            idm_act_chunks.append(act_exec.copy())

        # Per-chunk MSE BC vs IDM (unnormalized executed slice), kept per (chunk, env)
        bc_arr_full = np.stack(bc_act_chunks_full, axis=0)   # (n_chunks, num_envs, horizon, act_dim)
        idm_arr_full = np.stack(idm_act_chunks_full, axis=0)
        bc_arr_exec = np.stack(bc_act_chunks_exec, axis=0)   # (n_chunks, num_envs, act_steps, act_dim)
        idm_arr_exec = np.stack(idm_act_chunks, axis=0)
        per_chunk_env_mse = ((bc_arr_exec - idm_arr_exec) ** 2).mean(axis=(2, 3))  # (n_chunks, num_envs)

        # === Phase C: replay IDM action chunks in env from same seeds ===
        obs, _ = envs.reset(seed=seeds)
        # Verify the Phase-C reset puts envs in the same initial state as Phase A.
        # bc_obs_chunks[0] is the first stacked-obs Phase A saw right after reset
        # (shape (num_envs, To, ...) per key, with To frames replicated from the
        # single deque element). Compare element-wise against `obs` here.
        diffs = {}
        for k in obs:
            a = bc_obs_chunks[0][k].astype(np.float32)
            b = obs[k].astype(np.float32)
            diffs[k] = float(np.max(np.abs(a - b)))
        reset_verify_log.append({"round": round_idx, "max_abs_diff": diffs})
        if any(v > 1e-5 for v in diffs.values()):
            loguru.logger.warning(
                f"[round {round_idx + 1}] env reset NOT bit-identical, "
                f"max-abs-diff per key: {diffs}"
            )
        idm_reward = np.zeros(num_envs, dtype=np.float32)
        for i in range(n_chunks):
            obs, reward, term, trunc, info = envs.step(idm_act_chunks[i])
            idm_reward += reward
        idm_succ_mask = idm_reward > 0

        # Record per-BC-successful trajectory only
        for env_idx in range(num_envs):
            if not bc_succ_mask[env_idx]:
                continue
            succ_records.append({
                "round": round_idx,
                "env_idx": env_idx,
                "seed": seeds[env_idx],
                "idm_succ": int(idm_succ_mask[env_idx]),
                "idm_reward": float(idm_reward[env_idx]),
                "mse_per_chunk": per_chunk_env_mse[:, env_idx].copy(),  # (n_chunks,)
                "bc_act_full": bc_arr_full[:, env_idx].copy(),          # (n_chunks, horizon, act_dim)
                "idm_act_full": idm_arr_full[:, env_idx].copy(),
            })

        n_bc_succ_so_far = sum(bc_success)
        n_bc_replay_succ_so_far = sum(bc_replay_success)
        n_idm_succ_on_bc_succ_so_far = sum(r["idm_succ"] for r in succ_records)
        running_n = (round_idx + 1) * num_envs
        loguru.logger.info(
            f"[round {round_idx + 1}/{N_ROUNDS}] BC={bc_succ_mask.astype(int).tolist()} "
            f"BC-replay={bc_replay_succ_mask.astype(int).tolist()} "
            f"IDM(on BC-succ envs)={[int(idm_succ_mask[e]) for e in range(num_envs) if bc_succ_mask[e]]} | "
            f"running BC={n_bc_succ_so_far}/{running_n}, "
            f"BC-replay={n_bc_replay_succ_so_far}/{running_n}, "
            f"IDM|BC-succ={n_idm_succ_on_bc_succ_so_far}/{n_bc_succ_so_far}"
        )

    # ---- Env reset verification summary ----
    if reset_verify_log:
        all_keys = sorted(reset_verify_log[0]["max_abs_diff"].keys())
        worst = {k: max(r["max_abs_diff"][k] for r in reset_verify_log) for k in all_keys}
        print()
        print("=" * 60)
        print(f"Env reset determinism (Phase-C vs Phase-A first obs)")
        print("=" * 60)
        print(f"rounds checked: {len(reset_verify_log)}")
        all_zero = all(all(v == 0.0 for v in r["max_abs_diff"].values()) for r in reset_verify_log)
        print(f"all bit-identical: {all_zero}")
        for k in all_keys:
            print(f"  {k:30s} max-abs-diff across rounds: {worst[k]:.6e}")

    # ---- Aggregate ----
    n_eps = len(bc_success)
    bc_arr = np.array(bc_success)
    n_bc_succ = int(bc_arr.sum())
    n_idm_succ_on_bc = sum(r["idm_succ"] for r in succ_records)

    print()
    print("=" * 60)
    print(f"Oracle-goal open-loop replay eval — {n_eps} episodes (NFE_BC={NFE_BC}, NFE_IDM={NFE_IDM})")
    print(f"IDM Phase B/C run only on BC-successful trajectories")
    print("=" * 60)
    bc_replay_arr = np.array(bc_replay_success)
    n_bc_replay_succ = int(bc_replay_arr.sum())
    print(f"BC closed-loop success:    {bc_arr.mean():.3f} ({n_bc_succ}/{n_eps})")
    print(f"BC action replay success:  {bc_replay_arr.mean():.3f} ({n_bc_replay_succ}/{n_eps})")
    bc_match = (bc_arr == bc_replay_arr).all()
    print(f"BC closed-loop == BC replay (per episode): {bc_match}")
    if not bc_match:
        diff_idx = np.where(bc_arr != bc_replay_arr)[0].tolist()
        print(f"  episodes that disagree: {diff_idx}")
    if n_bc_succ > 0:
        cond = n_idm_succ_on_bc / n_bc_succ
        print(f"IDM-replay | BC succeeded: {cond:.3f} ({n_idm_succ_on_bc}/{n_bc_succ})")

    def wilson(p, n, z=1.96):
        if n == 0:
            return (0.0, 0.0)
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (center - half, center + half)

    bc_lo, bc_hi = wilson(bc_arr.mean(), n_eps)
    print(f"\nBC 95% CI: [{bc_lo:.3f}, {bc_hi:.3f}]")
    if n_bc_succ > 0:
        idm_lo, idm_hi = wilson(n_idm_succ_on_bc / n_bc_succ, n_bc_succ)
        print(f"IDM | BC-succ 95% CI: [{idm_lo:.3f}, {idm_hi:.3f}]")

    # ---- Per-chunk action MSE (BC vs IDM, executed slice, BC-success subset) ----
    if n_bc_succ > 0:
        # mse_stack: (n_bc_succ_records, n_chunks)
        mse_stack = np.stack([r["mse_per_chunk"] for r in succ_records], axis=0)
        n_chunks_eff = mse_stack.shape[1]
        print()
        print("=" * 60)
        print(f"BC vs IDM action MSE on BC-successful trajectories ({n_bc_succ} traj)")
        print("(unnormalized actions, executed slice [1:9])")
        print("=" * 60)
        print(f"shape:                     ({mse_stack.shape[0]} traj, {n_chunks_eff} chunks)")
        print(f"global mean MSE:           {mse_stack.mean():.4f}")
        print(f"global median MSE:         {np.median(mse_stack):.4f}")
        print(f"per-chunk MSE (mean / median / std over BC-succ trajectories):")
        chunks_to_show = list(range(0, n_chunks_eff, max(1, n_chunks_eff // 10)))
        if chunks_to_show[-1] != n_chunks_eff - 1:
            chunks_to_show.append(n_chunks_eff - 1)
        for c in chunks_to_show:
            col = mse_stack[:, c]
            print(f"  chunk {c:3d}:  mean={col.mean():.4f}  "
                  f"median={np.median(col):.4f}  std={col.std():.4f}")

        # Split by per-trajectory IDM success (within BC-succ subset)
        idm_traj_succ = np.array([r["idm_succ"] for r in succ_records], dtype=bool)
        if idm_traj_succ.any():
            print(f"mean MSE | IDM also succeeded: {mse_stack[idm_traj_succ].mean():.4f} "
                  f"(n={int(idm_traj_succ.sum())})")
        if (~idm_traj_succ).any():
            print(f"mean MSE | IDM failed:         {mse_stack[~idm_traj_succ].mean():.4f} "
                  f"(n={int((~idm_traj_succ).sum())})")

        # Save raw arrays for offline plotting
        out_path = OUT_NPZ
        np.savez(
            out_path,
            per_chunk_mse=mse_stack,                                              # (n_bc_succ, n_chunks)
            bc_success=bc_arr,                                                    # (n_eps,)
            bc_replay_success=bc_replay_arr,                                      # (n_eps,)
            idm_succ_on_bc_succ=np.array([r["idm_succ"] for r in succ_records]),  # (n_bc_succ,)
            seeds=np.array([r["seed"] for r in succ_records]),                    # (n_bc_succ,)
            bc_act=np.stack([r["bc_act_full"] for r in succ_records], axis=0),    # (n_bc_succ, n_chunks, horizon, act_dim)
            idm_act=np.stack([r["idm_act_full"] for r in succ_records], axis=0),
            bc_reward=np.array(bc_reward_log),
            bc_replay_reward=np.array(bc_replay_reward_log),
        )
        print(f"\nSaved arrays to {out_path}")


if __name__ == "__main__":
    main()
