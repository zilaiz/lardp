"""Analyze diversity of rollout data vs expert demonstrations.

Computes per-task metrics on low-dimensional observations (eef_pos, eef_quat,
gripper_qpos) and actions:
  - State-space coverage: per-dim min/max range, volume ratio
  - Variance: per-dim and total variance
  - Episode-level diversity: mean pairwise L2 distance between trajectories
  - Action diversity: entropy of discretized action distributions
  - Success rate of rollouts (from rewards)
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import pdist


# ── helpers ──────────────────────────────────────────────────────────────────

def load_lowdim_and_actions(path, max_demos=None):
    """Load low-dim obs and actions from an HDF5 file.

    Returns:
        all_states: list of (T, D) arrays (per-episode concatenated low-dim obs)
        all_actions: list of (T, A) arrays
        all_rewards: list of (T,) arrays (or None if not present)
    """
    obs_keys_single = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    obs_keys_dual = obs_keys_single + ["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]

    all_states, all_actions, all_rewards = [], [], []
    with h5py.File(str(path), "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if max_demos is not None:
            demos = demos[:max_demos]
        for demo_name in demos:
            d = f["data"][demo_name]
            obs = d["obs"]
            available = [k for k in obs_keys_dual if k in obs]
            if not available:
                available = [k for k in obs_keys_single if k in obs]
            parts = [np.array(obs[k]) for k in available]
            state = np.concatenate(parts, axis=-1)
            all_states.append(state)
            all_actions.append(np.array(d["actions"]))
            if "rewards" in d:
                all_rewards.append(np.array(d["rewards"]))
    return all_states, all_actions, all_rewards if all_rewards else None


def compute_coverage(trajectories):
    all_data = np.concatenate(trajectories, axis=0)
    mins = all_data.min(axis=0)
    maxs = all_data.max(axis=0)
    ranges = maxs - mins
    return mins, maxs, ranges


def compute_variance(trajectories):
    all_data = np.concatenate(trajectories, axis=0)
    per_dim = all_data.var(axis=0)
    total = per_dim.sum()
    return per_dim, total


def compute_action_entropy(actions_list, n_bins=20):
    all_actions = np.concatenate(actions_list, axis=0)
    entropies = []
    for dim in range(all_actions.shape[1]):
        vals = all_actions[:, dim]
        counts, _ = np.histogram(vals, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log(probs)))
    return np.mean(entropies), np.array(entropies)


def compute_trajectory_diversity(trajectories, max_trajs=100, subsample_steps=20):
    embeddings = []
    trajs = trajectories[:max_trajs]
    for traj in trajs:
        T = traj.shape[0]
        idx = np.linspace(0, T - 1, subsample_steps, dtype=int)
        embeddings.append(traj[idx].flatten())
    embeddings = np.array(embeddings)
    if len(embeddings) < 2:
        return 0.0
    dists = pdist(embeddings, metric="euclidean")
    return float(np.mean(dists))


def success_rate(rewards_list):
    if rewards_list is None:
        return None
    successes = sum(1 for r in rewards_list if r.max() > 0.5)
    return successes / len(rewards_list)


# ── main ─────────────────────────────────────────────────────────────────────

TASKS = {
    "can": {
        "expert": "data/robomimic/can/ph/image_v15.hdf5",
        "old_rollout": "data/robomimic/can/image_rollouts_9.hdf5",
        "new_rollout": "data/robomimic/can/image_rollouts.hdf5",
    },
    "square": {
        "expert": "data/robomimic/square/ph/image_v15.hdf5",
        "old_rollout": "data/robomimic/square/image_rollouts_9.hdf5",
        "new_rollout": "data/robomimic/square/image_rollouts.hdf5",
    },
    "tool_hang": {
        "expert": "data/robomimic/tool_hang/ph/image_v15.hdf5",
        "old_rollout": "data/robomimic/tool_hang/image_rollouts_9.hdf5",
        "new_rollout": "data/robomimic/tool_hang/image_rollouts.hdf5",
    },
    "transport": {
        "expert": "data/robomimic/transport/ph/image_v15.hdf5",
        "old_rollout": "data/robomimic/transport/image_rollouts_9.hdf5",
        "new_rollout": "data/robomimic/transport/image_rollouts.hdf5",
    },
}


def compute_metrics(states, actions, rewards):
    """Compute all diversity metrics for a dataset."""
    n_eps = len(states)
    n_steps = sum(s.shape[0] for s in states)
    sr = success_rate(rewards)
    _, var_total = compute_variance(states)
    act_ent, _ = compute_action_entropy(actions)
    traj_div = compute_trajectory_diversity(states)
    _, act_var_total = compute_variance(actions)
    return {
        "eps": n_eps, "steps": n_steps, "sr": sr,
        "state_var": var_total, "act_ent": act_ent,
        "traj_div": traj_div, "act_var": act_var_total,
    }


def analyze_one(name, paths, max_expert_demos=None):
    label = f"{name.upper()}"
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    sources = {
        "Expert": (paths["expert"], max_expert_demos),
        "Old rollout (9-step)": (paths["old_rollout"], None),
        "New rollout (3-step)": (paths["new_rollout"], None),
    }

    results = {}
    for src_label, (path, max_d) in sources.items():
        if not Path(path).exists():
            print(f"  {src_label}: MISSING")
            continue
        states, actions, rewards = load_lowdim_and_actions(path, max_demos=max_d)
        results[src_label] = compute_metrics(states, actions, rewards)

    if len(results) < 2:
        print("  Not enough datasets to compare")
        return

    # Summary table
    header = f"  {'':<22} {'Episodes':>8} {'Timesteps':>10} {'Success':>8} {'State Var':>10} {'Act Var':>10} {'Act Ent':>8} {'Traj Div':>9}"
    print(f"\n{header}")
    print(f"  {'-'*80}")
    for src_label, r in results.items():
        sr_str = f"{r['sr']:.1%}" if r['sr'] is not None else "N/A"
        print(f"  {src_label:<22} {r['eps']:>8} {r['steps']:>10} {sr_str:>8} {r['state_var']:>10.4f} {r['act_var']:>10.4f} {r['act_ent']:>8.3f} {r['traj_div']:>9.3f}")

    # Ratios
    if "Old rollout (9-step)" in results and "New rollout (3-step)" in results:
        old = results["Old rollout (9-step)"]
        new = results["New rollout (3-step)"]
        print(f"\n  New vs Old rollout ratios:")
        print(f"    State variance: {new['state_var']/old['state_var']:.2f}x")
        print(f"    Action variance: {new['act_var']/old['act_var']:.2f}x")
        print(f"    Action entropy: {new['act_ent']/old['act_ent']:.2f}x")
        print(f"    Traj diversity: {new['traj_div']/old['traj_div']:.2f}x")

    if "Expert" in results and "New rollout (3-step)" in results:
        exp = results["Expert"]
        new = results["New rollout (3-step)"]
        print(f"\n  New rollout vs Expert ratios:")
        print(f"    State variance: {new['state_var']/exp['state_var']:.2f}x")
        print(f"    Action variance: {new['act_var']/exp['act_var']:.2f}x")
        print(f"    Action entropy: {new['act_ent']/exp['act_ent']:.2f}x")
        print(f"    Traj diversity: {new['traj_div']/exp['traj_div']:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout vs expert diversity")
    parser.add_argument("--task", type=str, default=None, help="Analyze a single task")
    parser.add_argument("--max-expert-demos", type=int, default=None, help="Limit expert demos")
    args = parser.parse_args()

    tasks = {args.task: TASKS[args.task]} if args.task else TASKS
    for name, paths in tasks.items():
        analyze_one(name, paths, max_expert_demos=args.max_expert_demos)
    print()


if __name__ == "__main__":
    main()
