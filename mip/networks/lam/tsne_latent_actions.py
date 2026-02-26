"""t-SNE visualization of precomputed LAM latent actions.

Supports two analysis modes:
  1. Same task, compare latent actions across different timesteps
  2. Cross-task comparison at specified timesteps

Usage examples:
    # Mode 1: Same task, compare timesteps 0, 10, 20 across demos
    python mip/networks/lam/tsne_latent_actions.py \
        --hdf5_paths data/robomimic/can/ph/image_v15_lam_256.hdf5 \
        --timesteps 0 10 20 \
        --frame_skip 1 \
        --camera_key agentview_image

    # Mode 2: Compare two tasks at timestep 5
    python mip/networks/lam/tsne_latent_actions.py \
        --hdf5_paths data/robomimic/can/ph/image_v15_lam_256.hdf5 \
                     data/robomimic/square/ph/image_v15_lam_256.hdf5 \
        --task_names can square \
        --timesteps 5 \
        --frame_skip 1 \
        --camera_key agentview_image

    # Use relative timesteps (fraction of episode length)
    python mip/networks/lam/tsne_latent_actions.py \
        --hdf5_paths data/robomimic/can/ph/image_v15_lam_256.hdf5 \
        --timesteps_frac 0.0 0.25 0.5 0.75 1.0 \
        --frame_skip 8 \
        --camera_key agentview_image

    # Limit number of demos for faster visualization
    python mip/networks/lam/tsne_latent_actions.py \
        --hdf5_paths data/robomimic/can/ph/image_v15_lam_256.hdf5 \
        --timesteps 0 5 10 \
        --max_demos 50 \
        --frame_skip 1
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import TSNE


def get_demo_keys(f):
    """Get sorted demo keys from HDF5 file."""
    return sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))


def extract_latent_actions_at_timesteps(
    f, demo_keys, timesteps, frame_skip, camera_key, latent_type="bn",
    timesteps_frac=None,
):
    """Extract latent actions and RGB images at specified timesteps.

    Args:
        f: Open HDF5 file handle.
        demo_keys: List of demo keys to process.
        timesteps: List of absolute timestep indices. Ignored if timesteps_frac is set.
        frame_skip: Frame skip value used during precomputation.
        camera_key: RGB camera key (e.g. "agentview_image").
        latent_type: "bn" (32-dim) or "prebn" (1024-dim).
        timesteps_frac: List of fractional timesteps in [0, 1]. Overrides timesteps.

    Returns:
        latents: (N, latent_dim) array of latent actions.
        images: list of (H, W, 3) uint8 arrays (RGB observations).
        labels: list of timestep labels (one per sample).
        demo_ids: list of demo index (one per sample).
    """
    if latent_type == "bn":
        la_group = "latent_actions"
    elif latent_type == "prebn":
        la_group = "latent_actions_prebn"
    else:
        raise ValueError(f"Unknown latent_type: {latent_type}")

    latents, images, labels, demo_ids = [], [], [], []

    for demo_key in demo_keys:
        demo = f["data"][demo_key]
        la_path = f"{la_group}/fs{frame_skip}/{camera_key}"
        if la_path not in demo:
            print(f"  Warning: {la_path} not found in {demo_key}, skipping")
            continue

        la = demo[la_path][:]  # (T, latent_dim)
        rgb = demo["obs"][camera_key][:]  # (T, 256, 256, 3)
        T = la.shape[0]
        demo_idx = int(demo_key.split("_")[1])

        # Resolve timesteps
        if timesteps_frac is not None:
            ts = [(frac, min(int(frac * (T - 1)), T - 1)) for frac in timesteps_frac]
        else:
            ts = [(t, t) for t in timesteps if t < T]

        for lbl, t in ts:
            latents.append(la[t])
            images.append(rgb[t])
            labels.append(lbl)
            demo_ids.append(demo_idx)

    latents = np.stack(latents, axis=0) if latents else np.empty((0, 0))
    return latents, images, labels, demo_ids


def run_tsne(latents, perplexity=30, random_state=42):
    """Run t-SNE on latent action array."""
    n = latents.shape[0]
    perplexity = min(perplexity, max(5, n // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
    )
    return tsne.fit_transform(latents)


def plot_tsne_scatter(
    embeddings, labels, title, output_path, label_name="timestep", cmap="tab10",
):
    """Plot a colored t-SNE scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap(cmap, len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = np.array([l == lbl for l in labels])
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=[colors(i)], label=f"{label_name}={lbl}",
            s=20, alpha=0.7, edgecolors="none",
        )

    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tsne_with_images(
    embeddings, labels, images, title, output_path,
    label_name="timestep", cmap="tab10", max_images=100, image_zoom=0.15,
):
    """Plot t-SNE with thumbnail RGB observations overlaid."""
    fig, ax = plt.subplots(figsize=(14, 12))
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap(cmap, len(unique_labels))
    label_to_color = {lbl: colors(i) for i, lbl in enumerate(unique_labels)}

    # Draw scatter for all points
    for i, lbl in enumerate(unique_labels):
        mask = np.array([l == lbl for l in labels])
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=[colors(i)], label=f"{label_name}={lbl}",
            s=10, alpha=0.3, edgecolors="none",
        )

    # Overlay image thumbnails (subsample if too many)
    n = len(images)
    if n > max_images:
        indices = np.linspace(0, n - 1, max_images, dtype=int)
    else:
        indices = np.arange(n)

    for idx in indices:
        img = images[idx]
        imagebox = OffsetImage(img, zoom=image_zoom)
        lbl = labels[idx]
        ab = AnnotationBbox(
            imagebox, (embeddings[idx, 0], embeddings[idx, 1]),
            frameon=True,
            bboxprops=dict(edgecolor=label_to_color[lbl], linewidth=1.5),
            pad=0.1,
        )
        ax.add_artist(ab)

    ax.legend(fontsize=9, markerscale=3, loc="best")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sample_observations(images, labels, title, output_path, label_name="timestep", max_per_label=5):
    """Show a grid of sample observations grouped by label."""
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    n_cols = min(max_per_label, max(sum(1 for l in labels if l == unique_labels[0]) for _ in [0]))

    # Gather images per label
    images_by_label = {}
    for img, lbl in zip(images, labels):
        images_by_label.setdefault(lbl, []).append(img)

    n_cols = min(max_per_label, max(len(v) for v in images_by_label.values()))
    fig, axes = plt.subplots(n_labels, n_cols, figsize=(2.5 * n_cols, 2.5 * n_labels))
    if n_labels == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, lbl in enumerate(unique_labels):
        imgs = images_by_label[lbl][:n_cols]
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(imgs):
                ax.imshow(imgs[col])
            ax.axis("off")
            if col == 0:
                ax.set_title(f"{label_name}={lbl}", fontsize=9)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def analyze_single_task(args):
    """Mode 1: Same task, compare latent actions at different timesteps."""
    hdf5_path = args.hdf5_paths[0]
    task_name = args.task_names[0] if args.task_names else os.path.basename(os.path.dirname(os.path.dirname(hdf5_path)))

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = get_demo_keys(f)
        if args.max_demos:
            demo_keys = demo_keys[: args.max_demos]
        print(f"Task: {task_name}, demos: {len(demo_keys)}")

        for latent_type in args.latent_types:
            print(f"\n--- Latent type: {latent_type} ---")
            latents, images, labels, demo_ids = extract_latent_actions_at_timesteps(
                f, demo_keys, args.timesteps, args.frame_skip, args.camera_key,
                latent_type=latent_type, timesteps_frac=args.timesteps_frac,
            )
            if latents.shape[0] == 0:
                print(f"  No data extracted for {latent_type}, skipping")
                continue

            print(f"  Extracted {latents.shape[0]} samples, dim={latents.shape[1]}")

            emb = run_tsne(latents, perplexity=args.perplexity)

            prefix = f"{task_name}_fs{args.frame_skip}_{args.camera_key}_{latent_type}"

            # Scatter plot colored by timestep
            plot_tsne_scatter(
                emb, labels,
                title=f"t-SNE of {latent_type} latent actions — {task_name} (fs={args.frame_skip})",
                output_path=os.path.join(args.output_dir, f"{prefix}_tsne_scatter.png"),
                label_name="t",
            )

            # Scatter with image thumbnails
            plot_tsne_with_images(
                emb, labels, images,
                title=f"t-SNE of {latent_type} latent actions — {task_name} (fs={args.frame_skip})",
                output_path=os.path.join(args.output_dir, f"{prefix}_tsne_images.png"),
                label_name="t",
                max_images=args.max_images,
                image_zoom=args.image_zoom,
            )

            # Sample observations grid
            plot_sample_observations(
                images, labels,
                title=f"Sample observations — {task_name}",
                output_path=os.path.join(args.output_dir, f"{prefix}_obs_grid.png"),
                label_name="t",
            )


def analyze_cross_task(args):
    """Mode 2: Compare latent actions from different tasks."""
    tasks_str = "_vs_".join(args.task_names or [f"task{i}" for i in range(len(args.hdf5_paths))])

    for latent_type in args.latent_types:
        print(f"\n--- Latent type: {latent_type} ---")
        all_latents, all_images, all_task_labels, all_timestep_labels = [], [], [], []

        for i, hdf5_path in enumerate(args.hdf5_paths):
            task_name = args.task_names[i] if args.task_names else os.path.basename(
                os.path.dirname(os.path.dirname(hdf5_path))
            )
            print(f"  Loading task: {task_name} from {hdf5_path}")

            with h5py.File(hdf5_path, "r") as f:
                demo_keys = get_demo_keys(f)
                if args.max_demos:
                    demo_keys = demo_keys[: args.max_demos]

                latents, images, ts_labels, _ = extract_latent_actions_at_timesteps(
                    f, demo_keys, args.timesteps, args.frame_skip, args.camera_key,
                    latent_type=latent_type, timesteps_frac=args.timesteps_frac,
                )
                if latents.shape[0] == 0:
                    continue

                all_latents.append(latents)
                all_images.extend(images)
                all_task_labels.extend([task_name] * latents.shape[0])
                all_timestep_labels.extend(ts_labels)

        if not all_latents:
            print(f"  No data extracted for {latent_type}, skipping")
            continue

        combined_latents = np.concatenate(all_latents, axis=0)
        print(f"  Combined: {combined_latents.shape[0]} samples, dim={combined_latents.shape[1]}")

        emb = run_tsne(combined_latents, perplexity=args.perplexity)

        prefix = f"cross_{tasks_str}_fs{args.frame_skip}_{args.camera_key}_{latent_type}"

        # Color by task
        plot_tsne_scatter(
            emb, all_task_labels,
            title=f"t-SNE {latent_type} — cross-task (fs={args.frame_skip})",
            output_path=os.path.join(args.output_dir, f"{prefix}_tsne_by_task.png"),
            label_name="task",
        )

        # Color by timestep
        plot_tsne_scatter(
            emb, all_timestep_labels,
            title=f"t-SNE {latent_type} — cross-task by timestep (fs={args.frame_skip})",
            output_path=os.path.join(args.output_dir, f"{prefix}_tsne_by_timestep.png"),
            label_name="t",
        )

        # Color by task@timestep (joint label)
        combined_labels = [
            f"{task}@{t}" for task, t in zip(all_task_labels, all_timestep_labels)
        ]
        plot_tsne_scatter(
            emb, combined_labels,
            title=f"t-SNE {latent_type} — cross-task x timestep (fs={args.frame_skip})",
            output_path=os.path.join(args.output_dir, f"{prefix}_tsne_by_task_timestep.png"),
            label_name="task@t",
        )

        # With image thumbnails, colored by task@timestep
        plot_tsne_with_images(
            emb, combined_labels, all_images,
            title=f"t-SNE {latent_type} — cross-task x timestep (fs={args.frame_skip})",
            output_path=os.path.join(args.output_dir, f"{prefix}_tsne_images.png"),
            label_name="task@t",
            max_images=args.max_images,
            image_zoom=args.image_zoom,
        )

        # Sample observations grid (by task)
        plot_sample_observations(
            all_images, all_task_labels,
            title=f"Sample observations — cross-task",
            output_path=os.path.join(args.output_dir, f"{prefix}_obs_grid.png"),
            label_name="task",
        )


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE analysis of precomputed LAM latent actions"
    )
    parser.add_argument(
        "--hdf5_paths", type=str, nargs="+", required=True,
        help="Path(s) to augmented HDF5 file(s) with precomputed latent actions. "
             "One path = single-task mode; multiple = cross-task mode.",
    )
    parser.add_argument(
        "--task_names", type=str, nargs="+", default=None,
        help="Human-readable task names (one per hdf5_path). Auto-inferred if omitted.",
    )
    parser.add_argument(
        "--timesteps", type=int, nargs="+", default=None,
        help="Absolute timestep indices to extract (e.g. 0 10 20 50).",
    )
    parser.add_argument(
        "--timesteps_frac", type=float, nargs="+", default=None,
        help="Fractional timesteps in [0,1] (e.g. 0.0 0.25 0.5 0.75 1.0). "
             "Overrides --timesteps. Useful for variable-length episodes.",
    )
    parser.add_argument(
        "--frame_skip", type=int, default=1,
        help="Frame skip used during precomputation (default: 1).",
    )
    parser.add_argument(
        "--camera_key", type=str, default="agentview_image",
        help="RGB camera key (default: agentview_image).",
    )
    parser.add_argument(
        "--latent_types", type=str, nargs="+", default=["bn", "prebn"],
        choices=["bn", "prebn"],
        help="Which latent types to analyze (default: both bn and prebn).",
    )
    parser.add_argument(
        "--max_demos", type=int, default=None,
        help="Max number of demos to use (default: all).",
    )
    parser.add_argument(
        "--perplexity", type=float, default=30,
        help="t-SNE perplexity (default: 30).",
    )
    parser.add_argument(
        "--max_images", type=int, default=100,
        help="Max image thumbnails to overlay on t-SNE plot (default: 100).",
    )
    parser.add_argument(
        "--image_zoom", type=float, default=0.15,
        help="Zoom level for image thumbnails (default: 0.15).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./tsne_output",
        help="Directory for output plots (default: ./tsne_output).",
    )
    args = parser.parse_args()

    # Validate timestep args
    if args.timesteps is None and args.timesteps_frac is None:
        parser.error("Must specify either --timesteps or --timesteps_frac")

    if args.task_names and len(args.task_names) != len(args.hdf5_paths):
        parser.error("--task_names must match number of --hdf5_paths")

    os.makedirs(args.output_dir, exist_ok=True)

    if len(args.hdf5_paths) == 1:
        print("=== Single-task analysis ===")
        analyze_single_task(args)
    else:
        print("=== Cross-task analysis ===")
        analyze_cross_task(args)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
