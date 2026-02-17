"""Precompute LAM latent actions and store them in an augmented HDF5 file.

Usage:
    python mip/networks/lam/precompute_lam.py \
        --input_hdf5 /path/to/image_v15_256.hdf5 \
        --output_hdf5 /path/to/image_v15_lam_256.hdf5 \
        --lam_ckpt_path /path/to/lam.ckpt \
        --frame_skips 1,8 \
        --store_prebn
"""

import argparse
import os
import shutil

import h5py
import numpy as np
import torch
from tqdm import tqdm


def discover_rgb_keys(demo_obs_group):
    """Auto-detect RGB camera keys by finding datasets with shape (T, 256, 256, 3)."""
    rgb_keys = []
    for key in demo_obs_group:
        ds = demo_obs_group[key]
        if len(ds.shape) == 4 and ds.shape[1] == 256 and ds.shape[2] == 256 and ds.shape[3] == 3:
            rgb_keys.append(key)
    return sorted(rgb_keys)


def compute_latent_actions(lam, images, frame_skip, batch_size, device, store_prebn=False):
    """Compute dense latent actions for a sequence of images with given frame_skip.

    For every timestep t in [0, T-1], computes LAM(o_t, o_{min(t+frame_skip, T-1)}).
    This produces a dense (T, latent_dim) array so that any sampled window
    [s, s+horizon] can read latent[s:s+horizon] directly.

    Args:
        lam: LatentActionModel in eval mode
        images: (T, H, W, C) uint8 numpy array
        frame_skip: look-ahead interval for frame pairs
        batch_size: number of pairs per forward pass
        device: torch device
        store_prebn: whether to also return z_rep_prebn

    Returns:
        z_mu: (T, 32) float32 numpy array
        z_prebn: (T, 1024) float32 numpy array or None
    """
    T = images.shape[0]
    if T < 2:
        return None, None

    # Build dense pairs: (t, min(t + frame_skip, T - 1)) for every t
    pairs = [(t, min(t + frame_skip, T - 1)) for t in range(T)]

    # Process in batches
    all_z_mu = []
    all_z_prebn = []

    n_batches = (len(pairs) + batch_size - 1) // batch_size
    for batch_start in tqdm(
        range(0, len(pairs), batch_size),
        total=n_batches,
        desc=f"  LAM inference (fs={frame_skip})",
        leave=False,
    ):
        batch_pairs = pairs[batch_start : batch_start + batch_size]

        # Build video tensor: (B, 2, H, W, C) float32
        video_batch = np.stack(
            [
                np.stack(
                    [images[a].astype(np.float32) / 255.0, images[b].astype(np.float32) / 255.0],
                    axis=0,
                )
                for a, b in batch_pairs
            ],
            axis=0,
        )
        video_tensor = torch.from_numpy(video_batch).to(device)

        with torch.no_grad():
            enc_out = lam.encode(video_tensor)
            # z_mu: (B, 32) since T=2 per pair -> T-1=1 latent per pair
            z_mu = enc_out["z_mu"].cpu().numpy()  # (B, 32)
            all_z_mu.append(z_mu)
            if store_prebn:
                z_prebn = enc_out["z_rep_prebn"].cpu().numpy()  # (B, 1024)
                all_z_prebn.append(z_prebn)

    z_mu_dense = np.concatenate(all_z_mu, axis=0)  # (T, 32)
    assert z_mu_dense.shape[0] == T, f"Expected {T}, got {z_mu_dense.shape[0]}"

    z_prebn_dense = None
    if store_prebn:
        z_prebn_dense = np.concatenate(all_z_prebn, axis=0)  # (T, 1024)
        assert z_prebn_dense.shape[0] == T

    return z_mu_dense, z_prebn_dense


def load_lam(lam_ckpt_path, device, **lam_kwargs):
    """Load LAM model from checkpoint.

    Uses the same pattern as mip/network_utils.py:get_lam().
    """
    from mip.networks.lam.modules import LatentActionModel

    defaults = {
        "in_dim": 3,
        "model_dim": 1024,
        "latent_dim": 32,
        "patch_size": 16,
        "enc_blocks": 16,
        "dec_blocks": 16,
        "num_heads": 16,
        "dropout": 0.0,
    }
    defaults.update(lam_kwargs)

    lam = LatentActionModel(**defaults)

    ckpt_state_dict = torch.load(lam_ckpt_path, map_location="cpu")["state_dict"]
    lam_state_dict = {
        k: v for k, v in ckpt_state_dict.items() if k.startswith("lam.")
    }
    lam_weights = {k.removeprefix("lam."): v for k, v in lam_state_dict.items()}
    lam.load_state_dict(lam_weights)
    lam = lam.to(device)
    lam.eval()
    print(f"Loaded LAM checkpoint from {lam_ckpt_path}")
    return lam


def main():
    parser = argparse.ArgumentParser(
        description="Precompute LAM latent actions into augmented HDF5"
    )
    parser.add_argument("--input_hdf5", type=str, required=True, help="Path to source robomimic HDF5")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path for augmented output HDF5")
    parser.add_argument("--lam_ckpt_path", type=str, required=True, help="Path to LAM checkpoint")
    parser.add_argument("--frame_skips", type=str, default="1", help='Comma-separated list, e.g. "1,8"')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--store_prebn", action="store_true", help="Also store 1024-dim z_rep_prebn")
    args = parser.parse_args()

    frame_skips = [int(x) for x in args.frame_skips.split(",")]

    # Step 1: Copy input to output to preserve all original data
    # If output already exists, skip copy to allow resuming with new frame_skips/cameras
    if os.path.exists(args.output_hdf5):
        print(f"Output {args.output_hdf5} already exists, resuming (will add/overwrite requested frame_skips)")
    else:
        print(f"Copying {args.input_hdf5} -> {args.output_hdf5}")
        shutil.copy2(args.input_hdf5, args.output_hdf5)

    # Step 2: Load LAM
    device = torch.device(args.device)
    lam = load_lam(args.lam_ckpt_path, device)

    # Step 3: Process each demo
    with h5py.File(args.output_hdf5, "a") as f:
        demos = f["data"]
        demo_keys = sorted(demos.keys(), key=lambda x: int(x.split("_")[1]))

        # Auto-discover RGB camera keys from first demo
        first_demo = demos[demo_keys[0]]
        rgb_keys = discover_rgb_keys(first_demo["obs"])
        assert len(rgb_keys) > 0, "No RGB camera keys found (expected 256x256x3 datasets)"
        print(f"Discovered RGB cameras: {rgb_keys}")
        print(f"Frame skips: {frame_skips}")
        print(f"Processing {len(demo_keys)} demos...")

        total_tasks = len(demo_keys) * len(rgb_keys) * len(frame_skips)
        pbar = tqdm(total=total_tasks, desc="Processing demos")
        for demo_key in demo_keys:
            demo = demos[demo_key]

            for camera_key in rgb_keys:
                # Load images for this camera (one camera at a time for memory efficiency)
                images = demo["obs"][camera_key][:]  # (T, 256, 256, 3) uint8

                for fs in frame_skips:
                    pbar.set_postfix(demo=demo_key, cam=camera_key, fs=fs)
                    z_mu, z_prebn = compute_latent_actions(
                        lam, images, fs, args.batch_size, device, store_prebn=args.store_prebn
                    )

                    if z_mu is None:
                        # Too few frames for this frame_skip; skip
                        pbar.update(1)
                        continue

                    # Write z_mu: demo_i/latent_actions/fs{k}/{camera_key}
                    ds_path = f"{demo_key}/latent_actions/fs{fs}/{camera_key}"
                    if ds_path in f["data"]:
                        del f["data"][ds_path]
                    demo.create_dataset(
                        f"latent_actions/fs{fs}/{camera_key}",
                        data=z_mu,
                        dtype=np.float32,
                    )

                    # Optionally write z_prebn
                    if args.store_prebn and z_prebn is not None:
                        ds_prebn_path = f"{demo_key}/latent_actions_prebn/fs{fs}/{camera_key}"
                        if ds_prebn_path in f["data"]:
                            del f["data"][ds_prebn_path]
                        demo.create_dataset(
                            f"latent_actions_prebn/fs{fs}/{camera_key}",
                            data=z_prebn,
                            dtype=np.float32,
                        )

                    pbar.update(1)
        pbar.close()

        # Step 4: Store metadata as HDF5 root attributes
        # Merge with existing frame_skips to support incremental runs
        existing_fs = list(f.attrs.get("lam_frame_skips", []))
        merged_fs = sorted(set(existing_fs + frame_skips))
        f.attrs["lam_frame_skips"] = merged_fs
        f.attrs["lam_latent_dim"] = lam.latent_dim
        f.attrs["lam_ckpt_path"] = args.lam_ckpt_path
        existing_rgb = list(f.attrs.get("lam_rgb_keys", []))
        merged_rgb = sorted(set(existing_rgb + rgb_keys))
        f.attrs["lam_rgb_keys"] = merged_rgb

    print("Done! Precomputed latent actions saved to:", args.output_hdf5)

    # Print summary
    with h5py.File(args.output_hdf5, "r") as f:
        first_demo = f["data"][demo_keys[0]]
        print("\nSummary for first demo:")
        for fs in frame_skips:
            for cam in rgb_keys:
                path = f"latent_actions/fs{fs}/{cam}"
                if path in first_demo:
                    print(f"  {path}: {first_demo[path].shape}")
                prebn_path = f"latent_actions_prebn/fs{fs}/{cam}"
                if prebn_path in first_demo:
                    print(f"  {prebn_path}: {first_demo[prebn_path].shape}")


if __name__ == "__main__":
    main()
