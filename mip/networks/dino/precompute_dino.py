"""Precompute DINOv3 CLS and mean-pooled patch features and store them in an augmented HDF5 file.

Usage:
    python mip/networks/dino/precompute_dino.py \
        --input_hdf5 /path/to/image_v15_256.hdf5 \
        --output_hdf5 /path/to/image_v15_dino_256.hdf5 \
        --dino_model vitb16 \
        --dino_repo /path/to/dinov3

Available models (checkpoints in --dino_ckpt_dir):
    vits16      ViT-S/16   (embed_dim=384,  21M params)
    vits16plus  ViT-S+/16  (embed_dim=384,  29M params, SwiGLU FFN)
    vitb16      ViT-B/16   (embed_dim=768,  86M params)
    vitl16      ViT-L/16   (embed_dim=1024, 300M params)
"""

import argparse
import os
import shutil

import h5py
import numpy as np
import torch
import torchvision  # noqa: F401
from torchvision.transforms import v2
from tqdm import tqdm

DEFAULT_CKPT_DIR = "/oscar/data/csun45/zzeng28/cache/torch/dinov3"

DINO_MODELS = {
    "vits16": {"hub_name": "dinov3_vits16", "embed_dim": 384, "ckpt_file": "dinov3_vits16.pth"},
    "vits16plus": {"hub_name": "dinov3_vits16plus", "embed_dim": 384, "ckpt_file": "dinov3_vits16plus.pth"},
    "vitb16": {"hub_name": "dinov3_vitb16", "embed_dim": 768, "ckpt_file": "dinov3_vitb16.pth"},
    "vitl16": {"hub_name": "dinov3_vitl16", "embed_dim": 1024, "ckpt_file": "dinov3_vitl16.pth"},
}


def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def discover_rgb_keys(demo_obs_group):
    """Auto-detect RGB camera keys by finding datasets with shape (T, 256, 256, 3)."""
    rgb_keys = []
    for key in demo_obs_group:
        ds = demo_obs_group[key]
        if len(ds.shape) == 4 and ds.shape[1] == 256 and ds.shape[2] == 256 and ds.shape[3] == 3:
            rgb_keys.append(key)
    return sorted(rgb_keys)


def compute_dino_features(model, images, batch_size, device, transform):
    """Extract DINOv3 CLS and mean-pooled patch features for every frame.

    Args:
        model: DINOv3 ViT in eval mode.
        images: (T, H, W, C) uint8 numpy array.
        batch_size: images per forward pass.
        device: torch device.
        transform: torchvision transform for preprocessing.

    Returns:
        cls_features: (T, embed_dim) float32 numpy array.
        patch_features: (T, embed_dim) float32 numpy array (mean-pooled over patches).
    """
    T = images.shape[0]
    all_cls = []
    all_patch = []

    n_batches = (T + batch_size - 1) // batch_size
    for batch_start in tqdm(
        range(0, T, batch_size),
        total=n_batches,
        desc="  DINO inference",
        leave=False,
    ):
        batch_uint8 = images[batch_start : batch_start + batch_size]  # (B, H, W, C)
        # transform expects (B, C, H, W) uint8 input for ToImage
        batch_tensor = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2)  # (B, C, H, W)
        batch_tensor = transform(batch_tensor).to(device)

        with torch.no_grad():
            out = model.forward_features(batch_tensor)
            cls = out["x_norm_clstoken"]  # (B, embed_dim)
            patches = out["x_norm_patchtokens"]  # (B, N_patches, embed_dim)
            patch_mean = patches.mean(dim=1)  # (B, embed_dim)
            all_cls.append(cls.cpu().numpy())
            all_patch.append(patch_mean.cpu().numpy())

    cls_features = np.concatenate(all_cls, axis=0)  # (T, embed_dim)
    patch_features = np.concatenate(all_patch, axis=0)  # (T, embed_dim)
    assert cls_features.shape[0] == T, f"Expected {T}, got {cls_features.shape[0]}"
    assert patch_features.shape[0] == T
    return cls_features, patch_features


def load_dino(model_name, ckpt_dir, device, dino_repo):
    """Load a DINOv3 ViT model and populate it with local checkpoint weights.

    Args:
        model_name: key in DINO_MODELS (e.g. "vitb16").
        ckpt_dir: directory that contains the .pth checkpoint files.
        device: torch device.
        dino_repo: path to a local clone of facebookresearch/dinov3.

    Returns:
        (model, embed_dim)
    """
    if model_name not in DINO_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(DINO_MODELS.keys())}")

    info = DINO_MODELS[model_name]
    ckpt_path = os.path.join(ckpt_dir, info["ckpt_file"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model and load weights via torch.hub from local repo
    model = torch.hub.load(
        dino_repo, info["hub_name"], source="local", weights=ckpt_path,
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded DINOv3 {model_name} (embed_dim={info['embed_dim']}) from {ckpt_path}")
    return model, info["embed_dim"]


def main():
    parser = argparse.ArgumentParser(
        description="Precompute DINOv3 CLS features into augmented HDF5"
    )
    parser.add_argument("--input_hdf5", type=str, required=True, help="Path to source robomimic HDF5")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path for augmented output HDF5")
    parser.add_argument(
        "--dino_model",
        type=str,
        default="vitb16",
        choices=list(DINO_MODELS.keys()),
        help="DINOv3 model variant (default: vitb16)",
    )
    parser.add_argument(
        "--dino_ckpt_dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help=f"Directory containing DINOv3 .pth files (default: {DEFAULT_CKPT_DIR})",
    )
    parser.add_argument(
        "--dino_repo",
        type=str,
        default="/oscar/data/csun45/zzeng28/cache/torch/dinov3/dinov3",
        help="Path to local dinov3 repo clone",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Step 1: Copy input to output to preserve all original data
    if os.path.exists(args.output_hdf5):
        print(f"Output {args.output_hdf5} already exists, resuming (will add/overwrite DINO features)")
    else:
        print(f"Copying {args.input_hdf5} -> {args.output_hdf5}")
        shutil.copy2(args.input_hdf5, args.output_hdf5)

    # Step 2: Load DINOv3 and build transform
    device = torch.device(args.device)
    model, embed_dim = load_dino(args.dino_model, args.dino_ckpt_dir, device, args.dino_repo)
    transform = make_transform(resize_size=256)

    # Step 3: Process each demo
    with h5py.File(args.output_hdf5, "a") as f:
        demos = f["data"]
        demo_keys = sorted(demos.keys(), key=lambda x: int(x.split("_")[1]))

        # Auto-discover RGB camera keys from first demo
        first_demo = demos[demo_keys[0]]
        rgb_keys = discover_rgb_keys(first_demo["obs"])
        assert len(rgb_keys) > 0, "No RGB camera keys found (expected 256x256x3 datasets)"
        print(f"Discovered RGB cameras: {rgb_keys}")
        print(f"Processing {len(demo_keys)} demos...")

        total_tasks = len(demo_keys) * len(rgb_keys)
        pbar = tqdm(total=total_tasks, desc="Processing demos")
        for demo_key in demo_keys:
            demo = demos[demo_key]

            for camera_key in rgb_keys:
                pbar.set_postfix(demo=demo_key, cam=camera_key)
                # Load images for this camera (one camera at a time for memory efficiency)
                images = demo["obs"][camera_key][:]  # (T, 256, 256, 3) uint8

                cls_features, patch_features = compute_dino_features(
                    model, images, args.batch_size, device, transform
                )

                # Write CLS: demo_i/dino_cls/{args.dino_model}/{camera_key}  →  (T, embed_dim)
                cls_path = f"{demo_key}/dino_cls/{args.dino_model}/{camera_key}"
                if cls_path in f["data"]:
                    del f["data"][cls_path]
                demo.create_dataset(
                    f"dino_cls/{args.dino_model}/{camera_key}",
                    data=cls_features,
                    dtype=np.float32,
                )

                # Write mean-pooled patches: demo_i/dino_patch_mean/{args.dino_model}/{camera_key}  →  (T, embed_dim)
                patch_path = f"{demo_key}/dino_patch_mean/{args.dino_model}/{camera_key}"
                if patch_path in f["data"]:
                    del f["data"][patch_path]
                demo.create_dataset(
                    f"dino_patch_mean/{args.dino_model}/{camera_key}",
                    data=patch_features,
                    dtype=np.float32,
                )

                pbar.update(1)
        pbar.close()

        # Step 4: Store metadata as HDF5 root attributes
        # Merge with existing models to support multiple model runs
        existing_models = list(f.attrs.get("dino_models", []))
        existing_embed_dims = dict(zip(
            existing_models,
            f.attrs.get("dino_embed_dims", []), strict=False,
        ))
        existing_embed_dims[args.dino_model] = embed_dim
        merged_models = sorted(set(existing_models + [args.dino_model]))
        f.attrs["dino_models"] = merged_models
        f.attrs["dino_embed_dims"] = [existing_embed_dims[m] for m in merged_models]

        existing_rgb = list(f.attrs.get("dino_rgb_keys", []))
        merged_rgb = sorted(set(existing_rgb + rgb_keys))
        f.attrs["dino_rgb_keys"] = merged_rgb

    print("Done! Precomputed DINO features saved to:", args.output_hdf5)

    # Print summary
    with h5py.File(args.output_hdf5, "r") as f:
        first_demo = f["data"][demo_keys[0]]
        print("\nSummary for first demo:")
        for cam in rgb_keys:
            cls_path = f"dino_cls/{args.dino_model}/{cam}"
            if cls_path in first_demo:
                print(f"  {cls_path}: {first_demo[cls_path].shape}")
            patch_path = f"dino_patch_mean/{args.dino_model}/{cam}"
            if patch_path in first_demo:
                print(f"  {patch_path}: {first_demo[patch_path].shape}")


if __name__ == "__main__":
    main()
