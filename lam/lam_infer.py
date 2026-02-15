
import os

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from lam.modules import LatentActionModel
from tqdm import tqdm


def save_comparison(prev_frame, orig_frame, recon_frame, frame_idx, output_dir):
    """Save a side-by-side comparison of previous, original, and reconstructed frames."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(prev_frame)
    axes[0].set_title("Previous")
    axes[0].axis("off")
    axes[1].imshow(orig_frame)
    axes[1].set_title("Original")
    axes[1].axis("off")
    axes[2].imshow(recon_frame)
    axes[2].set_title("Reconstructed")
    axes[2].axis("off")
    fig.suptitle(f"Frame {frame_idx}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"compare_{frame_idx:04d}.png"), dpi=150)
    plt.close(fig)


@hydra.main(version_base=None, config_path="./config/", config_name="lam")
def main(config):
    lam = LatentActionModel(
        in_dim=config.model.image_channels,
        model_dim=config.model.lam_model_dim,
        latent_dim=config.model.lam_latent_dim,
        patch_size=config.model.lam_patch_size,
        enc_blocks=config.model.lam_enc_blocks,
        dec_blocks=config.model.lam_dec_blocks,
        num_heads=config.model.lam_num_heads,
        dropout=config.model.lam_dropout
    )

    ckpt_state_dict = torch.load(config.model.lam_ckpt_path, map_location=torch.device("cpu"))['state_dict']
    lam_state_dict = {k: v for k, v in ckpt_state_dict.items() if k.startswith("lam.")}
    lam_weights_compatible = {k.removeprefix('lam.'): v for k, v in lam_state_dict.items()}
    lam.load_state_dict(lam_weights_compatible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = lam.to(device)
    lam.eval()

    print('LAM Checkpoint is loaded!')
    task = 'can'
    dataset_path = f'/users/zzeng28/data/zzeng28/repo/lardp/data/robomimic/{task}/ph/image_v15_256.hdf5'
    with h5py.File(dataset_path) as file:
        obs = file["data"]["demo_0"]["obs"]["agentview_image"][()]
        # obs = file["data"]["demo_0"]["obs"]["shouldercamera0_image"][()]
        # obs = file["data"]["demo_0"]["obs"]["sideview_image"][()]

    # Apply frame_skip=2 to match LAM training distribution
    frame_skip = 8
    obs = obs[::frame_skip]
    print(f"Subsampled frames: {obs.shape[0]} (from original with frame_skip={frame_skip})")

    # Preprocess: uint8 [0, 255] -> float32 [0, 1]
    obs_float = obs.astype(np.float32) / 255.0

    output_dir = f"./lam_infer_output/{task}_skip{frame_skip}"
    os.makedirs(output_dir, exist_ok=True)

    all_latent_actions = []
    num_windows = obs_float.shape[0] - 1

    with torch.no_grad():
        for i in tqdm(range(num_windows), desc="LAM inference"):
            # Shape: (1, 2, 256, 256, 3)
            pair = obs_float[i : i + 2][np.newaxis]
            batch = {"videos": torch.from_numpy(pair).to(device)}

            output = lam(batch)

            # Collect latent action
            z_rep = output["z_rep"]
            all_latent_actions.append(z_rep.cpu())

            # Save per-frame comparison
            recon = output["recon"]  # (1, 1, H, W, C)
            recon_frame = recon[0, 0].clamp(0, 1).cpu().numpy()
            prev_frame = obs_float[i]
            target_frame = obs_float[i + 1]
            save_comparison(prev_frame, target_frame, recon_frame, i + 1, output_dir)

    all_latent_actions = torch.cat(all_latent_actions, dim=0)
    print(f"\nDone! Saved {num_windows} reconstructed frames to {output_dir}/")
    print(f"Latent actions shape: {all_latent_actions.shape}")


if __name__ == "__main__":
    main()

