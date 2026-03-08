"""Utilities for on-the-fly DINO feature extraction at eval time."""

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2


def make_dino_transform(resize_size: int = 256):
    """Build the preprocessing transform for env images (float CHW in [0,1])."""
    return v2.Compose([
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class DINOFeatureExtractor:
    """Extracts DINO features from raw env images on-the-fly.

    Produces a dict matching PrecomputedDINOEncoder's expected input format:
        {f"dino_{type}_{model}_{cam}": (num_envs, obs_steps, embed_dim), ...}

    Args:
        model: DINOv3 model in eval mode (from get_dino).
        dino_model: Model name string (e.g. "vitb16").
        dino_types: List of feature types to extract (e.g. ["cls", "patch_mean"]).
        camera_keys: List of camera keys from the env (e.g. ["agentview_image"]).
        device: Torch device.
        low_dim_keys: List of low_dim observation keys to pass through (e.g. ["robot0_eef_pos"]).
        normalizers: Dict of normalizers for low_dim keys (from dataset.normalizer["obs"]).
    """

    def __init__(
        self,
        model: nn.Module,
        dino_model: str,
        dino_types: list[str],
        camera_keys: list[str],
        device: str = "cuda",
        low_dim_keys: list[str] | None = None,
        normalizers: dict | None = None,
    ):
        self.model = model
        self.dino_model = dino_model
        self.dino_types = dino_types
        self.camera_keys = camera_keys
        self.device = device
        self.transform = make_dino_transform()
        self.low_dim_keys = low_dim_keys or []
        self.normalizers = normalizers or {}

    @torch.no_grad()
    def extract(self, obs_raw: dict) -> dict:
        """Extract DINO features from raw env observation dict.

        Args:
            obs_raw: Dict from env, each value is (num_envs, obs_steps, C, H, W) float numpy array.

        Returns:
            Dict keyed like "dino_cls_vitb16_agentview_image", each (num_envs, obs_steps, embed_dim) torch tensor.
        """
        result = {}
        for cam_key in self.camera_keys:
            images = obs_raw[cam_key]  # (num_envs, obs_steps, C, H, W)
            num_envs, obs_steps = images.shape[0], images.shape[1]
            import pdb
            pdb.set_trace()

            # Flatten to (num_envs * obs_steps, C, H, W)
            flat = images.reshape(-1, *images.shape[2:])

            # Already in (N, C, H, W) format from env
            tensor = torch.from_numpy(flat)
            tensor = self.transform(tensor).to(self.device)

            # Forward pass
            out = self.model.forward_features(tensor)

            for dino_type in self.dino_types:
                if dino_type == "cls":
                    feat = out["x_norm_clstoken"]  # (N, embed_dim)
                elif dino_type == "patch_mean":
                    feat = out["x_norm_patchtokens"].mean(dim=1)  # (N, embed_dim)
                else:
                    raise ValueError(f"Unknown dino_type: {dino_type}")

                # Reshape back to (num_envs, obs_steps, embed_dim)
                feat = feat.view(num_envs, obs_steps, -1)
                key = f"dino_{dino_type}_{self.dino_model}_{cam_key}"
                result[key] = feat

        # Pass through low_dim keys (normalize + convert to tensor)
        for key in self.low_dim_keys:
            if key in obs_raw:
                val = obs_raw[key].astype(np.float32)
                if key in self.normalizers:
                    val = self.normalizers[key].normalize(val)
                result[key] = torch.tensor(val, device=self.device, dtype=torch.float32)

        return result
