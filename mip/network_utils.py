"""Utility functions for setting up, save and load networks.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import loguru
import torch
import torch.nn as nn

from mip.config import LAMConfig, NetworkConfig, TaskConfig
from mip.encoders import (
    IdentityEncoder,
    MLPEncoder,
    MultiImageObsEncoder,
    PerStepMLPEncoder,
    PrecomputedDINOEncoder,
    PrecomputedLAMEncoder,
)


def get_network(network_config: NetworkConfig, task_config: TaskConfig):
    # Import inside function to avoid circular imports
    from mip.networks.chitfm import ChiTransformer
    from mip.networks.chiunet import ChiUNet
    from mip.networks.jannerunet import JannerUNet
    from mip.networks.lbmdit import LBMDiT, LBMDiTIDM, LBMDiTIDMv2
    from mip.networks.mlp import MLP, VanillaMLP
    from mip.networks.rnn import RNN, VanillaRNN
    from mip.networks.sudeepdit import SudeepDiT
    from mip.networks.sudeepdit_og import SudeepDiTOG
    from mip.networks.sudeepdit_og_condistill import SudeepDiTOGCondistill
    from mip.networks.sudeepdit_og_condistill_xattn import SudeepDiTOGCondistillXAttn
    from mip.networks.sudeepdit_og_xattn import SudeepDiTOGXAttn
    from mip.networks.sudeepdit_reg import SudeepDiTREG
    from mip.networks.sudeepdit_repa import SudeepDiTREPA
    from mip.networks.sudeepdit_repa_agg import SudeepDiTREPAAgg

    network_class = {
        "mlp": MLP,
        "vanilla_mlp": VanillaMLP,
        "chitransformer": ChiTransformer,
        "chiunet": ChiUNet,
        "jannerunet": JannerUNet,
        "rnn": RNN,
        "vanilla_rnn": VanillaRNN,
        "sudeepdit": SudeepDiT,
        "sudeepdit_og": SudeepDiTOG,
        "sudeepdit_og_condistill": SudeepDiTOGCondistill,
        "sudeepdit_og_condistill_xattn": SudeepDiTOGCondistillXAttn,
        "sudeepdit_og_xattn": SudeepDiTOGXAttn,
        "sudeepdit_repa": SudeepDiTREPA,
        "sudeepdit_reg": SudeepDiTREG,
        "sudeepdit_repa_agg": SudeepDiTREPAAgg,
        "lbmdit": LBMDiT,
        "lbmidm": LBMDiTIDM,
        "lbmidm_v2": LBMDiTIDMv2,
    }[network_config.network_type]

    # Common parameters for all networks
    common_params = {
        "act_dim": task_config.act_dim,
        "Ta": task_config.horizon,
        "obs_dim": network_config.emb_dim,  # all encoder will encode obs to emb_dim
        "To": task_config.obs_steps,
    }

    if network_config.network_type == "mlp":
        return network_class(
            **common_params,
            emb_dim=network_config.emb_dim,
            n_layers=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_dim=network_config.timestep_emb_dim,
        )
    elif network_config.network_type == "vanilla_mlp":
        return network_class(
            **common_params,
            emb_dim=network_config.emb_dim,
            n_layers=network_config.num_layers,
            dropout=network_config.dropout,
        )
    elif network_config.network_type == "chitransformer":
        return network_class(
            **common_params,
            d_model=network_config.emb_dim,
            nhead=network_config.n_heads,
            num_layers=network_config.num_layers,
            p_drop_emb=network_config.dropout,
            p_drop_attn=network_config.attn_dropout,
            n_cond_layers=network_config.n_cond_layers,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type == "chiunet":
        return network_class(
            **common_params,
            model_dim=network_config.model_dim,
            emb_dim=network_config.emb_dim,
            kernel_size=network_config.kernel_size,
            cond_predict_scale=network_config.cond_predict_scale,
            obs_as_global_cond=network_config.obs_as_global_cond,
            dim_mult=network_config.dim_mult,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type == "jannerunet":
        return network_class(
            **common_params,
            model_dim=network_config.model_dim,
            emb_dim=network_config.emb_dim,
            kernel_size=network_config.kernel_size,
            dim_mult=network_config.dim_mult,
            norm_type=network_config.norm_type,
            attention=network_config.attention,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type in ["rnn", "vanilla_rnn"]:
        rnn_params = {
            **common_params,
            "rnn_hidden_dim": network_config.emb_dim,
            "rnn_num_layers": network_config.num_layers,
            "rnn_type": network_config.rnn_type,
            "dropout": network_config.dropout,
        }
        if network_config.network_type == "rnn":
            rnn_params.update(
                {
                    "timestep_emb_dim": network_config.timestep_emb_dim,
                    "max_freq": network_config.max_freq,
                }
            )
        return network_class(**rnn_params)
    elif network_config.network_type in ("sudeepdit", "sudeepdit_og", "sudeepdit_og_xattn", "sudeepdit_og_condistill", "sudeepdit_og_condistill_xattn"):
        return network_class(
            **common_params,
            d_model=network_config.emb_dim,
            n_heads=network_config.n_heads,
            depth=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_type=network_config.timestep_emb_type,
        )

    elif network_config.network_type == "lbmdit":
        lbmdit_params = dict(common_params)
        enc_out_dim = _get_encoder_out_dim(network_config)
        lbmdit_params["obs_dim"] = enc_out_dim
        return network_class(
            **lbmdit_params,
            d_model=network_config.emb_dim,
            n_heads=network_config.n_heads,
            depth=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_type=network_config.timestep_emb_type,
        )

    elif network_config.network_type == "lbmidm":
        lbmidm_params = dict(common_params)
        enc_out_dim = _get_encoder_out_dim(network_config)
        lbmidm_params["obs_dim"] = enc_out_dim
        lbmidm_params["To"] = task_config.obs_steps + 1  # +1 for goal frame
        return network_class(
            **lbmidm_params,
            d_model=network_config.emb_dim,
            n_heads=network_config.n_heads,
            depth=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_type=network_config.timestep_emb_type,
        )

    elif network_config.network_type == "lbmidm_v2":
        enc_out_dim = _get_encoder_out_dim(network_config)
        return network_class(
            act_dim=task_config.act_dim,
            Ta=task_config.horizon,
            obs_dim=enc_out_dim,
            To=task_config.obs_steps + 1,  # obs frames + 1 goal frame
            To_obs=task_config.obs_steps,
            d_model=network_config.emb_dim,
            timestep_emb_dim=network_config.timestep_emb_dim,
            n_heads=network_config.n_heads,
            depth=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_type=network_config.timestep_emb_type,
            obs_summarizer_hidden=network_config.obs_summarizer_hidden,
            action_proj_hidden=network_config.action_proj_hidden,
            fdm_hidden=network_config.fdm_hidden,
        )

    elif "sudeepdit_repa" in network_config.network_type  or "sudeepdit_reg" in network_config.network_type:
        loguru.logger.info(f"REPA config - projector_dim: {network_config.projector_dim} | z_dims: {network_config.z_dims}")
        return network_class(
            **common_params,
            d_model=network_config.emb_dim,
            n_heads=network_config.n_heads,
            depth=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_type=network_config.timestep_emb_type,
            projector_dim=network_config.projector_dim,
            z_dims=network_config.z_dims,
        )


def _get_encoder_out_dim(network_config: NetworkConfig) -> int:
    """Get the encoder output dimension (encoder_out_dim if set, else emb_dim)."""
    return network_config.encoder_out_dim or network_config.emb_dim


def get_encoder(network_config: NetworkConfig, task_config: TaskConfig):
    enc_out_dim = _get_encoder_out_dim(network_config)
    if task_config.obs_type == "image":
        # Force image encoder for image observations
        encoder_type = getattr(network_config, "encoder_type", "image") or "image"
    elif task_config.obs_type in ["state", "keypoint"]:
        # For state/keypoint, default to configured encoder
        encoder_type = getattr(network_config, "encoder_type", "mlp") or "mlp"
    else:
        raise ValueError(f"Invalid observation type: {task_config.obs_type}")
    loguru.logger.info(f"Using encoder type: {encoder_type} | encoder_out_dim: {enc_out_dim}")

    if encoder_type == "identity":
        return IdentityEncoder(dropout=network_config.encoder_dropout)
    elif encoder_type == "mlp":
        return MLPEncoder(
            obs_dim=task_config.obs_dim,
            To=task_config.obs_steps,
            emb_dim=network_config.emb_dim,
            hidden_dims=[network_config.emb_dim] * network_config.num_encoder_layers,
            dropout=network_config.encoder_dropout,
        )
    elif encoder_type == "per_step_mlp":
        return PerStepMLPEncoder(
            obs_dim=task_config.obs_dim,
            emb_dim=network_config.emb_dim,
            hidden_dims=[network_config.emb_dim] * network_config.num_encoder_layers,
            dropout=network_config.encoder_dropout,
        )
    elif encoder_type == "image":
        kwargs = {
            "shape_meta": task_config.shape_meta,
            "rgb_model_name": network_config.rgb_model_name,
            "emb_dim": enc_out_dim,
            "use_seq": network_config.use_seq,
            "keep_horizon_dims": network_config.keep_horizon_dims,
            "resize_shape": task_config.resize_shape,
            "crop_shape": task_config.crop_shape,
            "random_crop": task_config.random_crop,
            "use_group_norm": task_config.use_group_norm,
        }
        return MultiImageObsEncoder(**kwargs)
    elif encoder_type == "dino":
        dino_embed_dims = {
            "vits16": 384, "vits16plus": 384,
            "vitb16": 768, "vitl16": 1024,
        }
        dino_embed_dim = dino_embed_dims[task_config.dino_model]
        dino_types = task_config.dino_types or ["cls"]
        if task_config.camera_keys:
            num_cams = len(task_config.camera_keys)
        else:
            num_cams = sum(
                1 for attr in task_config.shape_meta["obs"].values()
                if attr.get("type", "low_dim") == "rgb"
            )
        num_views = num_cams * len(dino_types)
        # Collect low_dim keys and their total dimension from shape_meta
        low_dim_keys = []
        low_dim_total_dim = 0
        for key, attr in task_config.shape_meta["obs"].items():
            if attr.get("type", "low_dim") == "low_dim":
                low_dim_keys.append(key)
                low_dim_total_dim += attr["shape"][0]
        return PrecomputedDINOEncoder(
            num_views=num_views,
            dino_embed_dim=dino_embed_dim,
            emb_dim=enc_out_dim,
            dropout=network_config.encoder_dropout,
            low_dim_keys=low_dim_keys,
            low_dim_total_dim=low_dim_total_dim,
        )
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")


def get_extra_cond_encoder(network_config: NetworkConfig, task_config: TaskConfig):
    extra_cond_encoder_type = getattr(task_config, "latent_type", "image") or "image"
    loguru.logger.info(f"Using extra cond encoder type: {extra_cond_encoder_type}")

    if extra_cond_encoder_type == "image":
        # Filter shape_meta to only include rgb keys (no low_dim)
        rgb_only_shape_meta = {
            "obs": {
                k: v for k, v in task_config.shape_meta["obs"].items()
                if v.get("type", "low_dim") != "low_dim"
            },
        }
        kwargs = {
            "shape_meta": rgb_only_shape_meta,
            "rgb_model_name": network_config.rgb_model_name,
            "emb_dim": network_config.emb_dim,
            "use_seq": network_config.use_seq,
            "keep_horizon_dims": network_config.keep_horizon_dims,
            "resize_shape": task_config.resize_shape,
            "crop_shape": task_config.crop_shape,
            "random_crop": task_config.random_crop,
            "use_group_norm": task_config.use_group_norm,
            "dropout": network_config.extra_cond_encoder_dropout,
        }
        return MultiImageObsEncoder(**kwargs)
    elif extra_cond_encoder_type == "dino":
        dino_embed_dims = {
            "vits16": 384, "vits16plus": 384,
            "vitb16": 768, "vitl16": 1024,
        }
        dino_embed_dim = dino_embed_dims[task_config.dino_model]
        dino_types = task_config.dino_types or ["cls"]
        if task_config.camera_keys:
            num_cams = len(task_config.camera_keys)
        else:
            num_cams = sum(
                1 for attr in task_config.shape_meta["obs"].values()
                if attr.get("type", "low_dim") == "rgb"
            )
        num_views = num_cams * len(dino_types)
        return PrecomputedDINOEncoder(
            num_views=num_views,
            dino_embed_dim=dino_embed_dim,
            emb_dim=network_config.emb_dim,
            dropout=network_config.extra_cond_encoder_dropout,
        )
    elif extra_cond_encoder_type == "lam":
        lam_latent_dims = {"bn": 32, "prebn": 1024}
        latent_dim = lam_latent_dims[task_config.lam_latent_type]
        if task_config.lam_camera_keys:
            num_cams = len(task_config.lam_camera_keys)
        else:
            num_cams = sum(
                1 for attr in task_config.shape_meta["obs"].values()
                if attr.get("type", "low_dim") == "rgb"
            )
        num_views = num_cams * len(task_config.lam_frame_skips)
        return PrecomputedLAMEncoder(
            num_views=num_views,
            latent_dim=latent_dim,
            emb_dim=network_config.emb_dim,
            dropout=network_config.extra_cond_encoder_dropout,
        )
    else:
        raise ValueError(f"Invalid extra cond encoder type: {extra_cond_encoder_type}")


def get_lam(lam_config: LAMConfig):
    from mip.networks.lam.modules import LatentActionModel

    lam = LatentActionModel(
        in_dim=lam_config.lam_image_channels,
        model_dim=lam_config.lam_model_dim,
        latent_dim=lam_config.lam_latent_dim,
        patch_size=lam_config.lam_patch_size,
        enc_blocks=lam_config.lam_enc_blocks,
        dec_blocks=lam_config.lam_dec_blocks,
        num_heads=lam_config.lam_num_heads,
        dropout=lam_config.lam_dropout
    )

    if lam_config.lam_ckpt_path:
        ckpt_state_dict = torch.load(lam_config.lam_ckpt_path, map_location=torch.device("cpu"))['state_dict']
        lam_state_dict = {k: v for k, v in ckpt_state_dict.items() if k.startswith("lam.")}
        lam_weights_compatible = {k.removeprefix('lam.'): v for k, v in lam_state_dict.items()}
        lam.load_state_dict(lam_weights_compatible)
        lam.eval()
        loguru.logger.info("Pretrained LAM is loaded")
        return lam
    else:
        raise ValueError("No pretrained LAM checkpoint provided")


def get_dino(task_config: TaskConfig):
    """Load a pretrained DINOv3 ViT model for on-the-fly feature extraction.

    Args:
        task_config: TaskConfig with dino_model, dino_ckpt_dir, dino_repo fields.

    Returns:
        nn.Module: DINOv3 model in eval mode with requires_grad=False.
    """
    import os

    DINO_MODELS = {
        "vits16": {"hub_name": "dinov3_vits16", "embed_dim": 384, "ckpt_file": "dinov3_vits16.pth"},
        "vits16plus": {"hub_name": "dinov3_vits16plus", "embed_dim": 384, "ckpt_file": "dinov3_vits16plus.pth"},
        "vitb16": {"hub_name": "dinov3_vitb16", "embed_dim": 768, "ckpt_file": "dinov3_vitb16.pth"},
        "vitl16": {"hub_name": "dinov3_vitl16", "embed_dim": 1024, "ckpt_file": "dinov3_vitl16.pth"},
    }

    model_name = task_config.dino_model
    if model_name not in DINO_MODELS:
        raise ValueError(f"Unknown DINO model '{model_name}'. Choose from: {list(DINO_MODELS.keys())}")

    info = DINO_MODELS[model_name]
    ckpt_dir = getattr(task_config, "dino_ckpt_dir", "/oscar/data/csun45/zzeng28/cache/torch/dinov3")
    dino_repo = getattr(task_config, "dino_repo", "/oscar/data/csun45/zzeng28/cache/torch/dinov3/dinov3")
    ckpt_path = os.path.join(ckpt_dir, info["ckpt_file"])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DINO checkpoint not found: {ckpt_path}")

    model = torch.hub.load(dino_repo, info["hub_name"], source="local", weights=ckpt_path)
    model.eval()
    model.requires_grad_(False)
    loguru.logger.info(f"Loaded DINOv3 {model_name} (embed_dim={info['embed_dim']}) from {ckpt_path}")
    return model

class GroupNorm1d(nn.Module):
    def __init__(self, dim, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, dim // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)
