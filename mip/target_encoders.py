"""Pluggable frozen target-feature encoders for the frozen-target ablation.

A ``TargetEncoder`` maps goal observations to the flow-matching state-flow
regression target — the "state representation" the joint policy denoises toward.
Each implementation owns its own loading, input preprocessing, forward, and
pooling, so different pretrained visual features (a DP/LBMDiT encoder, LeWM,
DINOv2, ...) can be swapped behind a single interface via
``get_target_encoder``. The agent applies the precomputed per-dim z-score
(``_normalize_goal`` / ``goal_stats_path``) on top of ``embed``'s raw output, so
``embed`` returns the *unnormalized* features.

All target encoders are frozen (``requires_grad_(False)``) and eval-always:
``train()`` is a no-op so they keep deterministic (center-crop, running-stats)
behavior regardless of the agent's mode.

Author: Zilai Zeng
"""

from __future__ import annotations

import os

import loguru
import torch
import torch.nn as nn

from mip.config import Config
from mip.network_utils import get_encoder


class TargetEncoder(nn.Module):
    """Frozen encoder: ``goal_obs -> (B, 1, output_dim)`` raw target features.

    Subclasses set ``output_dim`` and implement ``embed``. The module is frozen
    and forced to eval; ``train(mode)`` is overridden to a no-op so it never
    leaves eval (deterministic targets — no CropRandomizer / dropout / BN-batch
    noise in the regression target).
    """

    output_dim: int

    def embed(self, goal_obs) -> torch.Tensor:  # pragma: no cover - interface
        """Return the raw (pre-z-score) target features, shape ``(B, 1, D)``."""
        raise NotImplementedError

    def input_encoder_init_state_dict(self) -> dict | None:
        """A state_dict loadable into the agent's input ``MultiImageObsEncoder``
        for the ``init_input_encoder_from_dp`` warm-start, or ``None`` when the
        target encoder is architecturally incompatible (e.g. a foreign ViT).
        """
        return None

    def train(self, mode: bool = True):
        # Frozen target stays in eval regardless of the agent's train/eval.
        return super().train(False)

    # --- shared helpers for image-only foreign target encoders ---
    @staticmethod
    def _resolve_image_key(config: Config, opt) -> str:
        """The rgb obs key an image-only target encoder consumes:
        ``opt.target_encoder_image_key``, else the single rgb key in shape_meta
        (errors if 0 or >1 — e.g. multi-camera robomimic; set it explicitly).
        """
        key = getattr(opt, "target_encoder_image_key", None)
        if key is not None:
            return key
        rgb = [
            k
            for k, a in config.task.shape_meta["obs"].items()
            if a.get("type") == "rgb"
        ]
        if len(rgb) != 1:
            raise ValueError(
                f"target_encoder_image_key is null but the task has {len(rgb)} "
                f"rgb keys ({rgb}); set optimization.target_encoder_image_key "
                f"explicitly."
            )
        return rgb[0]

    def _flatten_images(self, goal_obs) -> tuple[torch.Tensor, int, int]:
        """Pull ``self._image_key`` and flatten ``(B,T,C,H,W) -> (B*T,C,H,W)``
        (also accepts ``(B,C,H,W)``). Returns ``(flat, B, T)``.
        """
        img = goal_obs[self._image_key]
        if img.dim() == 5:
            B, T = img.shape[0], img.shape[1]
            flat = img.reshape(B * T, *img.shape[2:])
        elif img.dim() == 4:
            B, T = img.shape[0], 1
            flat = img
        else:
            raise ValueError(
                f"unexpected image ndim {img.dim()} for key {self._image_key!r}"
            )
        return flat, B, T


class DPTargetEncoder(TargetEncoder):
    """Frozen DP/LBMDiT encoder — the same ``MultiImageObsEncoder`` architecture
    as the agent's input encoder, loaded from a checkpoint's ``encoder`` /
    ``encoder_ema`` key. This is the original frozen-target behavior.
    """

    def __init__(self, config: Config):
        super().__init__()
        opt = config.optimization
        device = opt.device

        dp_path = opt.dp_checkpoint_path
        if dp_path is None:
            raise ValueError(
                "optimization.dp_checkpoint_path must be set for "
                "target_encoder_type='dp' — it is the frozen encoder that "
                "produces the FM state-flow target."
            )
        loguru.logger.info(f"Loading FROZEN target encoder from {dp_path} (DP/LBMDiT)")
        state_dict = torch.load(dp_path, map_location=device, weights_only=False)
        encoder_key = "encoder_ema" if opt.dp_use_encoder_ema else "encoder"
        if encoder_key not in state_dict:
            available = sorted(k for k in state_dict if "encoder" in k)
            raise KeyError(
                f"DP checkpoint {dp_path} has no '{encoder_key}' key. "
                f"Available encoder-like keys: {available}. Toggle "
                f"optimization.dp_use_encoder_ema or pick another checkpoint."
            )
        encoder_sd = state_dict[encoder_key]
        # LBMDiT does not wrap the encoder; a leading 'encoder.' prefix or an
        # 'uncond_emb' means this is an IDM / GoalDropout checkpoint by mistake.
        if "uncond_emb" in encoder_sd or any(
            k.startswith("encoder.") for k in encoder_sd
        ):
            raise RuntimeError(
                f"DP checkpoint {dp_path} key '{encoder_key}' looks like a "
                f"GoalDropoutEncoder / IDM state dict (has 'encoder.' prefix "
                f"or 'uncond_emb'). Point dp_checkpoint_path at an LBMDiT (DP) "
                f"checkpoint instead."
            )

        # Build with the SAME network/task config as the trainable encoder so
        # the output dim equals obs_dim (no trunk change needed). load_state_dict
        # is the architecture guardrail — it errors loudly on any mismatch.
        self.encoder = get_encoder(config.network, config.task).to(device)
        self.encoder.load_state_dict(encoder_sd)
        self.output_dim = config.network.encoder_out_dim or config.network.emb_dim

        self.requires_grad_(False)
        self.eval()
        loguru.logger.info(
            f"Loaded {encoder_key} into frozen target encoder "
            f"({sum(v.numel() for v in encoder_sd.values()):,} params, "
            f"D={self.output_dim})"
        )

    def embed(self, goal_obs) -> torch.Tensor:
        return self.encoder(goal_obs, None)  # (B, 1, output_dim)

    def input_encoder_init_state_dict(self) -> dict | None:
        # Same arch as the agent's input encoder (both get_encoder(network,task)).
        return self.encoder.state_dict()


class LeWMTargetEncoder(TargetEncoder):
    """Frozen LeWM target features: a HuggingFace ViT-tiny encoder + a
    BatchNorm projector, loaded from the published ``quentinll/lewm-*``
    checkpoint (a dir with ``weights.pt`` + ``config.json``, or a direct
    ``weights.pt`` path via ``target_encoder_path``).

    ``embed(goal)`` = ``projector(encoder(img).last_hidden_state[:, 0])`` ->
    ``(B, 1, 192)`` — LeWM's operative latent (the space its world model
    predicts and plans in; the CLS is an intermediate the loss never uses).

    Vendored: only depends on ``transformers`` (ViTModel/ViTConfig) + torch.
    The ViT factory, the projector MLP, and the ImageNet preprocessing are
    reimplemented here, so neither stable-worldmodel nor stable-pretraining is
    required. Preprocessing matches LeWM's ``get_img_preprocessor`` exactly: the
    image is taken from a single rgb obs key, our ``x*2-1`` normalization is
    inverted to [0,1], ImageNet-normalized, then resized to 224 with the same
    torchvision v2 op LeWM uses (bilinear + antialias).
    """

    # ViT-tiny preset (vit_hf size map "tiny" + the lewm-pusht config.json).
    _HIDDEN = 192
    _LAYERS = 12
    _HEADS = 3
    _PATCH = 14
    _IMG = 224
    _PROJ_HIDDEN = 2048
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, config: Config):
        super().__init__()
        # Lazy import so the DP path never requires transformers.
        from transformers import ViTConfig, ViTModel

        opt = config.optimization
        device = opt.device

        path = opt.target_encoder_path
        if path is None:
            raise ValueError(
                "optimization.target_encoder_path must be set for "
                "target_encoder_type='lewm' (the LeWM checkpoint dir with "
                "weights.pt + config.json, or a direct weights.pt path)."
            )
        path = os.path.expanduser(path)
        weights = os.path.join(path, "weights.pt") if os.path.isdir(path) else path
        loguru.logger.info(f"Loading FROZEN LeWM target encoder from {weights}")
        sd = torch.load(weights, map_location=device, weights_only=False)

        # --- ViT-tiny encoder (vendored vit_hf factory) ---
        vit_cfg = ViTConfig(
            hidden_size=self._HIDDEN,
            num_hidden_layers=self._LAYERS,
            num_attention_heads=self._HEADS,
            intermediate_size=self._HIDDEN * 4,
            image_size=self._IMG,
            patch_size=self._PATCH,
        )
        self.vit = ViTModel(
            vit_cfg,
            add_pooling_layer=False,
            use_mask_token=False,
        ).to(device)
        enc_sd = {
            k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")
        }
        missing, unexpected = self.vit.load_state_dict(enc_sd, strict=False)
        # Tolerate missing non-persistent buffers (e.g. position_ids); every
        # real parameter must load, and nothing may be left over.
        param_names = set(dict(self.vit.named_parameters()))
        missing_params = [k for k in missing if k in param_names]
        if missing_params or unexpected:
            raise RuntimeError(
                f"LeWM ViT load mismatch from {weights}: "
                f"missing_params={missing_params}, unexpected={unexpected}"
            )

        # --- BatchNorm projector (vendored MLP: Linear->BN1d->GELU->Linear) ---
        self.projector = nn.Sequential(
            nn.Linear(self._HIDDEN, self._PROJ_HIDDEN),
            nn.BatchNorm1d(self._PROJ_HIDDEN),
            nn.GELU(),
            nn.Linear(self._PROJ_HIDDEN, self._HIDDEN),
        ).to(device)
        proj_sd = {
            k[len("projector.net.") :]: v
            for k, v in sd.items()
            if k.startswith("projector.net.")
        }
        self.projector.load_state_dict(proj_sd, strict=True)

        self.output_dim = self._HIDDEN  # 192

        # ImageNet preprocessing constants (move with the module via buffers).
        self.register_buffer(
            "_mean", torch.tensor(self._IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor(self._IMAGENET_STD, device=device).view(1, 3, 1, 1)
        )

        self._image_key = self._resolve_image_key(config, opt)

        self.requires_grad_(False)
        self.eval()
        loguru.logger.info(
            f"Loaded LeWM ViT-tiny + projector (D={self.output_dim}, "
            f"image_key={self._image_key!r})"
        )

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        # Functionally identical to LeWM's get_img_preprocessor
        # (ToImage(scale->ImageNet-normalize) THEN Resize): reconstruct the
        # [0,1] image from our x*2-1 normalization, ImageNet-normalize, then
        # resize with the SAME op LeWM uses — torchvision v2 bilinear +
        # antialias, int size (shorter-side; identity-square for square inputs).
        # Normalize-before-resize matches LeWM's order (and commutes with the
        # linear resize to ~1e-7 anyway). img: (N, 3, H, W).
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvf

        img = (img + 1.0) / 2.0  # our x*2-1 -> [0,1]
        img = (img - self._mean) / self._std  # ImageNet (x-mean)/std
        return tvf.resize(
            img,
            self._IMG,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def embed(self, goal_obs) -> torch.Tensor:
        flat, B, T = self._flatten_images(goal_obs)
        flat = self._preprocess(flat.to(self._mean.dtype))
        # interpolate_pos_encoding=True matches LeWM's encode() verbatim; an
        # identity at 224 (the trained size) but faithful for any input size.
        cls = self.vit(
            pixel_values=flat, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]  # (B*T, 192)
        emb = self.projector(cls)  # (B*T, 192)
        return emb.reshape(B, T, self.output_dim)


class DINOv2TargetEncoder(TargetEncoder):
    """Frozen DINOv2 (HF ``Dinov2Model``) target features. Loaded from a local
    dir via ``from_pretrained`` (no key surgery; version-robust). The target is
    the **CLS token** ``last_hidden_state[:, 0]`` (DINOv2's global descriptor;
    its ``pooler_output`` is the same CLS). ``output_dim`` = the model's
    ``hidden_size`` (DINOv2-S = 384).

    Preprocessing matches DINOv2's BitImageProcessor exactly: reconstruct [0,1]
    from our ``x*2-1``, resize shortest-edge -> 256 (BICUBIC), center-crop 224,
    ImageNet-normalize. The model's native image size is 518, so we pass
    ``interpolate_pos_encoding=True`` when feeding 224.
    """

    _RESIZE = 256
    _CROP = 224
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, config: Config):
        super().__init__()
        from transformers import Dinov2Model

        opt = config.optimization
        device = opt.device
        path = opt.target_encoder_path
        if path is None:
            raise ValueError(
                "optimization.target_encoder_path must be set for "
                "target_encoder_type='dinov2' (a local HF Dinov2 model dir)."
            )
        path = os.path.expanduser(path)
        loguru.logger.info(f"Loading FROZEN DINOv2 target encoder from {path}")
        self.vit = Dinov2Model.from_pretrained(path).to(device)
        self.output_dim = self.vit.config.hidden_size

        self.register_buffer(
            "_mean", torch.tensor(self._IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor(self._IMAGENET_STD, device=device).view(1, 3, 1, 1)
        )
        self._image_key = self._resolve_image_key(config, opt)

        self.requires_grad_(False)
        self.eval()
        loguru.logger.info(
            f"Loaded DINOv2 (D={self.output_dim}, image_key={self._image_key!r})"
        )

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvf

        img = (img + 1.0) / 2.0  # our x*2-1 -> [0,1]
        img = tvf.resize(  # shortest-edge -> 256, BICUBIC
            img,
            self._RESIZE,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        img = tvf.center_crop(img, self._CROP)  # 224
        return (img - self._mean) / self._std  # ImageNet (x-mean)/std

    def embed(self, goal_obs) -> torch.Tensor:
        flat, B, T = self._flatten_images(goal_obs)
        flat = self._preprocess(flat.to(self._mean.dtype))
        cls = self.vit(
            pixel_values=flat, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]  # (B*T, D)
        return cls.reshape(B, T, self.output_dim)


class SiglipTargetEncoder(TargetEncoder):
    """Frozen SigLIP vision tower (HF ``SiglipVisionModel``) target features.
    Loaded from a local dir via ``from_pretrained``. The target is the
    attention-pooled **``pooler_output``** (SigLIP's canonical image embedding;
    SigLIP has no CLS token). ``output_dim`` = ``hidden_size`` (base = 768).

    Preprocessing matches SiglipImageProcessor exactly: reconstruct [0,1] from
    our ``x*2-1``, resize to 224x224 (BICUBIC, no crop), normalize with
    mean=std=0.5 (-> [-1,1]). Native size is 224, so no pos-emb interpolation.
    """

    _IMG = 224
    _MEAN = (0.5, 0.5, 0.5)
    _STD = (0.5, 0.5, 0.5)

    def __init__(self, config: Config):
        super().__init__()
        from transformers import SiglipVisionModel

        opt = config.optimization
        device = opt.device
        path = opt.target_encoder_path
        if path is None:
            raise ValueError(
                "optimization.target_encoder_path must be set for "
                "target_encoder_type='siglip' (a local HF SigLIP model dir)."
            )
        path = os.path.expanduser(path)
        loguru.logger.info(f"Loading FROZEN SigLIP target encoder from {path}")
        self.vit = SiglipVisionModel.from_pretrained(path).to(device)
        self.output_dim = self.vit.config.hidden_size

        self.register_buffer(
            "_mean", torch.tensor(self._MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor(self._STD, device=device).view(1, 3, 1, 1)
        )
        self._image_key = self._resolve_image_key(config, opt)

        self.requires_grad_(False)
        self.eval()
        loguru.logger.info(
            f"Loaded SigLIP (D={self.output_dim}, image_key={self._image_key!r})"
        )

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvf

        img = (img + 1.0) / 2.0  # our x*2-1 -> [0,1]
        img = tvf.resize(  # 224x224 square, BICUBIC
            img,
            [self._IMG, self._IMG],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return (img - self._mean) / self._std  # (x-0.5)/0.5 -> [-1,1]

    def embed(self, goal_obs) -> torch.Tensor:
        flat, B, T = self._flatten_images(goal_obs)
        flat = self._preprocess(flat.to(self._mean.dtype))
        pooled = self.vit(pixel_values=flat).pooler_output  # (B*T, D)
        return pooled.reshape(B, T, self.output_dim)


class SharedBackboneTargetEncoder(TargetEncoder):
    """FM target = native pooled descriptor of an ALREADY-LOADED frozen backbone.

    Unlike the other ``TargetEncoder``s, this one does NOT ``from_pretrained`` a
    fresh model: it holds a reference to the ``FrozenVisionBackbone`` already
    instantiated by the agent's input ``FrozenViTMultiObsEncoder`` (a frozen
    DINOv2 / SigLIP backbone built with ``with_pooled=True``). The target is that
    backbone's native global descriptor (``backbone.pooled`` — DINOv2 CLS /
    SigLIP ``pooler_output``), so it is byte-for-byte identical to the standalone
    ``DINOv2TargetEncoder`` / ``SiglipTargetEncoder`` (same weights, pooling,
    preprocessing, and target camera) — meaning their precomputed goal z-score
    stats are directly reusable.

    This powers the frozen-ViT frozen-target ablation: the same pretrained
    backbone serves both the input adapter (via patch tokens) AND the FM state
    target (via this pooled descriptor), with only ONE backbone in memory. The
    input-side crop augmentation is deliberately NOT applied here — the target
    is the deterministic full-image descriptor, matching the standalone target
    encoders the goal stats were computed with.

    ``output_dim`` = ``backbone.token_dim`` (DINOv2-S = 384, SigLIP-B = 768).
    """

    def __init__(self, backbone, image_key: str):
        super().__init__()
        # Shared, frozen, eval-always — the SAME object as the input encoder's
        # backbone (no duplicate load). It is already requires_grad_(False).
        self.backbone = backbone
        self.output_dim = int(backbone.token_dim)
        self._image_key = image_key

        self.requires_grad_(False)
        self.eval()
        loguru.logger.info(
            f"SharedBackboneTargetEncoder reusing input backbone "
            f"(D={self.output_dim}, image_key={self._image_key!r})"
        )

    @torch.no_grad()
    def embed(self, goal_obs) -> torch.Tensor:
        flat, B, T = self._flatten_images(goal_obs)
        # backbone.preprocess owns resize/crop + the backbone's normalization;
        # backbone.pooled runs fp32 (no bf16 autocast) so the target matches the
        # standalone *TargetEncoder the goal stats were exported with.
        flat = self.backbone.preprocess(flat)
        # Disable any LoRA adapters for the target pass so the FM target stays the
        # NATIVE frozen pooled descriptor even when the input/condition path is
        # LoRA-adapted — keeping it stationary and matching the precomputed goal
        # z-score stats. A no-op when the backbone has no LoRA.
        with self.backbone.adapters_disabled():
            pooled = self.backbone.pooled(flat)  # (B*T, D)
        return pooled.reshape(B, T, self.output_dim)


def get_target_encoder(config: Config) -> TargetEncoder:
    """Factory: build the frozen ``TargetEncoder`` selected by
    ``config.optimization.target_encoder_type`` (default ``"dp"``).
    """
    t = getattr(config.optimization, "target_encoder_type", "dp")
    if t == "dp":
        return DPTargetEncoder(config)
    if t == "lewm":
        return LeWMTargetEncoder(config)
    if t == "dinov2":
        return DINOv2TargetEncoder(config)
    if t == "siglip":
        return SiglipTargetEncoder(config)
    raise ValueError(
        f"Unknown target_encoder_type={t!r}. "
        "Supported: 'dp', 'lewm', 'dinov2', 'siglip'."
    )
