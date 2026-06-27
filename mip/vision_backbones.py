"""Pluggable FROZEN vision backbones that expose patch tokens.

A ``FrozenVisionBackbone`` maps a single rgb image (in mip's ``x*2-1`` obs
normalization) to its sequence of **patch tokens** ``(N, N_patch, D)``. It owns
its own preprocessing (invert ``x*2-1``, resize/crop, backbone normalization)
and token extraction (which tokens count as patches — DINOv2 drops CLS +
register tokens, SigLIP keeps all tokens). The downstream encoder
(``FrozenViTMultiObsEncoder`` in ``mip/encoders.py``) is backbone-agnostic: it
attentively pools the patch tokens per camera view, concatenates views + low-dim
state, and projects to ``encoder_out_dim``. Swapping backbones is therefore a
single ``get_vision_backbone`` factory call / config string — no change to the
encoder, the pooling heads, the trunk, or the agent.

This is the "image -> patch tokens" seam, deliberately one level BELOW
``mip/target_encoders.py`` (whose ``TargetEncoder.embed`` returns an already
*pooled* vector). The preprocessing here mirrors the matching ``TargetEncoder``
exactly so the two stay consistent.

All backbones are frozen (``requires_grad_(False)``) and eval-always
(``train()`` is a no-op), so they produce deterministic features regardless of
the agent's mode. ``__deepcopy__`` returns ``self``: the agent's EMA copy of the
encoder shares the one frozen backbone instead of duplicating its weights (and
EMA-ing constant params).

Author: Zilai Zeng
"""

from __future__ import annotations

import os

import loguru
import torch
import torch.nn as nn


class FrozenVisionBackbone(nn.Module):
    """Frozen backbone: ``image (x*2-1) -> patch tokens (N, N_patch, D)``.

    Subclasses set ``token_dim`` (= D) and implement ``preprocess`` +
    ``patch_tokens``. The module is frozen and forced to eval.
    """

    token_dim: int

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Map ``(N, 3, H, W)`` in mip's ``x*2-1`` space to the backbone's
        expected input (resize/crop + the backbone's own normalization).
        """
        raise NotImplementedError

    @torch.no_grad()
    def patch_tokens(self, img: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Return the patch tokens ``(N, N_patch, token_dim)`` (no CLS /
        register / pooling tokens). Runs under ``no_grad`` — the backbone is
        frozen, so the trainable pooling head downstream treats these as
        constant inputs (gradient flows into the head, not the backbone).
        """
        raise NotImplementedError

    @torch.no_grad()
    def pooled(self, img: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Return the backbone's NATIVE global descriptor ``(N, token_dim)``
        (DINOv2 CLS token / SigLIP attention-pooled ``pooler_output``).

        This is the fixed, pretraining-defined pooling — NOT the trainable MAP
        adapter the input encoder learns. It exists so the
        ``SharedBackboneTargetEncoder`` can reuse this one frozen backbone to
        produce the FM state-flow target, byte-for-byte matching the standalone
        ``DINOv2TargetEncoder`` / ``SiglipTargetEncoder`` (so their precomputed
        goal z-score stats are reusable). Runs fp32 / no autocast to match those
        target encoders (the bf16 path is input-side only). Subclasses must be
        built with the pooling available (see ``with_pooled`` for SigLIP).
        """
        raise NotImplementedError

    def train(self, mode: bool = True):
        # Frozen backbone stays in eval regardless of the agent's train/eval.
        return super().train(False)

    def __deepcopy__(self, memo):
        # Frozen + shared: the agent's encoder_ema = deepcopy(encoder) references
        # the SAME backbone (no duplicated weights, no redundant EMA of constant
        # params). Safe because the backbone never changes.
        memo[id(self)] = self
        return self


class DINOv2Backbone(FrozenVisionBackbone):
    """Frozen DINOv2 (HF ``Dinov2Model``) patch tokens.

    Preprocessing matches ``DINOv2TargetEncoder`` / DINOv2's BitImageProcessor:
    reconstruct [0,1] from ``x*2-1``, resize shortest-edge -> 256 (BICUBIC),
    center-crop 224, ImageNet-normalize. Native image size is 518, so feed
    ``interpolate_pos_encoding=True`` at 224. Patch tokens = ``last_hidden_state``
    with the leading CLS (and any register tokens) dropped.
    """

    _RESIZE = 256
    _CROP = 224
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, path: str, with_pooled: bool = False):
        super().__init__()
        from transformers import Dinov2Model

        # ``with_pooled`` is accepted for a uniform factory signature but is a
        # no-op for DINOv2: the CLS token is always in last_hidden_state, so
        # ``pooled`` works regardless.
        if path is None:
            raise ValueError(
                "frozen_vit_path must be set for frozen_vit_backbone='dinov2' "
                "(a local HF Dinov2 model dir)."
            )
        path = os.path.expanduser(path)
        loguru.logger.info(f"Loading FROZEN DINOv2 backbone from {path}")
        self.vit = Dinov2Model.from_pretrained(path)
        self.token_dim = self.vit.config.hidden_size
        # Plain dinov2 has no registers; the *_reg variants set this > 0.
        self.num_register_tokens = int(
            getattr(self.vit.config, "num_register_tokens", 0)
        )

        self.register_buffer(
            "_mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer("_std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1))

        self.requires_grad_(False)
        self.eval()
        loguru.logger.info(
            f"DINOv2 backbone ready (token_dim={self.token_dim}, "
            f"register_tokens={self.num_register_tokens})"
        )

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvf

        img = (img + 1.0) / 2.0  # x*2-1 -> [0,1]
        img = tvf.resize(
            img,
            self._RESIZE,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        img = tvf.center_crop(img, self._CROP)
        return (img - self._mean) / self._std  # ImageNet (x-mean)/std

    @torch.no_grad()
    def patch_tokens(self, img: torch.Tensor) -> torch.Tensor:
        hs = self.vit(
            pixel_values=img, interpolate_pos_encoding=True
        ).last_hidden_state  # (N, 1[+reg]+N_patch, D)
        return hs[:, 1 + self.num_register_tokens :]  # drop CLS (+ registers)

    @torch.no_grad()
    def pooled(self, img: torch.Tensor) -> torch.Tensor:
        # CLS token (index 0) — DINOv2's global descriptor (== pooler_output);
        # matches DINOv2TargetEncoder.embed exactly.
        return self.vit(
            pixel_values=img, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]  # (N, D)


class SiglipBackbone(FrozenVisionBackbone):
    """Frozen SigLIP vision tower (HF ``SiglipVisionModel``) patch tokens.

    SigLIP has NO CLS token, so every token in ``last_hidden_state`` is a patch
    token. Loaded with ``vision_use_head=False`` so the pretraining MAP pooler is
    not even instantiated (we attach our own trainable pooling downstream).
    Preprocessing matches ``SiglipTargetEncoder`` / SiglipImageProcessor:
    reconstruct [0,1] from ``x*2-1``, resize to 224x224 (BICUBIC, no crop),
    normalize with mean=std=0.5 (-> [-1,1]).
    """

    _IMG = 224
    _MEAN = (0.5, 0.5, 0.5)
    _STD = (0.5, 0.5, 0.5)

    def __init__(self, path: str, with_pooled: bool = False):
        super().__init__()
        from transformers import SiglipVisionConfig, SiglipVisionModel

        if path is None:
            raise ValueError(
                "frozen_vit_path must be set for frozen_vit_backbone='siglip' "
                "(a local HF SigLIP model dir)."
            )
        path = os.path.expanduser(path)
        loguru.logger.info(f"Loading FROZEN SigLIP backbone from {path}")
        # ``with_pooled`` decides whether the pretraining MAP pooler head is
        # kept. Default (input-encoder use): vision_use_head=False skips it
        # entirely — we only consume last_hidden_state (the patch tokens), and
        # the head.* checkpoint keys load as harmless "unexpected" entries. When
        # the backbone is shared with the frozen-target ablation (with_pooled),
        # keep the head so ``pooled`` can return pooler_output (matching
        # SiglipTargetEncoder). It must be set on the config (from_pretrained
        # doesn't take it as a kwarg).
        self._has_pooled = bool(with_pooled)
        cfg = SiglipVisionConfig.from_pretrained(path)
        cfg.vision_use_head = self._has_pooled
        self.vit = SiglipVisionModel.from_pretrained(path, config=cfg)
        self.token_dim = self.vit.config.hidden_size

        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1))

        self.requires_grad_(False)
        self.eval()
        loguru.logger.info(
            f"SigLIP backbone ready (token_dim={self.token_dim}, "
            f"pooler_head={self._has_pooled})"
        )

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvf

        img = (img + 1.0) / 2.0  # x*2-1 -> [0,1]
        img = tvf.resize(
            img,
            [self._IMG, self._IMG],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return (img - self._mean) / self._std  # (x-0.5)/0.5 -> [-1,1]

    @torch.no_grad()
    def patch_tokens(self, img: torch.Tensor) -> torch.Tensor:
        # No CLS token in SigLIP — all of last_hidden_state are patch tokens.
        return self.vit(pixel_values=img).last_hidden_state  # (N, N_patch, D)

    @torch.no_grad()
    def pooled(self, img: torch.Tensor) -> torch.Tensor:
        # Attention-pooled pooler_output — SigLIP's canonical image embedding
        # (no CLS token); matches SiglipTargetEncoder.embed exactly. Requires
        # the head, which is only loaded when built with_pooled=True.
        if not self._has_pooled:
            raise RuntimeError(
                "SiglipBackbone.pooled() requires the pooler head, but this "
                "backbone was built with with_pooled=False (vision_use_head "
                "dropped). Rebuild it with with_pooled=True (set "
                "network.frozen_vit_expose_pooled=true)."
            )
        return self.vit(pixel_values=img).pooler_output  # (N, D)


def get_vision_backbone(
    name: str, path: str, with_pooled: bool = False
) -> FrozenVisionBackbone:
    """Factory: build the frozen ``FrozenVisionBackbone`` selected by ``name``.

    Mirrors ``mip/target_encoders.get_target_encoder``. Add a new backbone
    family by writing one ``FrozenVisionBackbone`` subclass and a branch here;
    the encoder / pooling / agent are unchanged.

    ``with_pooled`` keeps the backbone's native global-descriptor pooling so the
    shared-backbone frozen-target ablation can call ``pooled`` (no-op for
    DINOv2; keeps the pooler head for SigLIP).
    """
    if name == "dinov2":
        return DINOv2Backbone(path, with_pooled=with_pooled)
    if name == "siglip":
        return SiglipBackbone(path, with_pooled=with_pooled)
    raise ValueError(
        f"Unknown frozen_vit_backbone={name!r}. Supported: 'dinov2', 'siglip'."
    )
