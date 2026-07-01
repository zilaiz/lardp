"""Minimal, self-contained LoRA (Low-Rank Adaptation) for ``nn.Linear``.

``LoRALinear`` wraps a *frozen* ``nn.Linear`` and adds a trainable low-rank
correction ``scaling * B @ A`` to its output. It is the standard LoRA
formulation (``B`` zero-initialized so the wrapper is an exact no-op at
construction), hand-rolled instead of pulled from ``peft`` so it composes with
this repo's custom ``FrozenVisionBackbone`` hooks (``__deepcopy__`` returning
self, backbone-stripping ``state_dict``, eval-locked frozen backbone) without
``peft``'s module-tree rewriting fighting them.

The LoRA parameters are named ``lora_A`` / ``lora_B`` so callers can identify
them in a ``state_dict`` by the ``".lora_"`` substring (the encoder keeps these
while stripping the frozen backbone base weights from checkpoints).

``inject_lora`` replaces selected child ``nn.Linear``s in a module tree in
place, preserving their pretrained weights (the wrapped Linear IS the original
object). ``iter_lora_modules`` / ``adapters_disabled`` support toggling the
adapters on/off per forward (used to keep the FM target the native frozen
descriptor while the condition path stays adapted).

Author: Zilai Zeng
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` + trainable low-rank update.

    ``y = base(x) + scaling * (dropout(x) @ Aᵀ) @ Bᵀ``  with
    ``A: [r, in]`` (kaiming-uniform init) and ``B: [out, r]`` (zero init), so at
    construction the update is exactly zero and the layer reproduces ``base``.
    Only ``lora_A`` / ``lora_B`` are trainable; ``base`` is frozen. Set
    ``enabled=False`` to skip the update for a forward (native frozen behavior).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRALinear rank r must be > 0, got {r}")
        self.base = base
        self.base.requires_grad_(False)  # the pretrained weight never moves
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r))
        # Match the reference LoRA / PEFT init: A kaiming-uniform, B zeros ->
        # B@A == 0 at step 0 (wrapper is a no-op until trained).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.enabled = True
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled and not self._merged:
            # (..., in) @ (in, r) -> (..., r) @ (r, out) -> (..., out)
            lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
            out = out + self.scaling * lora
        return out

    @torch.no_grad()
    def merge_(self) -> LoRALinear:
        """Fold the low-rank update into ``base.weight`` in place (idempotent).

        After merging, the update is baked into the base weight and the LoRA
        path is skipped, so inference has zero overhead. Only useful when the
        base weight is actually serialized (the frozen ViT is normally
        reconstructed from ``from_pretrained`` and the LoRA params are restored
        from the checkpoint instead — see the encoder's state_dict handling).
        """
        if not self._merged:
            self.base.weight.data.add_(self.scaling * (self.lora_B @ self.lora_A))
            self._merged = True
        return self

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.4g}, enabled={self.enabled}"


def iter_lora_modules(root: nn.Module) -> Iterator[LoRALinear]:
    """Yield every ``LoRALinear`` in ``root`` (including ``root`` itself)."""
    for m in root.modules():
        if isinstance(m, LoRALinear):
            yield m


def inject_lora(
    root: nn.Module,
    target_names: Iterable[str],
    r: int,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """Replace every child ``nn.Linear`` whose attribute name is in
    ``target_names`` with a ``LoRALinear`` wrapping it, in place.

    The wrapped Linear IS the original module, so its pretrained weights are
    preserved (call this AFTER ``from_pretrained``). Returns the number of
    layers wrapped (0 usually means ``target_names`` don't match this model's
    attribute naming). Idempotent w.r.t. already-wrapped modules: a
    ``LoRALinear`` is not itself an ``nn.Linear``, so it is never re-wrapped.
    """
    targets = set(target_names)
    count = 0
    for module in root.modules():
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, r=r, alpha=alpha, dropout=dropout),
                )
                count += 1
    return count


@contextlib.contextmanager
def adapters_disabled(root: nn.Module):
    """Temporarily disable all ``LoRALinear`` adapters under ``root``.

    Restores each adapter's prior ``enabled`` state on exit. A no-op (empty
    context) when ``root`` contains no ``LoRALinear`` — so callers can wrap a
    forward unconditionally and it costs nothing when LoRA is off.
    """
    mods = list(iter_lora_modules(root))
    prev = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, p in zip(mods, prev, strict=True):
            m.enabled = p
