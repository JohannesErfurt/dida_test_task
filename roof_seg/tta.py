"""Test-Time Augmentation (TTA) for inference.

Averages predictions over the 8-element dihedral group (horizontal flip x
90-degree rotation), the same transform family already used and verified as
an *exact, invertible pixel permutation* for training augmentation (see
roof_seg/augmentation.py's mask-safety notes for "flip"/"rotate90"). Because
each view is an exact permutation of the original grid, the model's
per-pixel probability map for a transformed view can be mapped back to
original-image pixel coordinates exactly (no interpolation), then averaged
with the other 7 views' probabilities.

This trades 8x inference compute for a lower-variance probability estimate --
standard practice for small-data segmentation models where a single forward
pass is cheap relative to training.
"""

from __future__ import annotations

from typing import Optional

import torch

DIHEDRAL_TRANSFORMS: tuple[tuple[bool, int], ...] = tuple(
    (flip, k) for flip in (False, True) for k in range(4)
)


def _forward_transform(image: torch.Tensor, flip: bool, k: int) -> torch.Tensor:
    """Apply (optional horizontal flip, then k*90-degree rotation) to a (..., H, W) tensor."""
    if flip:
        image = torch.flip(image, dims=[-1])
    return torch.rot90(image, k, dims=[-2, -1])


def _inverse_transform(image: torch.Tensor, flip: bool, k: int) -> torch.Tensor:
    """Undo `_forward_transform` exactly (inverse rotation, then inverse flip)."""
    image = torch.rot90(image, -k, dims=[-2, -1])
    if flip:
        image = torch.flip(image, dims=[-1])
    return image


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: Optional[torch.device] = None,
    transforms: tuple[tuple[bool, int], ...] = DIHEDRAL_TRANSFORMS,
) -> torch.Tensor:
    """Average sigmoid probability maps over `transforms` dihedral views of `image`.

    `image` is a single (3, H, W) tensor (already normalized, as produced by
    `roof_seg.dataset.image_to_tensor`). Returns a (1, H, W) probability map
    in original-image pixel coordinates, on the CPU.
    """
    model.eval()
    if device is not None:
        model = model.to(device)
        image = image.to(device)

    probs_sum = None
    for flip, k in transforms:
        view = _forward_transform(image, flip, k).unsqueeze(0)
        logits = model(view)
        probs = torch.sigmoid(logits)[0]
        probs = _inverse_transform(probs, flip, k)
        probs_sum = probs if probs_sum is None else probs_sum + probs

    return (probs_sum / len(transforms)).cpu()
