"""Test-Time Augmentation (TTA) for inference.

Two families of views, combined by simple averaging of sigmoid probability
maps:

- **Geometric (dihedral) views** -- horizontal flip x 90-degree rotation (8
  views). The same transform family already used and verified as an *exact,
  invertible pixel permutation* for training augmentation (see
  roof_seg/augmentation.py's mask-safety notes for "flip"/"rotate90").
  Because each view is an exact permutation of the original grid, the
  model's per-pixel probability map for a transformed view can be mapped
  back to original-image pixel coordinates exactly (no interpolation),
  then averaged with the other views' probabilities.

- **Color views** (optional) -- fixed brightness/gamma perturbations. Unlike
  the geometric views, these never move a pixel, only its value: a
  prediction made on a color-perturbed image is already pixel-aligned with
  the original, so no inverse mapping is needed at all -- it's just another
  view to average in. Perspective and RandomResizedCrop (the other two
  training-augmentation features) are deliberately *not* offered here: both
  require resampling/cropping, so recovering an exact per-pixel inverse for
  their predictions isn't possible without reintroducing interpolation --
  the same reason those two need `cv2.INTER_NEAREST` special-casing to stay
  mask-safe during training at all.

Combining geometric and color views trades more inference compute for a
lower-variance probability estimate -- standard practice for small-data
segmentation models where a single forward pass is cheap relative to
training.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from roof_seg.config import IMAGENET_MEAN, IMAGENET_STD

DIHEDRAL_TRANSFORMS: tuple[tuple[bool, int], ...] = tuple(
    (flip, k) for flip in (False, True) for k in range(4)
)

# Fixed (non-random) brightness/gamma perturbations -- deterministic so TTA
# output is reproducible. Named so they're identifiable in logs/notebooks.
DEFAULT_COLOR_VARIANTS: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor]], ...] = (
    ("brightness_up", lambda x: x * 1.15),
    ("brightness_down", lambda x: x * 0.85),
    ("gamma_up", lambda x: x.clamp(min=1e-6) ** 0.8),
    ("gamma_down", lambda x: x.clamp(min=1e-6) ** 1.2),
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


def _denormalize(image: torch.Tensor) -> torch.Tensor:
    """(3, H, W) ImageNet-normalized -> (3, H, W) in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def _normalize(image_01: torch.Tensor) -> torch.Tensor:
    """(3, H, W) in [0, 1] -> ImageNet-normalized."""
    mean = torch.tensor(IMAGENET_MEAN, device=image_01.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_01.device).view(3, 1, 1)
    return (image_01 - mean) / std


def _apply_color_variant(image: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Apply a color perturbation `fn` (operating on [0,1] pixel values) to a normalized image tensor."""
    image_01 = fn(_denormalize(image)).clamp(0.0, 1.0)
    return _normalize(image_01)


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: Optional[torch.device] = None,
    transforms: tuple[tuple[bool, int], ...] = DIHEDRAL_TRANSFORMS,
    color_variants: Optional[Sequence[tuple[str, Callable[[torch.Tensor], torch.Tensor]]]] = None,
) -> torch.Tensor:
    """Average sigmoid probability maps over dihedral views of `image`, plus optional color views.

    `image` is a single (3, H, W) tensor (already normalized, as produced by
    `roof_seg.dataset.image_to_tensor`). `transforms` controls the geometric
    (flip, rotate90) views, each inverse-mapped back to original pixel
    coordinates before averaging. `color_variants` (e.g. `DEFAULT_COLOR_VARIANTS`)
    adds extra views that perturb brightness/gamma only -- these need no
    inverse mapping, since they never move a pixel. Pass `color_variants=None`
    (the default) to use geometric-only TTA.

    Returns a (1, H, W) probability map in original-image pixel coordinates,
    on the CPU.
    """
    model.eval()
    if device is not None:
        model = model.to(device)
        image = image.to(device)

    probs_sum = None
    n_views = 0

    for flip, k in transforms:
        view = _forward_transform(image, flip, k).unsqueeze(0)
        logits = model(view)
        probs = torch.sigmoid(logits)[0]
        probs = _inverse_transform(probs, flip, k)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        n_views += 1

    for _name, fn in color_variants or ():
        view = _apply_color_variant(image, fn).unsqueeze(0)
        logits = model(view)
        probs = torch.sigmoid(logits)[0]  # color views are pixel-aligned already, no inverse needed
        probs_sum = probs if probs_sum is None else probs_sum + probs
        n_views += 1

    return (probs_sum / n_views).cpu()
