"""Training-time data augmentation (SPEC §3.5).

Six named, independently toggleable features. Kept as named toggles (not one
monolithic pipeline) so `scripts/search_augmentation.py` can run a greedy
forward-selection search over them via the §3.4 CV harness, rather than the
set being locked in by assumption -- see that script and DATA_REPORT.md /
SPEC.md for the search results.

  - "flip"          -- horizontal + vertical flip
  - "rotate90"       -- 90-degree rotation
  - "color_jitter"   -- brightness/contrast/saturation/hue
  - "rgb_gamma"      -- per-channel RGB shift + gamma
  - "perspective"    -- mild shear + slight perspective warp
  - "resized_crop"   -- random-scale crop, resized back to 256x256

**Mask safety.** Every geometric feature above is safe for a strictly binary
{0,255} mask, but for two different reasons:
  - "flip" and "rotate90" are exact pixel permutations -- no interpolation
    happens at all, so there is nothing to introduce.
  - "perspective" and "resized_crop" *do* resample the image, but
    Albumentations 1.3's `Affine`, `Perspective`, and `RandomResizedCrop` all
    force `cv2.INTER_NEAREST` for the mask target specifically, regardless of
    what interpolation the image uses (verified against this project's
    pinned version: each overrides `apply_to_mask` to substitute
    `cv2.INTER_NEAREST` for the `interpolation` kwarg before calling
    `apply`). Nearest-neighbor never blends, so the mask stays exactly
    {0,255} either way. `data/data_convert/labels_bin_128/` was binarized
    specifically to eliminate antialiased boundary values (DATA_REPORT.md
    §5); bilinear resampling on the mask would have silently reintroduced
    them, which is why this was worth checking rather than assuming.
  - "color_jitter" and "rgb_gamma" are pixel-level transforms and
    Albumentations only ever applies those to the `image` key, never `mask`.

Built on `roof_seg.dataset.RoofTrainDataset`'s existing `transform` hook:
`transform(image=image, mask=mask) -> {"image": ..., "mask": ...}` is the
native Albumentations `Compose.__call__` signature, so no dataset changes
were needed to wire this in.
"""

from __future__ import annotations

from typing import Iterable, Optional

import albumentations as A

from roof_seg.config import IMAGE_SIZE

ALL_FEATURES: tuple[str, ...] = (
    "flip",
    "rotate90",
    "color_jitter",
    "rgb_gamma",
    "perspective",
    "resized_crop",
)

# The set actually used by the deliverable pipeline (scripts/train.py,
# scripts/run_cv.py's non-search paths) when `features=None`. This is
# deliberately *not* ALL_FEATURES -- it's whatever scripts/search_augmentation.py
# most recently concluded is the best-performing subset. Update this constant
# (and cite the search results) once a search run concludes; see
# DATA_REPORT.md / SPEC.md §3.5 for the current rationale.
DEFAULT_FEATURES: tuple[str, ...] = ("flip", "rotate90", "color_jitter")

# Small hue jitter: aerial imagery brightness/exposure varies a lot between
# captures, but a large hue shift would produce unrealistic (e.g. blue/green)
# roof colors no real photo would show -- so hue gets a much smaller range
# than brightness/contrast/saturation.
DEFAULT_COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

# Small per-channel shift (roughly +/-6% of the 0-255 range) and a mild gamma
# range: simulates cross-provider tonal/white-balance drift -- the training
# and test tiles carry different imagery-source watermarks (see
# outputs/inspection/test_set_preview.png), so this isn't a hypothetical
# distribution shift, it's one already present between train and test.
DEFAULT_RGB_SHIFT = dict(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15)
DEFAULT_GAMMA = dict(gamma_limit=(80, 120))

# Shear/perspective magnitudes kept small: real off-nadir aerial capture does
# produce mild perspective skew on rectangular roofs, but extreme values would
# create geometry no real capture angle produces.
DEFAULT_SHEAR = dict(shear=(-8, 8), scale=1.0, translate_percent=0, rotate=0)
DEFAULT_PERSPECTIVE = dict(scale=(0.02, 0.06))

# Crop 70-100% of the frame area before resizing back to 256x256 -- mild zoom
# jitter. Not more aggressive than 0.7: several training tiles already have
# roofs reaching close to the frame edge, and a smaller minimum scale risks
# cropping a roof out entirely rather than just changing its apparent size.
DEFAULT_RESIZED_CROP = dict(scale=(0.7, 1.0), ratio=(0.9, 1.0 / 0.9))


def _build_feature(name: str, p_geometric: float, p_color: float) -> list[A.BasicTransform]:
    if name == "flip":
        return [A.HorizontalFlip(p=p_geometric), A.VerticalFlip(p=p_geometric)]
    if name == "rotate90":
        return [A.RandomRotate90(p=p_geometric)]
    if name == "color_jitter":
        return [A.ColorJitter(**DEFAULT_COLOR_JITTER, p=p_color)]
    if name == "rgb_gamma":
        return [
            A.RGBShift(**DEFAULT_RGB_SHIFT, p=p_color),
            A.RandomGamma(**DEFAULT_GAMMA, p=p_color),
        ]
    if name == "perspective":
        return [
            A.Affine(**DEFAULT_SHEAR, p=p_geometric),
            A.Perspective(**DEFAULT_PERSPECTIVE, p=p_geometric),
        ]
    if name == "resized_crop":
        return [
            A.RandomResizedCrop(
                height=IMAGE_SIZE, width=IMAGE_SIZE, **DEFAULT_RESIZED_CROP, p=p_geometric
            )
        ]
    raise ValueError(f"Unknown augmentation feature '{name}' (expected one of {ALL_FEATURES})")


def build_train_transform(
    features: Optional[Iterable[str]] = None,
    p_geometric: float = 0.5,
    p_color: float = 0.5,
) -> A.Compose:
    """Augmentation pipeline for `RoofTrainDataset(transform=...)`.

    Args:
        features: which named features to include (see module docstring).
            `None` (the default) uses `DEFAULT_FEATURES` -- the currently
            established set, not all six; pass an explicit iterable
            (including `()` for no augmentation at all) to override, as
            `scripts/search_augmentation.py` does.
        p_geometric / p_color: per-call probability applied to every
            geometric / pixel-level transform respectively.

    Train-only: never pass this to `RoofTestDataset`, nor to the validation
    fold of a cross-validation split -- the SPEC §3.5 "done when" checklist
    requires augmentation to be active during training only.

    Reproducibility: Albumentations 1.3 draws randomness from the global
    `random`/`numpy.random` state, which `roof_seg.seed.set_seed()` already
    seeds -- call it before iterating a DataLoader that uses this transform.
    """
    selected = list(DEFAULT_FEATURES if features is None else features)
    unknown = set(selected) - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown augmentation feature(s) {sorted(unknown)}; expected a subset of {ALL_FEATURES}")

    transforms: list[A.BasicTransform] = []
    for name in selected:
        transforms.extend(_build_feature(name, p_geometric, p_color))

    return A.Compose(transforms)
