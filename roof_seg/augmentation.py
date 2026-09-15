"""Training-time data augmentation (SPEC §3.5).

Geometric transforms are restricted to exact pixel permutations -- flips and
90-degree rotations -- deliberately avoiding arbitrary-angle rotation or any
resize/crop that would require interpolation. `data/data_convert/labels_bin_128/`
was binarized specifically to eliminate the antialiased boundary values
present in the original labels (see DATA_REPORT.md §5); interpolating the
mask during augmentation (bilinear resize, arbitrary-angle rotation) would
silently reintroduce exactly that kind of soft, non-{0,255} boundary,
undoing that decision. Flips and 90-degree rotations need no interpolation
at all -- every output pixel is copied unchanged from some input pixel, so
the mask stays exactly binary after augmentation. This is the reason for the
90-degree choice, not just convenience -- see `SPEC.md §3.5`'s "or arbitrary
rotation -- document choice" note.

Color jitter (brightness/contrast/saturation/hue) only ever applies to the
image -- Albumentations does this automatically for pixel-level transforms,
leaving the mask key untouched.

Built on `roof_seg.dataset.RoofTrainDataset`'s existing `transform` hook:
`transform(image=image, mask=mask) -> {"image": ..., "mask": ...}` is the
native Albumentations `Compose.__call__` signature, so no dataset changes
were needed to wire this in.
"""

from __future__ import annotations

import albumentations as A

# Small hue jitter: aerial imagery brightness/exposure varies a lot between
# captures, but a large hue shift would produce unrealistic (e.g. blue/green)
# roof colors no real photo would show -- so hue gets a much smaller range
# than brightness/contrast/saturation.
DEFAULT_COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)


def build_train_transform(p_geometric: float = 0.5, p_color: float = 0.5) -> A.Compose:
    """Augmentation pipeline for `RoofTrainDataset(transform=...)`.

    Train-only: never pass this to `RoofTestDataset`, nor to the validation
    fold of a cross-validation split -- the SPEC §3.5 "done when" checklist
    requires augmentation to be active during training only.

    Reproducibility: Albumentations 1.3 draws randomness from the global
    `random`/`numpy.random` state, which `roof_seg.seed.set_seed()` already
    seeds -- call it before iterating a DataLoader that uses this transform.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=p_geometric),
            A.VerticalFlip(p=p_geometric),
            A.RandomRotate90(p=p_geometric),
            A.ColorJitter(**DEFAULT_COLOR_JITTER, p=p_color),
        ]
    )
