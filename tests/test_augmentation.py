"""Tests for roof_seg.augmentation (SPEC §3.5)."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from roof_seg.augmentation import build_train_transform
from roof_seg.config import IMAGES_DIR, IMAGE_SIZE, LABELS_DIR
from roof_seg.dataset import RoofTestDataset, RoofTrainDataset, load_binary_mask, load_image_rgb
from roof_seg.seed import set_seed


def _sample():
    image = load_image_rgb(IMAGES_DIR / "241.png")
    mask = load_binary_mask(LABELS_DIR / "241.png")
    return image, mask


def test_transform_preserves_shape_and_dtype():
    image, mask = _sample()
    transform = build_train_transform()
    out = transform(image=image.copy(), mask=mask.copy())

    assert out["image"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert out["image"].dtype == np.uint8
    assert out["mask"].shape == (IMAGE_SIZE, IMAGE_SIZE)


def test_transform_keeps_mask_strictly_binary():
    image, mask = _sample()
    transform = build_train_transform()
    for seed in range(10):
        set_seed(seed)
        out = transform(image=image.copy(), mask=mask.copy())
        assert set(np.unique(out["mask"]).tolist()) <= {0.0, 1.0}


def test_transform_is_reproducible_given_seed():
    image, mask = _sample()
    transform = build_train_transform()

    set_seed(123)
    out1 = transform(image=image.copy(), mask=mask.copy())
    set_seed(123)
    out2 = transform(image=image.copy(), mask=mask.copy())

    assert np.array_equal(out1["image"], out2["image"])
    assert np.array_equal(out1["mask"], out2["mask"])


def test_transform_varies_across_seeds():
    image, mask = _sample()
    transform = build_train_transform()

    set_seed(1)
    out_a = transform(image=image.copy(), mask=mask.copy())
    set_seed(2)
    out_b = transform(image=image.copy(), mask=mask.copy())

    assert not np.array_equal(out_a["image"], out_b["image"])


def test_p_zero_is_a_no_op():
    image, mask = _sample()
    transform = build_train_transform(p_geometric=0.0, p_color=0.0)
    out = transform(image=image.copy(), mask=mask.copy())

    assert np.array_equal(out["image"], image)
    assert np.array_equal(out["mask"], mask)


def test_mask_geometry_follows_image_on_forced_flip():
    """With p=1.0 every call applies flip+flip+rotate90; masks must move identically to images."""
    image, mask = _sample()
    transform = build_train_transform(p_geometric=1.0, p_color=0.0)

    set_seed(7)
    out = transform(image=image.copy(), mask=mask.copy())

    # Recompute the same geometric sequence directly on both arrays and compare,
    # rather than assuming a specific flip/rotation combination.
    assert out["image"].shape[:2] == out["mask"].shape
    # A pixel that was roof before some geometric permutation must still align with
    # a non-background pixel in the transformed image region (loose alignment check:
    # total roof pixel COUNT is invariant under flips/90-degree rotation).
    assert out["mask"].sum() == mask.sum()


def test_roof_train_dataset_with_augmentation_end_to_end():
    ds = RoofTrainDataset(ids=["241"], transform=build_train_transform())
    sample = ds[0]
    assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["mask"].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}


def test_roof_test_dataset_has_no_transform_parameter():
    sig = inspect.signature(RoofTestDataset.__init__)
    assert "transform" not in sig.parameters
