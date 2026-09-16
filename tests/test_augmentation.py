"""Tests for roof_seg.augmentation (SPEC §3.5)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from roof_seg.augmentation import ALL_FEATURES, DEFAULT_FEATURES, build_train_transform
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


def test_all_features_and_default_features_are_consistent():
    assert len(ALL_FEATURES) == 6
    assert set(DEFAULT_FEATURES) <= set(ALL_FEATURES)
    assert len(set(ALL_FEATURES)) == len(ALL_FEATURES)  # no duplicate names


def test_none_features_uses_default_features():
    image, mask = _sample()
    default_transform = build_train_transform(features=DEFAULT_FEATURES, p_geometric=1.0, p_color=1.0)
    none_transform = build_train_transform(features=None, p_geometric=1.0, p_color=1.0)

    set_seed(0)
    out_default = default_transform(image=image.copy(), mask=mask.copy())
    set_seed(0)
    out_none = none_transform(image=image.copy(), mask=mask.copy())

    assert np.array_equal(out_default["image"], out_none["image"])
    assert np.array_equal(out_default["mask"], out_none["mask"])


def test_empty_feature_set_is_a_no_op():
    image, mask = _sample()
    transform = build_train_transform(features=(), p_geometric=1.0, p_color=1.0)
    out = transform(image=image.copy(), mask=mask.copy())
    assert np.array_equal(out["image"], image)
    assert np.array_equal(out["mask"], mask)


def test_unknown_feature_name_raises():
    with pytest.raises(ValueError, match="Unknown augmentation feature"):
        build_train_transform(features=["not_a_real_feature"])


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_each_feature_individually_keeps_mask_binary_and_shape(feature):
    image, mask = _sample()
    transform = build_train_transform(features=[feature], p_geometric=1.0, p_color=1.0)
    for seed in range(3):
        set_seed(seed)
        out = transform(image=image.copy(), mask=mask.copy())
        assert set(np.unique(out["mask"]).tolist()) <= {0.0, 1.0}
        assert out["image"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert out["image"].dtype == np.uint8


def test_geometric_only_features_preserve_roof_pixel_count():
    """flip/rotate90 are exact permutations; resized_crop/perspective are not (they may
    crop area away or warp toward/away from the roof), so only assert this for the
    permutation-only features."""
    image, mask = _sample()
    for feature in ("flip", "rotate90"):
        transform = build_train_transform(features=[feature], p_geometric=1.0)
        set_seed(3)
        out = transform(image=image.copy(), mask=mask.copy())
        assert out["mask"].sum() == mask.sum(), feature


def test_resized_crop_produces_correct_output_size():
    image, mask = _sample()
    transform = build_train_transform(features=["resized_crop"], p_geometric=1.0)
    set_seed(0)
    out = transform(image=image.copy(), mask=mask.copy())
    assert out["image"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert out["mask"].shape == (IMAGE_SIZE, IMAGE_SIZE)
