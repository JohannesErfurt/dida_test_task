"""Tests for roof_seg.dataset (SPEC §3.3)."""

from __future__ import annotations

import numpy as np
import torch

from roof_seg.config import IMAGE_SIZE, TEST_IDS
from roof_seg.dataset import (
    RoofTestDataset,
    RoofTrainDataset,
    get_test_ids,
    get_train_ids,
    image_to_tensor,
    load_binary_mask,
    load_image_rgb,
    mask_to_tensor,
)


def test_get_train_ids_excludes_test_ids_and_278():
    train_ids = get_train_ids()
    assert len(train_ids) == 24
    assert not (set(train_ids) & set(TEST_IDS))
    assert "278" not in train_ids


def test_get_test_ids_matches_config():
    assert get_test_ids() == sorted(TEST_IDS)
    assert len(get_test_ids()) == 6


def test_load_image_rgb_drops_alpha():
    from roof_seg.config import IMAGES_DIR

    arr = load_image_rgb(IMAGES_DIR / "121.png")
    assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert arr.dtype == np.uint8


def test_load_binary_mask_is_strictly_binary():
    from roof_seg.config import LABELS_DIR

    mask = load_binary_mask(LABELS_DIR / "241.png")
    assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
    # 241 has known antialiased boundary pixels (1-254) that must be folded into roof=1
    assert mask.sum() > 0


def test_image_to_tensor_shape_and_normalization():
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[..., 0] = 255  # pure red
    tensor = image_to_tensor(image)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    # normalized red channel for a pure-red pixel: (1.0 - mean_r) / std_r
    expected_r = (1.0 - 0.485) / 0.229
    assert torch.allclose(tensor[0, 0, 0], torch.tensor(expected_r), atol=1e-4)


def test_mask_to_tensor_shape():
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    tensor = mask_to_tensor(mask)
    assert tensor.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_train_dataset_length_and_sample_shapes():
    ds = RoofTrainDataset()
    assert len(ds) == 24

    sample = ds[0]
    assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["mask"].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert isinstance(sample["id"], str)


def test_train_dataset_all_samples_load():
    ds = RoofTrainDataset()
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert sample["mask"].shape == (1, IMAGE_SIZE, IMAGE_SIZE)


def test_test_dataset_length_and_no_mask():
    ds = RoofTestDataset()
    assert len(ds) == 6

    sample = ds[0]
    assert "mask" not in sample
    assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    ids = sorted(ds[i]["id"] for i in range(len(ds)))
    assert ids == sorted(TEST_IDS)


def test_train_and_test_datasets_use_identical_preprocessing():
    """Same image_id would produce the same tensor whether loaded via either path."""
    from roof_seg.config import IMAGES_DIR

    image = load_image_rgb(IMAGES_DIR / "278.png")
    tensor_direct = image_to_tensor(image)

    test_ds = RoofTestDataset(ids=["278"])
    tensor_from_dataset = test_ds[0]["image"]

    assert torch.equal(tensor_direct, tensor_from_dataset)
