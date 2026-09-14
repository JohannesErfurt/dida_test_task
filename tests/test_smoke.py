"""Smoke tests for project setup: paths, config, and seeding."""

from __future__ import annotations

import random

import numpy as np
import pytest

from roof_seg.config import (
    IMAGES_DIR,
    IMAGE_SIZE,
    LABELS_DIR,
    RANDOM_SEED,
    RGB_IMAGES_DIR,
    RGB_LABELS_DIR,
    TEST_IDS,
)
from roof_seg.paths import ensure_output_dirs, CHECKPOINTS_DIR, INSPECTION_DIR, PREDICTIONS_DIR
from roof_seg.seed import set_seed


def test_data_dirs_exist():
    assert IMAGES_DIR.is_dir()
    assert LABELS_DIR.is_dir()


def test_image_inventory():
    images = sorted(p.stem for p in IMAGES_DIR.glob("*.png"))
    labels = sorted(p.stem for p in LABELS_DIR.glob("*.png"))
    assert len(images) == 30
    assert len(labels) == 24
    assert set(labels).issubset(set(images))
    assert "278" not in labels


def test_wrong_label_was_deleted_and_image_moved_to_test_set():
    """278's label was wrong (a copy of 270's); it's deleted and 278 is now a test image."""
    assert (IMAGES_DIR / "278.png").is_file()
    assert not (LABELS_DIR / "278.png").exists()
    assert "278" in TEST_IDS


def test_test_ids_have_no_labels():
    label_stems = {p.stem for p in LABELS_DIR.glob("*.png")}
    for test_id in TEST_IDS:
        assert test_id not in label_stems


def test_image_size_constant():
    assert IMAGE_SIZE == 256


def test_ensure_output_dirs_creates_directories(tmp_path, monkeypatch):
    import roof_seg.paths as paths_module

    fake_checkpoints = tmp_path / "checkpoints"
    fake_predictions = tmp_path / "predictions"
    fake_inspection = tmp_path / "inspection"
    monkeypatch.setattr(paths_module, "CHECKPOINTS_DIR", fake_checkpoints)
    monkeypatch.setattr(paths_module, "PREDICTIONS_DIR", fake_predictions)
    monkeypatch.setattr(paths_module, "INSPECTION_DIR", fake_inspection)

    paths_module.ensure_output_dirs()

    assert fake_checkpoints.is_dir()
    assert fake_predictions.is_dir()
    assert fake_inspection.is_dir()


def test_rgb_dataset_matches_original_minus_alpha():
    """data_rgb/ (built by scripts/build_rgb_dataset.py) mirrors data/ with alpha dropped."""
    from PIL import Image

    if not RGB_IMAGES_DIR.is_dir():
        pytest.skip("data_rgb/ not built yet — run scripts/build_rgb_dataset.py")

    rgb_images = sorted(p.stem for p in RGB_IMAGES_DIR.glob("*.png"))
    rgb_labels = sorted(p.stem for p in RGB_LABELS_DIR.glob("*.png"))
    orig_images = sorted(p.stem for p in IMAGES_DIR.glob("*.png"))
    orig_labels = sorted(p.stem for p in LABELS_DIR.glob("*.png"))
    assert rgb_images == orig_images
    assert rgb_labels == orig_labels

    with Image.open(IMAGES_DIR / "121.png") as orig, Image.open(RGB_IMAGES_DIR / "121.png") as rgb:
        assert orig.mode == "RGBA"
        assert rgb.mode == "RGB"
        assert np.array_equal(np.array(orig)[..., :3], np.array(rgb))


def test_set_seed_is_reproducible():
    set_seed(RANDOM_SEED)
    a = (random.random(), np.random.rand())

    set_seed(RANDOM_SEED)
    b = (random.random(), np.random.rand())

    assert a[0] == b[0]
    assert a[1] == b[1]
