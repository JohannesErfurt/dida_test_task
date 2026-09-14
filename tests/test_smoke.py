"""Smoke tests for project setup: paths, config, and seeding."""

from __future__ import annotations

import random

import numpy as np

from roof_seg.config import (
    IMAGES_DIR,
    IMAGE_SIZE,
    LABELS_DIR,
    RANDOM_SEED,
    TEST_IDS,
    WRONG_LABEL_DIR,
)
from roof_seg.paths import ensure_output_dirs, CHECKPOINTS_DIR, INSPECTION_DIR, PREDICTIONS_DIR
from roof_seg.seed import set_seed


def test_data_dirs_exist():
    assert IMAGES_DIR.is_dir()
    assert LABELS_DIR.is_dir()


def test_image_inventory():
    images = sorted(p.stem for p in IMAGES_DIR.glob("*.png"))
    labels = sorted(p.stem for p in LABELS_DIR.glob("*.png"))
    assert len(images) == 29
    assert len(labels) == 24
    assert set(labels).issubset(set(images))
    assert "278" not in images
    assert "278" not in labels


def test_excluded_pair_kept_for_reference():
    """278 was dropped (duplicate label of 270); its original image is kept in wrong_label/."""
    assert (WRONG_LABEL_DIR / "278.png").is_file()
    assert not (IMAGES_DIR / "278.png").exists()
    assert not (LABELS_DIR / "278.png").exists()


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


def test_wrong_label_image_is_valid():
    """The excluded 278 image is kept intact (256x256 RGBA) even though it's unused."""
    from PIL import Image

    with Image.open(WRONG_LABEL_DIR / "278.png") as im:
        assert im.size == (256, 256)
        assert im.mode == "RGBA"


def test_set_seed_is_reproducible():
    set_seed(RANDOM_SEED)
    a = (random.random(), np.random.rand())

    set_seed(RANDOM_SEED)
    b = (random.random(), np.random.rand())

    assert a[0] == b[0]
    assert a[1] == b[1]
