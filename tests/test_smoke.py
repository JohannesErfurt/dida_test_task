"""Smoke tests for project setup: paths, config, and seeding."""

from __future__ import annotations

import random

import numpy as np

from roof_seg.config import (
    IMAGES_DIR,
    IMAGE_SIZE,
    LABELS_DIR,
    ORG_IMAGES_DIR,
    ORG_LABELS_DIR,
    RANDOM_SEED,
    TEST_IDS,
)
from roof_seg.paths import ensure_output_dirs, CHECKPOINTS_DIR, INSPECTION_DIR, PREDICTIONS_DIR
from roof_seg.seed import set_seed


def test_data_dirs_exist():
    assert ORG_IMAGES_DIR.is_dir()
    assert ORG_LABELS_DIR.is_dir()
    assert IMAGES_DIR.is_dir()
    assert LABELS_DIR.is_dir()


def test_org_image_inventory():
    images = sorted(p.stem for p in ORG_IMAGES_DIR.glob("*.png"))
    labels = sorted(p.stem for p in ORG_LABELS_DIR.glob("*.png"))
    assert len(images) == 30
    assert len(labels) == 24
    assert set(labels).issubset(set(images))
    assert "278" not in labels


def test_convert_image_inventory():
    images = sorted(p.stem for p in IMAGES_DIR.glob("*.png"))
    labels = sorted(p.stem for p in LABELS_DIR.glob("*.png"))
    assert len(images) == 30
    assert len(labels) == 24
    assert set(labels).issubset(set(images))
    assert "278" not in labels
    # data_convert/ mirrors data_org/ exactly in which IDs are present
    assert images == sorted(p.stem for p in ORG_IMAGES_DIR.glob("*.png"))
    assert labels == sorted(p.stem for p in ORG_LABELS_DIR.glob("*.png"))


def test_wrong_label_was_deleted_and_image_moved_to_test_set():
    """278's label was wrong (a copy of 270's); it's deleted and 278 is now a test image."""
    assert (ORG_IMAGES_DIR / "278.png").is_file()
    assert not (ORG_LABELS_DIR / "278.png").exists()
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


def test_convert_images_match_org_minus_alpha():
    """data_convert/images_RGB mirrors data_org/images_RGBA with alpha dropped."""
    from PIL import Image

    with Image.open(ORG_IMAGES_DIR / "121.png") as org, Image.open(IMAGES_DIR / "121.png") as conv:
        assert org.mode == "RGBA"
        assert conv.mode == "RGB"
        assert np.array_equal(np.array(org)[..., :3], np.array(conv))


def test_convert_labels_match_org_thresholded_at_128():
    """data_convert/labels_bin_128 equals 255 * (org_label > 128) (DATA_REPORT.md §5, revised)."""
    from PIL import Image

    with Image.open(ORG_LABELS_DIR / "241.png") as org, Image.open(LABELS_DIR / "241.png") as conv:
        org_arr = np.array(org)
        conv_arr = np.array(conv)
        assert set(np.unique(conv_arr).tolist()) <= {0, 255}
        assert np.array_equal(conv_arr, (255 * (org_arr > 128).astype(np.uint8)).astype(np.uint8))
        # boundary pixels (1-254 in the raw label) exist and get split by the threshold
        boundary = (org_arr > 0) & (org_arr < 255)
        assert boundary.any()


def test_set_seed_is_reproducible():
    set_seed(RANDOM_SEED)
    a = (random.random(), np.random.rand())

    set_seed(RANDOM_SEED)
    b = (random.random(), np.random.rand())

    assert a[0] == b[0]
    assert a[1] == b[1]
