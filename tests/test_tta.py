"""Tests for roof_seg.tta (test-time augmentation)."""

from __future__ import annotations

import torch

from roof_seg.dataset import RoofTestDataset
from roof_seg.model import build_model
from roof_seg.tta import DIHEDRAL_TRANSFORMS, _forward_transform, _inverse_transform, predict_with_tta


def test_dihedral_transforms_has_8_unique_entries():
    assert len(DIHEDRAL_TRANSFORMS) == 8
    assert len(set(DIHEDRAL_TRANSFORMS)) == 8


def test_forward_then_inverse_is_identity():
    x = torch.arange(3 * 5 * 5).reshape(3, 5, 5).float()
    for flip, k in DIHEDRAL_TRANSFORMS:
        transformed = _forward_transform(x, flip, k)
        restored = _inverse_transform(transformed, flip, k)
        assert torch.equal(restored, x), (flip, k)


def test_predict_with_tta_returns_valid_probability_map():
    model = build_model()
    image = RoofTestDataset()[0]["image"]
    probs = predict_with_tta(model, image, transforms=DIHEDRAL_TRANSFORMS[:2])

    assert probs.shape == (1, image.shape[1], image.shape[2])
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_predict_with_tta_matches_manual_average_of_two_views():
    model = build_model()
    model.eval()
    image = RoofTestDataset()[0]["image"]

    with torch.no_grad():
        p_identity = torch.sigmoid(model(image.unsqueeze(0)))[0]
        flipped = _forward_transform(image, flip=True, k=0).unsqueeze(0)
        p_flipped = _inverse_transform(torch.sigmoid(model(flipped))[0], flip=True, k=0)
        expected = (p_identity + p_flipped) / 2

    actual = predict_with_tta(model, image, transforms=((False, 0), (True, 0)))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_single_identity_transform_matches_plain_forward_pass():
    model = build_model()
    model.eval()
    image = RoofTestDataset()[0]["image"]

    with torch.no_grad():
        expected = torch.sigmoid(model(image.unsqueeze(0)))[0]

    actual = predict_with_tta(model, image, transforms=((False, 0),))
    assert torch.allclose(actual, expected, atol=1e-6)
