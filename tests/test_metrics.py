"""Tests for roof_seg.metrics."""

from __future__ import annotations

import numpy as np

from roof_seg.metrics import dice_score, iou_score


def test_identical_masks_score_one():
    mask = np.zeros((10, 10))
    mask[2:6, 2:6] = 1
    assert iou_score(mask, mask) == 1.0
    assert dice_score(mask, mask) == 1.0


def test_disjoint_masks_score_zero():
    a = np.zeros((10, 10))
    a[0:4, 0:4] = 1
    b = np.zeros((10, 10))
    b[6:10, 6:10] = 1
    assert iou_score(a, b) == 0.0
    assert dice_score(a, b) == 0.0


def test_both_empty_scores_one():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    assert iou_score(a, b) == 1.0
    assert dice_score(a, b) == 1.0


def test_partial_overlap_known_value():
    # pred: 4x4 block at (0,0); target: 4x4 block at (2,2) -> overlap is 2x2 = 4
    pred = np.zeros((10, 10))
    pred[0:4, 0:4] = 1
    target = np.zeros((10, 10))
    target[2:6, 2:6] = 1

    intersection = 4  # 2x2 overlap
    union = 16 + 16 - intersection  # 28
    assert iou_score(pred, target) == intersection / union

    dice_denom = 16 + 16  # 32
    assert dice_score(pred, target) == 2 * intersection / dice_denom


def test_accepts_float_masks_not_just_bool():
    pred = np.array([[0.0, 1.0], [1.0, 0.0]])
    target = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert iou_score(pred, target) == 1 / 2
    assert dice_score(pred, target) == 2 * 1 / (2 + 1)
