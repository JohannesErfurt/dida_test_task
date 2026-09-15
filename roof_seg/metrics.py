"""Segmentation metrics: IoU and Dice (used by SPEC §3.4 CV harness and §3.8 evaluation)."""

from __future__ import annotations

import numpy as np


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection-over-Union between two same-shape binary masks.

    Both empty (no predicted or true roof pixels) is defined as 1.0 (perfect
    agreement), not 0/0 -- the conventional definition, and correct here since
    a same-shape comparison never happens on out-of-range inputs.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice / F1 coefficient between two same-shape binary masks. Both empty -> 1.0."""
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2 * intersection) / float(denom)
