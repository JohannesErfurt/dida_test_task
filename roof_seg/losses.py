"""Loss functions for imbalanced binary segmentation (SPEC §3.7).

Roof pixels are 5-28% of each frame (mean 14%, DATA_REPORT.md §6), so the
background class dominates. Two complementary losses are combined:

  - **BCE** (`BCEWithLogitsLoss`) gives smooth, well-conditioned per-pixel
    gradients everywhere, which matters most early in training. On its own
    it is averaged uniformly over pixels, so the ~86% background pixels
    dominate the signal and the model can reach a deceptively low loss while
    being mediocre on the minority roof class.

  - **Soft Dice** measures region overlap rather than per-pixel correctness,
    so a correctly-predicted sea of background contributes almost nothing to
    it. That makes it naturally insensitive to the class imbalance and
    aligned with the metric actually being reported (§3.8). On its own it
    has a harder gradient landscape early on, when predictions are near-zero
    everywhere and the intersection term is ~0.

Summing them gets BCE's stable optimization plus Dice's explicit pressure
toward region overlap. Both operate on **raw logits** (the model returns
logits by design -- see roof_seg/model.py), applying the sigmoid internally,
which is numerically stabler than sigmoid-then-BCE in float32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss on logits: 1 - Dice(sigmoid(logits), target).

    "Soft" because it uses the continuous predicted probabilities rather than
    thresholded ones, keeping the loss differentiable. `smooth` guards the
    degenerate case where prediction and target are both empty (which does
    happen per-batch on tiles with little roof area) -- without it that batch
    would be 0/0.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten per sample so each image contributes its own Dice, then average:
        # pooling the whole batch into one score would let large-roof images
        # dominate small-roof ones.
        probs = probs.flatten(1)
        target = target.flatten(1)

        intersection = (probs * target).sum(dim=1)
        denominator = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """`bce_weight * BCEWithLogits + dice_weight * SoftDice`, both on logits.

    Equal weights (1.0/1.0) by default: the standard combination, and not a
    hyperparameter worth tuning on 24 images unless training actually shows a
    problem (see SPEC §3.4 on over-tuning at this sample size).
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(name: str = "bce_dice") -> nn.Module:
    """Loss factory, so §3.4 CV comparisons can switch loss by name."""
    if name == "bce_dice":
        return BCEDiceLoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    raise ValueError(f"Unknown loss '{name}' (expected one of: bce_dice, bce, dice)")
