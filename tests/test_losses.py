"""Tests for roof_seg.losses (SPEC §3.7)."""

from __future__ import annotations

import pytest
import torch

from roof_seg.losses import BCEDiceLoss, DiceLoss, build_loss

BIG_LOGIT = 20.0  # sigmoid(20) ~= 1.0, sigmoid(-20) ~= 0.0


def _logits_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Turn a 0/1 mask into near-saturated logits predicting exactly that mask."""
    return (mask * 2 - 1) * BIG_LOGIT


def test_dice_loss_near_zero_for_perfect_prediction():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    loss = DiceLoss()(_logits_from_mask(target), target)
    assert loss.item() < 0.01


def test_dice_loss_near_one_for_inverted_prediction():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    loss = DiceLoss()(_logits_from_mask(1 - target), target)
    assert loss.item() > 0.9


def test_dice_loss_handles_empty_prediction_and_target():
    """Both empty is the 0/0 case the smoothing term exists for."""
    target = torch.zeros(1, 1, 16, 16)
    loss = DiceLoss()(_logits_from_mask(target), target)
    assert torch.isfinite(loss)
    assert loss.item() < 0.01


def test_dice_loss_averages_per_sample_not_over_pooled_batch():
    """A big-roof sample must not drown out a small-roof one in the same batch."""
    target = torch.zeros(2, 1, 16, 16)
    target[0, :, 0:14, 0:14] = 1.0  # large roof
    target[1, :, 0:2, 0:2] = 1.0    # small roof

    # Predict sample 0 perfectly, sample 1 entirely wrong.
    logits = torch.stack([_logits_from_mask(target[0]), _logits_from_mask(1 - target[1])])

    loss = DiceLoss()(logits, target).item()
    # Per-sample averaging => ~ (0 + 1) / 2. Pooling the batch would be dominated
    # by the large correct roof and land far below 0.4.
    assert 0.4 < loss < 0.6


def test_bce_dice_is_sum_of_parts():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()

    combined = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)(logits, target)
    bce = torch.nn.BCEWithLogitsLoss()(logits, target)
    dice = DiceLoss()(logits, target)

    assert torch.allclose(combined, bce + dice, atol=1e-6)


def test_bce_dice_weights_are_applied():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()

    only_bce = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)(logits, target)
    assert torch.allclose(only_bce, torch.nn.BCEWithLogitsLoss()(logits, target), atol=1e-6)


def test_losses_are_differentiable():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    BCEDiceLoss()(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("name", ["bce_dice", "bce", "dice"])
def test_build_loss_returns_usable_module(name):
    loss_fn = build_loss(name)
    logits = torch.randn(1, 1, 8, 8)
    target = (torch.rand(1, 1, 8, 8) > 0.5).float()
    assert torch.isfinite(loss_fn(logits, target))


def test_build_loss_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("focal")
