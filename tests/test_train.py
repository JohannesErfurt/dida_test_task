"""Tests for roof_seg.train (SPEC §3.7).

Training a ResNet34 U-Net is slow on CPU, so these tests use tiny ID subsets
and 1-2 epochs. They check the *contract* (no leakage, augmentation only on
the training split, checkpoint round-trip, CV glue), not model quality --
quality is what the §3.4 CV harness measures, separately.
"""

from __future__ import annotations

import pytest
import torch

from roof_seg.config import TEST_IDS
from roof_seg.dataset import get_train_ids
from roof_seg.train import (
    TrainConfig,
    evaluate,
    load_checkpoint,
    make_cv_train_fn,
    save_checkpoint,
    train_model,
)

IDS = get_train_ids()
TINY = TrainConfig(epochs=1, batch_size=2)


def test_train_model_runs_and_records_loss_per_epoch():
    config = TrainConfig(epochs=2, batch_size=2)
    result = train_model(train_ids=IDS[:4], config=config, progress=False)

    assert len(result.train_losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.train_losses)
    assert result.val_metrics == []  # no val_ids given
    assert result.train_ids == IDS[:4]


def test_train_model_reports_val_metrics_when_given_val_ids():
    result = train_model(
        train_ids=IDS[:4], val_ids=IDS[4:6], config=TINY, progress=False
    )
    assert len(result.val_metrics) == 1
    assert set(result.final_val_metrics()) == {"iou", "dice"}
    assert 0.0 <= result.final_val_metrics()["iou"] <= 1.0


def test_train_model_rejects_overlapping_train_and_val_ids():
    with pytest.raises(ValueError, match="overlap"):
        train_model(train_ids=IDS[:4], val_ids=IDS[3:6], config=TINY, progress=False)


def test_default_train_ids_exclude_every_test_id():
    """SPEC §3.7: no test image may appear in training."""
    assert not set(get_train_ids()) & set(TEST_IDS)


def test_augmentation_applies_to_training_split_only(monkeypatch):
    """Validation data must be loaded unaugmented even when config.augment is True."""
    import roof_seg.train as train_module

    seen_transforms = []
    original_dataset_cls = train_module.RoofTrainDataset

    def spy(*args, **kwargs):
        seen_transforms.append(kwargs.get("transform"))
        return original_dataset_cls(*args, **kwargs)

    monkeypatch.setattr(train_module, "RoofTrainDataset", spy)
    train_model(
        train_ids=IDS[:4],
        val_ids=IDS[4:6],
        config=TrainConfig(epochs=1, batch_size=2, augment=True),
        progress=False,
    )

    assert seen_transforms[0] is not None, "training split should be augmented"
    assert all(t is None for t in seen_transforms[1:]), "validation split must not be augmented"


def test_augment_false_disables_the_transform(monkeypatch):
    import roof_seg.train as train_module

    seen_transforms = []
    original_dataset_cls = train_module.RoofTrainDataset

    def spy(*args, **kwargs):
        seen_transforms.append(kwargs.get("transform"))
        return original_dataset_cls(*args, **kwargs)

    monkeypatch.setattr(train_module, "RoofTrainDataset", spy)
    train_model(
        train_ids=IDS[:4], config=TrainConfig(epochs=1, batch_size=2, augment=False), progress=False
    )
    assert all(t is None for t in seen_transforms)


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    result = train_model(train_ids=IDS[:4], config=TINY, progress=False)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, result)

    before = evaluate(result.model, IDS[4:6])
    reloaded, checkpoint = load_checkpoint(path)
    after = evaluate(reloaded, IDS[4:6])

    assert before == after
    assert checkpoint["config"]["epochs"] == TINY.epochs
    assert checkpoint["train_ids"] == IDS[:4]


def test_evaluate_returns_metrics_in_valid_range():
    result = train_model(train_ids=IDS[:4], config=TINY, progress=False)
    metrics = evaluate(result.model, IDS[4:8])
    assert set(metrics) == {"iou", "dice"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_make_cv_train_fn_matches_harness_interface():
    """Must be callable as train_fn(train_ids, val_ids) -> {metric: value} for §3.4."""
    train_fn = make_cv_train_fn(TINY)
    metrics = train_fn(IDS[:4], IDS[4:6])
    assert set(metrics) == {"iou", "dice"}


def _patch_evaluate_with_scripted_dice(monkeypatch, dices: list[float]) -> list[float]:
    """Replace roof_seg.train.evaluate with one returning a scripted Dice sequence,
    recording a real weight value alongside each call so tests can check *which*
    epoch's weights `train_model` actually kept.
    """
    import roof_seg.train as train_module

    snapshots: list[float] = []
    calls = {"i": 0}

    def fake_evaluate(model, ids, threshold=0.5, device=None):
        i = calls["i"]
        calls["i"] += 1
        snapshots.append(model.encoder.conv1.weight[0, 0, 0, 0].item())
        return {"iou": dices[i] / 2, "dice": dices[i]}

    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)
    return snapshots


def test_best_checkpoint_is_kept_not_last_epoch(monkeypatch):
    """Dice peaks at epoch 2 then declines; the returned model must be epoch 2's, not epoch 4's."""
    dices = [0.3, 0.6, 0.5, 0.4]
    snapshots = _patch_evaluate_with_scripted_dice(monkeypatch, dices)

    config = TrainConfig(epochs=4, batch_size=2, early_stopping_patience=0)
    result = train_model(train_ids=IDS[:4], val_ids=IDS[4:6], config=config, progress=False)

    assert result.best_epoch == 2
    assert len(result.train_losses) == 4  # ran the full budget (early stopping disabled)
    final_weight = result.model.encoder.conv1.weight[0, 0, 0, 0].item()
    assert final_weight == pytest.approx(snapshots[1]), "model should hold epoch 2's weights"
    assert result.final_val_metrics()["dice"] == pytest.approx(0.6)


def test_early_stopping_triggers_after_patience_epochs_without_improvement(monkeypatch):
    # Improves at epoch 2 (0.6), then 3 non-improving epochs -> stop after epoch 5.
    dices = [0.5, 0.6, 0.55, 0.55, 0.55, 0.9, 0.9]  # trailing values must never be reached
    _patch_evaluate_with_scripted_dice(monkeypatch, dices)

    config = TrainConfig(epochs=10, batch_size=2, early_stopping_patience=3)
    result = train_model(train_ids=IDS[:4], val_ids=IDS[4:6], config=config, progress=False)

    assert result.stopped_early is True
    assert len(result.train_losses) == 5
    assert result.best_epoch == 2


def test_early_stopping_patience_zero_runs_full_budget_but_still_tracks_best(monkeypatch):
    dices = [0.5, 0.6, 0.55, 0.55, 0.55, 0.55]
    _patch_evaluate_with_scripted_dice(monkeypatch, dices)

    config = TrainConfig(epochs=6, batch_size=2, early_stopping_patience=0)
    result = train_model(train_ids=IDS[:4], val_ids=IDS[4:6], config=config, progress=False)

    assert result.stopped_early is False
    assert len(result.train_losses) == 6
    assert result.best_epoch == 2


def test_no_val_ids_means_no_best_epoch_tracking():
    """Final deliverable case: last-epoch weights, no best-checkpoint selection possible."""
    config = TrainConfig(epochs=2, batch_size=2)
    result = train_model(train_ids=IDS[:4], config=config, progress=False)

    assert result.best_epoch is None
    assert result.stopped_early is False
    assert result.final_val_metrics() == {}


def test_checkpoint_records_best_epoch_and_stopped_early(tmp_path, monkeypatch):
    dices = [0.5, 0.6, 0.55, 0.55, 0.55]
    _patch_evaluate_with_scripted_dice(monkeypatch, dices)

    config = TrainConfig(epochs=10, batch_size=2, early_stopping_patience=3)
    result = train_model(train_ids=IDS[:4], val_ids=IDS[4:6], config=config, progress=False)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, result)
    _, checkpoint = load_checkpoint(path)

    assert checkpoint["best_epoch"] == 2
    assert checkpoint["stopped_early"] is True


def test_epoch_callback_fires_once_per_completed_epoch_with_model_in_eval_mode():
    seen_epochs = []
    seen_training_flags = []

    def callback(epoch, model):
        seen_epochs.append(epoch)
        seen_training_flags.append(model.training)

    config = TrainConfig(epochs=3, batch_size=2)
    train_model(train_ids=IDS[:4], config=config, progress=False, epoch_callback=callback)

    assert seen_epochs == [1, 2, 3]
    assert all(flag is False for flag in seen_training_flags), "model should be in eval() mode in the callback"


def test_epoch_callback_stops_firing_after_early_stop(monkeypatch):
    dices = [0.5, 0.6, 0.55, 0.55, 0.55]  # best at epoch 2, stop after epoch 5 (patience=3)
    snapshots = _patch_evaluate_with_scripted_dice(monkeypatch, dices)
    del snapshots  # unused here; only need the scripted dice sequence

    seen_epochs = []
    config = TrainConfig(epochs=10, batch_size=2, early_stopping_patience=3)
    train_model(
        train_ids=IDS[:4], val_ids=IDS[4:6], config=config, progress=False,
        epoch_callback=lambda epoch, model: seen_epochs.append(epoch),
    )

    assert seen_epochs == [1, 2, 3, 4, 5]  # fired for every epoch actually run, then stopped
