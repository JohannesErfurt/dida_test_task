"""Integration tests for scripts/run_cv.py's placeholder baselines on real data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_cv  # noqa: E402
from roof_seg.cross_validation import cross_validate, paired_compare  # noqa: E402


@pytest.mark.parametrize("baseline_fn", [run_cv.spatial_prior_baseline, run_cv.centered_square_baseline])
def test_baseline_returns_valid_metrics(baseline_fn):
    ids = ["121", "241", "270", "272"]
    train_ids, val_ids = ids[:3], ids[3:]
    metrics = baseline_fn(train_ids, val_ids)

    assert set(metrics.keys()) == {"iou", "dice"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_spatial_prior_baseline_via_cross_validate_on_real_data():
    result = cross_validate(run_cv.spatial_prior_baseline, n_folds=6, seed=0)
    assert result.n_folds == 6
    assert len(result.fold_results) == 6
    summary = result.summary()
    # a non-degenerate, class-balance-aware baseline should beat 0 by a clear margin
    # but not come close to a real trained model
    assert 0.05 < summary["iou"]["mean"] < 0.5
    assert 0.05 < summary["dice"]["mean"] < 0.6


def test_paired_compare_between_baselines_uses_identical_folds():
    result_a, result_b = paired_compare(
        run_cv.spatial_prior_baseline, run_cv.centered_square_baseline, n_folds=4, seed=0,
    )
    assert [fr.val_ids for fr in result_a.fold_results] == [fr.val_ids for fr in result_b.fold_results]
