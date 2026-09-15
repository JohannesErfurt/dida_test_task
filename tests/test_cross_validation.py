"""Tests for roof_seg.cross_validation (SPEC §3.4)."""

from __future__ import annotations

import pytest

from roof_seg.config import TEST_IDS
from roof_seg.cross_validation import cross_validate, make_folds, paired_compare
from roof_seg.dataset import get_train_ids

TRAIN_IDS = get_train_ids()  # 24 real IDs, for a couple of integration-style checks


def test_make_folds_covers_every_id_exactly_once():
    ids = [f"id{i}" for i in range(12)]
    folds = make_folds(ids, n_folds=4, seed=0)

    assert len(folds) == 4
    all_val_ids = [v for _, val_ids in folds for v in val_ids]
    assert sorted(all_val_ids) == sorted(ids)  # every id appears as val exactly once
    assert len(all_val_ids) == len(set(all_val_ids))  # no duplicates


def test_make_folds_train_val_disjoint_and_complete():
    ids = [f"id{i}" for i in range(12)]
    for train_ids, val_ids in make_folds(ids, n_folds=4, seed=0):
        assert set(train_ids) & set(val_ids) == set()
        assert set(train_ids) | set(val_ids) == set(ids)


def test_make_folds_is_deterministic():
    ids = [f"id{i}" for i in range(12)]
    folds_a = make_folds(ids, n_folds=4, seed=42)
    folds_b = make_folds(ids, n_folds=4, seed=42)
    assert folds_a == folds_b


def test_make_folds_different_seeds_differ():
    ids = [f"id{i}" for i in range(12)]
    folds_a = make_folds(ids, n_folds=4, seed=1)
    folds_b = make_folds(ids, n_folds=4, seed=2)
    assert folds_a != folds_b


def test_make_folds_rejects_test_id_overlap():
    ids = [f"id{i}" for i in range(10)] + [TEST_IDS[0]]
    with pytest.raises(ValueError, match="TEST_IDS"):
        make_folds(ids, n_folds=3, seed=0)


def test_make_folds_rejects_invalid_n_folds():
    ids = [f"id{i}" for i in range(5)]
    with pytest.raises(ValueError):
        make_folds(ids, n_folds=1, seed=0)  # too few
    with pytest.raises(ValueError):
        make_folds(ids, n_folds=6, seed=0)  # too many (> len(ids))


def test_make_folds_nearly_equal_sizes():
    ids = [f"id{i}" for i in range(10)]
    folds = make_folds(ids, n_folds=3, seed=0)  # 10 -> sizes 4,3,3
    sizes = sorted(len(val_ids) for _, val_ids in folds)
    assert sizes == [3, 3, 4]


def test_loocv_gives_one_val_id_per_fold():
    ids = [f"id{i}" for i in range(8)]
    folds = make_folds(ids, n_folds=8, seed=0)  # LOOCV
    assert len(folds) == 8
    for train_ids, val_ids in folds:
        assert len(val_ids) == 1
        assert len(train_ids) == 7


def test_default_ids_are_the_24_real_train_ids():
    folds = make_folds(n_folds=6, seed=0)
    all_val_ids = sorted(v for _, val_ids in folds for v in val_ids)
    assert all_val_ids == sorted(TRAIN_IDS)
    assert not (set(all_val_ids) & set(TEST_IDS))


def test_cross_validate_aggregates_fold_metrics():
    ids = [f"id{i}" for i in range(6)]

    def fake_train_fn(train_ids, val_ids):
        # deterministic "metric" derived from val set so we can check aggregation exactly
        return {"score": float(len(val_ids))}

    result = cross_validate(fake_train_fn, ids=ids, n_folds=3, seed=0)
    assert result.n_folds == 3
    assert len(result.fold_results) == 3
    # 6 ids into 3 folds -> 2 val ids per fold -> score always 2.0
    assert result.mean("score") == 2.0
    assert result.std("score") == 0.0
    assert result.summary() == {"score": {"mean": 2.0, "std": 0.0}}


def test_cross_validate_fold_results_track_correct_val_ids():
    ids = [f"id{i}" for i in range(6)]
    expected_folds = make_folds(ids, n_folds=3, seed=7)

    def fake_train_fn(train_ids, val_ids):
        return {"n_train": float(len(train_ids))}

    result = cross_validate(fake_train_fn, ids=ids, n_folds=3, seed=7)
    for fold_result, (exp_train, exp_val) in zip(result.fold_results, expected_folds):
        assert fold_result.train_ids == exp_train
        assert fold_result.val_ids == exp_val


def test_paired_compare_uses_identical_folds_for_both_configs():
    ids = [f"id{i}" for i in range(8)]
    seen_val_ids_a = []
    seen_val_ids_b = []

    def train_fn_a(train_ids, val_ids):
        seen_val_ids_a.append(tuple(val_ids))
        return {"score": 1.0}

    def train_fn_b(train_ids, val_ids):
        seen_val_ids_b.append(tuple(val_ids))
        return {"score": 2.0}

    result_a, result_b = paired_compare(train_fn_a, train_fn_b, ids=ids, n_folds=4, seed=3)

    assert seen_val_ids_a == seen_val_ids_b  # exact same fold assignment for both
    assert [fr.val_ids for fr in result_a.fold_results] == [fr.val_ids for fr in result_b.fold_results]
    assert result_a.mean("score") == 1.0
    assert result_b.mean("score") == 2.0
