"""Cross-validation harness (SPEC §3.4).

A single fixed train/val split is too high-variance to trust at N=24 (which
handful of images land in validation can swing IoU/Dice more than the effect
of whatever is actually being tested). This module provides a deterministic
fold splitter and a generic CV runner over the 24 training IDs, so config
choices (augmentation, loss, architecture, ...) can be compared via *paired*
runs -- identical folds and seed, only one setting changed -- rather than
guessed at or judged from a single noisy split.

The runner is deliberately decoupled from any specific model: callers pass a
`train_fn(train_ids, val_ids) -> {metric_name: value}` that trains/evaluates
however it likes for that fold. This lets §3.6/§3.7's real segmentation model
plug in here later with zero changes to this module -- see scripts/run_cv.py
for the current placeholder-baseline demonstration (no real model exists
yet), which uses these exact same functions.

Never touches TEST_IDS: folds are only ever carved out of the 24 training IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from roof_seg.config import RANDOM_SEED, TEST_IDS
from roof_seg.dataset import get_train_ids

TrainFn = Callable[[list[str], list[str]], dict[str, float]]


def make_folds(
    ids: Optional[list[str]] = None,
    n_folds: int = 6,
    seed: int = RANDOM_SEED,
) -> list[tuple[list[str], list[str]]]:
    """Deterministically split `ids` into `n_folds` (train_ids, val_ids) pairs.

    Every ID is used as validation exactly once, across folds of nearly equal
    size (as equal as an integer split allows). `n_folds == len(ids)` gives
    leave-one-out CV. Shuffling is seeded, so the same `seed` always produces
    the same folds -- required for paired comparisons to be meaningful.
    """
    ids = sorted(ids if ids is not None else get_train_ids())
    if set(ids) & set(TEST_IDS):
        raise ValueError("Fold IDs must not include any TEST_IDS")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in fold input")
    if not (2 <= n_folds <= len(ids)):
        raise ValueError(f"n_folds must be between 2 and {len(ids)} (got {n_folds})")

    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    base, remainder = divmod(len(shuffled), n_folds)
    chunks: list[list[str]] = []
    start = 0
    for i in range(n_folds):
        size = base + (1 if i < remainder else 0)
        chunks.append(shuffled[start:start + size])
        start += size

    folds = []
    for i in range(n_folds):
        val_ids = sorted(chunks[i])
        train_ids = sorted(
            id_ for j, chunk in enumerate(chunks) if j != i for id_ in chunk
        )
        folds.append((train_ids, val_ids))
    return folds


@dataclass
class FoldResult:
    fold: int
    train_ids: list[str]
    val_ids: list[str]
    metrics: dict[str, float]


@dataclass
class CVResult:
    fold_results: list[FoldResult]
    n_folds: int
    seed: int

    def values(self, metric: str) -> list[float]:
        return [fr.metrics[metric] for fr in self.fold_results]

    def mean(self, metric: str) -> float:
        return float(np.mean(self.values(metric)))

    def std(self, metric: str) -> float:
        return float(np.std(self.values(metric)))

    def summary(self) -> dict[str, dict[str, float]]:
        """{metric_name: {"mean": ..., "std": ...}} for every metric returned by train_fn."""
        if not self.fold_results:
            return {}
        metric_names = self.fold_results[0].metrics.keys()
        return {m: {"mean": self.mean(m), "std": self.std(m)} for m in metric_names}


def cross_validate(
    train_fn: TrainFn,
    ids: Optional[list[str]] = None,
    n_folds: int = 6,
    seed: int = RANDOM_SEED,
) -> CVResult:
    """Run `train_fn` on each of `n_folds` folds and collect the per-fold metrics.

    Reports mean +/- std across folds (via CVResult.summary()) rather than a
    single aggregate number -- individual fold metrics are noisy on their
    own (a handful of validation images each) and shouldn't be read in
    isolation.
    """
    folds = make_folds(ids, n_folds=n_folds, seed=seed)
    results = []
    for i, (train_ids, val_ids) in enumerate(folds):
        metrics = train_fn(train_ids, val_ids)
        results.append(FoldResult(fold=i, train_ids=train_ids, val_ids=val_ids, metrics=metrics))
    return CVResult(fold_results=results, n_folds=n_folds, seed=seed)


def paired_compare(
    train_fn_a: TrainFn,
    train_fn_b: TrainFn,
    ids: Optional[list[str]] = None,
    n_folds: int = 6,
    seed: int = RANDOM_SEED,
) -> tuple[CVResult, CVResult]:
    """Run two configs on *identical* folds and seed.

    This isolates the effect of whatever differs between train_fn_a and
    train_fn_b from fold-composition noise -- a far more sensitive comparison
    than judging two independently-run CVResults against each other.
    """
    result_a = cross_validate(train_fn_a, ids=ids, n_folds=n_folds, seed=seed)
    result_b = cross_validate(train_fn_b, ids=ids, n_folds=n_folds, seed=seed)
    return result_a, result_b
