"""Run the cross-validation harness (SPEC §3.4).

Two modes:

  - `--mode model` (default): cross-validates the **real** U-Net (§3.6)
    trained by §3.7, one model per fold. This is the harness doing its
    actual job -- e.g. `--compare augmentation` answers §3.5's deferred
    question (does augmentation help?) as a paired comparison on identical
    folds and seed.

  - `--mode baseline`: the original no-learning placeholders, kept because
    they cost seconds rather than minutes and still serve as a sanity floor
    to compare the trained model against:
      * "spatial-prior" -- the pixels most frequently marked as roof across
        the training fold, sized to that fold's average roof-area fraction;
      * "centered-square" -- a fixed centered block of the same area.
    Neither learns anything; they exist to bound "how good is trivial?".

Nothing in roof_seg/cross_validation.py changed to support the real model --
`train_fn` was always model-agnostic, and roof_seg.train.make_cv_train_fn()
just supplies a different one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.config import IMAGE_SIZE, LABELS_DIR, RANDOM_SEED  # noqa: E402
from roof_seg.cross_validation import CVResult, cross_validate, paired_compare  # noqa: E402
from roof_seg.dataset import load_binary_mask  # noqa: E402
from roof_seg.metrics import dice_score, iou_score  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402
from roof_seg.train import TrainConfig, make_cv_train_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run (or paired-compare) the CV harness.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--n-folds", type=int, default=6,
        help="Number of CV folds (default: 6). Use 24 for leave-one-out.",
    )
    parser.add_argument(
        "--mode", choices=["model", "baseline"], default="model",
        help="'model' cross-validates the real U-Net (slow: trains one model per fold); "
        "'baseline' runs the no-learning placeholders (seconds).",
    )
    parser.add_argument(
        "--epochs", type=int, default=TrainConfig().epochs,
        help="Epochs per fold in --mode model.",
    )
    parser.add_argument(
        "--compare", choices=["augmentation", "baselines"], default=None,
        help="Paired comparison on identical folds/seed: 'augmentation' trains with vs "
        "without §3.5 augmentation (--mode model); 'baselines' compares the two "
        "placeholders (--mode baseline).",
    )
    return parser.parse_args()


def _score_fixed_prediction(pred: np.ndarray, val_ids: list[str]) -> dict[str, float]:
    ious, dices = [], []
    for val_id in val_ids:
        target = load_binary_mask(LABELS_DIR / f"{val_id}.png")
        ious.append(iou_score(pred, target))
        dices.append(dice_score(pred, target))
    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices))}


def spatial_prior_baseline(train_ids: list[str], val_ids: list[str]) -> dict[str, float]:
    masks = np.stack([load_binary_mask(LABELS_DIR / f"{i}.png") for i in train_ids])
    avg_mask = masks.mean(axis=0)
    area_fraction = float(masks.mean())
    k = max(1, round(area_fraction * avg_mask.size))
    threshold = np.partition(avg_mask.ravel(), -k)[-k]
    pred = (avg_mask >= threshold).astype(np.float32)
    return _score_fixed_prediction(pred, val_ids)


def centered_square_baseline(train_ids: list[str], val_ids: list[str]) -> dict[str, float]:
    masks = np.stack([load_binary_mask(LABELS_DIR / f"{i}.png") for i in train_ids])
    area_fraction = float(masks.mean())
    side = int(round(IMAGE_SIZE * area_fraction ** 0.5))
    pred = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    start = (IMAGE_SIZE - side) // 2
    pred[start:start + side, start:start + side] = 1.0
    return _score_fixed_prediction(pred, val_ids)


def print_result(label: str, result: CVResult) -> None:
    print(f"\n=== {label} ({result.n_folds} folds, seed={result.seed}) ===")
    for fr in result.fold_results:
        m = fr.metrics
        print(f"  fold {fr.fold}: val={fr.val_ids} -> "
              f"iou={m['iou']:.3f} dice={m['dice']:.3f}")
    summary = result.summary()
    print(f"  mean IoU  = {summary['iou']['mean']:.3f} +/- {summary['iou']['std']:.3f}")
    print(f"  mean Dice = {summary['dice']['mean']:.3f} +/- {summary['dice']['std']:.3f}")


def print_paired_difference(label_a: str, label_b: str, result_a: CVResult, result_b: CVResult) -> None:
    print(f"\n=== Paired difference ({label_a} minus {label_b}) ===")
    for metric in ("iou", "dice"):
        diffs = [a - b for a, b in zip(result_a.values(metric), result_b.values(metric))]
        wins = sum(d > 0 for d in diffs)
        print(f"  {metric}: mean diff = {np.mean(diffs):+.3f}, "
              f"{label_a} better on {wins}/{len(diffs)} folds, per-fold = "
              f"{[f'{d:+.3f}' for d in diffs]}")
    print("\n  Reminder (SPEC §3.4): at N=24 treat differences below roughly 0.05-0.10 "
          "as noise rather than a real effect.")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "baseline":
        print("Mode: no-learning placeholder baselines (a sanity floor, not a real model).")
        if args.compare == "augmentation":
            raise SystemExit("--compare augmentation requires --mode model")
        if args.compare == "baselines":
            result_a, result_b = paired_compare(
                spatial_prior_baseline, centered_square_baseline,
                n_folds=args.n_folds, seed=args.seed,
            )
            print_result("spatial-prior baseline", result_a)
            print_result("centered-square baseline", result_b)
            print_paired_difference("spatial-prior", "centered-square", result_a, result_b)
        else:
            result = cross_validate(spatial_prior_baseline, n_folds=args.n_folds, seed=args.seed)
            print_result("spatial-prior baseline", result)
        return 0

    print(f"Mode: real U-Net (§3.6) trained per fold for {args.epochs} epochs -- this is slow.")

    if args.compare == "baselines":
        raise SystemExit("--compare baselines requires --mode baseline")

    if args.compare == "augmentation":
        with_aug = make_cv_train_fn(TrainConfig(epochs=args.epochs, augment=True, seed=args.seed))
        without_aug = make_cv_train_fn(TrainConfig(epochs=args.epochs, augment=False, seed=args.seed))
        result_a, result_b = paired_compare(
            with_aug, without_aug, n_folds=args.n_folds, seed=args.seed
        )
        print_result("U-Net WITH augmentation", result_a)
        print_result("U-Net WITHOUT augmentation", result_b)
        print_paired_difference("with-aug", "without-aug", result_a, result_b)
        return 0

    train_fn = make_cv_train_fn(TrainConfig(epochs=args.epochs, seed=args.seed))
    result = cross_validate(train_fn, n_folds=args.n_folds, seed=args.seed)
    print_result(f"U-Net (resnet34), {args.epochs} epochs", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
