"""Exercise the cross-validation harness (SPEC §3.4).

No real segmentation model exists yet (that's §3.6/§3.7), so this script
demonstrates the harness end-to-end with two trivial, no-learning baseline
predictors instead -- both plug into the exact same
roof_seg.cross_validation.cross_validate() / paired_compare() functions the
real model will use later, so nothing here needs to change once it exists:

  - "spatial-prior": predicts the pixels most frequently marked as roof
    across the training fold (a fixed mask, independent of the validation
    image's actual content), sized to match the training fold's average
    roof-area fraction.
  - "centered-square": predicts a single square block centered in the frame,
    sized to match the same area fraction. A different placeholder, used
    only so --compare has two distinguishable configs to demonstrate the
    paired-comparison mechanism on.

Neither is a real baseline worth reporting as "how good is roof
segmentation" -- they exist purely to prove the harness (fold isolation,
metric aggregation, paired comparison) works correctly before the real
model is built.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run (or paired-compare) the CV harness with placeholder baselines.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--n-folds", type=int, default=6,
        help="Number of CV folds (default: 6). Use 24 for leave-one-out.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both baselines on identical folds/seed and report the paired difference "
        "(default: run just the spatial-prior baseline).",
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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    print("NOTE: no real segmentation model exists yet (SPEC §3.6/§3.7). This run uses "
          "placeholder, no-learning baselines purely to exercise the CV harness end-to-end.")

    if not args.compare:
        result = cross_validate(spatial_prior_baseline, n_folds=args.n_folds, seed=args.seed)
        print_result("spatial-prior baseline", result)
        return 0

    result_a, result_b = paired_compare(
        spatial_prior_baseline, centered_square_baseline,
        n_folds=args.n_folds, seed=args.seed,
    )
    print_result("spatial-prior baseline", result_a)
    print_result("centered-square baseline", result_b)

    print(f"\n=== Paired difference (spatial-prior minus centered-square) ===")
    for metric in ("iou", "dice"):
        diffs = [a - b for a, b in zip(result_a.values(metric), result_b.values(metric))]
        print(f"  {metric}: mean diff = {np.mean(diffs):+.3f}, per-fold = "
              f"{[f'{d:+.3f}' for d in diffs]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
