"""Greedy forward-selection search over the 6 augmentation features (SPEC §3.5).

Algorithm: start with the empty feature set as `current`. In each round, for
every feature not yet in `current`, run a **paired** CV comparison (6-fold,
identical folds/seed) of `current + {candidate}` against `current` alone.
Whichever candidate gives the largest mean-IoU improvement is added to
`current` if that improvement is positive; any candidate whose paired IoU
difference is at or below `-NOISE_THRESHOLD` is discarded permanently (a real
sign of harm, not just no help). The search stops when no remaining
candidate improves on `current`, or every feature has been added.

This is the exact procedure requested: round 1 tests each of the 6 features
individually against no augmentation; round 2 takes the best feature from
round 1 and tests it plus each remaining candidate; and so on.

Every comparison is written to `outputs/inspection/augmentation_search.json`
immediately after it completes (not just at the end), so a partial run's
results are never lost and the search is inspectable while still running.

Cost note: this trains 2 models per candidate per round (12 per round at
6-fold), so a full worst-case search (6+5+4+3+2+1 = 21 comparisons) is
expensive. The per-run epoch budget below is deliberately smaller than the
"official" §3.5 augmentation on/off check (25 epochs/patience 10) specifically
to keep search rounds tractable; the winning feature set is worth re-confirming
later at the full budget if the result matters enough to justify the extra
compute (see README.md's Cross-validation harness section on this tradeoff).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.augmentation import ALL_FEATURES  # noqa: E402
from roof_seg.config import INSPECTION_DIR, RANDOM_SEED  # noqa: E402
from roof_seg.cross_validation import paired_compare  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402
from roof_seg.train import TrainConfig, make_cv_train_fn  # noqa: E402

NOISE_THRESHOLD = 0.05  # mean IoU difference below which a result isn't trusted (SPEC §3.4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Per-fold epoch budget for search rounds (smaller than the 25 used for the "
        "confirmed augmentation-on/off check, to keep the search tractable).",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early-stopping patience for search rounds (smaller than the usual 10).",
    )
    parser.add_argument(
        "--log-path", type=Path, default=INSPECTION_DIR / "augmentation_search.json",
    )
    return parser.parse_args()


def config_for(features: frozenset, epochs: int, patience: int, seed: int) -> TrainConfig:
    if not features:
        return TrainConfig(epochs=epochs, early_stopping_patience=patience, seed=seed, augment=False)
    return TrainConfig(
        epochs=epochs, early_stopping_patience=patience, seed=seed,
        augment=True, augment_features=tuple(sorted(features)),
    )


def load_log(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"comparisons": [], "rounds": [], "final_features": None, "seed": None, "n_folds": None}


def save_log(path: Path, log: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def run_comparison(
    current: frozenset, candidate: str, args: argparse.Namespace, log: dict,
) -> dict:
    """Run (or reuse from the log) the paired comparison for adding `candidate` to `current`."""
    key = f"{sorted(current)}+{candidate}"
    for existing in log["comparisons"]:
        if existing["key"] == key:
            print(f"    [cached] {key}: diff_iou={existing['diff_iou']:+.3f}")
            return existing

    with_features = frozenset(current) | {candidate}
    cfg_with = config_for(with_features, args.epochs, args.patience, args.seed)
    cfg_without = config_for(frozenset(current), args.epochs, args.patience, args.seed)

    print(f"    Running {key} ...", flush=True)
    t0 = time.time()
    result_with, result_without = paired_compare(
        make_cv_train_fn(cfg_with), make_cv_train_fn(cfg_without),
        n_folds=args.n_folds, seed=args.seed,
    )
    elapsed = time.time() - t0

    diff_iou = result_with.mean("iou") - result_without.mean("iou")
    diff_dice = result_with.mean("dice") - result_without.mean("dice")
    entry = {
        "key": key,
        "current": sorted(current),
        "candidate": candidate,
        "with_iou_mean": result_with.mean("iou"),
        "with_iou_std": result_with.std("iou"),
        "with_dice_mean": result_with.mean("dice"),
        "without_iou_mean": result_without.mean("iou"),
        "without_iou_std": result_without.std("iou"),
        "without_dice_mean": result_without.mean("dice"),
        "diff_iou": diff_iou,
        "diff_dice": diff_dice,
        "per_fold_diff_iou": [
            a - b for a, b in zip(result_with.values("iou"), result_without.values("iou"))
        ],
        "elapsed_seconds": elapsed,
    }
    log["comparisons"].append(entry)
    save_log(args.log_path, log)
    print(f"    {key}: with={entry['with_iou_mean']:.3f} without={entry['without_iou_mean']:.3f} "
          f"diff_iou={diff_iou:+.3f} diff_dice={diff_dice:+.3f} ({elapsed:.0f}s)")
    return entry


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    log = load_log(args.log_path)
    log["seed"] = args.seed
    log["n_folds"] = args.n_folds

    current: frozenset = frozenset(log["rounds"][-1]["current_after"]) if log["rounds"] else frozenset()
    remaining = set(ALL_FEATURES) - current
    for entry in log.get("discarded", []):
        remaining.discard(entry)
    discarded: list[str] = list(log.get("discarded", []))

    round_num = len(log["rounds"])
    print(f"Starting from current={sorted(current)}, remaining={sorted(remaining)}, "
          f"already-discarded={sorted(discarded)}")

    while remaining:
        round_num += 1
        print(f"\n=== Round {round_num}: current={sorted(current) or '(none)'}, "
              f"testing {sorted(remaining)} ===")

        round_results = {}
        for candidate in sorted(remaining):
            entry = run_comparison(current, candidate, args, log)
            round_results[candidate] = entry

        best_candidate = max(round_results, key=lambda c: round_results[c]["diff_iou"])
        best_diff = round_results[best_candidate]["diff_iou"]

        newly_discarded = [
            c for c, e in round_results.items() if e["diff_iou"] <= -NOISE_THRESHOLD
        ]
        for c in newly_discarded:
            remaining.discard(c)
            discarded.append(c)

        round_summary = {
            "round": round_num,
            "current_before": sorted(current),
            "candidates_tested": sorted(round_results),
            "results": {c: round_results[c]["diff_iou"] for c in round_results},
            "best_candidate": best_candidate,
            "best_diff_iou": best_diff,
            "newly_discarded": newly_discarded,
        }

        if best_diff > 0:
            current = current | {best_candidate}
            remaining.discard(best_candidate)
            round_summary["decision"] = f"added '{best_candidate}' (diff_iou={best_diff:+.3f})"
            print(f"  -> ADDED '{best_candidate}' (diff_iou={best_diff:+.3f}). "
                  f"current={sorted(current)}")
        else:
            round_summary["decision"] = f"stopped: no candidate improved (best was {best_candidate} at {best_diff:+.3f})"
            print(f"  -> STOPPING: no candidate improved current set "
                  f"(best was '{best_candidate}' at {best_diff:+.3f})")
            round_summary["current_after"] = sorted(current)
            log["rounds"].append(round_summary)
            log["discarded"] = discarded
            save_log(args.log_path, log)
            break

        round_summary["current_after"] = sorted(current)
        log["rounds"].append(round_summary)
        log["discarded"] = discarded
        save_log(args.log_path, log)

    log["final_features"] = sorted(current)
    log["discarded"] = discarded
    log["never_selected_no_signal"] = sorted(set(ALL_FEATURES) - current - set(discarded))
    save_log(args.log_path, log)

    print(f"\n=== FINAL FEATURE SET: {sorted(current) or '(none)'} ===")
    print(f"Discarded (real degradation): {sorted(discarded) or '(none)'}")
    print(f"Log written to {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
