"""Generate roof predictions on test images (SPEC §3.8)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.config import DEFAULT_CHECKPOINT, PREDICTIONS_DIR, RANDOM_SEED  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict roof masks for held-out test images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint path (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()

    print(f"Random seed: {args.seed}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Predictions directory: {PREDICTIONS_DIR}")
    print("Inference is not implemented yet (SPEC §3.8).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
