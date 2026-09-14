"""Train roof segmentation model (SPEC §3.6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.config import CHECKPOINTS_DIR, RANDOM_SEED  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a roof segmentation model on labeled satellite images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()

    print(f"Random seed: {args.seed}")
    print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
    print("Training is not implemented yet (SPEC §3.6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
