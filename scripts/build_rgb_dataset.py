"""Materialize an RGB-only copy of the dataset (SPEC §3.2/§3.3 alpha decision).

DATA_REPORT.md §4 decided to drop the alpha channel for training/inference
(the non-opaque pixels are already-black censorship boxes with negligible
overlap with roof labels). This script writes that decision out as a real,
inspectable dataset at data_rgb/, converting every image from RGBA to RGB
and copying labels through unchanged (they have no alpha channel). The
original data/ directory is left untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.config import IMAGES_DIR, LABELS_DIR, RGB_IMAGES_DIR, RGB_LABELS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an RGB-only (alpha-dropped) copy of the dataset at data_rgb/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in data_rgb/ (default: skip them).",
    )
    return parser.parse_args()


def convert_images(overwrite: bool) -> int:
    RGB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for src in sorted(IMAGES_DIR.glob("*.png")):
        dst = RGB_IMAGES_DIR / src.name
        if dst.exists() and not overwrite:
            continue
        with Image.open(src) as im:
            im.convert("RGB").save(dst)
        written += 1
    return written


def copy_labels(overwrite: bool) -> int:
    RGB_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for src in sorted(LABELS_DIR.glob("*.png")):
        dst = RGB_LABELS_DIR / src.name
        if dst.exists() and not overwrite:
            continue
        with Image.open(src) as im:
            im.save(dst)
        written += 1
    return written


def main() -> int:
    args = parse_args()

    n_images = convert_images(args.overwrite)
    n_labels = copy_labels(args.overwrite)

    print(f"Wrote {n_images} RGB images to {RGB_IMAGES_DIR}")
    print(f"Wrote {n_labels} labels to {RGB_LABELS_DIR} (unchanged; grayscale has no alpha)")
    print(f"Original dataset left untouched at {IMAGES_DIR.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
