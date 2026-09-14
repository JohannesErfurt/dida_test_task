"""Materialize a binarized copy of the labels (mask = label > 0) and visualize the diff.

DATA_REPORT.md §5 decided to binarize labels with `label > 0` before training,
folding all antialiased boundary pixels (see DATA_REPORT.md §3.2 discussion)
into the roof class. This script writes that decision out as a real dataset
at data_binarized/labels/ (pixel values strictly 0 or 1 — true binary, not
0/255) and saves a visual comparison so the effect of binarization on the
antialiased boundary is easy to see. The original data/labels/ is untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.config import BINARIZED_LABELS_DIR, IMAGES_DIR, INSPECTION_DIR, LABELS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a binarized (0/1) copy of the labels and visualize the boundary diff.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in data_binarized/labels/ (default: skip them).",
    )
    parser.add_argument(
        "--compare-samples",
        type=int,
        default=4,
        help="Number of labels to include in the visual comparison figure (default: 4).",
    )
    return parser.parse_args()


def binarize_labels(overwrite: bool) -> list[str]:
    BINARIZED_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for src in sorted(LABELS_DIR.glob("*.png")):
        dst = BINARIZED_LABELS_DIR / src.name
        if dst.exists() and not overwrite:
            written.append(src.stem)
            continue
        arr = np.array(Image.open(src))
        binary = (arr > 0).astype(np.uint8)  # strictly {0, 1}, not {0, 255}
        Image.fromarray(binary).save(dst)
        written.append(src.stem)
    return written


def visual_comparison(label_ids: list[str], n_samples: int) -> None:
    rng = np.random.default_rng(0)
    n_samples = min(n_samples, len(label_ids))
    sample_ids = sorted(rng.choice(label_ids, size=n_samples, replace=False).tolist())

    fig, axes = plt.subplots(n_samples, 4, figsize=(11, 2.6 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    print("Boundary pixels reclassified as roof=1 by binarization:")
    for row, image_id in enumerate(sample_ids):
        raw = np.array(Image.open(LABELS_DIR / f"{image_id}.png"))
        binary = np.array(Image.open(BINARIZED_LABELS_DIR / f"{image_id}.png")) * 255
        image = np.array(Image.open(IMAGES_DIR / f"{image_id}.png"))[..., :3]

        # Diff: pixels that were 1-254 (antialiased boundary) and got rounded up to roof=1.
        raw_boundary = (raw > 0) & (raw < 255)
        diff = np.zeros((*raw.shape, 3), dtype=np.uint8)
        diff[...] = 40  # dark gray background
        diff[raw == 255] = (255, 255, 255)  # was already solid roof
        diff[raw_boundary] = (255, 60, 60)  # antialiased boundary pixels, now forced to roof=1

        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(image_id, fontsize=9)
        axes[row, 1].imshow(raw, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].imshow(binary, cmap="gray", vmin=0, vmax=255)
        axes[row, 3].imshow(diff)

        n_boundary = int(raw_boundary.sum())
        axes[row, 3].set_xlabel(f"{n_boundary} px changed", fontsize=8)
        print(f"  {image_id}: {n_boundary} antialiased boundary pixels pulled into roof=1")

        for col, title in enumerate(["image", "raw label\n(0-254-255)", "binarized\n(label > 0)", "boundary pixels\npulled into roof=1"]):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)

    fig.tight_layout()
    out_path = INSPECTION_DIR / "binarization_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved binarization comparison for {sample_ids} to {out_path}")


def main() -> int:
    args = parse_args()
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

    label_ids = binarize_labels(args.overwrite)
    print(f"Wrote {len(label_ids)} binarized labels (values strictly 0/1) to {BINARIZED_LABELS_DIR}")
    print(f"Original labels left untouched at {LABELS_DIR}")

    visual_comparison(label_ids, args.compare_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
