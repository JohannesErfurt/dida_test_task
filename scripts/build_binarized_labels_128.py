"""Materialize a second binarized labels dataset using threshold 128 and compare it to >0.

DATA_REPORT.md §5 compared two binarization rules — `label > 0` (used for
scripts/build_binarized_labels.py -> data_binarized/) and the SPEC's other
candidate, `label >= 128`. This script builds the second one, using the
rule as specified: `label = 255 * (label > 128)` (pixel values strictly
{0, 255}), and writes it to data_binarized_128/labels/. It also renders a
three-way visual comparison (raw | >0 rule | >128 rule) so the difference
between the two rules — not just each one vs. the raw label — is visible.
Neither the original data/labels/ nor data_binarized/labels/ is touched.
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

from roof_seg.config import (  # noqa: E402
    BINARIZED_128_LABELS_DIR,
    BINARIZED_LABELS_DIR,
    IMAGES_DIR,
    INSPECTION_DIR,
    LABELS_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a binarized (0/255, threshold 128) copy of the labels and compare it "
        "to the label > 0 binarization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in data_binarized_128/labels/ (default: skip them).",
    )
    parser.add_argument(
        "--compare-samples",
        type=int,
        default=4,
        help="Number of labels to include in the visual comparison figure (default: 4).",
    )
    return parser.parse_args()


def binarize_labels(overwrite: bool) -> list[str]:
    BINARIZED_128_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for src in sorted(LABELS_DIR.glob("*.png")):
        dst = BINARIZED_128_LABELS_DIR / src.name
        if dst.exists() and not overwrite:
            written.append(src.stem)
            continue
        arr = np.array(Image.open(src))
        binary = (255 * (arr > 128).astype(np.uint8)).astype(np.uint8)
        Image.fromarray(binary).save(dst)
        written.append(src.stem)
    return written


def visual_comparison(label_ids: list[str], n_samples: int) -> None:
    rng = np.random.default_rng(0)
    n_samples = min(n_samples, len(label_ids))
    sample_ids = sorted(rng.choice(label_ids, size=n_samples, replace=False).tolist())

    fig, axes = plt.subplots(n_samples, 5, figsize=(13.5, 2.6 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    print("Pixels where the '>0' and '>128' binarization rules disagree:")
    for row, image_id in enumerate(sample_ids):
        image = np.array(Image.open(IMAGES_DIR / f"{image_id}.png"))[..., :3]
        raw = np.array(Image.open(LABELS_DIR / f"{image_id}.png"))
        gt0 = np.array(Image.open(BINARIZED_LABELS_DIR / f"{image_id}.png"))
        gt128 = np.array(Image.open(BINARIZED_128_LABELS_DIR / f"{image_id}.png"))

        # Disagreement: pixels counted as roof by '>0' but not by '>128' (0 < label <= 128).
        disagree = (gt0 == 255) & (gt128 == 0)
        diff = np.zeros((*raw.shape, 3), dtype=np.uint8)
        diff[...] = 40
        diff[gt128 == 255] = (255, 255, 255)  # roof under both rules
        diff[disagree] = (255, 60, 60)  # roof under '>0' only

        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(image_id, fontsize=9)
        axes[row, 1].imshow(raw, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].imshow(gt0, cmap="gray", vmin=0, vmax=255)
        axes[row, 3].imshow(gt128, cmap="gray", vmin=0, vmax=255)
        axes[row, 4].imshow(diff)

        n_disagree = int(disagree.sum())
        axes[row, 4].set_xlabel(f"{n_disagree} px differ", fontsize=8)
        print(f"  {image_id}: {n_disagree} pixels are roof under '>0' but not under '>128'")

        titles = ["image", "raw label\n(0-254-255)", "binarized\n(label > 0)",
                  "binarized\n(label > 128)", "'>0' vs '>128'\ndisagreement"]
        for col, title in enumerate(titles):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)

    fig.tight_layout()
    out_path = INSPECTION_DIR / "binarization_threshold_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved threshold comparison for {sample_ids} to {out_path}")


def main() -> int:
    args = parse_args()
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

    if not BINARIZED_LABELS_DIR.is_dir() or not any(BINARIZED_LABELS_DIR.glob("*.png")):
        print(f"Note: {BINARIZED_LABELS_DIR} not found — run scripts/build_binarized_labels.py "
              "first for the full 3-way comparison. Continuing with just the >128 dataset.")

    label_ids = binarize_labels(args.overwrite)
    print(f"Wrote {len(label_ids)} binarized labels (label = 255 * (label > 128)) to "
          f"{BINARIZED_128_LABELS_DIR}")
    print(f"Original labels left untouched at {LABELS_DIR}")
    print(f"'>0' binarized labels (data_binarized/) also left untouched at {BINARIZED_LABELS_DIR}")

    if BINARIZED_LABELS_DIR.is_dir() and any(BINARIZED_LABELS_DIR.glob("*.png")):
        visual_comparison(label_ids, args.compare_samples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
