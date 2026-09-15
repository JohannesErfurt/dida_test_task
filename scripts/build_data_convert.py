"""Build data/data_convert/ from data/data_org/ (SPEC §3.2/§3.3 preprocessing decisions).

Materializes both preprocessing decisions from DATA_REPORT.md as an actual,
inspectable dataset rather than only a load-time transform:
  - images: RGBA -> RGB (alpha dropped, DATA_REPORT.md §4)
  - labels: grayscale (0-254-255, antialiased boundary) -> binarized with
    `label = 255 * (label > 128)` (DATA_REPORT.md §5, revised decision --
    see the note there for why this supersedes the earlier `label > 0` default)

data/data_org/ (the original RGBA images and raw labels) is left untouched.
This is the single source of truth for data/data_convert/ -- if you need to
regenerate it (e.g. after re-cloning, or to verify it wasn't hand-edited),
run this script with --overwrite.
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
    DATA_ORG_DIR,
    IMAGES_DIR,
    INSPECTION_DIR,
    LABELS_DIR,
    ORG_IMAGES_DIR,
    ORG_LABELS_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/data_convert/ (RGB images + label>128 binarized labels) "
        "from data/data_org/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in data_convert/ (default: skip them).",
    )
    parser.add_argument(
        "--compare-samples",
        type=int,
        default=4,
        help="Number of labels to include in the visual comparison figure (default: 4).",
    )
    return parser.parse_args()


def convert_images(overwrite: bool) -> int:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for src in sorted(ORG_IMAGES_DIR.glob("*.png")):
        dst = IMAGES_DIR / src.name
        if dst.exists() and not overwrite:
            continue
        with Image.open(src) as im:
            im.convert("RGB").save(dst)
        written += 1
    return written


def binarize_labels(overwrite: bool) -> int:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for src in sorted(ORG_LABELS_DIR.glob("*.png")):
        dst = LABELS_DIR / src.name
        if dst.exists() and not overwrite:
            continue
        arr = np.array(Image.open(src))
        binary = (255 * (arr > 128).astype(np.uint8)).astype(np.uint8)
        Image.fromarray(binary).save(dst)
        written += 1
    return written


def visual_comparison(n_samples: int, seed: int = 0) -> None:
    label_ids = sorted(p.stem for p in LABELS_DIR.glob("*.png"))
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(label_ids))
    sample_ids = sorted(rng.choice(label_ids, size=n_samples, replace=False).tolist())

    fig, axes = plt.subplots(n_samples, 4, figsize=(11, 2.6 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, image_id in enumerate(sample_ids):
        org_image = np.array(Image.open(ORG_IMAGES_DIR / f"{image_id}.png"))[..., :3]
        org_label = np.array(Image.open(ORG_LABELS_DIR / f"{image_id}.png"))
        conv_image = np.array(Image.open(IMAGES_DIR / f"{image_id}.png"))
        conv_label = np.array(Image.open(LABELS_DIR / f"{image_id}.png"))

        axes[row, 0].imshow(org_image)
        axes[row, 0].set_ylabel(image_id, fontsize=9)
        axes[row, 1].imshow(conv_image)
        axes[row, 2].imshow(org_label, cmap="gray", vmin=0, vmax=255)
        axes[row, 3].imshow(conv_label, cmap="gray", vmin=0, vmax=255)

        titles = ["org image\n(RGBA)", "convert image\n(RGB)",
                  "org label\n(0-254-255)", "convert label\n(255*(label>128))"]
        for col, title in enumerate(titles):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)

    fig.tight_layout()
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INSPECTION_DIR / "data_convert_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved org-vs-convert comparison for {sample_ids} to {out_path}")


def main() -> int:
    args = parse_args()

    n_images = convert_images(args.overwrite)
    n_labels = binarize_labels(args.overwrite)

    print(f"Wrote {n_images} new RGB images to {IMAGES_DIR} "
          f"({len(list(IMAGES_DIR.glob('*.png')))} total)")
    print(f"Wrote {n_labels} new binarized labels to {LABELS_DIR} "
          f"({len(list(LABELS_DIR.glob('*.png')))} total)")
    print(f"Original dataset left untouched at {DATA_ORG_DIR}")

    visual_comparison(args.compare_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
