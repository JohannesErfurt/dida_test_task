"""Inspect dataset quality before training (SPEC §3.2).

Performs the checks listed in SPEC.md §3.2: file inventory, dimension
checks, image/label alignment, duplicate-label detection, alpha-channel
audit, label binarization comparison, class balance, and test-set
inspection. Writes plots/tables to outputs/inspection/ and prints a
findings summary that documents the preprocessing decisions used by
later stages (§3.3 onward).
"""

from __future__ import annotations

import argparse
import hashlib
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

from roof_seg.config import IMAGES_DIR, INSPECTION_DIR, LABELS_DIR, RANDOM_SEED, TEST_IDS  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dataset quality and document preprocessing decisions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--overlay-samples",
        type=int,
        default=6,
        help="Number of train pairs to show in the alignment grid (default: 6).",
    )
    return parser.parse_args()


def load_image(image_id: str) -> np.ndarray:
    return np.array(Image.open(IMAGES_DIR / f"{image_id}.png"))


def load_label(image_id: str) -> np.ndarray:
    return np.array(Image.open(LABELS_DIR / f"{image_id}.png"))


def file_inventory() -> tuple[list[str], list[str]]:
    image_ids = sorted(p.stem for p in IMAGES_DIR.glob("*.png"))
    label_ids = sorted(p.stem for p in LABELS_DIR.glob("*.png"))

    print("=== 1. File inventory ===")
    print(f"Images found: {len(image_ids)} (expected 30)")
    print(f"Labels found: {len(label_ids)} (expected 25)")

    orphan_labels = sorted(set(label_ids) - set(image_ids))
    train_only_images = sorted(set(image_ids) - set(label_ids))
    expected_test = sorted(TEST_IDS)

    print(f"Orphan labels (label without image): {orphan_labels or 'none'}")
    print(f"Images without a label (candidate test set): {train_only_images}")
    print(f"Expected test IDs from config: {expected_test}")
    assert train_only_images == expected_test, (
        "Images without labels do not match TEST_IDS in roof_seg/config.py"
    )
    print("Image/label inventory matches expectations: OK")
    return image_ids, label_ids


def check_dimensions(image_ids: list[str], label_ids: list[str]) -> None:
    print("\n=== 2. Dimension checks ===")
    bad_images = []
    for image_id in image_ids:
        with Image.open(IMAGES_DIR / f"{image_id}.png") as im:
            if im.size != (256, 256) or im.mode != "RGBA":
                bad_images.append((image_id, im.size, im.mode))
    bad_labels = []
    for label_id in label_ids:
        with Image.open(LABELS_DIR / f"{label_id}.png") as im:
            if im.size != (256, 256) or im.mode != "L":
                bad_labels.append((label_id, im.size, im.mode))

    print(f"Images with unexpected size/mode: {bad_images or 'none'}")
    print(f"Labels with unexpected size/mode: {bad_labels or 'none'}")
    assert not bad_images and not bad_labels, "Found images/labels with unexpected shape or mode"
    print("All images are 256x256 RGBA; all labels are 256x256 grayscale: OK")


def find_duplicate_labels(label_ids: list[str]) -> dict[str, list[str]]:
    print("\n=== 3. Duplicate / inconsistent labels ===")
    hashes: dict[str, list[str]] = {}
    for label_id in label_ids:
        arr = load_label(label_id)
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        hashes.setdefault(digest, []).append(label_id)

    duplicates = {h: ids for h, ids in hashes.items() if len(ids) > 1}
    if duplicates:
        for ids in duplicates.values():
            print(f"Identical label pixels shared by images: {ids}")
    else:
        print("No duplicate labels found.")
    return duplicates


def alpha_audit(image_ids: list[str]) -> dict[str, float]:
    print("\n=== 4. Alpha-channel audit ===")
    opacity: dict[str, float] = {}
    for image_id in image_ids:
        arr = load_image(image_id)
        alpha = arr[..., 3]
        opacity[image_id] = float((alpha == 255).mean() * 100)

    flagged = {k: v for k, v in opacity.items() if v < 100.0}
    print(f"{len(flagged)}/{len(image_ids)} images contain non-opaque pixels (alpha < 255).")
    for image_id, pct in sorted(flagged.items(), key=lambda kv: kv[1]):
        marker = " (TEST)" if image_id in TEST_IDS else ""
        print(f"  {image_id}{marker}: {pct:.2f}% opaque")
    return opacity


def masked_region_roof_overlap(label_ids: list[str]) -> dict[str, float]:
    """% of roof-label pixels that fall inside non-opaque (masked) regions, per train image."""
    overlap_pct: dict[str, float] = {}
    for label_id in label_ids:
        image = load_image(label_id)
        label = load_label(label_id)
        low_alpha = image[..., 3] < 255
        roof = label > 0
        if roof.any():
            overlap_pct[label_id] = float((roof & low_alpha).sum() / roof.sum() * 100)
        else:
            overlap_pct[label_id] = 0.0
    return overlap_pct


def label_value_distribution(label_ids: list[str]) -> dict:
    print("\n=== 5. Label value distribution & binarization comparison ===")
    total_bg = total_fg = total_edge = 0
    gt0_ratios = []
    ge128_ratios = []
    all_values = []

    for label_id in label_ids:
        arr = load_label(label_id)
        all_values.append(arr.ravel())
        total_bg += int((arr == 0).sum())
        total_fg += int((arr == 255).sum())
        total_edge += int(((arr > 0) & (arr < 255)).sum())
        gt0_ratios.append(float((arr > 0).mean() * 100))
        ge128_ratios.append(float((arr >= 128).mean() * 100))

    total_px = total_bg + total_fg + total_edge
    print(f"Aggregate over {len(label_ids)} labels: background={total_bg}, "
          f"roof-interior(255)={total_fg}, boundary(1-254)={total_edge}")
    print(f"Boundary pixels are {total_edge / total_px * 100:.2f}% of all pixels.")

    diffs = [g - h for g, h in zip(gt0_ratios, ge128_ratios)]
    print("Roof-area % per image using threshold '>0'   -> "
          f"mean={np.mean(gt0_ratios):.2f}%, range=[{min(gt0_ratios):.2f}, {max(gt0_ratios):.2f}]")
    print("Roof-area % per image using threshold '>=128' -> "
          f"mean={np.mean(ge128_ratios):.2f}%, range=[{min(ge128_ratios):.2f}, {max(ge128_ratios):.2f}]")
    print(f"'>0' includes on average {np.mean(diffs):.2f} percentage points more roof area than "
          f"'>=128' (max difference {max(diffs):.2f}pp) by counting all boundary pixels as roof.")

    values = np.concatenate(all_values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=256, range=(0, 255), log=True, color="#3b6ea5")
    ax.axvline(128, color="crimson", linestyle="--", label="threshold = 128")
    ax.set_xlabel("Label pixel value")
    ax.set_ylabel("Pixel count (log scale)")
    ax.set_title("Label value distribution across all 25 training labels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(INSPECTION_DIR / "label_value_histogram.png", dpi=120)
    plt.close(fig)
    print(f"Saved label value histogram to {INSPECTION_DIR / 'label_value_histogram.png'}")

    return {"gt0_ratios": gt0_ratios, "ge128_ratios": ge128_ratios}


def class_balance_table(label_ids: list[str], gt0_ratios: list[float]) -> None:
    print("\n=== 6. Class balance (roof foreground %, using '>0') ===")
    pairs = sorted(zip(label_ids, gt0_ratios), key=lambda kv: kv[1])
    for image_id, ratio in pairs:
        print(f"  {image_id}: {ratio:.2f}%")
    ratios = np.array(gt0_ratios)
    print(f"Mean={ratios.mean():.2f}%  Std={ratios.std():.2f}%  "
          f"Min={ratios.min():.2f}% ({pairs[0][0]})  Max={ratios.max():.2f}% ({pairs[-1][0]})")


def alignment_grid(label_ids: list[str], n_samples: int, seed: int) -> None:
    print("\n=== 7. Image/label alignment grid ===")
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(label_ids))
    sample_ids = sorted(rng.choice(label_ids, size=n_samples, replace=False).tolist())

    fig, axes = plt.subplots(n_samples, 3, figsize=(6, 2 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    for row, image_id in enumerate(sample_ids):
        image = load_image(image_id)[..., :3]
        label = load_label(image_id)
        mask = label > 0

        overlay = image.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(image_id, fontsize=9)
        axes[row, 1].imshow(label, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].imshow(overlay)
        for col, title in enumerate(["image", "label", "overlay"]):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)

    fig.tight_layout()
    out_path = INSPECTION_DIR / "alignment_grid.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved alignment grid for {sample_ids} to {out_path}")


def test_set_inspection(image_ids: list[str], opacity: dict[str, float]) -> None:
    print("\n=== 8. Test-set inspection ===")
    fig, axes = plt.subplots(1, len(TEST_IDS), figsize=(3 * len(TEST_IDS), 3.2))
    for col, test_id in enumerate(TEST_IDS):
        image = load_image(test_id)
        axes[col].imshow(image[..., :3])
        axes[col].set_title(f"{test_id}\n{opacity[test_id]:.1f}% opaque", fontsize=9)
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        print(f"  {test_id}: {opacity[test_id]:.2f}% opaque, "
              f"shape={image.shape}, dtype={image.dtype}")
    fig.tight_layout()
    out_path = INSPECTION_DIR / "test_set_preview.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved test-set preview to {out_path}")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()

    image_ids, label_ids = file_inventory()
    check_dimensions(image_ids, label_ids)
    find_duplicate_labels(label_ids)
    opacity = alpha_audit(image_ids)

    overlap = masked_region_roof_overlap(label_ids)
    flagged_overlap = {k: v for k, v in overlap.items() if v > 0.5}
    print("\n=== Masked-region / roof-label overlap (train only) ===")
    if flagged_overlap:
        for image_id, pct in sorted(flagged_overlap.items(), key=lambda kv: -kv[1]):
            print(f"  {image_id}: {pct:.1f}% of roof pixels fall inside masked (alpha<255) regions")
    else:
        print("  No train image has meaningful roof/mask overlap.")

    dist = label_value_distribution(label_ids)
    class_balance_table(label_ids, dist["gt0_ratios"])
    alignment_grid(label_ids, args.overlay_samples, args.seed)
    test_set_inspection(image_ids, opacity)

    print(f"\nInspection outputs written to: {INSPECTION_DIR}")
    print("See DATA_REPORT.md for the findings summary and preprocessing decisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
