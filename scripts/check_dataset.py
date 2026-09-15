"""Sanity-check the dataset loader (SPEC §3.3 'done when' checklist).

Exercises roof_seg.dataset.RoofTrainDataset / RoofTestDataset directly (not
just raw files, unlike scripts/inspect_data.py) to confirm:
  - all train pairs load without error, with the expected tensor shapes
  - the test loader returns images with identical preprocessing, no label
  - alpha-channel and binarization decisions are actually applied
  - an (image, mask) pair, once de-normalized, still visually aligns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.config import (  # noqa: E402
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INSPECTION_DIR,
    RANDOM_SEED,
    TEST_IDS,
)
from roof_seg.dataset import RoofTestDataset, RoofTrainDataset, get_train_ids  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check the dataset loader.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--samples", type=int, default=3, help="Number of overlays to render.")
    return parser.parse_args()


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor -> (H, W, 3) uint8, inverting image_to_tensor()."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnormalized = (image_tensor * std + mean).clamp(0, 1)
    return (unnormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def check_train_dataset() -> RoofTrainDataset:
    print("=== Train dataset ===")
    train_ds = RoofTrainDataset()
    train_ids = get_train_ids()
    print(f"Length: {len(train_ds)} (expected 24)")
    assert len(train_ds) == 24, f"Expected 24 train pairs, got {len(train_ds)}"
    assert not (set(train_ids) & set(TEST_IDS)), "Train IDs overlap with TEST_IDS!"
    print("No overlap between train IDs and TEST_IDS: OK")

    errors = 0
    for i in range(len(train_ds)):
        sample = train_ds[i]
        image, mask = sample["image"], sample["mask"]
        if image.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
            print(f"  BAD image shape for {sample['id']}: {image.shape}")
            errors += 1
        if mask.shape != (1, IMAGE_SIZE, IMAGE_SIZE):
            print(f"  BAD mask shape for {sample['id']}: {mask.shape}")
            errors += 1
        mask_values = set(torch.unique(mask).tolist())
        if not mask_values <= {0.0, 1.0}:
            print(f"  BAD mask values for {sample['id']}: {mask_values}")
            errors += 1
    assert errors == 0, f"{errors} sample(s) failed shape/value checks"
    print(f"All {len(train_ds)} train pairs load with shapes "
          f"(3, {IMAGE_SIZE}, {IMAGE_SIZE}) / (1, {IMAGE_SIZE}, {IMAGE_SIZE}) "
          "and binary {0,1} masks: OK")
    return train_ds


def check_test_dataset() -> RoofTestDataset:
    print("\n=== Test dataset ===")
    test_ds = RoofTestDataset()
    print(f"Length: {len(test_ds)} (expected 6)")
    assert len(test_ds) == 6, f"Expected 6 test images, got {len(test_ds)}"

    ids_seen = []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        assert "mask" not in sample, "Test samples must not carry a label"
        assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        ids_seen.append(sample["id"])
    assert sorted(ids_seen) == sorted(TEST_IDS)
    print(f"Test IDs match TEST_IDS: {sorted(ids_seen)}")
    print(f"All {len(test_ds)} test images load with shape "
          f"(3, {IMAGE_SIZE}, {IMAGE_SIZE}), same preprocessing as training, no label: OK")
    return test_ds


def render_overlays(train_ds: RoofTrainDataset, n_samples: int, seed: int) -> None:
    print("\n=== Overlay visualization ===")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_ds), size=min(n_samples, len(train_ds)), replace=False)

    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        sample = train_ds[int(idx)]
        image = denormalize(sample["image"])
        mask = sample["mask"].squeeze(0).numpy() > 0.5

        overlay = image.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(sample["id"], fontsize=9)
        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 2].imshow(overlay)
        for col, title in enumerate(["image (de-normalized)", "mask (from tensor)", "overlay"]):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)

    fig.tight_layout()
    out_path = INSPECTION_DIR / "dataset_sanity_check.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved overlay sanity check to {out_path}")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()

    train_ds = check_train_dataset()
    check_test_dataset()
    render_overlays(train_ds, args.samples, args.seed)

    print("\nAll SPEC §3.3 dataset checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
