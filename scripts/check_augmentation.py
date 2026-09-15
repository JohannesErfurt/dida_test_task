"""Sanity-check the augmentation pipeline (SPEC §3.5 'done when' checklist).

Exercises roof_seg.augmentation.build_train_transform() through
RoofTrainDataset directly (not a standalone transform call) to confirm:
  - augmented (image, mask) pairs stay spatially aligned (mask geometry
    follows the image through flips/rotation)
  - masks stay strictly binary {0,1} after every augmentation (no
    interpolation artifacts reintroducing antialiased boundary values)
  - repeated sampling of the same image produces visibly different
    augmented views (the pipeline isn't a no-op)
  - RoofTestDataset has no augmentation hook at all -- augmentation is
    structurally train-only, not just "usually" disabled elsewhere
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

from roof_seg.augmentation import build_train_transform  # noqa: E402
from roof_seg.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, INSPECTION_DIR, RANDOM_SEED  # noqa: E402
from roof_seg.dataset import RoofTestDataset, RoofTrainDataset, get_train_ids  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check the augmentation pipeline.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--views", type=int, default=5, help="Augmented views per sample image.")
    return parser.parse_args()


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnormalized = (image_tensor * std + mean).clamp(0, 1)
    return (unnormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def check_masks_stay_binary(n_samples: int = 24) -> None:
    print("=== Mask binarity under augmentation ===")
    ds = RoofTrainDataset(transform=build_train_transform())
    bad = []
    for i in range(min(n_samples, len(ds))):
        for _ in range(3):  # a few augmented draws per image
            sample = ds[i]
            values = set(torch.unique(sample["mask"]).tolist())
            if not values <= {0.0, 1.0}:
                bad.append((sample["id"], values))
    if bad:
        print(f"  BAD: {len(bad)} augmented samples had non-binary mask values: {bad[:5]}")
    else:
        print(f"  All augmented masks across {min(n_samples, len(ds))} images x 3 draws stayed strictly {{0,1}}: OK")
    assert not bad, "Augmentation reintroduced non-binary mask values"


def check_test_dataset_has_no_transform_hook() -> None:
    print("\n=== Test dataset has no augmentation hook ===")
    import inspect

    sig = inspect.signature(RoofTestDataset.__init__)
    assert "transform" not in sig.parameters, "RoofTestDataset must not accept a transform argument"
    print("  RoofTestDataset.__init__ has no `transform` parameter at all: OK")


def render_augmentation_grid(n_views: int, seed: int) -> None:
    print("\n=== Augmentation grid ===")
    rng = np.random.default_rng(seed)
    sample_ids = sorted(rng.choice(get_train_ids(), size=3, replace=False).tolist())

    plain_ds = RoofTrainDataset(ids=sample_ids)  # no transform: the un-augmented reference
    aug_ds = RoofTrainDataset(ids=sample_ids, transform=build_train_transform())

    n_cols = 1 + n_views
    fig, axes = plt.subplots(len(sample_ids), n_cols, figsize=(2.2 * n_cols, 2.4 * len(sample_ids)))
    if len(sample_ids) == 1:
        axes = axes[np.newaxis, :]

    for row, image_id in enumerate(sample_ids):
        set_seed(seed)  # reset so the "original" column and grid are reproducible run-to-run
        original = plain_ds[row]
        image = denormalize(original["image"])
        mask = original["mask"].squeeze(0).numpy() > 0.5
        overlay = image.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        axes[row, 0].imshow(overlay)
        axes[row, 0].set_ylabel(image_id, fontsize=9)
        if row == 0:
            axes[row, 0].set_title("original", fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for col in range(1, n_cols):
            sample = aug_ds[row]  # re-draws with a fresh random augmentation each call
            image = denormalize(sample["image"])
            mask = sample["mask"].squeeze(0).numpy() > 0.5
            overlay = image.copy()
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            axes[row, col].imshow(overlay)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f"aug {col}", fontsize=9)

    fig.tight_layout()
    out_path = INSPECTION_DIR / "augmentation_grid.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved augmentation grid for {sample_ids} to {out_path}")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()

    check_masks_stay_binary()
    check_test_dataset_has_no_transform_hook()
    render_augmentation_grid(args.views, args.seed)

    print("\nAll SPEC §3.5 augmentation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
