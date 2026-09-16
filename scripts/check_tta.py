"""Visualize plain vs. test-time-augmented (TTA) predictions on the 6 test images.

Loads the final deliverable checkpoint (outputs/checkpoints/best_model.pt by
default) and, for each of the 6 real test images, renders three panels:
  - the plain single-pass prediction (threshold 0.5)
  - the 8-way dihedral TTA-averaged prediction (threshold 0.5)
  - the per-pixel |plain - TTA| disagreement map, to show where TTA actually
    changes the decision (usually thin boundary pixels, not the roof interior)

Usage: python scripts/check_tta.py [--checkpoint PATH]
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

from roof_seg.config import DEFAULT_CHECKPOINT, IMAGENET_MEAN, IMAGENET_STD, INSPECTION_DIR  # noqa: E402
from roof_seg.dataset import RoofTestDataset  # noqa: E402
from roof_seg.train import get_device, load_checkpoint  # noqa: E402
from roof_seg.tta import DEFAULT_COLOR_VARIANTS, predict_with_tta  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-path", type=Path, default=INSPECTION_DIR / "tta_comparison.png",
    )
    return parser.parse_args()


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnormalized = (image_tensor * std + mean).clamp(0, 1)
    return (unnormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    out = image.copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
    return out


def main() -> int:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    test_ds = RoofTestDataset()
    n = len(test_ds)
    row_labels = ["plain", "TTA (8-way geometric)", "TTA (+color: 12-way)", "disagreement\n(geometric vs +color)"]
    fig, axes = plt.subplots(len(row_labels), n, figsize=(2.6 * n, 4 * len(row_labels)))

    agreement_geo = []
    agreement_geo_vs_color = []
    with torch.no_grad():
        for col in range(n):
            sample = test_ds[col]
            image_tensor = sample["image"]
            image_np = denormalize(image_tensor)

            plain_probs = torch.sigmoid(model(image_tensor.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            plain_mask = plain_probs > 0.5

            tta_geo_probs = predict_with_tta(model, image_tensor, device=device)[0].numpy()
            tta_geo_mask = tta_geo_probs > 0.5

            tta_color_probs = predict_with_tta(
                model, image_tensor, device=device, color_variants=DEFAULT_COLOR_VARIANTS
            )[0].numpy()
            tta_color_mask = tta_color_probs > 0.5

            geo_disagreement = plain_mask != tta_geo_mask
            color_disagreement = tta_geo_mask != tta_color_mask
            agreement_geo.append(1.0 - geo_disagreement.mean())
            agreement_geo_vs_color.append(1.0 - color_disagreement.mean())

            axes[0, col].imshow(overlay(image_np, plain_mask))
            axes[1, col].imshow(overlay(image_np, tta_geo_mask))
            axes[2, col].imshow(overlay(image_np, tta_color_mask))
            axes[3, col].imshow(color_disagreement, cmap="hot", vmin=0, vmax=1)

            for row in range(len(row_labels)):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            axes[0, col].set_title(sample["id"], fontsize=9)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    fig.suptitle("Plain vs. TTA predictions (geometric-only vs. +color variants)", fontsize=12)
    fig.tight_layout()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=120)
    plt.close(fig)

    print(f"\nSaved comparison grid to {args.out_path}")
    print("Per-image pixel agreement (plain vs 8-way geometric TTA):")
    for sample_id, frac in zip((test_ds[i]["id"] for i in range(n)), agreement_geo):
        print(f"  {sample_id}: {frac:.4%} of pixels agree")
    print("Per-image pixel agreement (geometric TTA vs geometric+color TTA):")
    for sample_id, frac in zip((test_ds[i]["id"] for i in range(n)), agreement_geo_vs_color):
        print(f"  {sample_id}: {frac:.4%} of pixels agree")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
