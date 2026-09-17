"""Generate roof predictions on test images (SPEC §3.9)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.config import DEFAULT_CHECKPOINT, IMAGENET_MEAN, IMAGENET_STD, PREDICTIONS_DIR, RANDOM_SEED  # noqa: E402
from roof_seg.dataset import RoofTestDataset  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402
from roof_seg.train import get_device, load_checkpoint  # noqa: E402
from roof_seg.tta import DIHEDRAL_TRANSFORMS, predict_with_tta  # noqa: E402


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnormalized = (image_tensor * std + mean).clamp(0, 1)
    return (unnormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    out = image.copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
    return out


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

    device = get_device()
    print(f"Random seed: {args.seed}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Predictions directory: {PREDICTIONS_DIR}")
    print(f"Device: {device}")

    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    test_ds = RoofTestDataset()
    for i in range(len(test_ds)):
        sample = test_ds[i]
        sample_id = sample["id"]
        probs = predict_with_tta(
            model, sample["image"], device=device, transforms=DIHEDRAL_TRANSFORMS, color_variants=None
        )[0].numpy()
        mask_bool = probs > 0.5
        mask = mask_bool.astype(np.uint8) * 255

        out_path = PREDICTIONS_DIR / f"{sample_id}.png"
        Image.fromarray(mask, mode="L").save(out_path)

        image_np = denormalize(sample["image"])
        overlay_np = overlay(image_np, mask_bool)
        overlay_path = PREDICTIONS_DIR / f"{sample_id}_overlay.png"
        Image.fromarray(overlay_np, mode="RGB").save(overlay_path)

        print(f"  {sample_id}: saved prediction to {out_path}, overlay to {overlay_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
