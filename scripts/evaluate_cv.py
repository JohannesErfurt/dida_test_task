"""Internal CV evaluation of the final chosen config (SPEC §3.8).

Runs the config actually used for the final deliverable checkpoint (§3.7:
augment=True, features=flip/rotate90/color_jitter, best-Dice checkpoint per
fold) across the same 6-fold split used throughout §3.4/3.5 (seed 42), and:

  1. Reports the official §3.8 evaluation metric: mean +/- std IoU/Dice across
     folds, each of the 24 training images validated exactly once by a model
     that never trained on it.
  2. Renders an image | ground truth | prediction grid for all 24 training
     images (each from its own fold's held-out prediction), satisfying the
     "input | ground truth | prediction on a CV validation sample" requirement.

Checkpoints are cached under outputs/checkpoints/cv_eval/ so the run is
resumable (mirrors scripts/compare_augmentation_ensemble.py's approach).

Usage: python scripts/evaluate_cv.py
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

from roof_seg.config import CHECKPOINTS_DIR, IMAGENET_MEAN, IMAGENET_STD, INSPECTION_DIR, RANDOM_SEED  # noqa: E402
from roof_seg.cross_validation import make_folds  # noqa: E402
from roof_seg.dataset import RoofTrainDataset  # noqa: E402
from roof_seg.metrics import dice_score, iou_score  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402
from roof_seg.train import TrainConfig, get_device, load_checkpoint, save_checkpoint, train_model  # noqa: E402

FEATURES = ("flip", "rotate90", "color_jitter")  # final deliverable's augmentation set (§3.5/§3.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=CHECKPOINTS_DIR / "cv_eval")
    parser.add_argument(
        "--grid-path", type=Path, default=INSPECTION_DIR / "cv_validation_grid.png",
    )
    return parser.parse_args()


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnormalized = (image_tensor * std + mean).clamp(0, 1)
    return (unnormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def render_validation_grid(samples: list[dict], out_path: Path) -> None:
    """samples: list of {"id", "image" (3,H,W tensor), "gt" (H,W bool), "pred" (H,W bool)}."""
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(6, 2.2 * n))
    col_titles = ["image", "ground truth", "prediction"]

    for row, sample in enumerate(samples):
        image_np = denormalize(sample["image"])
        axes[row, 0].imshow(image_np)
        axes[row, 1].imshow(sample["gt"], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(sample["pred"], cmap="gray", vmin=0, vmax=1)
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(sample["id"], fontsize=9)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    fig.suptitle("CV validation predictions: image | ground truth | prediction (all 24, held-out per fold)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    fold_splits = make_folds(n_folds=args.n_folds, seed=args.seed)
    print(f"Using {len(fold_splits)} folds (seed={args.seed}).")

    config = TrainConfig(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        seed=args.seed,
        augment=True,
        augment_features=FEATURES,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_metrics = []
    val_samples = []

    for i, (train_ids, val_ids) in enumerate(fold_splits):
        ckpt_path = out_dir / f"fold{i}.pt"
        if ckpt_path.exists():
            model, ck = load_checkpoint(ckpt_path, device=device)
            best_epoch = ck["best_epoch"]
            metrics = ck["val_metrics"][best_epoch - 1] if best_epoch else ck["val_metrics"][-1]
            print(f"  [cached] fold {i}: best_epoch={best_epoch} iou={metrics['iou']:.3f} dice={metrics['dice']:.3f}")
        else:
            result = train_model(train_ids=train_ids, val_ids=val_ids, config=config, progress=False, device=device)
            save_checkpoint(ckpt_path, result)
            model = result.model
            best_epoch = result.best_epoch
            metrics = result.final_val_metrics()
            print(f"  fold {i}: best_epoch={best_epoch}/{len(result.train_losses)} "
                  f"iou={metrics['iou']:.3f} dice={metrics['dice']:.3f}"
                  f"{' (stopped early)' if result.stopped_early else ''}")
        model.eval()
        per_fold_metrics.append(metrics)

        val_ds = RoofTrainDataset(ids=val_ids)  # transform=None: never augment validation data
        with torch.no_grad():
            for j in range(len(val_ds)):
                sample = val_ds[j]
                logits = model(sample["image"].unsqueeze(0).to(device))
                pred = (torch.sigmoid(logits)[0, 0] > config.threshold).cpu().numpy()
                gt = sample["mask"][0].numpy().astype(bool)
                val_samples.append({"id": sample["id"], "image": sample["image"], "gt": gt, "pred": pred})

    ious = [m["iou"] for m in per_fold_metrics]
    dices = [m["dice"] for m in per_fold_metrics]
    print(f"\n=== SPEC §3.8 evaluation metric ===")
    print(f"Config: augment_features={FEATURES}, {args.n_folds}-fold CV, seed={args.seed}, "
          f"epochs<= {args.epochs}, patience={args.patience}, best-Dice checkpoint per fold")
    print(f"Mean IoU:  {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
    print(f"Mean Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f}")

    val_samples.sort(key=lambda s: s["id"])
    render_validation_grid(val_samples, args.grid_path)
    print(f"\nSaved image|ground-truth|prediction grid ({len(val_samples)} images) to {args.grid_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
