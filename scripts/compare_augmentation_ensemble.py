"""Compare 3 augmentation configs via 6-fold-ensemble majority-vote test predictions.

Replaces the greedy per-feature search with a direct, fixed 3-way comparison
(per user request):

  - "no_augmentation"  -- augment=False
  - "3_features"       -- flip, rotate90, color_jitter (the original SPEC §3.5 set)
  - "6_features"       -- + rgb_gamma, perspective, resized_crop

All three use the **same 6-fold split** (`roof_seg.cross_validation.make_folds`,
same seed), so any difference between them is attributable to the augmentation
config, not to different folds. Each fold's model keeps its own best-Dice
epoch's weights (SPEC §3.7 best-checkpoint tracking) -- these are genuinely
validated checkpoints, not the arbitrary-stopping-point single model the
"brainstorm" conversation was about.

For each config: train 6 fold-models (one per fold, checkpointed to disk and
skipped if already cached, so the run survives interruption), then run all 6
on the 6 real test images and **majority-vote** each pixel (roof if > 3 of 6
models say roof; a 3-3 tie resolves to background) into one consensus mask
per test image. This sidesteps the "which epoch is the deliverable" question
for these 18 models entirely -- each one has a real, CV-validated stopping
point, and the ensemble combines them rather than picking just one.

Reports each config's aggregate CV IoU/Dice (mean +/- std across folds, using
each fold's best epoch) alongside the qualitative majority-vote overlays.
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
    CHECKPOINTS_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INSPECTION_DIR,
    RANDOM_SEED,
)
from roof_seg.cross_validation import make_folds  # noqa: E402
from roof_seg.dataset import RoofTestDataset  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402
from roof_seg.train import TrainConfig, get_device, load_checkpoint, save_checkpoint, train_model  # noqa: E402

CONFIGS: dict[str, tuple[str, ...] | None] = {
    "no_augmentation": None,  # augment=False entirely
    "3_features": ("flip", "rotate90", "color_jitter"),
    "6_features": ("flip", "rotate90", "color_jitter", "rgb_gamma", "perspective", "resized_crop"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--out-dir", type=Path, default=CHECKPOINTS_DIR / "ensemble_compare",
        help="Where per-fold checkpoints are cached (resumable across runs).",
    )
    return parser.parse_args()


def config_for(features, args: argparse.Namespace) -> TrainConfig:
    if features is None:
        return TrainConfig(epochs=args.epochs, early_stopping_patience=args.patience, seed=args.seed, augment=False)
    return TrainConfig(
        epochs=args.epochs, early_stopping_patience=args.patience, seed=args.seed,
        augment=True, augment_features=features,
    )


def train_fold_models(config_name: str, features, fold_splits, args, device):
    """Train (or load cached) one model per fold. Returns (models, per_fold_metrics)."""
    models = []
    per_fold = []
    out_dir = args.out_dir / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (train_ids, val_ids) in enumerate(fold_splits):
        ckpt_path = out_dir / f"fold{i}.pt"
        if ckpt_path.exists():
            model, ck = load_checkpoint(ckpt_path, device=device)
            best_epoch = ck["best_epoch"]
            metrics = ck["val_metrics"][best_epoch - 1] if best_epoch else ck["val_metrics"][-1]
            print(f"  [cached] fold {i}: best_epoch={best_epoch} "
                  f"iou={metrics['iou']:.3f} dice={metrics['dice']:.3f}")
        else:
            cfg = config_for(features, args)
            result = train_model(
                train_ids=train_ids, val_ids=val_ids, config=cfg, progress=False, device=device
            )
            save_checkpoint(ckpt_path, result)
            model = result.model
            best_epoch = result.best_epoch
            metrics = result.final_val_metrics()
            print(f"  fold {i}: best_epoch={best_epoch}/{len(result.train_losses)} "
                  f"iou={metrics['iou']:.3f} dice={metrics['dice']:.3f}"
                  f"{' (stopped early)' if result.stopped_early else ''}")
        model.eval()
        models.append(model)
        per_fold.append({"fold": i, "best_epoch": best_epoch, **metrics})

    return models, per_fold


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnormalized = (image_tensor * std + mean).clamp(0, 1)
    return (unnormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def majority_vote_predict(models: list[torch.nn.Module], device: torch.device) -> dict:
    """{test_id: {"image": tensor, "mask": bool array, "vote_count": int array}}."""
    test_ds = RoofTestDataset()
    results = {}
    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            votes = []
            for model in models:
                logits = model(image_tensor)
                pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
                votes.append(pred)
            vote_count = np.stack(votes).sum(axis=0)  # 0..len(models) per pixel
            majority = vote_count > (len(models) / 2)  # ties (== half) resolve to background
            results[sample["id"]] = {
                "image": sample["image"].cpu(), "mask": majority, "vote_count": vote_count,
            }
    return results


def render_comparison_grid(all_predictions: dict[str, dict], out_path: Path) -> None:
    config_names = list(all_predictions.keys())
    test_ids = list(next(iter(all_predictions.values())).keys())

    fig, axes = plt.subplots(
        len(config_names), len(test_ids), figsize=(2.6 * len(test_ids), 2.8 * len(config_names))
    )

    for row, config_name in enumerate(config_names):
        for col, test_id in enumerate(test_ids):
            pred = all_predictions[config_name][test_id]
            image = denormalize(pred["image"])
            overlay = image.copy()
            overlay[pred["mask"]] = (0.5 * overlay[pred["mask"]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

            ax = axes[row, col]
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(test_id, fontsize=9)
            if col == 0:
                ax.set_ylabel(config_name, fontsize=9)

    fig.suptitle("Majority-vote test predictions (6-fold ensemble), by augmentation config", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Identical folds for every config -- computed once, reused for all three.
    fold_splits = make_folds(n_folds=args.n_folds, seed=args.seed)
    print(f"Using {len(fold_splits)} folds (seed={args.seed}), identical across all 3 configs.")

    all_predictions = {}
    config_summaries = {}

    for config_name, features in CONFIGS.items():
        print(f"\n=== {config_name} (features={features}) ===")
        models, per_fold = train_fold_models(config_name, features, fold_splits, args, device)

        ious = [f["iou"] for f in per_fold]
        dices = [f["dice"] for f in per_fold]
        best_epochs = [f["best_epoch"] for f in per_fold]
        print(f"  CV: mean IoU={np.mean(ious):.3f}+/-{np.std(ious):.3f}  "
              f"mean Dice={np.mean(dices):.3f}+/-{np.std(dices):.3f}  "
              f"best_epochs={best_epochs}")
        config_summaries[config_name] = {
            "per_fold": per_fold, "mean_iou": float(np.mean(ious)), "std_iou": float(np.std(ious)),
            "mean_dice": float(np.mean(dices)), "std_dice": float(np.std(dices)),
        }

        print(f"  Running majority-vote inference on the 6 test images ({len(models)} models)...")
        all_predictions[config_name] = majority_vote_predict(models, device)

        # Free the 6 models before training the next config's 6 -- keeps peak memory bounded.
        del models

    grid_path = INSPECTION_DIR / "augmentation_ensemble_comparison.png"
    render_comparison_grid(all_predictions, grid_path)
    print(f"\nSaved comparison grid to {grid_path}")

    print("\n=== Summary ===")
    for config_name, summary in config_summaries.items():
        print(f"  {config_name}: IoU={summary['mean_iou']:.3f}+/-{summary['std_iou']:.3f}  "
              f"Dice={summary['mean_dice']:.3f}+/-{summary['std_dice']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
