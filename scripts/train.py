"""Train the final roof segmentation model (SPEC §3.7).

Trains on **all 24 labeled pairs** with no held-out split and saves the
checkpoint used for test-set inference (§3.9). Per SPEC §3.7, cross-validation
(§3.4) selects the configuration beforehand; it is not used to permanently
reserve a validation slice from a dataset this small.

Use `--val-fraction` only for a quick sanity run that reports a metric while
training -- it shrinks the training set, so it should not produce the
deliverable checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_seg.config import DEFAULT_CHECKPOINT, INSPECTION_DIR, RANDOM_SEED, TEST_IDS  # noqa: E402
from roof_seg.cross_validation import make_folds  # noqa: E402
from roof_seg.dataset import get_train_ids  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.train import (  # noqa: E402
    TrainConfig,
    TrainResult,
    get_device,
    save_checkpoint,
    train_model,
)


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train the final roof segmentation model on all 24 labeled pairs.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--loss", default=defaults.loss, choices=["bce_dice", "bce", "dice"])
    parser.add_argument("--no-augment", action="store_true", help="Disable §3.5 augmentation.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train the decoder only.")
    parser.add_argument(
        "--val-fraction",
        type=int,
        default=0,
        metavar="N_FOLDS",
        help="Sanity-run only: hold out 1/N of the training IDs for per-epoch metrics. "
        "0 (default) trains on all 24 -- the deliverable checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def plot_history(result: TrainResult, out_path: Path) -> None:
    has_val = bool(result.val_metrics)
    fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(10 if has_val else 5.5, 4))
    axes = axes if has_val else [axes]

    epochs = range(1, len(result.train_losses) + 1)
    axes[0].plot(epochs, result.train_losses, color="#3b6ea5")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(f"train loss ({result.config.loss})")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    if has_val:
        axes[1].plot(epochs, [m["iou"] for m in result.val_metrics], label="IoU")
        axes[1].plot(epochs, [m["dice"] for m in result.val_metrics], label="Dice")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("validation metric")
        axes[1].set_title("Validation (sanity split, not the deliverable setup)")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss=args.loss,
        augment=not args.no_augment,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
    )

    all_ids = get_train_ids()
    if args.val_fraction:
        train_ids, val_ids = make_folds(all_ids, n_folds=args.val_fraction, seed=args.seed)[0]
        print(f"SANITY RUN: holding out {len(val_ids)} of {len(all_ids)} images "
              f"({val_ids}). This is NOT the deliverable checkpoint setup.")
    else:
        train_ids, val_ids = all_ids, None

    # SPEC §3.7: no test image may appear in training.
    leaked = set(train_ids) & set(TEST_IDS)
    assert not leaked, f"Test IDs leaked into training: {sorted(leaked)}"

    print(f"Device: {get_device()}")
    print(f"Training on {len(train_ids)} images")
    print(f"Config: {config}")
    print(f"Test IDs excluded from training: {sorted(TEST_IDS)}")

    result = train_model(train_ids=train_ids, val_ids=val_ids, config=config)

    save_checkpoint(args.checkpoint, result)
    print(f"\nSaved checkpoint to {args.checkpoint}")

    plot_path = INSPECTION_DIR / "training_history.png"
    plot_history(result, plot_path)
    print(f"Saved training history plot to {plot_path}")

    print(f"Final train loss: {result.train_losses[-1]:.4f}")
    if result.val_metrics:
        final = result.final_val_metrics()
        print(f"Final sanity-split metrics: IoU={final['iou']:.3f} Dice={final['dice']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
