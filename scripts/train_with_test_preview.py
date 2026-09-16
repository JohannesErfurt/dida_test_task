"""Train the deliverable model while watching predictions evolve on the real test set.

Same configuration as `scripts/train.py`'s default run (all 24 labeled
images, no held-out split, `epochs=40`, `early_stopping_patience=10`) --
this produces the same deliverable checkpoint, just with an added diagnostic:
after every epoch, a forward pass over the 6 real `TEST_IDS` images (no
ground truth exists for them; this is qualitative only) is rendered as an
image | predicted-mask-overlay grid and saved to
`outputs/inspection/epoch_progress/epoch_NNN.png`, so the progression of the
test-set predictions can be inspected directly rather than only inferred
from the training loss curve.

Note: since this run has no validation split (by design -- see
`roof_seg/train.py`'s module docstring / SPEC §3.7), `early_stopping_patience`
has no effect here and training always runs the full `--epochs` budget. The
per-epoch test overlays are shown purely for visual inspection, not used to
decide when to stop.
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
    DEFAULT_CHECKPOINT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INSPECTION_DIR,
    RANDOM_SEED,
    TEST_IDS,
)
from roof_seg.dataset import RoofTestDataset, get_train_ids  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.train import TrainConfig, get_device, save_checkpoint, train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=defaults.early_stopping_patience,
        help="Inert here: this run has no validation split (see module docstring).",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnormalized = (image_tensor * std + mean).clamp(0, 1)
    return (unnormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def make_test_overlay_callback(out_dir: Path, device: torch.device):
    """Returns an `epoch_callback(epoch, model)` that renders all 6 test images."""
    test_ds = RoofTestDataset()  # fixed order: same 6 images, same preprocessing, every epoch
    out_dir.mkdir(parents=True, exist_ok=True)

    def callback(epoch: int, model: torch.nn.Module) -> None:
        fig, axes = plt.subplots(1, len(test_ds), figsize=(3 * len(test_ds), 3.4))
        with torch.no_grad():
            for col in range(len(test_ds)):
                sample = test_ds[col]
                image = denormalize(sample["image"])
                logits = model(sample["image"].unsqueeze(0).to(device))
                pred_mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()

                overlay = image.copy()
                overlay[pred_mask] = (
                    0.5 * overlay[pred_mask] + 0.5 * np.array([255, 0, 0])
                ).astype(np.uint8)

                axes[col].imshow(overlay)
                axes[col].set_title(sample["id"], fontsize=9)
                axes[col].set_xticks([])
                axes[col].set_yticks([])

        fig.suptitle(f"Test-set prediction preview -- epoch {epoch}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"epoch_{epoch:03d}.png", dpi=110)
        plt.close(fig)

    return callback


def build_contact_sheet(out_dir: Path, n_epochs_run: int, contact_sheet_path: Path, max_rows: int = 9) -> None:
    """Stitch a subset of per-epoch grids into one image so progress is visible at a glance."""
    import matplotlib.image as mpimg

    if n_epochs_run <= max_rows:
        epochs_to_show = list(range(1, n_epochs_run + 1))
    else:
        # Evenly spaced epochs, always including the first and last.
        epochs_to_show = sorted(set(
            int(round(x)) for x in np.linspace(1, n_epochs_run, max_rows)
        ))

    fig, axes = plt.subplots(
        len(epochs_to_show), 1, figsize=(13, 2.3 * len(epochs_to_show))
    )
    if len(epochs_to_show) == 1:
        axes = [axes]

    for ax, epoch in zip(axes, epochs_to_show):
        img = mpimg.imread(out_dir / f"epoch_{epoch:03d}.png")
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(contact_sheet_path, dpi=110)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    device = get_device()

    config = TrainConfig(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        # augment / loss / lr / etc. left at TrainConfig defaults: flips + 90-degree
        # rotation + color jitter (SPEC §3.5), BCE+Dice, AdamW @ 3e-4 (SPEC §3.7).
    )

    train_ids = get_train_ids()
    leaked = set(train_ids) & set(TEST_IDS)
    assert not leaked, f"Test IDs leaked into training: {sorted(leaked)}"

    epoch_progress_dir = INSPECTION_DIR / "epoch_progress"
    callback = make_test_overlay_callback(epoch_progress_dir, device)

    print(f"Device: {device}")
    print(f"Training on {len(train_ids)} images (no held-out split -- see module docstring)")
    print(f"Config: {config}")
    print(f"Test IDs excluded from training: {sorted(TEST_IDS)}")
    print(f"Saving one test-set overlay grid per epoch to {epoch_progress_dir}")

    result = train_model(
        train_ids=train_ids, val_ids=None, config=config, epoch_callback=callback
    )

    save_checkpoint(args.checkpoint, result)
    print(f"\nSaved checkpoint to {args.checkpoint}")

    n_epochs_run = len(result.train_losses)
    contact_sheet_path = INSPECTION_DIR / "epoch_progress_contact_sheet.png"
    build_contact_sheet(epoch_progress_dir, n_epochs_run, contact_sheet_path)
    print(f"Saved a {min(n_epochs_run, 9)}-epoch contact sheet to {contact_sheet_path}")

    print(f"Ran {n_epochs_run} epoch(s)")
    print(f"Final train loss: {result.train_losses[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
