"""Training loop (SPEC §3.7).

Two entry points, deliberately sharing one implementation:

  - `train_model(train_ids, val_ids=[...])` -- trains on the given IDs and
    evaluates on a held-out set each epoch. Used by the §3.4 CV harness via
    `make_cv_train_fn()`, where `val_ids` is the fold's held-out slice.

  - `train_model(train_ids=<all 24>, val_ids=None)` -- the **final
    deliverable checkpoint**: trained on all 24 labeled pairs with nothing
    held back. Per SPEC §3.7, CV is used to *select* the configuration
    beforehand, not to permanently reserve a validation slice out of a
    dataset this small.

Sharing one implementation is the point: the configuration CV measures is
exactly the configuration the final model is trained with, with no second
code path that could quietly diverge.

Augmentation (§3.5) is applied to the training split only -- validation data
is loaded with `transform=None`, so a fold's held-out images are never
augmented, as SPEC §3.5 requires.

**Best-checkpoint tracking and early stopping** both require a validation
signal, so they only take effect when `val_ids` is non-empty -- i.e. for CV
folds and `--val-fraction` sanity runs. Whenever val Dice improves, the
model's weights at that epoch are snapshotted; the *final* returned model is
that best snapshot, not whatever the last epoch happened to produce (those
can differ -- validation metrics wobble late in training on a dataset this
small). If Dice hasn't improved for `early_stopping_patience` epochs in a
row, training stops early. The final deliverable run (`train_ids=<all 24>,
val_ids=None`) has no validation set to track improvement against, so both
features are inert there and it always runs the full `epochs` budget,
returning the last epoch's weights -- there is no "best" to pick from
without a held-out set, by design (see SPEC §3.7).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from roof_seg.augmentation import build_train_transform
from roof_seg.config import RANDOM_SEED
from roof_seg.dataset import RoofTrainDataset, get_train_ids
from roof_seg.losses import build_loss
from roof_seg.metrics import dice_score, iou_score
from roof_seg.model import DEFAULT_ENCODER, build_model
from roof_seg.seed import set_seed


@dataclass
class TrainConfig:
    """Everything that defines a training run.

    Kept as one object so the §3.4 CV harness can compare two configs that
    differ in exactly one field (a paired comparison), and so the checkpoint
    can record precisely what produced it.

    Defaults: AdamW at 3e-4 is a standard fine-tuning learning rate for a
    pretrained encoder -- high enough to adapt ImageNet features to aerial
    imagery, low enough not to destroy them. Batch size 4 gives 6 steps per
    epoch over 24 images, keeping BatchNorm statistics usable (batch 1-2
    would make them very noisy). `epochs=40` is an upper budget: with a
    validation set, `early_stopping_patience` can end a run sooner, and the
    best-Dice epoch's weights are kept regardless of how long training ran.
    The final deliverable run has no validation split, so it always uses the
    full 40-epoch budget -- see module docstring.

    `early_stopping_patience=10`: stop once val Dice hasn't improved for 10
    consecutive epochs. Set to 0 to disable early stopping while still
    tracking the best checkpoint (train the full epoch budget, keep the best
    epoch's weights). Only takes effect when `val_ids` is given.
    """

    epochs: int = 40
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    loss: str = "bce_dice"
    augment: bool = True
    freeze_encoder: bool = False
    encoder_name: str = DEFAULT_ENCODER
    seed: int = RANDOM_SEED
    threshold: float = 0.5  # for turning probabilities into a binary mask when scoring
    early_stopping_patience: int = 10  # epochs without val Dice improvement before stopping; 0 disables


@dataclass
class TrainResult:
    model: torch.nn.Module
    config: TrainConfig
    train_ids: list[str]
    val_ids: list[str]
    train_losses: list[float]
    val_metrics: list[dict[str, float]]  # one entry per epoch actually run, chronological
    best_epoch: Optional[int] = None  # 1-indexed; None if val_ids was empty (nothing to track)
    stopped_early: bool = False

    def final_val_metrics(self) -> dict[str, float]:
        """Metrics for the epoch whose weights `model` actually holds.

        That's the best-Dice epoch when validation was used (`best_epoch` is
        set), not necessarily the last entry in `val_metrics` -- Dice can dip
        after its peak on a dataset this small, and `model` holds the best
        snapshot, not the last one. Falls back to the last epoch's metrics
        when there was no validation to select a best epoch from.
        """
        if not self.val_metrics:
            return {}
        if self.best_epoch is not None:
            return self.val_metrics[self.best_epoch - 1]
        return self.val_metrics[-1]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    ids: list[str],
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Mean IoU/Dice over `ids`, thresholded at `threshold`.

    Loads without augmentation (`transform=None`) -- evaluation must see the
    real images, not augmented ones.
    """
    device = device or get_device()
    model.eval()
    dataset = RoofTrainDataset(ids=ids)  # no transform: never augment evaluation data

    ious, dices = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        logits = model(sample["image"].unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits)[0, 0] > threshold).cpu().numpy()
        target = sample["mask"][0].numpy()
        ious.append(iou_score(pred, target))
        dices.append(dice_score(pred, target))

    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices))}


def train_model(
    train_ids: Optional[list[str]] = None,
    val_ids: Optional[list[str]] = None,
    config: Optional[TrainConfig] = None,
    progress: bool = True,
    device: Optional[torch.device] = None,
) -> TrainResult:
    """Train a model on `train_ids`, optionally scoring `val_ids` each epoch.

    Args:
        train_ids: IDs to train on. Defaults to all 24 labeled training IDs.
        val_ids: held-out IDs to evaluate each epoch. `None` (the default)
            means no validation -- the final-checkpoint case.
        config: hyperparameters; see `TrainConfig`.
        progress: print per-epoch loss (and val metrics, if any).
    """
    config = config or TrainConfig()
    device = device or get_device()
    train_ids = list(train_ids) if train_ids is not None else get_train_ids()
    val_ids = list(val_ids) if val_ids is not None else []

    if set(train_ids) & set(val_ids):
        raise ValueError("train_ids and val_ids overlap")

    set_seed(config.seed)

    transform = build_train_transform() if config.augment else None
    dataset = RoofTrainDataset(ids=train_ids, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,  # tiny dataset; workers would cost more than they save
    )

    model = build_model(
        encoder_name=config.encoder_name, freeze_encoder=config.freeze_encoder
    ).to(device)
    criterion = build_loss(config.loss)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_losses: list[float] = []
    val_metrics: list[dict[str, float]] = []

    best_dice = -1.0
    best_epoch: Optional[int] = None
    best_state_dict = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() * images.size(0)

        epoch_loss /= len(dataset)
        train_losses.append(epoch_loss)

        message = f"  epoch {epoch:3d}/{config.epochs}  train_loss={epoch_loss:.4f}"

        if val_ids:
            metrics = evaluate(model, val_ids, threshold=config.threshold, device=device)
            val_metrics.append(metrics)
            message += f"  val_iou={metrics['iou']:.3f}  val_dice={metrics['dice']:.3f}"

            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                message += "  (best)"
            else:
                epochs_without_improvement += 1

        if progress:
            print(message, flush=True)

        if (
            val_ids
            and config.early_stopping_patience
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            if progress:
                print(
                    f"  Early stopping: val Dice hasn't improved for "
                    f"{config.early_stopping_patience} epochs "
                    f"(best={best_dice:.3f} at epoch {best_epoch})."
                )
            break

    # Return the best-Dice snapshot, not whatever the loop ended on -- these can
    # differ, since val Dice can dip after its peak on a dataset this small.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return TrainResult(
        model=model,
        config=config,
        train_ids=train_ids,
        val_ids=val_ids,
        train_losses=train_losses,
        val_metrics=val_metrics,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
    )


def make_cv_train_fn(
    config: Optional[TrainConfig] = None, progress: bool = False
) -> Callable[[list[str], list[str]], dict[str, float]]:
    """Adapt `train_model` to the §3.4 CV harness's `train_fn` interface.

    Returns a callable `(train_ids, val_ids) -> {"iou": ..., "dice": ...}`,
    which is exactly what `roof_seg.cross_validation.cross_validate()` expects.
    Pass two of these (differing in one `TrainConfig` field) to
    `paired_compare()` to test that field on identical folds.
    """
    config = config or TrainConfig()

    def train_fn(train_ids: list[str], val_ids: list[str]) -> dict[str, float]:
        result = train_model(
            train_ids=train_ids, val_ids=val_ids, config=config, progress=progress
        )
        return result.final_val_metrics()

    return train_fn


def save_checkpoint(path, result: TrainResult) -> None:
    """Persist weights plus everything §3.9 needs to rebuild the same architecture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "config": asdict(result.config),
            "train_ids": result.train_ids,
            "val_ids": result.val_ids,
            "train_losses": result.train_losses,
            "val_metrics": result.val_metrics,
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
        },
        path,
    )


def load_checkpoint(path, device: Optional[torch.device] = None) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model described by a checkpoint and load its weights."""
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # encoder_weights=None: the saved state_dict supplies the trained weights,
    # so there is no reason to download ImageNet weights just to overwrite them.
    model = build_model(encoder_name=config["encoder_name"], encoder_weights=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
