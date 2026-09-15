"""Sanity-check the model definition (SPEC §3.6 'done when' checklist).

Builds the model and runs it against real batches from RoofTrainDataset to confirm:
  - `(B, 3, 256, 256)` in -> `(B, 1, 256, 256)` out
  - the output is raw logits, not probabilities (i.e. `activation=None` took
    effect) -- otherwise a logits-space loss in §3.7 would be silently wrong
  - the encoder is actually pretrained (weights are not at random init)
  - `freeze_encoder=True` really does stop encoder params training
  - the whole chain works together: augmentation -> dataset -> model

Also renders an untrained-model prediction figure, which is a useful
"before" reference to compare §3.7's trained output against.
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
from roof_seg.config import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    INSPECTION_DIR,
    RANDOM_SEED,
)
from roof_seg.dataset import RoofTrainDataset, get_train_ids  # noqa: E402
from roof_seg.model import DEFAULT_ENCODER, build_model, count_parameters  # noqa: E402
from roof_seg.paths import ensure_output_dirs  # noqa: E402
from roof_seg.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check the segmentation model.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnormalized = (image_tensor * std + mean).clamp(0, 1)
    return (unnormalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def check_shapes_and_logits(batch_size: int) -> torch.nn.Module:
    print("=== Model build and forward pass ===")
    model = build_model()
    trainable, total = count_parameters(model)
    print(f"Architecture: U-Net, encoder={DEFAULT_ENCODER} (imagenet), classes=1, activation=None")
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    ds = RoofTrainDataset(transform=build_train_transform())
    batch = torch.stack([ds[i]["image"] for i in range(batch_size)])
    assert batch.shape == (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), batch.shape

    model.eval()
    with torch.no_grad():
        logits = model(batch)

    print(f"Input  {tuple(batch.shape)} -> output {tuple(logits.shape)}")
    assert logits.shape == (batch_size, 1, IMAGE_SIZE, IMAGE_SIZE), logits.shape
    print(f"Shapes match SPEC §3.6 requirement (B,3,256,256) -> (B,1,256,256): OK")

    lo, hi = float(logits.min()), float(logits.max())
    print(f"Output range: [{lo:.3f}, {hi:.3f}]")
    if lo < 0.0 or hi > 1.0:
        print("  Output falls outside [0,1] -> these are raw logits, not probabilities: OK")
    else:
        raise AssertionError(
            "Output happens to lie inside [0,1]; cannot confirm it is logits rather than "
            "probabilities. Check that activation=None is set."
        )
    return model


def check_encoder_is_pretrained() -> None:
    """A randomly-initialized encoder and a pretrained one must not share weights."""
    print("\n=== Encoder is genuinely pretrained ===")
    set_seed(0)
    pretrained = build_model(encoder_weights="imagenet")
    set_seed(0)
    random_init = build_model(encoder_weights=None)

    w_pre = pretrained.encoder.conv1.weight.detach()
    w_rand = random_init.encoder.conv1.weight.detach()
    assert not torch.allclose(w_pre, w_rand), "Pretrained encoder matches random init"
    print(f"  encoder.conv1 pretrained mean|std = {w_pre.mean():+.5f}|{w_pre.std():.5f}, "
          f"random init = {w_rand.mean():+.5f}|{w_rand.std():.5f}")
    print("  Pretrained weights differ from random init: OK")


def check_freeze_encoder() -> None:
    print("\n=== freeze_encoder ===")
    unfrozen_trainable, total = count_parameters(build_model(freeze_encoder=False))
    frozen_trainable, _ = count_parameters(build_model(freeze_encoder=True))
    print(f"  freeze_encoder=False -> {unfrozen_trainable:,} / {total:,} trainable")
    print(f"  freeze_encoder=True  -> {frozen_trainable:,} / {total:,} trainable")
    assert frozen_trainable < unfrozen_trainable
    print("  Freezing reduces the trainable parameter count: OK")


def render_untrained_prediction(model: torch.nn.Module, seed: int) -> None:
    print("\n=== Untrained prediction figure ===")
    rng = np.random.default_rng(seed)
    sample_ids = sorted(rng.choice(get_train_ids(), size=3, replace=False).tolist())
    ds = RoofTrainDataset(ids=sample_ids)  # no augmentation for a clean reference

    model.eval()
    fig, axes = plt.subplots(len(sample_ids), 3, figsize=(9, 3 * len(sample_ids)))
    if len(sample_ids) == 1:
        axes = axes[np.newaxis, :]

    for row, image_id in enumerate(sample_ids):
        sample = ds[row]
        with torch.no_grad():
            prob = torch.sigmoid(model(sample["image"].unsqueeze(0)))[0, 0].numpy()

        axes[row, 0].imshow(denormalize(sample["image"]))
        axes[row, 0].set_ylabel(image_id, fontsize=9)
        axes[row, 1].imshow(sample["mask"].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        im = axes[row, 2].imshow(prob, cmap="magma", vmin=0, vmax=1)
        for col, title in enumerate(["image", "ground truth", "untrained model P(roof)"]):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)
    fig.colorbar(im, ax=axes[:, 2], fraction=0.046)

    out_path = INSPECTION_DIR / "model_untrained_prediction.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved untrained-model reference to {out_path}")
    print("  (Output is expected to be meaningless -- the model has not been trained yet.)")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()

    model = check_shapes_and_logits(args.batch_size)
    check_encoder_is_pretrained()
    check_freeze_encoder()
    render_untrained_prediction(model, args.seed)

    print("\nAll SPEC §3.6 model checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
