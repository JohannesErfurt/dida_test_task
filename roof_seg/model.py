"""Segmentation model definition (SPEC §3.6).

U-Net with an ImageNet-pretrained ResNet34 encoder, via
`segmentation-models-pytorch`. Single output channel = per-pixel roof logit.

Why this architecture for this task:

  - **U-Net, not a classifier-style / heavily-downsampling head.** Roof
    masks are judged on boundary accuracy (IoU/Dice over regions that are
    only 5-28% of the frame -- DATA_REPORT.md §6). U-Net's skip connections
    feed high-resolution encoder features directly into the decoder, so
    fine spatial detail lost to downsampling is recovered at the output.
    A model that only upsamples from a coarse bottleneck (e.g. plain FCN)
    gives blobbier boundaries, which is exactly where the metric is won or
    lost on rectangular, straight-edged roofs.

  - **Pretrained encoder, non-negotiable at N=24.** 24 training images is
    far too few to learn general visual features (edges, texture, shading)
    from scratch. The ResNet34 encoder arrives already knowing those from
    ImageNet; training only has to adapt them to "roof vs not-roof" on
    aerial imagery. This is the single most important design decision for
    a dataset this small.

  - **ResNet34 specifically, over a bigger backbone.** ResNet34 is deep
    enough to carry useful pretrained features but small enough (~24M total
    params here) not to overfit 24 images instantly. A ResNet101 or large
    EfficientNet would bring more capacity than this dataset can constrain.
    It's also `smp`'s default, is well-trodden for segmentation transfer
    learning, and its ImageNet preprocessing matches what
    `roof_seg/dataset.py` already applies (asserted in `build_model()`).

The model returns **raw logits**, not probabilities: `activation=None`. The
training loss (§3.7) is expected to be a logits-space BCE
(`BCEWithLogitsLoss`), which is numerically stabler than sigmoid-then-BCE,
and inference (§3.9) applies the sigmoid explicitly before thresholding.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_preprocessing_params

from roof_seg.config import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_ENCODER = "resnet34"
DEFAULT_ENCODER_WEIGHTS = "imagenet"


def _assert_preprocessing_matches_config(encoder_name: str, encoder_weights: str | None) -> None:
    """Fail loudly if the encoder expects different normalization than the dataset applies.

    `roof_seg/dataset.py` normalizes with `config.IMAGENET_MEAN/STD`, hard-coded.
    Swapping in an encoder pretrained with different statistics is a one-line
    change whose only symptom would be a quietly worse model -- so it's checked
    rather than trusted.
    """
    if encoder_weights is None:
        return  # random init: no preprocessing expectation to honour

    params = get_preprocessing_params(encoder_name, encoder_weights)
    expected_mean = tuple(round(v, 6) for v in params["mean"])
    expected_std = tuple(round(v, 6) for v in params["std"])

    if expected_mean != IMAGENET_MEAN or expected_std != IMAGENET_STD:
        raise ValueError(
            f"Encoder {encoder_name}/{encoder_weights} expects normalization "
            f"mean={expected_mean}, std={expected_std}, but roof_seg.dataset normalizes with "
            f"mean={IMAGENET_MEAN}, std={IMAGENET_STD}. Update config.IMAGENET_MEAN/STD "
            f"(and the dataset docstring) to match, or pick a different encoder."
        )


def build_model(
    encoder_name: str = DEFAULT_ENCODER,
    encoder_weights: str | None = DEFAULT_ENCODER_WEIGHTS,
    freeze_encoder: bool = False,
) -> nn.Module:
    """Build the roof segmentation model.

    Args:
        encoder_name: `smp` encoder identifier (default `resnet34`).
        encoder_weights: pretrained weights tag, or None for random init.
            Downloads on first use and is then cached by torch.hub.
        freeze_encoder: if True, encoder parameters get `requires_grad=False`
            so only the decoder trains. A plausible small-data tactic worth
            comparing via the §3.4 CV harness; off by default (full
            fine-tuning) since with a task this far from ImageNet's domain
            the encoder usually does need to adapt.

    Returns:
        Module mapping `(B, 3, H, W)` -> `(B, 1, H, W)` raw logits.

    Reproducibility: decoder weights are randomly initialized, so call
    `roof_seg.seed.set_seed()` before building if you need identical inits.
    """
    _assert_preprocessing_matches_config(encoder_name, encoder_weights)

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,       # single channel: per-pixel roof logit
        activation=None,  # raw logits -- see module docstring
    )

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
