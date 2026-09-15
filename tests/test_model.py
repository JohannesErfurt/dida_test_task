"""Tests for roof_seg.model (SPEC §3.6).

Most tests build with `encoder_weights=None` (random init): the architecture
contract being tested is identical either way, and skipping the pretrained
download keeps the suite fast and runnable offline. The two things that
genuinely require pretrained weights get their own tests.
"""

from __future__ import annotations

import pytest
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_params

from roof_seg.config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
from roof_seg.model import DEFAULT_ENCODER, DEFAULT_ENCODER_WEIGHTS, build_model, count_parameters
from roof_seg.seed import set_seed


@pytest.fixture(scope="module")
def model():
    """Random-init model, built once and shared (building is the slow part)."""
    set_seed(0)
    return build_model(encoder_weights=None)


def test_output_shape_is_single_channel_same_resolution(model):
    batch = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE)


def test_output_is_logits_not_probabilities(model):
    """activation=None must be in effect, or §3.7's logits-space loss breaks silently."""
    set_seed(1)
    batch = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.min() < 0.0 or out.max() > 1.0, "Output looks bounded; activation may not be None"


def test_batch_size_one_works(model):
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)


def test_gradients_flow_to_encoder_and_decoder():
    set_seed(0)
    m = build_model(encoder_weights=None, freeze_encoder=False)
    out = m(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    out.sum().backward()

    assert m.encoder.conv1.weight.grad is not None
    assert any(p.grad is not None for p in m.decoder.parameters())


def test_freeze_encoder_reduces_trainable_params_but_keeps_decoder():
    set_seed(0)
    unfrozen = build_model(encoder_weights=None, freeze_encoder=False)
    set_seed(0)
    frozen = build_model(encoder_weights=None, freeze_encoder=True)

    unfrozen_trainable, unfrozen_total = count_parameters(unfrozen)
    frozen_trainable, frozen_total = count_parameters(frozen)

    assert unfrozen_total == frozen_total  # same architecture
    assert frozen_trainable < unfrozen_trainable
    assert all(not p.requires_grad for p in frozen.encoder.parameters())
    assert all(p.requires_grad for p in frozen.decoder.parameters())


def test_count_parameters_matches_manual_sum(model):
    trainable, total = count_parameters(model)
    assert total == sum(p.numel() for p in model.parameters())
    assert trainable == total  # nothing frozen in this fixture


def test_encoder_preprocessing_matches_dataset_normalization():
    """The claim §3.3 makes about ImageNet stats must actually hold for this encoder.

    roof_seg/dataset.py normalizes with config.IMAGENET_MEAN/STD; if the default
    encoder ever expects different statistics, that normalization is wrong.
    """
    params = get_preprocessing_params(DEFAULT_ENCODER, DEFAULT_ENCODER_WEIGHTS)
    assert tuple(params["mean"]) == IMAGENET_MEAN
    assert tuple(params["std"]) == IMAGENET_STD


def test_build_model_rejects_encoder_with_mismatched_preprocessing():
    """inceptionv4 expects mean/std of 0.5, not ImageNet stats -> must fail loudly.

    The guard runs before any weight download, so this stays fast and offline-safe.
    """
    with pytest.raises(ValueError, match="expects normalization"):
        build_model(encoder_name="inceptionv4", encoder_weights="imagenet")


def test_random_init_skips_the_preprocessing_check():
    """encoder_weights=None has no pretrained preprocessing expectation to honour."""
    m = build_model(encoder_name="inceptionv4", encoder_weights=None)
    with torch.no_grad():
        out = m(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)


def test_pretrained_encoder_differs_from_random_init():
    """Confirms encoder_weights='imagenet' actually loads weights (needs cached download)."""
    set_seed(0)
    pretrained = build_model(encoder_weights="imagenet")
    set_seed(0)
    random_init = build_model(encoder_weights=None)

    assert not torch.allclose(
        pretrained.encoder.conv1.weight.detach(),
        random_init.encoder.conv1.weight.detach(),
    )
