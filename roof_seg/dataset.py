"""Dataset loading and preprocessing (SPEC §3.3).

IMAGES_DIR / LABELS_DIR (roof_seg/config.py) point at data/data_convert/, not
the original data/data_org/: the alpha-drop and label-binarization decisions
from data inspection (see DATA_REPORT.md for the full findings and reasoning)
are already materialized on disk there by scripts/build_data_convert.py:
  - Images are already RGB, alpha dropped (§4).
  - Labels are already binarized with `label = 255 * (label > 128)` (§5,
    revised from the earlier `label > 0` default).
  - 278's label was wrong (a copy of 270's) and was excluded from the
    training set entirely: it has no file in data_convert/labels_bin_128/,
    and 278 is in TEST_IDS instead (§3). Train IDs are therefore just "every
    stem present in LABELS_DIR" -- no extra filtering needed here.

load_image_rgb() / load_binary_mask() still call `.convert("RGB")` / threshold
the array rather than loading it raw -- this is a defensive no-op on already-
clean data_convert/ files, so the functions stay correct even if pointed at
different (e.g. raw) source directories.

Images are converted to tensors and normalized with ImageNet mean/std,
matching the pretrained encoder planned for §3.6 (segmentation-models-pytorch
models expect input normalized the same way their encoder was pretrained).
The same loading/normalization path is used for training and test images, so
train and inference preprocessing can never drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from roof_seg.config import IMAGENET_MEAN, IMAGENET_STD, IMAGES_DIR, LABELS_DIR, TEST_IDS


def load_image_rgb(path: Path) -> np.ndarray:
    """Load a satellite tile as RGB (data_convert/ images already have no alpha; §4)."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))  # (H, W, 3) uint8


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a label as a {0,1} mask (data_convert/ labels are already binarized; §5)."""
    with Image.open(path) as im:
        arr = np.array(im)
    return (arr > 0).astype(np.float32)  # (H, W), values in {0., 1.}


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float32, scaled to [0,1] then ImageNet-normalized."""
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """(H, W) float32 {0,1} -> (1, H, W) float32 tensor."""
    return torch.from_numpy(mask).unsqueeze(0).float()


def get_train_ids() -> list[str]:
    """IDs with a usable label. 278 is already excluded on disk (DATA_REPORT.md §3)."""
    ids = sorted(p.stem for p in LABELS_DIR.glob("*.png"))
    assert not (set(ids) & set(TEST_IDS)), "Train/test ID overlap detected"
    return ids


def get_test_ids() -> list[str]:
    return sorted(TEST_IDS)


class RoofTrainDataset(Dataset):
    """(image, mask) pairs for the labeled training images.

    `transform` is an optional Albumentations-style callable applied to the
    raw numpy arrays via `transform(image=image, mask=mask)`, run *before*
    normalization/tensor conversion. Left unset (no augmentation) here --
    wiring in an actual augmentation pipeline is SPEC §3.5, a separate
    subtask.
    """

    def __init__(self, ids: Optional[list[str]] = None, transform: Optional[Callable] = None):
        self.ids = ids if ids is not None else get_train_ids()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        image_id = self.ids[index]
        image = load_image_rgb(IMAGES_DIR / f"{image_id}.png")
        mask = load_binary_mask(LABELS_DIR / f"{image_id}.png")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return {
            "image": image_to_tensor(image),
            "mask": mask_to_tensor(mask),
            "id": image_id,
        }


class RoofTestDataset(Dataset):
    """Images only (no label) for the held-out test IDs -- identical preprocessing to training."""

    def __init__(self, ids: Optional[list[str]] = None):
        self.ids = ids if ids is not None else get_test_ids()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        image_id = self.ids[index]
        image = load_image_rgb(IMAGES_DIR / f"{image_id}.png")
        return {"image": image_to_tensor(image), "id": image_id}
