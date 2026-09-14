"""Project paths, dataset constants, and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

# RGB-only copy of the dataset (alpha channel dropped), built by
# scripts/build_rgb_dataset.py per the DATA_REPORT.md §4 decision. The
# original RGBA dataset above is left untouched.
RGB_DATA_DIR = PROJECT_ROOT / "data_rgb"
RGB_IMAGES_DIR = RGB_DATA_DIR / "images"
RGB_LABELS_DIR = RGB_DATA_DIR / "labels"

# Binarized copy of the labels (pixel values strictly {0, 1} via `label > 0`), built by
# scripts/build_binarized_labels.py per the DATA_REPORT.md §5 decision. The original
# labels (with antialiased 1-254 boundary values) above are left untouched.
BINARIZED_LABELS_DIR = PROJECT_ROOT / "data_binarized" / "labels"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
INSPECTION_DIR = OUTPUTS_DIR / "inspection"

# 278's label (data/labels/278.png) was deleted from the dataset: on inspection it clearly
# does not correspond to image 278 at all (it's a pixel-perfect duplicate of 270's label,
# almost certainly a copy/paste annotation error) and would degrade training quality if
# used. Rather than discard the image too, 278 has been added to TEST_IDS below, so it
# still gets a prediction once the model is trained — see DATA_REPORT.md §3.
TEST_IDS: tuple[str, ...] = ("278", "535", "537", "539", "551", "553")

IMAGE_SIZE = 256
RANDOM_SEED = 42

# Default checkpoint filename (used by train.py / predict.py)
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "best_model.pt"
