"""Project paths, dataset constants, and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
# 278's label was a byte-identical duplicate of 270's, so the pair was dropped from
# the dataset entirely; its original image is kept here for reference (see DATA_REPORT.md §3).
WRONG_LABEL_DIR = DATA_DIR / "wrong_label"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
INSPECTION_DIR = OUTPUTS_DIR / "inspection"

TEST_IDS: tuple[str, ...] = ("535", "537", "539", "551", "553")

IMAGE_SIZE = 256
RANDOM_SEED = 42

# Default checkpoint filename (used by train.py / predict.py)
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "best_model.pt"
