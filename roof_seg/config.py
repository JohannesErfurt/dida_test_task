"""Project paths, dataset constants, and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = PROJECT_ROOT / "images"
LABELS_DIR = PROJECT_ROOT / "labels"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
INSPECTION_DIR = OUTPUTS_DIR / "inspection"

TEST_IDS: tuple[str, ...] = ("535", "537", "539", "551", "553")

IMAGE_SIZE = 256
RANDOM_SEED = 42

# Default checkpoint filename (used by train.py / predict.py)
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "best_model.pt"
