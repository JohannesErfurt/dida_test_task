"""Project paths, dataset constants, and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
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
