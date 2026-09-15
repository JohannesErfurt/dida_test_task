"""Project paths, dataset constants, and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Original, unmodified data: RGBA images and raw grayscale labels (antialiased boundary
# pixels, values 0-254-255). Used only for data inspection (scripts/inspect_data.py) and as
# the source for scripts/build_data_convert.py -- the training pipeline does not read from
# here directly, see IMAGES_DIR / LABELS_DIR below.
DATA_ORG_DIR = DATA_DIR / "data_org"
ORG_IMAGES_DIR = DATA_ORG_DIR / "images_RGBA"
ORG_LABELS_DIR = DATA_ORG_DIR / "labels_org"

# Converted data: RGB images (alpha dropped, DATA_REPORT.md §4) and binarized labels
# (`label = 255 * (label > 128)`, DATA_REPORT.md §5). Built from DATA_ORG_DIR by
# scripts/build_data_convert.py. This is what the training pipeline actually reads --
# IMAGES_DIR / LABELS_DIR below point here, not at the original data.
DATA_CONVERT_DIR = DATA_DIR / "data_convert"
IMAGES_DIR = DATA_CONVERT_DIR / "images_RGB"
LABELS_DIR = DATA_CONVERT_DIR / "labels_bin_128"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
INSPECTION_DIR = OUTPUTS_DIR / "inspection"

# 278's label (data_org/labels_org/278.png) was deleted from the dataset: on inspection it
# clearly does not correspond to image 278 at all (it's a pixel-perfect duplicate of 270's
# label, almost certainly a copy/paste annotation error) and would degrade training quality
# if used. Rather than discard the image too, 278 has been added to TEST_IDS below, so it
# still gets a prediction once the model is trained -- see DATA_REPORT.md §3.
TEST_IDS: tuple[str, ...] = ("278", "535", "537", "539", "551", "553")

IMAGE_SIZE = 256
RANDOM_SEED = 42

# Image normalization (SPEC §3.3): scale to [0,1] then normalize with ImageNet stats, since the
# pretrained encoder planned for §3.6 (e.g. ResNet34 via segmentation-models-pytorch) was itself
# pretrained on ImageNet-normalized inputs. Applied identically at train and inference time.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Default checkpoint filename (used by train.py / predict.py)
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "best_model.pt"
