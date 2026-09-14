"""Ensure output directories exist."""

from roof_seg.config import CHECKPOINTS_DIR, INSPECTION_DIR, PREDICTIONS_DIR


def ensure_output_dirs() -> None:
    """Create standard output directories if they do not exist."""
    for directory in (CHECKPOINTS_DIR, PREDICTIONS_DIR, INSPECTION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
