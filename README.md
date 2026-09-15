# Roof Segmentation on Satellite Imagery

Take-home task: train a neural network to predict pixel-wise roof masks from aerial satellite images, evaluate on held-out test tiles, and document the approach for discussion.

> **Full specification:** see [`SPEC.md`](SPEC.md) for goals, subtasks, acceptance criteria, and open decisions.
> **Data inspection findings:** see [`DATA_REPORT.md`](DATA_REPORT.md) for the dataset audit and preprocessing decisions (alpha handling, label binarization, the wrong `278` label).

## Original task

> There are 30 satellite pictures of houses and 25 corresponding labels that indicate the roofs. Take those 25 data points and train a neural network on them — you are completely free about the architecture and are of course allowed to use any predefined version of networks, however, you should be able to explain what you are doing — in terms of code as well as in terms of why certain steps are good choices. The preferred language is Python, but you can also use other languages. Please evaluate your network on the 5 remaining test images by making predictions of the roofs — send us the predictions and ideally some comments on what you have been doing. Everything else we will discuss from there.

## Goal

Build a **binary semantic segmentation** pipeline that:

1. Inspects and documents the dataset
2. Trains on labeled image–mask pairs
3. Predicts roof masks for **6 test images** (no labels provided)
4. Delivers predictions plus a short write-up explaining model choice, preprocessing, and limitations

## Dataset

The dataset lives under `data/` in two stages:

| Asset | Path | Count | Format |
|---|---|---|---|
| Satellite images, **original** | `data/data_org/images_RGBA/{id}.png` | 30 | 256×256 RGBA |
| Roof labels, **original** | `data/data_org/labels_org/{id}.png` | 24 | 256×256 grayscale (antialiased, values 0–254–255) |
| Satellite images, **converted** (training input) | `data/data_convert/images_RGB/{id}.png` | 30 | 256×256 RGB |
| Roof labels, **converted** (training input) | `data/data_convert/labels_bin_128/{id}.png` | 24 | 256×256 grayscale, values strictly `{0, 255}` |

`data_org/` is the original data, never modified. `data_convert/` is generated from it by [`scripts/build_data_convert.py`](scripts/build_data_convert.py) (`make data-convert`) and is what the **training pipeline actually reads** (`roof_seg/dataset.py`) — `data_org/` is used only for inspection (`scripts/inspect_data.py`). Regenerate `data_convert/` any time with `make data-convert --overwrite`-equivalent (`python scripts/build_data_convert.py --overwrite`); by default it skips files that already exist.

Two preprocessing decisions turn `data_org/` into `data_convert/` (see [DATA_REPORT.md](DATA_REPORT.md) for the full reasoning):
- **Alpha dropped** (§4): images go from RGBA to RGB. The non-opaque regions are black censorship boxes with negligible overlap with roof pixels.
- **Labels binarized with `label = 255 * (label > 128)`** (§5, revised from an earlier `label > 0` default): a "majority-coverage" rule — a boundary pixel counts as roof only if more than half its area is covered by the annotated polygon. `outputs/inspection/data_convert_comparison.png` (org image | convert image | org label | convert label) shows the effect on a few samples.

Of the original 30 images / 25 labels, **`278`'s label was wrong**: on inspection it is clearly not `278`'s roof mask at all, but a byte-identical copy of `270`'s label — almost certainly a copy/paste annotation error. Training on it would teach the model an incorrect image→mask mapping and hurt prediction quality, so it has been **deleted** (no file in either `labels_org/` or `labels_bin_128/`). `278`'s image was kept and moved into the test set instead (`TEST_IDS` in `roof_seg/config.py`), so it still gets a prediction once the model is trained — it just never contributes a (wrong) training signal. See [`DATA_REPORT.md` §3](DATA_REPORT.md#3-duplicate--inconsistent-labels--278s-label-is-wrong) for the full writeup.

**Train set:** the 24 images in `data/data_convert/images_RGB/` that have a matching label in `data/data_convert/labels_bin_128/`.

**Test set (inference only):** `278`, `535`, `537`, `539`, `551`, `553`.

## Data loading (`roof_seg/dataset.py`)

[`roof_seg/dataset.py`](roof_seg/dataset.py) implements SPEC §3.3 on top of `data/data_convert/` (`roof_seg/config.py::IMAGES_DIR`/`LABELS_DIR`) — it never reads `data_org/`, so alpha-handling and binarization are fixed, materialized artifacts rather than something recomputed on every run:

- **`RoofTrainDataset`** — the 24 `(image, mask)` training pairs (`get_train_ids()` = every stem in `data_convert/labels_bin_128/`, which already excludes `278` — see above). Each item is `{"image": (3,256,256) tensor, "mask": (1,256,256) tensor, "id": str}`.
- **`RoofTestDataset`** — the 6 `TEST_IDS` images, same preprocessing, no mask. Each item is `{"image": (3,256,256) tensor, "id": str}`.
- `load_image_rgb()` / `load_binary_mask()` are defensive no-ops on the already-clean `data_convert/` files (`.convert("RGB")`, threshold the array) — they stay correct even if pointed at a different source directory.
- **Normalization:** `image_to_tensor()` scales to `[0,1]` then applies ImageNet mean/std (`roof_seg/config.py::IMAGENET_MEAN/STD`), matching the pretrained encoder planned for §3.6 — identical for train and test, so preprocessing can't drift between them.
- `RoofTrainDataset` takes an optional `transform` (Albumentations-style) for §3.5's augmentation pipeline; unset for now.

Run `python scripts/check_dataset.py` (or `make check-dataset`) to sanity-check the loader: verifies dataset lengths (24/6), tensor shapes, strictly-binary masks, and renders `outputs/inspection/dataset_sanity_check.png` (de-normalized image | mask | overlay) to confirm alignment survives the full tensor pipeline.

## Workflow

```
Project setup
  → Data inspection            ← RGB/alpha, label quirks, binarization rule
    → Data loading
      → Cross-validation harness   ← built early so it can test augmentation/hyperparameter choices
        → Augmentation              ← tested with the CV harness (paired on/off)
          → Model (pretrained encoder)
            → Training (final, full-data)
              → Internal evaluation     ← reports the CV results
                → Test inference
                  → Documentation & delivery
```

## Project structure

```
dida_test_task/
├── data/
│   ├── data_org/              # Original, unmodified data
│   │   ├── images_RGBA/         # 30 satellite tiles (RGBA)
│   │   └── labels_org/          # 24 roof masks (antialiased, 278's wrong label deleted)
│   └── data_convert/          # Generated from data_org/ — the training pipeline's actual input
│       ├── images_RGB/          # 30 satellite tiles (RGB, alpha dropped)
│       └── labels_bin_128/      # 24 roof masks, values strictly {0, 255} (label > 128)
├── roof_seg/                # Python package (config, seeds, pipeline modules)
│   ├── config.py            # Paths, test IDs, normalization stats, defaults (seed = 42)
│   ├── seed.py              # Reproducibility helper
│   ├── paths.py             # Output directory setup
│   └── dataset.py           # RoofTrainDataset / RoofTestDataset, preprocessing (§3.3)
├── scripts/
│   ├── inspect_data.py       # Data quality analysis on data_org/ (§3.2)
│   ├── build_data_convert.py # Build data_convert/ from data_org/ + comparison figure
│   ├── check_dataset.py      # Sanity-check the dataset loader (§3.3) + overlay figure
│   ├── train.py               # Model training (§3.7)
│   └── predict.py             # Test-set inference (§3.9)
├── tests/
│   ├── test_smoke.py        # Project setup + dataset-file smoke tests
│   └── test_dataset.py      # roof_seg.dataset unit tests (§3.3)
├── notebooks/               # Exploratory notebooks
├── outputs/
│   ├── checkpoints/         # Saved model weights
│   ├── predictions/         # Final test-set roof masks
│   └── inspection/          # Data inspection plots & reports
├── requirements.txt
├── pyproject.toml
├── SPEC.md
├── DATA_REPORT.md           # Data inspection findings (after §3.2)
└── README.md
```

## Setup

Requires **Python 3.10+**.

```bash
py -3.11 -m venv .venv        # Python 3.10+ required
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
pip install -e .              # install roof_seg package in editable mode
```

## Usage

All scripts accept `--seed` (default: **42**, defined in `roof_seg/config.py`).

```bash
# 1. Inspect dataset and document findings
python scripts/inspect_data.py

# 2. Build the training-input dataset (RGB images + binarized labels) from the original
python scripts/build_data_convert.py

# 3. (optional) Sanity-check the dataset loader
python scripts/check_dataset.py

# 4. Train segmentation model
python scripts/train.py --epochs 50

# 5. Generate predictions on test images
python scripts/predict.py --checkpoint outputs/checkpoints/best_model.pt
```

### Makefile shortcuts

Equivalent targets are available via `make` (Git Bash / WSL / any shell with `make`):

```bash
make install       # create .venv and install dependencies + roof_seg package
make inspect       # run dataset inspection on data/data_org/
make data-convert  # (re)build data/data_convert/ (RGB images + label>128 labels) from data/data_org/
make check-dataset # sanity-check the dataset loader (§3.3) + overlay figure
make train         # run training (EPOCHS=50 SEED=42 by default, e.g. make train EPOCHS=10)
make predict       # run inference on the 6 test images
make test          # run the test suite with pytest
make clean         # remove generated outputs (checkpoints, predictions, inspection)
make distclean     # clean + remove the virtualenv
```

Expected test outputs:

```
outputs/predictions/278.png
outputs/predictions/535.png
outputs/predictions/537.png
outputs/predictions/539.png
outputs/predictions/551.png
outputs/predictions/553.png
```

## Approach (planned)

| Component | Planned choice | Rationale |
|---|---|---|
| Task type | Binary semantic segmentation | Labels are pixel-wise roof masks |
| Model | U-Net + pretrained encoder (e.g. ResNet34 via `segmentation-models-pytorch`) | Strong baseline for small-data segmentation |
| Transfer learning | Required | 25 samples is too few to train from scratch |
| Augmentation | Flips, rotations, color jitter | Effective dataset expansion |
| Loss | BCE + Dice | Handles class imbalance (mostly background) |

Final choices are recorded in the write-up after data inspection and validation experiments.

## Deliverables

- [x] Data inspection report (`DATA_REPORT.md` or notebook)
- [ ] Reproducible training and inference code
- [ ] Model checkpoint
- [ ] 6 prediction PNGs for the test set
- [ ] Write-up covering approach, findings, metrics, and limitations

See the [acceptance checklist in SPEC.md](SPEC.md#4-acceptance-checklist-final-review) before submission.

## Status

| Step | Status |
|---|---|
| 3.1 Project setup | Done |
| 3.2 Data inspection | Done |
| 3.3 Data loading | Done |
| 3.4 Cross-validation harness | Not started |
| 3.5 Augmentation | Not started |
| 3.6 Model | Not started |
| 3.7 Training | Not started |
| 3.8 Internal evaluation | Not started |
| 3.9 Test inference | Not started |
| 3.10 Documentation | Not started |
