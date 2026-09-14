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

| Asset | Path | Count | Format |
|---|---|---|---|
| Satellite images | `data/images/{id}.png` | 30 | 256×256 RGBA |
| Roof labels | `data/labels/{id}.png` | 24 | 256×256 grayscale |

Of the original 30 images / 25 labels, **`278`'s label was wrong**: on inspection it is clearly not `278`'s roof mask at all, but a byte-identical copy of `270`'s label — almost certainly a copy/paste annotation error. Training on it would teach the model an incorrect image→mask mapping and hurt prediction quality, so `data/labels/278.png` has been **deleted**. `278`'s image was kept and moved into the test set instead (`TEST_IDS` in `roof_seg/config.py`), so it still gets a prediction once the model is trained — it just never contributes a (wrong) training signal. See [`DATA_REPORT.md` §3](DATA_REPORT.md#3-duplicate--inconsistent-labels--278s-label-is-wrong) for the full writeup.

**Train set:** the 24 images in `data/images/` that have a matching label in `data/labels/`.

**Test set (inference only):** `278`, `535`, `537`, `539`, `551`, `553`.

Labels use `0` for background, `255` for roof interior, and `1–254` for boundary pixels. The binarization rule is chosen during data inspection (see [`SPEC.md` §3.2](SPEC.md#32-data-inspection-and-quality-analysis)).

## Workflow

```
Project setup
  → Data inspection       ← RGB/alpha, label quirks, binarization rule
    → Data loading
      → Augmentation
        → Model (pretrained encoder)
          → Training
            → Internal evaluation
              → Test inference
                → Documentation & delivery
```

## Project structure

```
dida_test_task/
├── data/
│   ├── images/               # 30 satellite tiles
│   └── labels/                # 24 roof masks (278's wrong label deleted; 278 moved to TEST_IDS)
├── roof_seg/                # Python package (config, seeds, pipeline modules)
│   ├── config.py            # Paths, test IDs, defaults (seed = 42)
│   ├── seed.py              # Reproducibility helper
│   └── paths.py             # Output directory setup
├── scripts/
│   ├── inspect_data.py      # Data quality analysis (§3.2)
│   ├── train.py             # Model training (§3.6)
│   └── predict.py           # Test-set inference (§3.8)
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

# 2. Train segmentation model
python scripts/train.py --epochs 50

# 3. Generate predictions on test images
python scripts/predict.py --checkpoint outputs/checkpoints/best_model.pt
```

### Makefile shortcuts

Equivalent targets are available via `make` (Git Bash / WSL / any shell with `make`):

```bash
make install   # create .venv and install dependencies + roof_seg package
make inspect   # run dataset inspection
make train     # run training (EPOCHS=50 SEED=42 by default, e.g. make train EPOCHS=10)
make predict   # run inference on the 5 test images
make test      # run the test suite with pytest
make clean     # remove generated outputs (checkpoints, predictions, inspection)
make distclean # clean + remove the virtualenv
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
- [ ] 5 prediction PNGs for the test set
- [ ] Write-up covering approach, findings, metrics, and limitations

See the [acceptance checklist in SPEC.md](SPEC.md#4-acceptance-checklist-final-review) before submission.

## Status

| Step | Status |
|---|---|
| 3.1 Project setup | Done |
| 3.2 Data inspection | Done |
| 3.3 Data loading | Not started |
| 3.4 Augmentation | Not started |
| 3.5 Model | Not started |
| 3.6 Training | Not started |
| 3.7 Internal evaluation | Not started |
| 3.8 Test inference | Not started |
| 3.9 Documentation | Not started |
