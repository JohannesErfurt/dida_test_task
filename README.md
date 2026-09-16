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
- **Normalization:** `image_to_tensor()` scales to `[0,1]` then applies ImageNet mean/std (`roof_seg/config.py::IMAGENET_MEAN/STD`), matching the pretrained ResNet34 encoder used by §3.6's model (asserted in `build_model()`) — identical for train and test, so preprocessing can't drift between them.
- `RoofTrainDataset` takes an optional `transform` (Albumentations-style); pass §3.5's `build_train_transform()` for augmented training, leave unset for validation folds and reference loads.

Run `python scripts/check_dataset.py` (or `make check-dataset`) to sanity-check the loader: verifies dataset lengths (24/6), tensor shapes, strictly-binary masks, and renders `outputs/inspection/dataset_sanity_check.png` (de-normalized image | mask | overlay) to confirm alignment survives the full tensor pipeline.

## Cross-validation harness (`roof_seg/cross_validation.py`)

Implements SPEC §3.4. **Why it exists:** with only 24 labeled images, a single fixed train/val split (e.g. 19/5) is too high-variance to trust — which handful of images land in validation can swing IoU/Dice by more than the effect of whatever is actually being compared (augmentation on/off, loss choice, etc.). Cross-validation averages over all 24 images instead, and — used as a **paired comparison** (identical folds and seed, only one setting changed) — isolates a real effect from fold-composition noise far better than eyeballing two independent numbers.

- **`make_folds(ids, n_folds, seed)`** — deterministic k-fold split; every ID validates exactly once. `n_folds == 24` gives leave-one-out CV (affordable here given how small/fast each fold's training is). Raises if any `TEST_IDS` leak in.
- **`cross_validate(train_fn, n_folds, seed)`** — runs `train_fn(train_ids, val_ids) -> {metric: value}` once per fold and returns a `CVResult` with `.mean(metric)` / `.std(metric)` / `.summary()`. `train_fn` is fully decoupled from any specific model — §3.7's `make_cv_train_fn()` supplies one that trains the real U-Net, and nothing in `cross_validation.py` had to change to support it.
- **`paired_compare(train_fn_a, train_fn_b, ...)`** — runs two configs on identical folds/seed for a sensitive, noise-isolated comparison.

**Running it (`scripts/run_cv.py` / `make run-cv`):** `--mode model` (default) cross-validates the real U-Net, training one model per fold — that's what `make compare-augmentation` uses to answer §3.5's augmentation question on identical folds. `--mode baseline` keeps the original no-learning placeholders (`spatial-prior`, `centered-square`); they cost seconds instead of minutes and are retained as a sanity floor — "how good is trivial?" — to compare the trained model against.

Results (`make run-cv`, 6-fold, seed 42): spatial-prior baseline scores mean IoU 0.187 ± 0.020, Dice 0.309 ± 0.031. Paired comparison (`make run-cv COMPARE=1`) against the centered-square baseline on the *same* folds: mean IoU difference +0.005 (per-fold range −0.060 to +0.057), Dice difference +0.013 — i.e. **no detectable difference** between the two placeholders, which is the expected/correct outcome (they're both naive area-matched guesses) and demonstrates the harness's paired comparison is sensitive to real per-fold variation without over-claiming a winner from noise.

**Limitations of CV at N=24** (documented per SPEC §3.4, to be folded into the final write-up at §3.10): even with all 24 images used, per-fold metrics are noisy (LOOCV fold std ≈ 0.08 IoU vs 6-fold ≈ 0.02 — a single validation image is a high-variance estimate); folds share most of their training data with each other, so they aren't independent samples in the formal statistical sense; and reusing the same 24 images to both *select* a configuration and *report* its performance risks mild optimism bias (a full nested-CV would avoid this but isn't implemented here — see SPEC §3.4/§6 "Validation strategy" for the reasoning). Treat differences smaller than roughly 5–10 percentage points of Dice/IoU as noise, not a real effect.

## Data augmentation (`roof_seg/augmentation.py`)

Implements SPEC §3.5: `build_train_transform()` returns an Albumentations pipeline — horizontal flip, vertical flip, 90° rotation, and color jitter (brightness/contrast/saturation, small hue) — passed to `RoofTrainDataset(transform=...)`.

**Why only 90° rotation, not arbitrary-angle:** flips and 90° rotation are exact pixel permutations — every output pixel is copied unchanged from some input pixel, so the mask needs zero interpolation and stays exactly `{0, 255}`. `data/data_convert/labels_bin_128/` was binarized specifically to eliminate the antialiased boundary values in the original labels (DATA_REPORT.md §5); arbitrary-angle rotation (or any resize/crop) would interpolate the mask and silently reintroduce that same soft, non-binary boundary — undoing that decision. This isn't a convenience default, it's the reason 90°-only was chosen.

**Train-only, structurally, not by convention:** `RoofTestDataset` has no `transform` parameter at all (verified by `scripts/check_augmentation.py` via `inspect.signature`) — there's no way to accidentally augment a test or validation-fold image, since the class doesn't expose the hook to do so.

Run `python scripts/check_augmentation.py` (or `make check-augmentation`) to verify: masks stay strictly binary after augmentation (24 images × 3 draws each), `RoofTestDataset` has no augmentation hook, and to render `outputs/inspection/augmentation_grid.png` (original | 5 augmented views, mask overlaid) confirming mask geometry tracks the image through every transform.

**Does it actually help?** Checked, not assumed — `make compare-augmentation` runs the §3.4 harness as a paired comparison (6 folds, 25 epochs, identical folds and seed, augmentation the only difference):

| | mean IoU | mean Dice |
|---|---|---|
| **With** augmentation | 0.697 ± 0.050 | 0.818 ± 0.035 |
| **Without** augmentation | 0.664 ± 0.061 | 0.789 ± 0.050 |
| Paired difference | **+0.033** | +0.028 |

Augmentation is ahead on 4 of 6 folds (per-fold range −0.055 to +0.090). But **+0.033 IoU sits below the 0.05–0.10 threshold this project set in advance** for calling a difference real at N=24 — so the honest conclusion is *suggestive, not established*. Augmentation stays **on**: the point estimate favours it, fold-to-fold variance is lower with it (0.050 vs 0.061 IoU), it's well-motivated for a 24-image dataset, it costs nothing at inference, and there's no evidence it hurts. What this does **not** support is a claim that augmentation was measurably decisive — with 24 images and 6 folds, detecting an effect this size reliably would need more data than exists here.

This is also the project's first real generalization estimate: **IoU ≈ 0.70, Dice ≈ 0.82** on images the model never saw, against the ~0.19 IoU no-learning baseline from §3.4.

## Model (`roof_seg/model.py`)

Implements SPEC §3.6: `build_model()` returns a **U-Net with an ImageNet-pretrained ResNet34 encoder** (`segmentation-models-pytorch`), mapping `(B, 3, 256, 256)` → `(B, 1, 256, 256)` **raw logits**. ~24.4M parameters.

Why this architecture, for this task:

- **U-Net, for boundary detail.** The metric here (IoU/Dice over regions that are only 5–28% of the frame) is won or lost at roof boundaries. U-Net's skip connections feed high-resolution encoder features straight into the decoder, recovering spatial detail that downsampling destroys — important for the straight, rectangular edges these roofs actually have. An architecture that only upsamples from a coarse bottleneck produces blobbier outlines.
- **Pretrained encoder — the single most important choice at N=24.** 24 images is nowhere near enough to learn general visual features from scratch. The encoder arrives already knowing edges, texture and shading from ImageNet; training only has to adapt that to "roof vs not-roof" on aerial imagery.
- **ResNet34 rather than something larger.** Deep enough to carry useful pretrained features, small enough not to instantly overfit 24 images — a ResNet101 or large EfficientNet brings more capacity than this dataset can constrain.
- **Logits, not probabilities** (`activation=None`): §3.7's loss is expected to be a logits-space `BCEWithLogitsLoss` (numerically stabler than sigmoid-then-BCE), and §3.9's inference applies the sigmoid explicitly before thresholding.
- **`freeze_encoder=False` by default**, but available: freezing drops trainable parameters from 24.4M to 3.2M (decoder only). Plausible small-data tactic, but aerial imagery is far enough from ImageNet's domain that the encoder usually does need to adapt — so it's off by default and left as something to compare via the §3.4 CV harness rather than assumed either way.

`build_model()` also **asserts the encoder's expected normalization matches `config.IMAGENET_MEAN/STD`** (what `roof_seg/dataset.py` actually applies). Swapping in an encoder pretrained with different statistics — e.g. `inceptionv4`, which expects mean/std of 0.5 — is a one-line change whose only symptom would be a quietly worse model, so it raises instead of silently mis-normalizing.

Run `python scripts/check_model.py` (or `make check-model`) to verify shapes, that outputs really are logits, that the encoder weights are genuinely pretrained (vs. random init), and that `freeze_encoder` works — and to render `outputs/inspection/model_untrained_prediction.png`, an untrained-model "before" reference to compare §3.7's trained output against.

## Training (`roof_seg/train.py`, `roof_seg/losses.py`)

Implements SPEC §3.7. `make train` (or `python scripts/train.py`) trains on **all 24 labeled pairs** and writes `outputs/checkpoints/best_model.pt` plus `outputs/inspection/training_history.png`.

**Loss — BCE + soft Dice** (`roof_seg/losses.py`), both computed on logits. Roof pixels are only 5–28% of a frame, so plain BCE — averaged uniformly over pixels — lets the ~86% background dominate the gradient and can look "low loss, mediocre roofs". Dice measures region overlap instead, so correctly-predicted background contributes almost nothing to it, which makes it imbalance-insensitive and aligned with the reported metric; but on its own its gradients are poorly conditioned early, when predictions are near-zero and the intersection term is ~0. Summing them gets BCE's stable optimization plus Dice's pressure toward overlap. Dice is averaged **per sample**, not over a pooled batch, so a large-roof tile can't drown out a small-roof one.

**Hyperparameters** (`TrainConfig`): AdamW, lr 3e-4, weight decay 1e-4, batch size 4, 40 epochs, augmentation on. 3e-4 is a standard fine-tuning rate for a pretrained encoder — enough to adapt ImageNet features to aerial imagery without destroying them. Batch 4 gives 6 steps/epoch over 24 images, keeping BatchNorm statistics usable (batch 1–2 would make them very noisy).

**One implementation, two uses.** The same `train_model()` serves both the CV harness (`make_cv_train_fn()` adapts it to §3.4's `train_fn` interface) and the final checkpoint — so the configuration CV measures is exactly the configuration the deliverable is trained with, with no second code path to drift. The final model trains on all 24 images with **no held-out split**: per SPEC §3.7, CV selects the config beforehand rather than permanently reserving a validation slice from a dataset this small. `--val-fraction N` exists for a quick sanity run that reports per-epoch metrics, but it shrinks the training set and shouldn't produce the deliverable.

**Leakage protection:** `scripts/train.py` asserts no `TEST_IDS` reach training, `get_train_ids()` reads only the 24 labeled IDs, and `make_folds()` raises if a test ID enters a fold. Validation data is always loaded with `transform=None`, so held-out images are never augmented — verified by a test that spies on the dataset construction.

**Final run** (`make train`, 40 epochs, all 24 images): loss 1.52 → 0.198, converging smoothly (`outputs/inspection/training_history.png`). The checkpoint scores IoU 0.90 / Dice 0.95 **on its own training data** — that is a measure of fit, *not* generalization, and is reported only as evidence the model has the capacity to fit this task. The honest generalization estimate comes from cross-validation (§3.4/§3.8), where each model is scored on images it never saw.

An earlier sanity run (20 epochs, 4 images held out) climbed from IoU 0.14 to ~0.60–0.67, against the ~0.19 no-learning baseline from §3.4 — the model is clearly learning, and the fold-to-fold wobble in that range is exactly the small-N variance §3.4 was built to average over.

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
│   ├── dataset.py           # RoofTrainDataset / RoofTestDataset, preprocessing (§3.3)
│   ├── metrics.py           # IoU / Dice
│   ├── cross_validation.py  # Fold splitter, CV runner, paired comparison (§3.4)
│   ├── augmentation.py      # build_train_transform: flips, 90° rotation, color jitter (§3.5)
│   ├── model.py             # build_model: U-Net + pretrained ResNet34 encoder (§3.6)
│   ├── losses.py            # BCE + soft Dice, for class imbalance (§3.7)
│   └── train.py             # train_model, checkpointing, CV glue (§3.7)
├── scripts/
│   ├── inspect_data.py         # Data quality analysis on data_org/ (§3.2)
│   ├── build_data_convert.py   # Build data_convert/ from data_org/ + comparison figure
│   ├── check_dataset.py        # Sanity-check the dataset loader (§3.3) + overlay figure
│   ├── run_cv.py                # Exercise the CV harness with placeholder baselines (§3.4)
│   ├── check_augmentation.py   # Sanity-check the augmentation pipeline (§3.5) + grid figure
│   ├── check_model.py          # Sanity-check the model definition (§3.6) + untrained-prediction figure
│   ├── train.py                 # Model training (§3.7)
│   └── predict.py               # Test-set inference (§3.9)
├── tests/
│   ├── test_smoke.py           # Project setup + dataset-file smoke tests
│   ├── test_dataset.py         # roof_seg.dataset unit tests (§3.3)
│   ├── test_metrics.py         # roof_seg.metrics unit tests
│   ├── test_cross_validation.py # roof_seg.cross_validation unit tests (§3.4)
│   ├── test_run_cv.py          # scripts/run_cv.py baseline integration tests
│   ├── test_augmentation.py    # roof_seg.augmentation unit tests (§3.5)
│   ├── test_model.py           # roof_seg.model unit tests (§3.6)
│   ├── test_losses.py          # roof_seg.losses unit tests (§3.7)
│   └── test_train.py           # roof_seg.train unit tests (§3.7)
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
make run-cv        # cross-validate the real model (§3.4); MODE=baseline for the fast placeholder floor
make compare-augmentation # paired CV: augmentation on vs off, identical folds (slow)
make check-augmentation # sanity-check the augmentation pipeline (§3.5) + grid figure
make check-model   # sanity-check the model definition (§3.6) + untrained-prediction figure
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
| 3.4 Cross-validation harness | Done |
| 3.5 Augmentation | Done |
| 3.6 Model | Done |
| 3.7 Training | Done |
| 3.8 Internal evaluation | Not started |
| 3.9 Test inference | Not started |
| 3.10 Documentation | Not started |
