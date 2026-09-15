# Roof Segmentation — Task Specification

## 1. Goal

Train a neural network to predict **pixel-wise roof masks** from satellite imagery, using **24 labeled image–mask pairs** (25 originally; 278's label was wrong and dropped — see §2), and produce **roof predictions for 6 held-out test images** (5 originally, plus 278). Deliver the predictions together with a concise explanation of the approach suitable for a technical discussion.

### Success criteria (overall)

The task is complete when all of the following are true:

- [ ] Data has been inspected and preprocessing decisions (RGB, label threshold, flagged inconsistencies) are documented.
- [ ] A reproducible training pipeline trains on the 24 labeled pairs without using the 6 test images.
- [ ] Predicted roof masks exist for test images `278`, `535`, `537`, `539`, `551`, and `553`.
- [ ] Predictions are visually plausible roof outlines on the satellite tiles (not blank, not full-frame, not random noise).
- [ ] A short write-up explains model choice, preprocessing, training strategy, and limitations.
- [ ] Code can be run end-to-end by a reviewer with documented dependencies and commands.

### Out of scope

- Instance segmentation (separating individual roofs into distinct IDs).
- Roof type or material classification.
- Object detection with bounding boxes.
- Hyperparameter search at production scale or deployment/inference API.

---

## 2. Data specification

### Inputs

The dataset lives under `data/` in two stages (see [DATA_REPORT.md](DATA_REPORT.md) for the full reasoning):

| Asset | Location | Count | Format |
|---|---|---|---|
| Satellite images, original | `data/data_org/images_RGBA/{id}.png` | 30 | 256×256 RGBA, uint8 |
| Roof labels, original | `data/data_org/labels_org/{id}.png` | 24 (25 originally; `278`'s label deleted, see below) | 256×256 grayscale, uint8 |
| Satellite images, converted (training input) | `data/data_convert/images_RGB/{id}.png` | 30 | 256×256 RGB, uint8 |
| Roof labels, converted (training input) | `data/data_convert/labels_bin_128/{id}.png` | 24 | 256×256 grayscale, uint8, values strictly `{0, 255}` |

`data_convert/` is built from `data_org/` by [`scripts/build_data_convert.py`](scripts/build_data_convert.py) (`make data-convert`) and is what the training pipeline (`roof_seg/dataset.py`) actually reads; `data_org/` is used only for inspection (`scripts/inspect_data.py`).

### Splits

| Split | Image IDs | Count | Purpose |
|---|---|---|---|
| **Train** | All IDs in `data_convert/images_RGB/` with a matching label in `data_convert/labels_bin_128/` | 24 | Model training (and optional internal validation) |
| **Test** | `278`, `535`, `537`, `539`, `551`, `553` | 6 | Final inference only — no labels available |

### Label semantics (`data_org/labels_org/`)

- `0` → background (non-roof)
- `255` → roof interior
- `1–254` → boundary/edge pixels around roof polygons

**Canonical binary mask for training** (already applied in `data_convert/labels_bin_128/`; see [DATA_REPORT.md §5](DATA_REPORT.md#5-label-value-distribution--binarization-rule) for why this was chosen over the alternative):

```python
mask = 255 * (label > 128).astype(np.uint8)
```

### Known data quirks (to be verified in §3.2)

- Images `270` and `278` shared an identical label but different input images. On inspection this is clearly a mistake (almost certainly copy/paste during annotation) — `278`'s label does not correspond to `278`'s image at all. `278`'s label has been deleted and `278` moved into `TEST_IDS`, since training on it would teach the model a wrong mapping and hurt prediction quality. See [DATA_REPORT.md §3](DATA_REPORT.md#3-duplicate--inconsistent-labels--278s-label-is-wrong).
- Test image `535` (and possibly others) may contain non-opaque alpha values (masked/censored regions). Preprocessing must define how alpha is handled.

---

## 3. Subtasks

### 3.1 Project setup

**Objective:** Establish a minimal, reproducible Python environment and project layout.

**Deliverables:**

- Dependency file (`requirements.txt` or equivalent).
- Entry points for training and inference (scripts or notebook).
- Consistent directory conventions (e.g. `outputs/predictions/`, `outputs/checkpoints/`).

**Done when:**

- [x] Dependencies are pinned or version-bounded sufficiently for reproducibility.
- [x] `pip install -r requirements.txt` (or documented equivalent) succeeds on a clean environment.
- [x] Project structure is documented (in README or this spec's run section).
- [x] Random seeds are set and documented for training.

---

### 3.2 Data inspection and quality analysis

**Objective:** Systematically analyze the dataset *before* building the pipeline, document inconsistencies, and make informed preprocessing decisions (especially RGB conversion and label binarization).

**Deliverables:**

- Inspection script or notebook (e.g. `notebooks/data_inspection.ipynb` or `scripts/inspect_data.py`).
- Short findings summary (section in README, `DATA_REPORT.md`, or notebook markdown) that records results and decisions.

**Checks to perform:**

| Area | What to verify |
|---|---|
| **File inventory** | 30 images, 25 labels originally (24 labels after deleting `278`'s); list train vs test IDs; no orphan labels without images |
| **Dimensions** | All images and labels are 256×256; no shape mismatches within pairs |
| **Image–label alignment** | Overlay each pair visually; confirm masks line up with roof structures |
| **Duplicate / inconsistent labels** | Hash or compare label arrays — flag pairs like `270`/`278` where different images share the same mask, or where masks look wrong for the scene |
| **Alpha channel** | Per image: count pixels with `alpha < 255`; classify train vs test; decide if RGB-only conversion is safe or if semi-transparent regions need special handling |
| **Label value distribution** | Histogram of grayscale values; quantify background (`0`), interior (`255`), and edge (`1–254`) pixels; compare effect of `> 0` vs `>= 128` binarization |
| **Class balance** | Roof pixel ratio per image (foreground %); note outliers |
| **Test-set specifics** | Inspect all 6 test images for censorship masks, unusual content, or format differences vs train |

**Done when:**

- [x] File inventory is complete and matches §2 (24 train pairs after dropping `278`'s label, 6 test-only images, no surprises).
- [x] All image–label pairs have matching spatial dimensions.
- [x] At least one grid visualization exists: image \| label \| overlay for a representative sample of train pairs (≥5).
- [x] Duplicate or mismatched label cases are explicitly listed (e.g. `270`/`278`) with a stated impact on training — `278`'s label was wrong and dropped.
- [x] Alpha-channel audit is complete for all 30 images: for each image, `% fully opaque` is recorded; any image with `alpha < 255` is flagged.
- [x] A clear **RGB conversion decision** is documented: either “safe to drop alpha everywhere” or “handle exceptions” with a per-image rule.
- [x] Label binarization rule (`> 0` vs `>= 128`) is chosen based on comparison plots/counts and recorded as the preprocessing default for §3.3.
- [x] Roof foreground ratio per labeled image is tabulated; extreme outliers are noted.
- [x] All 6 test images have been visually inspected and any anomalies (censorship, missing roofs, format quirks) are noted.
- [x] Findings summary is written and linked from the project README.

---

### 3.3 Data loading and preprocessing

**Objective:** Load image–label pairs correctly and apply consistent preprocessing for train and inference, following decisions from §3.2.

**Requirements:**

- Match images to labels by filename stem (e.g. `121.png` ↔ `121.png`).
- Use **RGB** as model input; apply the alpha-handling rule decided in §3.2.
- Normalize images (e.g. ImageNet mean/std or `[0, 1]` scaling — choice must be consistent train/inference).
- Binarize labels using the threshold rule chosen in §3.2.
- Exclude the 6 test IDs from the training dataset loader.
- Apply any dataset exclusions flagged during inspection (e.g. drop mismatched pairs) — must be justified in the findings summary.

**Done when:**

- [x] All 24 train pairs load without error and produce aligned `(image, mask)` tensors of shape `(3, 256, 256)` and `(1, 256, 256)` (or equivalent).
- [x] Test loader returns 6 images with the same spatial preprocessing as training (no label required).
- [x] Alpha-channel handling matches the §3.2 decision and is documented in code.
- [x] Label binarization matches the §3.2 decision.
- [x] A quick sanity check (script or notebook cell) visualizes ≥1 `(image, mask)` overlay confirming alignment.

---

### 3.4 Cross-validation harness

**Objective:** Build a reusable evaluation protocol — k-fold / leave-one-out cross-validation (CV) over the 24 training pairs — so that augmentation strategies, loss functions, model/optimizer choices, and other hyperparameters (§3.5 onward) can be compared against each other on evidence, not guesswork. Built *before* those decisions are finalized, not bolted on after training as an afterthought.

**Why this is its own subtask:** with only 24 labeled images, a single fixed train/val split (e.g. 19/5) gives a high-variance metric estimate — which handful of images land in validation can swing IoU/Dice by more than the effect of whatever is being tested. Cross-validation averages over all 24 images (each one validated exactly once across folds), giving a much more stable estimate and enabling meaningful *paired* comparisons between configurations (same folds, same seed, only one setting changed).

**Requirements:**

- A deterministic fold-splitting utility over the 24 training IDs (built on `roof_seg/config.py::TEST_IDS`-aware train ID list from §3.3), reproducible given `RANDOM_SEED`.
- Support both k-fold (e.g. `--n-folds 6`, default) and leave-one-out (`--n-folds 24`) via the same code path — LOOCV is affordable here given the tiny dataset and fast per-model training, and gives the lowest-variance estimate.
- A runner that, for a given training config, trains one model per fold (that fold held out), evaluates it on the held-out fold, and aggregates the metric (mean ± std across folds) rather than reporting single-fold numbers.
- Must never touch the 6 official `TEST_IDS` — folds are carved only out of the 24 training IDs.
- Usable as a **paired comparison tool**: running it twice with only one setting changed (e.g. augmentation on vs off, one loss variant vs another), using identical fold assignment and seed both times, to isolate that setting's effect from fold-composition noise.

**Done when:**

- [ ] Fold-splitting utility exists, is deterministic (same seed → same folds), and asserts no overlap with `TEST_IDS`.
- [ ] CV runner trains and evaluates across all folds for a given config and reports mean ± std IoU/Dice (not just a single aggregate number with no spread).
- [ ] At least one paired comparison (e.g. augmentation on vs off, §3.5) is run using identical folds/seed, with results recorded in the write-up.
- [ ] CV is documented as the internal evaluation protocol referenced in §3.8, including its limitations for a dataset this small (high variance, non-independent folds, risk of overfitting hyperparameters to only 24 images) — see write-up §3.10.

---

### 3.5 Data augmentation (training)

**Objective:** Expand effective training data to mitigate the small sample size (N=25).

**Minimum augmentation set:**

- Horizontal and vertical flips
- 90° rotations (or arbitrary rotation — document choice)
- Color jitter (brightness/contrast/saturation)

**Done when:**

- [ ] Augmentations apply to **both** image and mask jointly (mask geometry preserved).
- [ ] Augmentations are active during training only, not during validation/inference.
- [ ] Augmentation pipeline is implemented (e.g. Albumentations) and listed in the write-up.
- [ ] Whether augmentation actually helps is checked with the §3.4 CV harness (paired on/off comparison), not assumed.

---

### 3.6 Model definition

**Objective:** Define a segmentation model appropriate for small-data roof segmentation.

**Requirements:**

- Semantic segmentation with a **single output channel** (roof probability per pixel).
- Use a **pretrained encoder** (transfer learning required).
- Allowed: U-Net, FPN, DeepLabV3+, or equivalent from a library such as `segmentation-models-pytorch`.

**Done when:**

- [ ] Model accepts `(B, 3, 256, 256)` and outputs `(B, 1, 256, 256)` logits or probabilities.
- [ ] Pretrained backbone is identified by name (e.g. `resnet34`, `efficientnet-b0`).
- [ ] Architecture choice is justified in the write-up (spatial detail preservation, transfer learning, small-data fit).

---

### 3.7 Training loop

**Objective:** Train the final model on all 24 labeled pairs and persist a usable checkpoint, using the config selected via the §3.4 CV harness.

**Requirements:**

- Loss function suited to imbalanced segmentation (e.g. **BCE + Dice**, or Dice alone) — chosen/confirmed via §3.4 if compared against alternatives.
- Optimizer and learning rate documented.
- Training length defined (epochs and/or early stopping).
- The **final deliverable checkpoint** is trained on all 24 images (no held-out fold) — CV (§3.4) is used to *select* the config beforehand, not to reserve a permanent validation slice from a dataset this small.

**Done when:**

- [ ] Training runs to completion without errors on all 24 train pairs.
- [ ] A model checkpoint is saved to disk.
- [ ] Training loss is logged or plotted.
- [ ] No test-set images (`278`, `535`, `537`, `539`, `551`, `553`) appear in training or in any CV fold.

---

### 3.8 Evaluation (internal)

**Objective:** Quantify model quality on labeled data without touching the official test set, using the §3.4 CV harness as the primary protocol.

**Metrics (at least one):**

- **IoU** (Intersection over Union) on roof class
- **Dice coefficient**

**Done when:**

- [ ] Metric is computed via the §3.4 cross-validation harness across the 24 labels (mean ± std across folds), not a single fixed split.
- [ ] Metric value(s) and evaluation protocol (fold count, LOOCV vs k-fold, seed) are recorded in the write-up.
- [ ] At least one side-by-side visualization exists: input | ground truth | prediction (on a CV validation sample).

---

### 3.9 Inference on test set

**Objective:** Generate final roof predictions for the 6 held-out images.

**Requirements:**

- Load best/final checkpoint (the full-data model from §3.7, not a single CV fold model).
- Apply identical preprocessing as training (except augmentations).
- Post-process: sigmoid → threshold (default 0.5, document if tuned).
- Export binary masks as PNG.

**Done when:**

- [ ] Prediction files exist:
  - `outputs/predictions/278.png`
  - `outputs/predictions/535.png`
  - `outputs/predictions/537.png`
  - `outputs/predictions/539.png`
  - `outputs/predictions/551.png`
  - `outputs/predictions/553.png`
- [ ] Each output is 256×256, grayscale or binary PNG (white = roof, black = background).
- [ ] Optional overlay images exist for visual inspection (recommended).

---

### 3.10 Documentation and delivery

**Objective:** Enable the reviewer to understand and reproduce the work, and provide material for the follow-up discussion.

**Write-up must cover:**

1. Problem framing (semantic segmentation, binary roof mask).
2. Data inspection findings (inconsistencies, alpha/RGB decision, label threshold).
3. Cross-validation protocol (§3.4): fold count, LOOCV vs k-fold, what was compared with it, and its limitations for N=24.
4. Model architecture and pretrained backbone.
5. Why transfer learning and chosen augmentations (with CV evidence, not just intuition).
6. Loss function and training hyperparameters.
7. Internal validation results (CV metric mean ± std + qualitative examples).
8. Known limitations (dataset size, duplicate label pair, no test ground truth, CV variance/non-independence at this scale).

**Done when:**

- [ ] README (or `REPORT.md`) contains setup, train, and inference commands.
- [ ] Write-up addresses all eight points above (including data inspection summary).
- [ ] A reviewer can reproduce predictions by following the documented steps.
- [ ] Deliverable bundle is ready to send: prediction PNGs + write-up (+ code/repo link).

---

## 4. Acceptance checklist (final review)

Before submission, confirm:

| # | Check | Pass |
|---|---|---|
| 1 | Data inspection completed; findings documented (alpha, label quirks, binarization rule) | ☑ |
| 2 | 24 train pairs used (`278`'s wrong label dropped); 6 test images never seen during training or CV | ☐ |
| 3 | Cross-validation harness built and used for at least one paired comparison (e.g. augmentation on/off) | ☐ |
| 4 | Pretrained segmentation model with documented architecture | ☐ |
| 5 | Data augmentation applied during training, validated via CV rather than assumed | ☐ |
| 6 | Labels binarized consistently (per inspection decision) | ☐ |
| 7 | Checkpoint saved and reloadable | ☐ |
| 8 | 6 prediction PNGs exported | ☐ |
| 9 | Internal validation metric (CV mean ± std) or qualitative eval documented | ☐ |
| 10 | Write-up explains *what* and *why* | ☐ |
| 11 | End-to-end reproducible from documented commands | ☐ |

---

## 5. Suggested execution order

```
3.1 Project setup
  → 3.2 Data inspection        ← decisions feed into everything below
    → 3.3 Data loading
      → 3.4 Cross-validation harness   ← built early so it can test 3.5 onward
        → 3.5 Augmentation             ← tested with the 3.4 harness (paired on/off)
          → 3.6 Model
            → 3.7 Training (final, full-data)
              → 3.8 Internal evaluation      ← reports the 3.4 CV results
                → 3.9 Test inference
                  → 3.10 Documentation
```

---

## 6. Open decisions (resolve during implementation)

Record the chosen option in the write-up when decided:

| Decision | Options | Recommendation | Decide in |
|---|---|---|---|
| Label threshold | `> 0` vs `> 128` | **Decided: `> 128`** (revised from an earlier `> 0` default) — see [DATA_REPORT.md §5](DATA_REPORT.md#5-label-value-distribution--binarization-rule) | §3.2 |
| Alpha handling | Drop vs ignore-mask vs composite | **Decided: drop alpha, RGB only** — see [DATA_REPORT.md §4](DATA_REPORT.md#4-alpha-channel-audit) | §3.2 |
| Mismatched pairs | Keep / drop / relabel | **Decided: drop `278`'s label** (deleted; image moved into `TEST_IDS`) — see [DATA_REPORT.md §3](DATA_REPORT.md#3-duplicate--inconsistent-labels--278s-label-is-wrong) | §3.2 |
| Validation strategy | k-fold CV vs LOOCV vs fixed hold-out | **Decided: cross-validation (k-fold or LOOCV) via the §3.4 harness**, not a fixed hold-out — a single ~5-image split is too high-variance at N=24 to trust for comparisons | §3.4 |
| Model | U-Net vs DeepLabV3+ | U-Net + ResNet34 via `smp` | §3.6 |
| Loss | BCE, Dice, BCE+Dice | BCE + Dice — confirm via §3.4 CV if compared against Dice-only | §3.7 |
| Threshold | 0.5 vs tuned on val | 0.5 default; tune if CV suggests otherwise | §3.9 |
