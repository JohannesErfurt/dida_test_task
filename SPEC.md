# Roof Segmentation — Task Specification

## 1. Goal

Train a neural network to predict **pixel-wise roof masks** from satellite imagery, using **25 labeled image–mask pairs**, and produce **roof predictions for 5 held-out test images**. Deliver the predictions together with a concise explanation of the approach suitable for a technical discussion.

### Success criteria (overall)

The task is complete when all of the following are true:

- [ ] Data has been inspected and preprocessing decisions (RGB, label threshold, flagged inconsistencies) are documented.
- [ ] A reproducible training pipeline trains on the 25 labeled pairs without using the 5 test images.
- [ ] Predicted roof masks exist for test images `535`, `537`, `539`, `551`, and `553`.
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

| Asset | Location | Count | Format |
|---|---|---|---|
| Satellite images | `images/{id}.png` | 30 | 256×256 RGBA, uint8 |
| Roof labels | `labels/{id}.png` | 25 | 256×256 grayscale, uint8 |

### Splits

| Split | Image IDs | Count | Purpose |
|---|---|---|---|
| **Train** | All IDs with a matching label | 25 | Model training (and optional internal validation) |
| **Test** | `535`, `537`, `539`, `551`, `553` | 5 | Final inference only — no labels available |

### Label semantics

- `0` → background (non-roof)
- `255` → roof interior
- `1–254` → boundary/edge pixels around roof polygons

**Canonical binary mask for training:**

```python
mask = (label > 0).astype(np.float32)
```

Alternative (`label >= 128`) may be used if documented; both should be evaluated on a validation subset and the chosen rule stated in the write-up.

### Known data quirks (to be verified in §3.2)

- Images `270` and `278` may share an identical label but different input images.
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
| **File inventory** | 30 images, 25 labels; list train vs test IDs; no orphan labels without images |
| **Dimensions** | All images and labels are 256×256; no shape mismatches within pairs |
| **Image–label alignment** | Overlay each pair visually; confirm masks line up with roof structures |
| **Duplicate / inconsistent labels** | Hash or compare label arrays — flag pairs like `270`/`278` where different images share the same mask, or where masks look wrong for the scene |
| **Alpha channel** | Per image: count pixels with `alpha < 255`; classify train vs test; decide if RGB-only conversion is safe or if semi-transparent regions need special handling |
| **Label value distribution** | Histogram of grayscale values; quantify background (`0`), interior (`255`), and edge (`1–254`) pixels; compare effect of `> 0` vs `>= 128` binarization |
| **Class balance** | Roof pixel ratio per image (foreground %); note outliers |
| **Test-set specifics** | Inspect all 5 test images for censorship masks, unusual content, or format differences vs train |

**Done when:**

- [ ] File inventory is complete and matches §2 (25 train pairs, 5 test-only images, no surprises).
- [ ] All image–label pairs have matching spatial dimensions.
- [ ] At least one grid visualization exists: image \| label \| overlay for a representative sample of train pairs (≥5).
- [ ] Duplicate or mismatched label cases are explicitly listed (e.g. `270`/`278`) with a stated impact on training (keep both, drop one, etc.).
- [ ] Alpha-channel audit is complete for all 30 images: for each image, `% fully opaque` is recorded; any image with `alpha < 255` is flagged.
- [ ] A clear **RGB conversion decision** is documented: either “safe to drop alpha everywhere” or “handle exceptions” with a per-image rule.
- [ ] Label binarization rule (`> 0` vs `>= 128`) is chosen based on comparison plots/counts and recorded as the preprocessing default for §3.3.
- [ ] Roof foreground ratio per labeled image is tabulated; extreme outliers are noted.
- [ ] All 5 test images have been visually inspected and any anomalies (censorship, missing roofs, format quirks) are noted.
- [ ] Findings summary is written and linked from the project README.

---

### 3.3 Data loading and preprocessing

**Objective:** Load image–label pairs correctly and apply consistent preprocessing for train and inference, following decisions from §3.2.

**Requirements:**

- Match images to labels by filename stem (e.g. `121.png` ↔ `121.png`).
- Use **RGB** as model input; apply the alpha-handling rule decided in §3.2.
- Normalize images (e.g. ImageNet mean/std or `[0, 1]` scaling — choice must be consistent train/inference).
- Binarize labels using the threshold rule chosen in §3.2.
- Exclude the 5 test IDs from the training dataset loader.
- Apply any dataset exclusions flagged during inspection (e.g. drop mismatched pairs) — must be justified in the findings summary.

**Done when:**

- [ ] All 25 train pairs load without error and produce aligned `(image, mask)` tensors of shape `(3, 256, 256)` and `(1, 256, 256)` (or equivalent).
- [ ] Test loader returns 5 images with the same spatial preprocessing as training (no label required).
- [ ] Alpha-channel handling matches the §3.2 decision and is documented in code.
- [ ] Label binarization matches the §3.2 decision.
- [ ] A quick sanity check (script or notebook cell) visualizes ≥1 `(image, mask)` overlay confirming alignment.

---

### 3.4 Data augmentation (training)

**Objective:** Expand effective training data to mitigate the small sample size (N=25).

**Minimum augmentation set:**

- Horizontal and vertical flips
- 90° rotations (or arbitrary rotation — document choice)
- Color jitter (brightness/contrast/saturation)

**Done when:**

- [ ] Augmentations apply to **both** image and mask jointly (mask geometry preserved).
- [ ] Augmentations are active during training only, not during validation/inference.
- [ ] Augmentation pipeline is implemented (e.g. Albumentations) and listed in the write-up.

---

### 3.5 Model definition

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

### 3.6 Training loop

**Objective:** Train the model on the 25 labeled pairs and persist a usable checkpoint.

**Requirements:**

- Loss function suited to imbalanced segmentation (e.g. **BCE + Dice**, or Dice alone).
- Optimizer and learning rate documented.
- Training length defined (epochs and/or early stopping).
- Optional but recommended: hold out a small validation subset from the 25 (e.g. 4–5 images) for monitoring — **not** the 5 official test images.

**Done when:**

- [ ] Training runs to completion without errors on all 25 train pairs.
- [ ] A model checkpoint is saved to disk.
- [ ] Training loss (and validation metric, if used) is logged or plotted.
- [ ] No test-set images (`535`, `537`, `539`, `551`, `553`) appear in training or validation splits.
- [ ] Validation metric (IoU or Dice) on the internal hold-out is reported, if a hold-out is used.

---

### 3.7 Evaluation (internal)

**Objective:** Quantify model quality on labeled data without touching the official test set.

**Metrics (at least one):**

- **IoU** (Intersection over Union) on roof class
- **Dice coefficient**

**Done when:**

- [ ] At least one metric is computed on an internal validation subset or via cross-validation on the 25 labels.
- [ ] Metric value(s) and evaluation protocol are recorded in the write-up.
- [ ] At least one side-by-side visualization exists: input | ground truth | prediction (on a validation sample).

---

### 3.8 Inference on test set

**Objective:** Generate final roof predictions for the 5 held-out images.

**Requirements:**

- Load best/final checkpoint.
- Apply identical preprocessing as training (except augmentations).
- Post-process: sigmoid → threshold (default 0.5, document if tuned).
- Export binary masks as PNG.

**Done when:**

- [ ] Prediction files exist:
  - `outputs/predictions/535.png`
  - `outputs/predictions/537.png`
  - `outputs/predictions/539.png`
  - `outputs/predictions/551.png`
  - `outputs/predictions/553.png`
- [ ] Each output is 256×256, grayscale or binary PNG (white = roof, black = background).
- [ ] Optional overlay images exist for visual inspection (recommended).

---

### 3.9 Documentation and delivery

**Objective:** Enable the reviewer to understand and reproduce the work, and provide material for the follow-up discussion.

**Write-up must cover:**

1. Problem framing (semantic segmentation, binary roof mask).
2. Data inspection findings (inconsistencies, alpha/RGB decision, label threshold).
3. Model architecture and pretrained backbone.
4. Why transfer learning and chosen augmentations.
5. Loss function and training hyperparameters.
6. Internal validation results (metric + qualitative examples).
7. Known limitations (dataset size, duplicate label pair, no test ground truth).

**Done when:**

- [ ] README (or `REPORT.md`) contains setup, train, and inference commands.
- [ ] Write-up addresses all seven points above (including data inspection summary).
- [ ] A reviewer can reproduce predictions by following the documented steps.
- [ ] Deliverable bundle is ready to send: prediction PNGs + write-up (+ code/repo link).

---

## 4. Acceptance checklist (final review)

Before submission, confirm:

| # | Check | Pass |
|---|---|---|
| 1 | Data inspection completed; findings documented (alpha, label quirks, binarization rule) | ☐ |
| 2 | 25 train pairs used; 5 test images never seen during training | ☐ |
| 3 | Pretrained segmentation model with documented architecture | ☐ |
| 4 | Data augmentation applied during training | ☐ |
| 5 | Labels binarized consistently (per inspection decision) | ☐ |
| 6 | Checkpoint saved and reloadable | ☐ |
| 7 | 5 prediction PNGs exported | ☐ |
| 8 | Internal validation metric or qualitative eval documented | ☐ |
| 9 | Write-up explains *what* and *why* | ☐ |
| 10 | End-to-end reproducible from documented commands | ☐ |

---

## 5. Suggested execution order

```
3.1 Project setup
  → 3.2 Data inspection        ← decisions feed into everything below
    → 3.3 Data loading
      → 3.4 Augmentation
        → 3.5 Model
          → 3.6 Training
            → 3.7 Internal evaluation
              → 3.8 Test inference
                → 3.9 Documentation
```

---

## 6. Open decisions (resolve during implementation)

Record the chosen option in the write-up when decided:

| Decision | Options | Recommendation | Decide in |
|---|---|---|---|
| Label threshold | `> 0` vs `>= 128` | Compare on overlays; default `> 0` | §3.2 |
| Alpha handling | Drop vs ignore-mask vs composite | Audit all 30 images first | §3.2 |
| Mismatched pairs | Keep / drop / relabel | Document case-by-case (e.g. `270`/`278`) | §3.2 |
| Validation split | 5-fold CV vs fixed 20/5 hold-out | Fixed hold-out for speed | §3.6 |
| Model | U-Net vs DeepLabV3+ | U-Net + ResNet34 via `smp` | §3.5 |
| Loss | BCE, Dice, BCE+Dice | BCE + Dice | §3.6 |
| Threshold | 0.5 vs tuned on val | 0.5 default; tune if val set exists | §3.8 |
