# Image forgery detection using compression coefficient analysis

Re-saving a JPEG at a second quality factor leaves a measurable trace: the second
quantisation step superimposes its own divisibility structure on coefficients that were
already quantised once, producing a periodic "comb" pattern in the histograms of the AC
DCT coefficients. This project detects that trace. Every image is tiled into the standard
8×8 JPEG blocks, transformed with the 2-D DCT and quantised; for each of the 63 AC
coefficient positions a 100-bin histogram over [-50, 50] is collected, so each image is
reduced to a 6,300-dimensional **compression fingerprint** in which double compression
shows up as the comb pattern.

The fingerprints are classified by a small PyTorch **MLP** (hidden layers 512, 256 and 64 with dropout 0.3,
BCEWithLogitsLoss, Adam with early stopping on validation accuracy). Whole-image verdicts
use **patch voting**: the image is divided into 16×16-block (128×128 px) patches, each
patch is classified independently, and the image is flagged double-compressed when at
least half of the patches vote so — which also localises regions whose compression
history differs from the rest of the image.

## Dataset

**LV-MHP v1.0** — the NUS Learning and Vision (LV) Group Multiple-Human Parsing
benchmark, 4,980 multi-person photographs (Li, Zhao et al., 2017,
https://arxiv.org/abs/1705.07206). Only the photos are used, as raw material; the parsing
annotations are irrelevant here. Download the dataset and point `DATA_DIR` in
`src/config.py` at its image folder:

```
LV-MHP-v1/images/*.jpg      <- DATA_DIR = "D:/projects/datasets/LV-MHP-v1/images"
```

Any folder of ordinary images works as well. Training discovers
`.jpg/.jpeg/.png/.bmp/.tif/.tiff` directly under `DATA_DIR` and synthesises two samples
per image: a single-compressed copy (quality `QF_SINGLE` = 90) and a double-compressed
copy (two quality factors drawn per image from the `QF_FIRST`/`QF_SECOND` ranges). JPEG
inputs keep their original first compression; non-JPEG inputs get a fully simulated one.
`MAX_IMAGES` caps the run for quick experiments (0 = all images). As augmentation,
each sample is additionally rotated by a random multiple of 90° — a lossless pixel
permutation (albumentations `RandomRotate90`) that keeps the 8×8 DCT grid and the
comb pattern intact; toggle it with `ROTATION_AUGMENT` in `src/config.py`.
Distributed for non-commercial research use only.

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv/Scripts/python src/train_model.py            # Linux/macOS: .venv/bin/python
.venv/Scripts/python src/predict.py path/to/image.jpg
.venv/Scripts/python src/main.py
```

Run everything from the project root: the `weights/` and `assets/` output folders are
relative paths. `train_model.py` extracts features on the fly, trains the MLP and writes
`weights/mlp_model_best.pt`, `weights/mlp_model_last.pt` and `weights/scaler_stats.pt`,
plus diagnostic figures to `assets/`. `predict.py` classifies a single image from the
saved weights (train first); `main.py` runs the full six-phase pipeline — sample
creation, comb-pattern visualisation, training, model diagnostics, patch-based
image-level evaluation and image-level figures. All settings — dataset path, quality
factors, histogram bins, MLP shape, patch voting — live in `src/config.py`.

## Layout

- `src/config.py` — all tunable settings (paths, quality factors, rotation augmentation, histogram, training, patch voting)
- `src/dct_utils.py` — vectorised 8×8 block-wise 2-D DCT/IDCT and quantisation helpers
- `src/quant_tables.py` — standard JPEG luminance quantisation table scaled by quality factor
- `src/build_dataset.py` — single/double JPEG compression simulation and AC-histogram feature extraction
- `src/train_model.py` — feature extraction, MLP training, evaluation and weight saving
- `src/predict.py` — single-image inference from saved weights
- `src/main.py` — end-to-end six-phase pipeline including patch-based image evaluation

Attribution for the dataset and every library is in [RESOURCES.md](RESOURCES.md).

---

# Conversation & design log (Cline session, 2026-11-25)

> Preserved from the working chat that produced the current multi-source
> pipeline: the requested spec, every design decision and its reason, the
> feature-design probe, and the retraining results (end of section).

## 1. Goal

Retrain the JPEG compression-history detector on realistic multi-source data so
it survives real-world re-saves (phone photo -> app download -> re-save) instead
of recognizing only the synthetic in-memory pairs of the first version.

## 2. Diagnosis of the old pipeline (from the earlier analysis)

Three failure layers:

1. **Simulated "compression".** `build_dataset.process_image` never writes a
   real JPEG: it quantises DCT coefficients in NumPy (round at q1 ->
   dequantise -> round at q2). Real JPEGs are decoded by libjpeg's IDCT,
   chroma is subsampled and the decoded pixels land on the encoder's 8x8 grid
   - none of which the simulation reproduces, so the model learned to detect
   the simulation, not real double compression.
2. **Self-quantisation with a known table.** Features were built by quantising
   the analysis DCT with the same table logic as the sample generator (a fixed
   q=75 at inference), leaking generator knowledge and mismatching inference.
3. **Leaky split.** Samples were split per-sample with `stratify=y`, so the
   single- and double-compressed versions of the SAME source image could land
   in train and test - the reported ~0.99 accuracy measured content
   memorization, not generalization.

## 3. Data inventory

| Pool | Files | Assumed history | Role |
|---|---|---|---|
| `bmp_sources/archive_org` | 2,004 BMPs | k = 0 (never JPEG'd) | synthesize both classes with REAL JPEG rounds |
| `Photos_downloaded_by_AirDroid_2` | 949 JPEGs | k = 1 (phone saved once) | real label-0 singles + derived label-1 doubles |
| (future) multi-compressed folder | - | k >= 2 | real label-1, used as-is |

## 4. Requested training specification (user's words, condensed)

- Build a "3 folders" pipeline: never-compressed / compressed once /
  compressed multiple times.
- No-compress folder: create training versions with 1 and 2 (or more)
  compressions.
- Single-compress folder: train both with re-compressed versions AND use the
  originals as-is.
- Multi-compress folder: use as-is, count the original compressions.
- Label = double-compressed iff (source compressions + applied compressions)
  is at least 2.
- Every compression step is a REAL JPEG encode/decode (in-memory, PIL), each
  with a random quality; probe the feature design first, then train, log the
  whole chat into README.md and produce a results HTML in `doc/`.

## 5. Feature-design probe (120 BMPs, real PIL JPEG rounds, 2026-11-25)

Logistic-regression probe on 357 samples (119 source groups, 20% group-level
test) comparing raw DCT histograms vs quantise-at-75 vs their concatenation:

| Variant | Overall acc | uncompressed | single | double |
|---|---|---|---|---|
| raw DCT histograms | **0.722** | 1.000 | 0.458 | 0.708 |
| quantise@75        | 0.625 | 0.833 | 0.583 | 0.458 |
| concat (raw + q75) | 0.681 | 1.000 | 0.417 | 0.625 |

Decisions: `ANALYSIS_QF = None` (raw) for training AND inference - a fixed
analysis table destroys information. The single-vs-double boundary is hard
for a linear probe at this scale; the full MLP with ~15.8k samples does the
real work. Uncompressed vs JPEG separates perfectly, confirming the BMP pool
is genuinely never-compressed.

## 6. Q&A decisions (asked by Cline, answered by user)

### 6.1 Role of the single-compressed pool (AirDroid)

Use BOTH ways: the originals serve as real label-0 samples, AND one re-encode
(fresh random q != estimated source q, 25% chance of a second re-encode)
derives real label-1 doubles from them.

### 6.2 Rotation: before or after compression

- BMP (never-compressed) sources: rotate BEFORE compressing, so all 4
  orientations also occur compressed.
- Already-compressed sources: rotate AFTER decoding (harmless for real
  histories, gives augmentation parity).
Both are grid-preserving 90-degree multiples (no interpolation).

### 6.3 Counting rule for multi-compressed sources

Label = 1 iff (prior compressions k of the source + simulated rounds t) >= 2:

| Source (k) | Simulated t | k + t | Label |
|---|---|---|---|
| 0 (BMP) | 0 | 0 | 0 (uncompressed reference) |
| 0 (BMP) | 1 | 1 | 0 |
| 0 (BMP) | 2-3 | 2-3 | 1 |
| 1 (AirDroid) | 0 | 1 | 0 |
| 1 (AirDroid) | 1-2 | 2-3 | 1 |
| >= 2 (multi folder) | 0 | >= 2 | 1 (used as-is, no re-encode) |

Monotone by construction (rounds only add, labels can never flip 1 -> 0).
`MULTI_COMPRESS_FOLDERS = []` for this run; when a real multi-compressed
folder is supplied, its images are used as-is for label 1 and ground truth
rests on folder provenance.

## 7. Implemented pipeline (this retraining)

- `src/config.py` - new multi-source keys: `NO_COMPRESS_FOLDERS`,
  `SINGLE_COMPRESS_FOLDERS`, `MULTI_COMPRESS_FOLDERS`,
  `COMPRESS_Q_RANGE = (30, 100)`, `FORCE_DISTINCT_Q = True`,
  `TRIPLE_COMPRESS_PROB = 0.25`, `TRAIN_PATCH_SIZE = 256`, per-source sample
  counts, `VAL_SIZE = 0.1`, `ANALYSIS_QF = None`, `DOC_DIR = "doc"`;
  `NUM_WORKERS` 16 -> 4 (Windows spawn RAM).
- `src/dataset_sources.py` (new) - `apply_compress` (REAL in-memory
  encode/decode per round, distinct random q per round, first round avoids
  the source file's estimated quality via `estimate_qf`), grid-aligned random
  256-px patches, lossless rot90 augmentation, the k + t labeling, and the
  parallel worker entry point `process_source`.
- `src/train_model.py` - multi-source dataset build (8 feature workers),
  group-level 70/10/20 split by source image (zero content leakage), the
  unchanged MLP / training loop / checkpoint format, plus a self-contained
  HTML report `doc/training_results.html` (dataset composition, config,
  curves, confusion matrix, ROC, per-kind and per-QF-bucket accuracy, figures
  embedded as base64).
- `src/predict.py` - inference now uses the matching raw-feature path
  (`features_for_file`), no more assumed q=75.
- Dataset: 2,004 BMPs x 6 samples + 949 JPEGs x 4 samples = **15,820 samples,
  exactly 50/50** (7,910 single-class / 7,910 multi-class).
- Known boundary kept out of scope: aligned double compression (same quality
  twice) is near-undetectable by design; the distinct-q rule avoids training
  on it.

## 9. Port sync: ai-existence-check (2026-11-25)

The retrained weights and the raw-feature path were synced into the
`ai-existence-check` service (`src/pipeline/double_compression.py`):

- `models/double_compression_mlp.pt` + `models/double_compression_scaler.pt`
  replaced (previous ones kept in
  `backups/double_compression_*_pre_retrain.pt`).
- Feature extraction switched to raw DCT AC histograms; `qf_extract` is kept
  for API compatibility but ignored. The MLP class gained dropout 0.3 to
  match the new checkpoint layout.
- Spot check: Telegram-re-encoded watch photo p(double)=0.87; never-compressed
  PNG p(double)=0.02 (the old checkpoint saturated at ~1.0 for every input).
- Port test suite: 11/11 passed.
- Full benchmark (1,993 pairs / 3,986 JPEGs, same archive.org BMP sources,
  split by source image; results in
  `outputs/double_compression_bench_mlp_retrained/` of the port repo):

| Model | Thr | TN | FP | FN | TP | Recall | Spec | Acc | AUC | med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| old (pre-retrain) | 0.5 | 0 | 1794 | 0 | 1794 | 1.000 | 0.000 | 0.500 | 0.500 | 40 |
| retrained | 0.5 | 917 | 877 | 370 | 1424 | 0.794 | 0.511 | 0.652 | 0.739 | 30 |
| retrained | cal 0.555 | 1131 | 663 | 550 | 1244 | 0.693 | 0.630 | 0.662 | 0.739 | 30 |

The old model was a constant "double" predictor. The bench is exactly the
hardest class pair (BMP-synthesized singles vs doubles, random q in 30-100
without the distinct-q rule), and the 0.652 matches the internal blind-test
estimate for that pair (synthetic_single 0.45 / synthetic_multi 0.85).

## 10. DQ-residue features + detectable re-save chains + threshold calibration (2026-11-27)

Baseline run before this change: blind-test acc 0.8092, F1(double) 0.8280,
AUC 0.9022, with synthetic_single accuracy stuck at 0.522 (742 test FPs).

Diagnosis (probes in `logs/probe_*.py`): the generator drew q2 uniformly, so
~50% of doubles were descending re-saves with step ratios >= 3, which are
statistically near-identical to single compression (between-class DQ
histogram distance 0.2-0.4x the noise floor). Training on those pairs taught
the MLP the spurious shortcut "coarse last lattice -> double", which misfires
on low-quality genuine singles. Ascending re-saves (finer second grid) are
the strongly detectable regime (2.0x).

Changes:

- `src/config.py`: `SAMPLES_SINGLE_PER_UNCOMPRESSED` 2 -> 3 (synthetic
  classes now 1:1), new `Q2_ASCEND_SHARE = 0.5`, `Q2_STEP_RATIO_RANGE =
  (1.2, 2.5)` (follow-up passes draw a step RATIO, applied ascending or
  descending, so pairs always sit in the detectable band), and
  `Q2_MAX_QUALITY = 95` (near-lossless passes leave no trace). New feature
  keys `DQ_HIST_BINS/DQ_HIST_RANGE` and `RES_HIST_BINS/RES_HIST_RANGE`.
- `src/dataset_sources.py`: `draw_followup_quality` (ratio-based follow-up
  quality), feature vector 6300 -> 9702 dims: raw histograms + per-position
  stats (mean/std/zero-ratio/kurtosis) + DQ histograms (coefficients divided
  by the LAST pass steps -- the classic double-quantisation comb, exact for
  PIL-synthesised chains, the file's ACTUAL table for real files via new
  `load_quant_table`) + fractional-part residue histograms.
- `src/train_model.py`: `calibrate_threshold` sweeps the decision threshold
  on the VALIDATION split for max F1(double) after training and persists it
  as `decision_threshold` in both checkpoints; the blind-test evaluation
  uses it; `q_bucket_table` now reports last-pass ordering (re-save lower vs
  higher) and last-quality buckets; samples carry `q_prior` (estimated
  source quality of derived samples).
- `src/predict.py`: applies the checkpoint's calibrated threshold (falls
  back to `PATCH_VOTE_THRESHOLD` for legacy checkpoints).

Result (same trainer, MLP, split; `doc/training_results.html`,
`logs/train_v5.log`): blind-test **acc 0.8859, F1(double) 0.8816, AUC
0.9542** (threshold 0.520). Per kind: synthetic_single 0.804 (was 0.522),
synthetic_multi 0.880 (ascending re-saves 0.974), derived_multi 0.874,
real_single 0.999, real_multi 0.971, uncompressed 1.000. Remaining known
hard regime: descending re-saves (0.78-0.86) -- moderate step ratios are
detectable but weak; much-coarser re-saves and aligned (integer-ratio)
double compression remain near-undetectable by design. Baseline report and
weights preserved in `doc/training_results_baseline.html`,
`doc/training_results_v4.html`, and `weights_baseline/`.
