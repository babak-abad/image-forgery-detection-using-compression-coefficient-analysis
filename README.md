# Image forgery detection using compression coefficient analysis

Re-saving a JPEG at a second quality factor leaves a measurable trace: the second
quantisation step superimposes its own divisibility structure on coefficients that were
already quantised once, producing a periodic "comb" pattern in the histograms of the AC
DCT coefficients. This project detects that trace. Every image is tiled into the standard
8×8 JPEG blocks, transformed with the 2-D DCT and quantised; for each of the 63 AC
coefficient positions a 100-bin histogram over [-50, 50] is collected, so each image is
reduced to a 6,300-dimensional **compression fingerprint** in which double compression
shows up as the comb pattern.

The fingerprints are classified by a small PyTorch **MLP** (hidden layers 128 and 64,
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
`MAX_IMAGES` caps the run for quick experiments (0 = all images). Distributed for
non-commercial research use only.

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

- `src/config.py` — all tunable settings (paths, quality factors, histogram, training, patch voting)
- `src/dct_utils.py` — vectorised 8×8 block-wise 2-D DCT/IDCT and quantisation helpers
- `src/quant_tables.py` — standard JPEG luminance quantisation table scaled by quality factor
- `src/build_dataset.py` — single/double JPEG compression simulation and AC-histogram feature extraction
- `src/train_model.py` — feature extraction, MLP training, evaluation and weight saving
- `src/predict.py` — single-image inference from saved weights
- `src/main.py` — end-to-end six-phase pipeline including patch-based image evaluation

Attribution for the dataset and every library is in [RESOURCES.md](RESOURCES.md).
