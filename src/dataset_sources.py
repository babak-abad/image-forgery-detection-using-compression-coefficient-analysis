# dataset_sources.py
"""Multi-source training-sample generation for compression-history detection.

Each source folder declares how many JPEG compressions are already baked into
its images (the "prior k" of their pixel history):

  NO_COMPRESS_FOLDERS     k = 0   -> can synthesize BOTH classes
  SINGLE_COMPRESS_FOLDERS k = 1   -> originals are label 0; adding >= 1 round -> label 1
  MULTI_COMPRESS_FOLDERS  k >= 2  -> always label 1 (used as-is)

Current pools: uncompressed BMPs (archive.org scans) plus the VISION dataset
(https://lesc.dinfo.unifi.it/VISION/, CC BY-SA 4.0, fetched by
src/download_vision.py): images/nat/ native phone JPEGs are the
SINGLE_COMPRESS pool, images/natWA/ WhatsApp re-encodes of the SAME photos
are the MULTI_COMPRESS pool. A photo and its nat/natWA twin share one
split group (see source_group_key), so neither twin can leak across
train/val/test.

Label rule: a sample is 1 (multi-compressed) iff (k + applied rounds) >= 2.
Compression count is monotone: rounds can only be added, never removed, so a
label can never flip from 1 back to 0.

Every simulated compression round is a REAL JPEG encode/decode in memory
(PIL/libjpeg), so the artifacts match real re-saved files. The first pass on
pristine pixels draws a uniform quality from COMPRESS_Q_RANGE; any FOLLOW-UP
pass (and the first pass of a derived sample) is drawn so that the
double-quantisation comb is detectable: a share of Q2_ASCEND_SHARE passes
lands ABOVE the previous quality (finer second grid, strong comb), the rest
descend with a step ratio inside Q2_STEP_RATIO_RANGE (the still-detectable
band). Consecutive rounds (and a derived first pass vs the source file's
estimated quality) never share a quality, because re-compression at the SAME
quality leaves almost no detectable trace.

Augmentation per sample: one random grid-aligned crop (TRAIN_PATCH_SIZE) and
one random 90-degree-multiple rotation, applied BEFORE any simulated
compression for never-compressed sources (so all orientations occur
compressed) and AFTER decoding for already-compressed sources.
"""
import io
import os
import re


import numpy as np
from PIL import Image

from config import (
    BLOCK_SIZE,
    N_AC_COEFFS,
    N_HIST_BINS,
    RANDOM_STATE,
    NO_COMPRESS_FOLDERS,
    SINGLE_COMPRESS_FOLDERS,
    MULTI_COMPRESS_FOLDERS,
    COMPRESS_Q_RANGE,
    FORCE_DISTINCT_Q,
    Q2_ASCEND_SHARE,
    Q2_STEP_RATIO_RANGE,
    Q2_MAX_QUALITY,
    TRIPLE_COMPRESS_PROB,
    TRAIN_PATCH_SIZE,
    DQ_HIST_BINS,
    DQ_HIST_RANGE,
    RES_HIST_BINS,
    RES_HIST_RANGE,
    SAMPLES_UNCOMPRESSED_PER_UNCOMPRESSED,
    SAMPLES_SINGLE_PER_UNCOMPRESSED,
    SAMPLES_DOUBLE_PER_UNCOMPRESSED,
    SAMPLES_ORIGINAL_PER_COMPRESSED,
    SAMPLES_DOUBLE_PER_COMPRESSED,
    ANALYSIS_QF,
)
from dct_utils import blockify, dct_2d_batch, quantise
from quant_tables import STD_LUM_QT, get_quant_table
from build_dataset import _histograms_numpy, HIST_LO, HIST_HI

IMG_EXTS_UNCOMPRESSED = (".bmp", ".png", ".tif", ".tiff")
IMG_EXTS_COMPRESSED = (".jpg", ".jpeg")


def list_source_images(folders, exts):
    """Recursively collect sorted image paths under the given folders."""
    paths = []
    for folder in folders:
        if not folder:
            continue
        if not os.path.isdir(folder):
            print(f"  warning: source folder not found, skipping: {folder}")
            continue
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(exts):
                    paths.append(os.path.join(root, name))
    return sorted(paths)


VISION_FILE_RE = re.compile(
    r"^(D\d{2})_I_(?:nat|natWA)_(\d{4})\.jpe?g$", re.IGNORECASE
)


def source_group_key(path):
    """Split-group key of a source file.

    VISION photos are keyed by (device, index) so the nat and natWA twins of
    one photo (and everything derived from them) share one group_id and can
    never land in different train/val/test splits. All other files key by
    their full path.
    """
    match = VISION_FILE_RE.match(os.path.basename(path))
    if match:
        return f"vision/{match.group(1).upper()}/{match.group(2)}"
    return os.path.normcase(path)


def build_source_tasks():
    """One task per source image, each with a deterministic RNG seed.

    Tasks that share a split-group key (VISION nat/natWA twins) share one
    group_id.
    """
    tasks = []
    group_of_key = {}
    for source_type, folders, exts in (
        ("no_compress", NO_COMPRESS_FOLDERS, IMG_EXTS_UNCOMPRESSED),
        ("single_compress", SINGLE_COMPRESS_FOLDERS, IMG_EXTS_COMPRESSED),
        ("multi_compress", MULTI_COMPRESS_FOLDERS, IMG_EXTS_COMPRESSED),
    ):
        for path in list_source_images(folders, exts):
            key = source_group_key(path)
            group_id = group_of_key.setdefault(key, len(group_of_key))
            tasks.append(
                {
                    "path": path,
                    "source_type": source_type,
                    "group_id": group_id,
                    "seed": RANDOM_STATE + 1000 + group_id,
                }
            )
    return tasks


def load_gray(path):
    """Decode any supported image to a 2-D grayscale (Y) pixel array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def grid_crop(pixels):
    """Crop to a multiple of BLOCK_SIZE so the 8x8 grid stays aligned."""
    h, w = pixels.shape
    return pixels[: h - (h % BLOCK_SIZE), : w - (w % BLOCK_SIZE)]


def random_patch(pixels, rng, patch_size=TRAIN_PATCH_SIZE):
    """Random crop at an 8-pixel-aligned offset (grid-aligned patch)."""
    h, w = pixels.shape
    ph = min(patch_size, h)
    pw = min(patch_size, w)
    ph -= ph % BLOCK_SIZE
    pw -= pw % BLOCK_SIZE
    y = int(rng.integers(0, (h - ph) // BLOCK_SIZE + 1)) * BLOCK_SIZE
    x = int(rng.integers(0, (w - pw) // BLOCK_SIZE + 1)) * BLOCK_SIZE
    return pixels[y : y + ph, x : x + pw]


def random_rot90(pixels, rng):
    """Lossless 90-degree-multiple rotation (keeps the 8x8 grid)."""
    k = int(rng.integers(0, 4))
    if k == 0:
        return pixels
    return np.ascontiguousarray(np.rot90(pixels, k))


def jpeg_round(pixels, quality):
    """One REAL JPEG compression round: encode + decode in memory."""
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("L"))


def draw_quality(rng, avoid=None):
    """Draw a random quality, optionally forcing it to differ from `avoid`.

    Capped at Q2_MAX_QUALITY: near-lossless passes (q > ~95) leave no
    detectable compression trace, so they are never simulated.
    """
    lo, hi = COMPRESS_Q_RANGE
    hi = min(hi, Q2_MAX_QUALITY)
    q = int(rng.integers(lo, hi + 1))
    if FORCE_DISTINCT_Q and avoid is not None:
        while q == avoid:
            q = int(rng.integers(lo, hi + 1))
    return q


def _libjpeg_scale(q):
    """libjpeg luminance-table scale factor of quality q (inverse pair below)."""
    return 5000.0 / q if q < 50 else 200.0 - 2.0 * q


def _q_for_scale(s):
    """Quality whose libjpeg scale is s (inverse of _libjpeg_scale)."""
    return 5000.0 / s if s >= 100.0 else (200.0 - s) / 2.0


def draw_followup_quality(rng, q_prev):
    """Draw the quality of a re-compression pass that FOLLOWS a pass at q_prev.

    A step ratio is drawn from Q2_STEP_RATIO_RANGE and applied in the drawn
    direction: Q2_ASCEND_SHARE of passes make the grid FINER (q2 > q_prev --
    the strong-comb regime, periodic gaps between the last lattice's
    integers), the rest COARSER (realistic downward re-save, still inside the
    detectable band; much-coarser second grids are statistically near-identical
    to single compression and would only inject label noise).

    Both directions are clipped into [COMPRESS_Q_RANGE[0], Q2_MAX_QUALITY]
    (near-lossless qualities leave no trace). Results always differ from
    q_prev (strictly above or below before clipping, nudged after).
    """
    lo, hi = COMPRESS_Q_RANGE
    hi = min(hi, Q2_MAX_QUALITY)
    scale_prev = _libjpeg_scale(q_prev)
    rho = float(rng.uniform(Q2_STEP_RATIO_RANGE[0], Q2_STEP_RATIO_RANGE[1]))
    ascending = rng.random() < Q2_ASCEND_SHARE
    if scale_prev <= 1e-6:
        # Degenerate (near-lossless) previous quality: any real change works.
        q = int(rng.integers(lo, hi + 1))
    else:
        target = scale_prev / rho if ascending else scale_prev * rho
        q = int(round(_q_for_scale(target)))
        q = min(hi, max(lo, q))
    if FORCE_DISTINCT_Q and q == q_prev:
        q = q_prev - 1 if q_prev > lo else q_prev + 1
    return q


def apply_compress(pixels, rng, times, avoid_first=None):
    """Apply `times` real JPEG rounds, mostly at DESCENDING qualities.

    Returns (pixels, q_chain) where q_chain[i] is the quality of pass i+1.
    `avoid_first` is a quality to skip on the first pass (e.g. the estimated
    quality of the source file, guaranteeing q_new != q_source).

    Pass 0 on pristine pixels (avoid_first is None) draws uniformly. Any
    follow-up pass, and pass 0 of a re-save whose source quality is known
    (avoid_first given), is drawn via draw_followup_quality: a
    Q2_ASCEND_SHARE share of finer (higher-quality) re-saves with a strong
    comb, the rest descending inside the detectable step-ratio band.
    """
    out = pixels
    q_chain = []
    q_prev = avoid_first
    for i in range(times):
        if i == 0 and avoid_first is None:
            q = draw_quality(rng)
        else:
            q = draw_followup_quality(rng, q_prev=q_prev)
        out = jpeg_round(out, q)
        q_chain.append(q)
        q_prev = q
    return out, q_chain


def estimate_qf(path):
    """Estimate the quality a JPEG was saved at, from its quant table.

    Inverts the libjpeg quality scaling of the standard luminance table and
    takes the per-entry median. Returns None when no table is available.
    """
    try:
        with Image.open(path) as im:
            tables = getattr(im, "quantization", None)
            if not tables:
                return None
            table = np.asarray(tables[0], dtype=np.float64)
    except Exception:
        return None
    if table.size != 64:
        return None
    base = STD_LUM_QT.astype(np.float64).ravel()
    scale = table * 100.0 / base
    q = np.where(scale >= 100.0, (200.0 - scale) / 2.0, 5000.0 / scale)
    q = np.clip(q, 1.0, 100.0)
    return int(round(float(np.median(q))))


def load_quant_table(path):
    """Raster-order 8x8 luminance quantisation table of a JPEG file (or None).

    PIL exposes tables in raster order and they match get_quant_table for
    libjpeg-encoded files; real camera/messenger files may deviate, and the
    DQ-residue features must use the ACTUAL table so that genuine singles
    keep a clean zero-fraction spike.
    """
    try:
        with Image.open(path) as im:
            tables = getattr(im, "quantization", None)
            if not tables:
                return None
            table = np.asarray(tables[0], dtype=np.float64)
    except Exception:
        return None
    if table.size != BLOCK_SIZE * BLOCK_SIZE:
        return None
    return table.reshape(BLOCK_SIZE, BLOCK_SIZE)


def features_from_pixels(
    pixels, analysis_qf=ANALYSIS_QF, dq_qf=None, dq_table=None
):
    """AC-coefficient fingerprint: raw histograms + stats + DQ-residue features.

    Raw DCT coefficients by default (ANALYSIS_QF = None); an integer q
    quantises with table(q) first, matching the legacy pipeline.

    Feature blocks (concatenated):
      1. per-AC-position density histograms of the coefficients
         (N_AC_COEFFS * N_HIST_BINS dims);
      2. per-position statistics -- mean, std, zero-ratio (|c| < 0.5),
         kurtosis -- capturing lattice concentration and tails
         (N_AC_COEFFS * 4 dims);
      3. DQ histograms (N_AC_COEFFS * DQ_HIST_BINS dims): coefficients
         divided by the steps of the LAST JPEG pass, given either as a
         quality (`dq_qf`, standard table -- exact for PIL-synthesised
         chains) or as the file's actual table (`dq_table`, 8x8 raster);
      4. residue histograms (N_AC_COEFFS * RES_HIST_BINS dims): the
         fractional part of the dequantised values. Single compression
         concentrates it at 0; any earlier pass at a different quality
         spreads it periodically over [-0.5, 0.5] -- a quality-invariant
         double-compression signature.

    dq_qf=None and dq_table=None (never-compressed input, no table
    available) produce zero blocks for 3 and 4, identically at training and
    inference time.
    """
    blocks = blockify(grid_crop(pixels))
    d = dct_2d_batch(blocks.astype(np.float64))
    if analysis_qf is not None:
        d = quantise(d, get_quant_table(analysis_qf)).astype(np.float64)
    flat = d.reshape(d.shape[0], BLOCK_SIZE * BLOCK_SIZE)
    ac = np.ascontiguousarray(flat[:, 1:])

    out = np.zeros((N_AC_COEFFS, N_HIST_BINS), dtype=np.float64)
    _histograms_numpy(ac, HIST_LO, HIST_HI, N_HIST_BINS, out)
    parts = [out.reshape(-1)]

    mean = ac.mean(axis=0)
    std = ac.std(axis=0)
    zero_ratio = (np.abs(ac) < 0.5).mean(axis=0)
    var = std ** 2
    m4 = ((ac - mean) ** 4).mean(axis=0)
    kurtosis = np.where(var > 1e-12, m4 / np.maximum(var ** 2, 1e-24), 3.0)
    parts.append(
        np.concatenate([mean, std, zero_ratio, kurtosis]).astype(np.float64)
    )

    if dq_table is None and dq_qf is not None:
        dq_table = get_quant_table(dq_qf).astype(np.float64)
    steps = None
    if dq_table is not None:
        # Per-AC-position steps (raster order, DC skipped).
        steps = np.asarray(dq_table, dtype=np.float64).ravel()[1:]

    dq_out = np.zeros((N_AC_COEFFS, DQ_HIST_BINS), dtype=np.float64)
    res_out = np.zeros((N_AC_COEFFS, RES_HIST_BINS), dtype=np.float64)
    if steps is not None:
        # UNROUNDED division keeps the sub-integer comb structure.
        dq = ac / steps
        _histograms_numpy(dq, DQ_HIST_RANGE[0], DQ_HIST_RANGE[1], DQ_HIST_BINS, dq_out)
        # Fractional part of the dequantised values: the DQ residue.
        residue = dq - np.round(dq)
        _histograms_numpy(
            residue, RES_HIST_RANGE[0], RES_HIST_RANGE[1], RES_HIST_BINS, res_out
        )
    parts.append(dq_out.reshape(-1))
    parts.append(res_out.reshape(-1))

    return np.concatenate(parts)


def features_for_file(path, analysis_qf=ANALYSIS_QF):
    """Whole-image feature vector for inference (no rotation, no patching).

    The DQ blocks use the file's own quantization table (falling back to the
    standard table of the estimated quality), exactly like the training
    samples of real files.
    """
    dq_table = load_quant_table(path)
    dq_qf = None if dq_table is not None else estimate_qf(path)
    return features_from_pixels(
        grid_crop(load_gray(path)), analysis_qf, dq_qf=dq_qf, dq_table=dq_table
    )


def process_source(task):
    """Worker: generate all training samples for one source image.

    Returns a list of sample dicts (features/label/kind/source_type/group_id/
    q_chain/q_prior/path/sample_idx), or an empty list when the file is
    unreadable.
    """
    rng = np.random.default_rng(task["seed"])
    path = task["path"]
    source_type = task["source_type"]
    group_id = task["group_id"]
    samples = []

    def add(pixels, label, kind, q_chain=None, q_prior=None, dq_qf=None, dq_table=None):
        # DQ-residue blocks keyed on the LAST pass: known from the synthesis
        # chain for synthetic/derived samples; the file's actual table for
        # as-is real samples.
        if dq_qf is None and dq_table is None and q_chain:
            dq_qf = q_chain[-1]
        samples.append(
            {
                "features": features_from_pixels(
                    pixels, dq_qf=dq_qf, dq_table=dq_table
                ).astype(np.float32),
                "label": label,
                "kind": kind,
                "source_type": source_type,
                "group_id": group_id,
                "q_chain": q_chain,
                "q_prior": q_prior,
                "path": path,
                "sample_idx": len(samples),
            }
        )

    try:
        if source_type == "no_compress":
            base = grid_crop(load_gray(path))
            for _ in range(SAMPLES_UNCOMPRESSED_PER_UNCOMPRESSED):
                patch = random_rot90(random_patch(base, rng), rng)
                add(patch, 0, "uncompressed")
            for _ in range(SAMPLES_SINGLE_PER_UNCOMPRESSED):
                patch = random_rot90(random_patch(base, rng), rng)
                px, qs = apply_compress(patch, rng, 1)
                add(px, 0, "synthetic_single", qs)
            for _ in range(SAMPLES_DOUBLE_PER_UNCOMPRESSED):
                patch = random_rot90(random_patch(base, rng), rng)
                times = 3 if rng.random() < TRIPLE_COMPRESS_PROB else 2
                px, qs = apply_compress(patch, rng, times)
                add(px, 1, "synthetic_multi", qs)
        elif source_type == "single_compress":
            base = grid_crop(load_gray(path))
            q1_est = estimate_qf(path)
            q1_table = load_quant_table(path)
            for _ in range(SAMPLES_ORIGINAL_PER_COMPRESSED):
                patch = random_rot90(random_patch(base, rng), rng)
                add(patch, 0, "real_single", dq_table=q1_table)
            for _ in range(SAMPLES_DOUBLE_PER_COMPRESSED):
                patch = random_rot90(random_patch(base, rng), rng)
                times = 2 if rng.random() < TRIPLE_COMPRESS_PROB else 1
                px, qs = apply_compress(
                    pixels=patch, rng=rng, times=times, avoid_first=q1_est
                )
                add(px, 1, "derived_multi", qs, q_prior=q1_est)
        elif source_type == "multi_compress":
            base = grid_crop(load_gray(path))
            q_last_table = load_quant_table(path)
            for _ in range(SAMPLES_ORIGINAL_PER_COMPRESSED):
                patch = random_rot90(random_patch(base, rng), rng)
                add(patch, 1, "real_multi", dq_table=q_last_table)
        else:
            raise ValueError(f"unknown source_type: {source_type}")
    except Exception as exc:
        print(f"  skipping {path}: {exc}")
        return []
    return samples
