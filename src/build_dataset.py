# build_dataset.py
"""Feature-extraction utilities for DCT-histogram compression analysis.

This module exposes `extract_hist_features()` and `process_image()`. It is a
pure library: it does NOT read or write any dataset files. `train_model.py`
and `predict.py` call these helpers to compute feature vectors on-the-fly.

Performance notes
-----------------
* DCT/quantise run batched over all 8x8 blocks of an image in a single
  vectorised call (`dct_2d_batch`) instead of a per-block Python loop.
* The 63 per-position histograms are computed in a single vectorised
  `np.bincount` pass over `(N, 63)` integer coefficients, replacing the
  previous 63 separate `np.histogram` calls.
"""
import numpy as np
from PIL import Image

from config import N_HIST_BINS, N_AC_COEFFS, BLOCK_SIZE
from dct_utils import blockify, dct_2d_batch, quantise, dequantise
from quant_tables import get_quant_table

# Histogram range used for the per-coefficient density histograms.
HIST_LO = -50.0
HIST_HI = 50.0

# Lazily-built albumentations rotation transform (see _rotate_random_90).
_ROTATE_90 = None


def _rotate_random_90(img):
    """Rotate a 2D grayscale image by a random multiple of 90 degrees.

    Uses albumentations RandomRotate90, a lossless pixel permutation
    (np.rot90 under the hood), so no interpolation is introduced and the
    8x8 DCT block grid stays aligned with the image. albumentations is
    imported lazily so inference-only callers (predict.py) never load it
    or its dependencies.
    """
    global _ROTATE_90
    if _ROTATE_90 is None:
        import albumentations as A
        _ROTATE_90 = A.RandomRotate90()
    rotated = _ROTATE_90(image=img[:, :, np.newaxis])["image"]
    return rotated[:, :, 0]


def _histograms_numpy(ac, lo, hi, n_bins, out):
    """Compute per-position density histograms for AC coefficients.

    Parameters
    ----------
    ac      : (N, K) int32 quantised AC coefficients (K = N_AC_COEFFS).
    lo, hi  : float, histogram range. Values outside [lo, hi] are ignored;
              the last bin is closed on both ends (hi maps to the last bin),
              matching `np.histogram(..., range=(lo, hi))`.
    n_bins  : int, number of histogram bins.
    out     : (K, n_bins) float64 output buffer, zeroed by the caller.

    The result is `density=True` (count / (total * bin_width)), matching the
    previous `np.histogram(..., density=True)` semantics exactly. Computed in
    a single vectorised pass via `np.bincount`.
    """
    width = (hi - lo) / n_bins
    n_coeffs = ac.shape[1]

    # Keep only coefficients inside [lo, hi] (last bin is closed on both ends).
    mask = (ac >= lo) & (ac <= hi)

    # Bin index per coefficient, clipped to [0, n_bins - 1].
    b = np.floor((ac - lo) / width).astype(np.int64)
    np.clip(b, 0, n_bins - 1, out=b)

    # Offset each coefficient column by its own block so a single `bincount`
    # over the flattened (N, K) array produces all K histograms at once.
    offsets = (np.arange(n_coeffs) * n_bins).astype(np.int64)
    flat = (b + offsets)[mask]

    counts = np.bincount(flat, minlength=n_coeffs * n_bins)
    counts = counts.reshape(n_coeffs, n_bins)
    total = counts.sum(axis=1)

    nz = total > 0
    out[nz] = counts[nz] / (total[nz, None] * width)
    # leave pre-zeroed rows (total == 0) untouched
    return out


def extract_hist_features(blocks, qf):
    """Extract histogram feature vector for a set of 8x8 blocks.

    `blocks` is expected to be the (already quantised) coefficient blocks of
    shape (N, 8, 8). For backwards-compatible semantics with the original
    implementation, the blocks are DCT-ed and quantised again here before the
    histograms are built.

    Returns a 1D array of length N_AC_COEFFS * N_HIST_BINS.
    """
    q_table = get_quant_table(qf)

    # Batched DCT + quantise over all blocks (vectorised, single call).
    dct_blocks = dct_2d_batch(blocks.astype(np.float64))
    q_blocks = quantise(dct_blocks, q_table)

    # AC coefficients in raster order skipping the DC at flat index 0.
    # Original code: q_block[0, 1:].tolist() + q_block[1:, :].flatten().tolist()
    # which is exactly flat indices 1..63 for an 8x8 block.
    flat = q_blocks.reshape(q_blocks.shape[0], BLOCK_SIZE * BLOCK_SIZE)
    ac = np.ascontiguousarray(flat[:, 1:], dtype=np.int32)

    out = np.zeros((N_AC_COEFFS, N_HIST_BINS), dtype=np.float64)
    _histograms_numpy(ac, HIST_LO, HIST_HI, N_HIST_BINS, out)
    return out.reshape(-1)


def process_image(img_path, label, qf, qf2=None, rotate=False):
    """Load an image, produce a compressed version (single or double),
    and return its feature vector.

    `label` is accepted for API compatibility but unused (labels are assigned
    by the caller).

    When `rotate` is True, a random 90-degree-multiple rotation (albumentations
    RandomRotate90) is applied to the loaded image before cropping and DCT
    analysis. Training-time augmentation only; inference callers (predict.py,
    main.py) keep the default False. Each call draws an independent rotation,
    so the single and double samples of one image may end up differently
    rotated (each remains a valid sample of a rotated source image).

    Compression handling depends on the input format:
      * JPEG inputs (`.jpg`/`.jpeg`, any case) are ALREADY single-compressed
        (the first compression happened when the file was saved), so the first
        compression step is NEVER re-applied here. For the single class no
        additional compression is performed; for the double class ONLY the
        second compression (`qf2`) is applied.
      * Non-JPEG inputs (png/bmp/tiff/...) are uncompressed, so the first
        compression (`qf`) is applied (single), and the second (`qf2`) is
        applied on top (double), as before.
    """
    # Load grayscale (Y channel), optionally rotate (lossless 90-degree
    # multiples; a 90-degree turn swaps H and W so the crop runs AFTER the
    # rotation), and crop to a multiple of BLOCK_SIZE.
    img = Image.open(img_path).convert('L')
    img = np.array(img)
    if rotate:
        img = _rotate_random_90(img)
    h, w = img.shape
    h = h - (h % BLOCK_SIZE)
    w = w - (w % BLOCK_SIZE)
    img = img[:h, :w]

    # Vectorised: split into (N, 8, 8) blocks and run one batched DCT.
    blocks = blockify(img)
    dct_blocks = dct_2d_batch(blocks.astype(np.float64))

    # JPEG files already encode a first compression, so never re-apply it.
    is_jpeg = img_path.lower().endswith((".jpg", ".jpeg"))

    if is_jpeg:
        if qf2 is None:  # SINGLE class: no additional compression
            compressed_blocks = dct_blocks
        else:            # DOUBLE class: apply ONLY the second compression
            q_table2 = get_quant_table(qf2)
            compressed_blocks = quantise(dct_blocks, q_table2)
    else:
        # Non-JPEG input: apply the first (and optionally second) compression.
        q_table1 = get_quant_table(qf)
        if qf2 is None:  # SINGLE compression
            compressed_blocks = quantise(dct_blocks, q_table1)
        else:            # DOUBLE compression
            q_table2 = get_quant_table(qf2)
            q_b = quantise(dct_blocks, q_table1)
            dct_rec = dequantise(q_b, q_table1)   # dequantise
            compressed_blocks = quantise(dct_rec, q_table2)  # requantise

    # Extract histogram features from the quantised coefficients.
    return extract_hist_features(
        compressed_blocks, qf if qf2 is None else qf2
    )