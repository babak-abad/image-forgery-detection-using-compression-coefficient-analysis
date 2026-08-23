# dct_utils.py
import numpy as np
from scipy.fftpack import dct, idct
from config import BLOCK_SIZE


def blockify(image):
    """Split a 2D image into non-overlapping BxB blocks (vectorised).

    Returns an array of shape (N, BLOCK_SIZE, BLOCK_SIZE) where
    N = (H // B) * (W // B). The image is cropped to the nearest multiple
    of BLOCK_SIZE in both dimensions (incomplete edge blocks are dropped).
    """
    h, w = image.shape
    h = h - (h % BLOCK_SIZE)
    w = w - (w % BLOCK_SIZE)
    image = image[:h, :w]
    n_h = h // BLOCK_SIZE
    n_w = w // BLOCK_SIZE
    # (H, W) -> (n_h, B, n_w, B) -> (n_h, n_w, B, B) -> (N, B, B)
    blocks = image.reshape(n_h, BLOCK_SIZE, n_w, BLOCK_SIZE)
    blocks = blocks.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(blocks.reshape(n_h * n_w, BLOCK_SIZE, BLOCK_SIZE))


def dct_2d_batch(blocks):
    """Batched separable 2D DCT-II (ortho) over (N, B, B) blocks.

    Applies scipy.fftpack.dct along axis=-1 then axis=-2, which is
    mathematically identical to D @ X @ D^T (the standard 2D DCT).
    Replaces the per-block Python loop with a single vectorised call.
    """
    t = dct(blocks, axis=-1, norm='ortho')
    return dct(t, axis=-2, norm='ortho')


def idct_2d_batch(blocks):
    """Batched separable 2D inverse DCT over (N, B, B) blocks."""
    t = idct(blocks, axis=-1, norm='ortho')
    return idct(t, axis=-2, norm='ortho')


def dct_2d(block):
    """2D DCT on a single BxB block (kept for backward compatibility)."""
    return dct_2d_batch(block[None, ...])[0]


def idct_2d(block):
    """Inverse 2D DCT on a single BxB block (kept for backward compatibility)."""
    return idct_2d_batch(block[None, ...])[0]


def quantise(block, q_table):
    """Quantise DCT coefficients (works for single block or batch (N,B,B))."""
    return np.round(block / q_table).astype(np.int32)


def dequantise(q_block, q_table):
    """Dequantise (reconstruct) DCT coefficients."""
    return q_block * q_table