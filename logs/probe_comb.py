import sys

sys.path.insert(0, "src")
import numpy as np

from dataset_sources import (
    load_gray,
    grid_crop,
    random_patch,
    jpeg_round,
    features_from_pixels,
    list_source_images,
    IMG_EXTS_UNCOMPRESSED,
)
from config import NO_COMPRESS_FOLDERS

imgs = list_source_images(NO_COMPRESS_FOLDERS, IMG_EXTS_UNCOMPRESSED)[:60]
rng = np.random.default_rng(11)
patches = [random_patch(grid_crop(load_gray(p)), rng) for p in imgs]

DQ_OFF = 6300 + 252


def dq_hist(px, q):
    feats = features_from_pixels(px, dq_qf=q)
    return feats[DQ_OFF:DQ_OFF + 2520].reshape(63, 40)


def l1(h_a, h_b):
    return float(np.abs(h_a - h_b).sum(axis=1).mean())


def compare(q2, q1):
    singles = [dq_hist(jpeg_round(patch, q2), q2) for patch in patches]
    doubles = [
        dq_hist(jpeg_round(jpeg_round(patch, q1), q2), q2) for patch in patches
    ]
    h_s = np.mean(singles, axis=0)
    h_d = np.mean(doubles, axis=0)
    n = len(singles) // 2
    within_s = l1(np.mean(singles[:n], axis=0), np.mean(singles[n:], axis=0))
    within_d = l1(np.mean(doubles[:n], axis=0), np.mean(doubles[n:], axis=0))
    between = l1(h_s, h_d)
    print(
        f"q1={q1} q2={q2}: between-class L1={between:.3f}  "
        f"within-class L1: single={within_s:.3f} double={within_d:.3f}  "
        f"ratio={between / max(within_s, within_d):.1f}x"
    )


compare(50, 90)
compare(75, 90)
compare(75, 60)
compare(65, 90)
compare(85, 90)
