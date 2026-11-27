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

imgs = list_source_images(NO_COMPRESS_FOLDERS, IMG_EXTS_UNCOMPRESSED)[:30]
rng = np.random.default_rng(5)
patches = [random_patch(grid_crop(load_gray(p)), rng) for p in imgs]

OFF = 6300 + 252 + 2520


def res_hist(px, q):
    feats = features_from_pixels(px, dq_qf=q)
    return feats[OFF:OFF + 630].reshape(63, 10)


def near_zero(hist):
    return hist[:, 4:6].sum() / hist.sum()


def run(name, q2, q1):
    scores = []
    for patch in patches:
        px = jpeg_round(patch, q2) if q1 is None else jpeg_round(jpeg_round(patch, q1), q2)
        scores.append(near_zero(res_hist(px, q2)))
    print(f"{name:15s} near-zero fraction: {np.mean(scores):.3f}")


run("single q=35", 35, None)
run("single q=75", 75, None)
run("single q=90", 90, None)
run("double 90->75", 75, 90)
run("double 90->45", 45, 90)
run("double 60->75", 75, 60)
run("double 75->60", 60, 75)
