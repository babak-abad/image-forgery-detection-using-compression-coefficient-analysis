import sys

sys.path.insert(0, "src")
import numpy as np
import torch

from dataset_sources import (
    list_source_images,
    load_gray,
    grid_crop,
    random_patch,
    jpeg_round,
    features_from_pixels,
    IMG_EXTS_UNCOMPRESSED,
)
from config import NO_COMPRESS_FOLDERS
from train_model import MLP

ckpt = torch.load("weights/mlp_model_best.pt", map_location="cpu", weights_only=True)
stats = torch.load("weights/scaler_stats.pt", map_location="cpu", weights_only=True)
model = MLP(ckpt["input_dim"], ckpt["hidden_layers"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
mean = stats["mean"].squeeze(0)
std = stats["std"].squeeze(0)


def proba(feat):
    x = torch.from_numpy(feat.astype(np.float32))
    with torch.no_grad():
        logit = model(((x - mean) / std).unsqueeze(0))
        return torch.sigmoid(logit).item()


def scale_of(q):
    """libjpeg luminance-table scale factor for quality q."""
    return 5000.0 / q if q < 50 else 200.0 - 2.0 * q


def q_for_scale(s):
    return 5000.0 / s if s >= 100.0 else (200.0 - s) / 2.0


imgs = list_source_images(NO_COMPRESS_FOLDERS, IMG_EXTS_UNCOMPRESSED)[:40]
rng = np.random.default_rng(123)
patches = [random_patch(grid_crop(load_gray(p)), rng) for p in imgs]

print("=== synthetic SINGLES (label 0) at fixed q ===")
for q in (35, 50, 65, 80, 95):
    ps = [proba(features_from_pixels(jpeg_round(patch, q), dq_qf=q)) for patch in patches]
    print(
        f"single q={q:>3}  mean p={np.mean(ps):.3f}  frac>=0.5={np.mean(np.array(ps) >= 0.5):.2f}"
    )

print("=== synthetic DOUBLES (label 1) from q1=90 at step ratio rho ===")
for rho in (1.2, 1.5, 2.0, 2.5, 3.5, 5.5):
    ps = []
    q1 = 90
    q2 = int(round(min(100, max(30, q_for_scale(rho * scale_of(q1))))))
    if q2 == q1:
        q2 -= 1
    for patch in patches:
        px = jpeg_round(jpeg_round(patch, q1), q2)
        ps.append(proba(features_from_pixels(px, dq_qf=q2)))
    print(
        f"double q1=90 rho={rho:.1f} (q2={q2:>3})  mean p={np.mean(ps):.3f}  "
        f"frac>=0.5={np.mean(np.array(ps) >= 0.5):.2f}"
    )
