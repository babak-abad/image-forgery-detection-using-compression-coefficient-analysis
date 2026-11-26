# predict.py
"""Inference entry-point: classify a single image as single- or multi-compressed.

Loads the PyTorch MLP (`mlp_model_best.pt`, falling back to `mlp_model_last.pt`)
and its normalization stats (`scaler_stats.pt`) saved by `train_model.py`, and
applies them to a RAW DCT AC-histogram feature vector extracted from the whole
image (features_for_file in dataset_sources.py, ANALYSIS_QF = None) - exactly
the feature path used at training time, with no assumed quality factor.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from config import PATCH_VOTE_THRESHOLD, WEIGHTS_DIR, MLP_DROPOUT
from dataset_sources import features_for_file


# Match the MLP definition in train_model.py. Kept self-contained so predict.py
# does not import training-side code/dependencies.
class MLP(nn.Module):
    """Simple feed-forward binary classifier matching MLP_HIDDEN_LAYERS."""

    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(MLP_DROPOUT))
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict_image(img_path):
    """Predict whether `img_path` is single- or multi-compressed.

    Returns:
        (pred, proba): pred is 0 (single) or 1 (multi); proba is a length-2
        numpy array [P(single), P(multi)] for API parity with the old sklearn
        `predict_proba` output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"PyTorch is running on {device.type.upper()} ({_dev_name})")

    # Load model + reconstruct architecture from saved metadata.
    # Prefer best (lowest val loss) checkpoint; fall back to last if absent.
    best_path = os.path.join(WEIGHTS_DIR, "mlp_model_best.pt")
    last_path = os.path.join(WEIGHTS_DIR, "mlp_model_last.pt")
    if os.path.exists(best_path):
        model_path = best_path
    elif os.path.exists(last_path):
        model_path = last_path
    else:
        raise FileNotFoundError(
            f"No model weights found in {WEIGHTS_DIR}. "
            "Run train_model.py first."
        )
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = MLP(ckpt["input_dim"], ckpt["hidden_layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load normalization stats (mean, std over training set)
    stats = torch.load(
        os.path.join(WEIGHTS_DIR, "scaler_stats.pt"),
        map_location=device,
        weights_only=True,
    )
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)

    # Raw DCT histogram features, identical to the training feature path.
    feats = features_for_file(img_path)
    x = torch.from_numpy(feats.astype(np.float32)).to(device)

    with torch.no_grad():
        x = (x - mean.squeeze(0)) / std.squeeze(0)
        logit = model(x.unsqueeze(0))
        p_double = torch.sigmoid(logit).item()

    pred = 1 if p_double >= PATCH_VOTE_THRESHOLD else 0
    proba = np.array([1.0 - p_double, p_double], dtype=np.float32)

    label = "single-compressed" if pred == 0 else "multi-compressed"
    print(f"Prediction: {label}  (confidence: {max(proba):.3f})")
    return pred, proba


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py /path/to/suspect.jpg")
    else:
        predict_image(sys.argv[1])
