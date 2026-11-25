# train_model.py
"""Train a PyTorch MLP for single-vs-double JPEG compression detection.

There is NO "max iterations" knob: each epoch is a full pass over the training
set in BATCH_SIZE mini-batches (iterations_per_epoch = ceil(n_train / BATCH_SIZE)),
and training continues epoch-by-epoch up to MLP_NUM_EPOCHS, with early stopping
(MLP_PATIENCE, counted in epochs) as the practical stop condition.
"""
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless script runs
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init
from concurrent.futures import ProcessPoolExecutor, as_completed

colorama_init()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)

from config import (
    DATA_DIR,
    ASSETS_DIR,
    WEIGHTS_DIR,
    QF_SINGLE,
    QF_FIRST,
    QF_SECOND,
    MAX_IMAGES,
    TEST_SIZE,
    RANDOM_STATE,
    MLP_HIDDEN_LAYERS,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_FEATURE_WORKERS,
    MLP_LEARNING_RATE,
    MLP_NUM_EPOCHS,
    MLP_EARLY_STOPPING,
    MLP_PATIENCE,
    ROTATION_AUGMENT,
)
from build_dataset import process_image


def _process_one_image(path):
    """Worker callable for ProcessPoolExecutor (must be module-level on Windows).

    Returns ((feats_single, 0), (feats_double, 1)) or None on failure.

    QF_FIRST and QF_SECOND are [min, max] ranges; a random quality factor is
    drawn from each range per image to diversify double-compression artifacts.
    When ROTATION_AUGMENT is True, each sample additionally receives an
    independent random 90-degree-multiple rotation (albumentations).
    """
    try:
        feats_single = process_image(
            path, label=0, qf=QF_SINGLE, rotate=ROTATION_AUGMENT
        )
        qf_first = random.randint(QF_FIRST[0], QF_FIRST[1])
        qf_second = random.randint(QF_SECOND[0], QF_SECOND[1])
        feats_double = process_image(
            path, label=1, qf=qf_first, qf2=qf_second, rotate=ROTATION_AUGMENT
        )
        return feats_single, 0, feats_double, 1
    except Exception as exc:  # skip unreadable / corrupt images
        print(f"  skipping {path}: {exc}")
        return None

# Image extensions accepted as dataset inputs
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Device: use CUDA if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU diagnostics + cuDNN tuning
_dev_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
print(f"PyTorch is running on {DEVICE.type.upper()} ({_dev_name})")
torch.backends.cudnn.deterministic = True   # reproducible kernels
torch.backends.cudnn.benchmark = True       # autotune for fixed input size


class MLP(nn.Module):
    """Simple feed-forward binary classifier matching MLP_HIDDEN_LAYERS."""

    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        # Single logit output for BCEWithLogitsLoss (positive class = double)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def list_images(data_dir):
    """Return all image paths directly under data_dir (no subfolders)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {data_dir}. "
            "Update DATA_DIR in config.py to point at your image folder."
        )
    paths = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.lower().endswith(IMG_EXTS)
    ]
    if not paths:
        raise FileNotFoundError(
            f"No images ({', '.join(IMG_EXTS)}) found in DATA_DIR: {data_dir}"
        )
    return paths


def save_all_graphs(assets_dir, loss_curve, val_scores, y_test, y_pred, y_proba,
                    val_loss_curve=None):
    """Generate and save all diagnostic figures to the assets directory."""
    os.makedirs(assets_dir, exist_ok=True)

    # 1) Loss curves (training + validation, per epoch)
    plt.figure()
    plt.plot(loss_curve, label="training loss")
    if val_loss_curve is not None and len(val_loss_curve) > 0:
        plt.plot(val_loss_curve, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # 2) Validation score curve (available when early_stopping=True)
    if len(val_scores) > 0:
        plt.figure()
        plt.plot(val_scores, label="validation accuracy", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("MLP Validation Scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(assets_dir, "validation_scores.png"), dpi=150)
        plt.close()

    # 3) Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar(label="count")
    classes = ["single", "double"]
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 4) ROC curve + AUC (binary classification, use proba of the positive class)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "roc_curve.png"), dpi=150)
    plt.close()

    # 5) Per-class precision / recall / F1 bar chart
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    x = np.arange(len(classes))
    width = 0.25
    plt.figure()
    plt.bar(x - width, p, width, label="precision")
    plt.bar(x, r, width, label="recall")
    plt.bar(x + width, f, width, label="f1-score")
    plt.xticks(x, classes)
    plt.ylim(0, 1.05)
    plt.title("Per-class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "precision_recall_bar.png"), dpi=150)
    plt.close()


def _to_device_normalized(X_chunk, mean, std):
    """Normalize a CPU tensor and move it to DEVICE (async when CUDA)."""
    x = (X_chunk - mean) / std
    return x.to(DEVICE, non_blocking=True)


@torch.no_grad()
def evaluate(model, X, y, mean, std, device=DEVICE, batch_size=512):
    """Return (predictions, probabilities) for evaluation tensors X, y."""
    model.eval()
    preds, probs = [], []
    for i in range(0, X.shape[0], batch_size):
        xb = _to_device_normalized(X[i:i + batch_size], mean, std)
        logits = model(xb).cpu()
        p = torch.sigmoid(logits)
        preds.append((p >= 0.5).long())
        probs.append(p)
    preds = torch.cat(preds).numpy()
    probs = torch.cat(probs).numpy()
    return preds, probs


@torch.no_grad()
def validation_accuracy(model, X_val, y_val, mean, std, device=DEVICE):
    """Quick validation accuracy (val set is small)."""
    model.eval()
    xb = _to_device_normalized(X_val, mean, std)
    logits = model(xb).cpu()
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return (preds == y_val).float().mean().item()


@torch.no_grad()
def validation_metrics(model, X_val, y_val, mean, std, criterion):
    """Return (val_loss, val_acc, precision, recall) on the validation set."""
    model.eval()
    xb = _to_device_normalized(X_val, mean, std)
    logits = model(xb).cpu()
    val_loss = criterion(logits, y_val).item()
    preds = (torch.sigmoid(logits) >= 0.5).long()
    y_true = y_val.long()
    val_acc = (preds == y_true).float().mean().item()
    # Binary precision/recall (positive class = double compressed)
    tp = ((preds == 1) & (y_true == 1)).sum().item()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return val_loss, val_acc, precision, recall


def train_loop(model, X_tr, y_tr, X_val, y_val, mean, std, input_dim):
    """Train `model` epoch by epoch with manual early stopping.

    iterations_per_epoch = ceil(n_train / BATCH_SIZE). There is no global
    iteration cap; the outer loop runs up to MLP_NUM_EPOCHS and stops early
    (MLP_PATIENCE epochs without val-accuracy improvement) when
    MLP_EARLY_STOPPING is True.

    Data stays on the CPU (normalized lazily per batch); worker processes
    prefetch batches and pin them so GPU transfer overlaps with compute.
    """
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

    # Keep data on CPU; workers prefetch + pin_memory enables async H2D copy.
    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    iters_per_epoch = int(np.ceil(len(ds) / BATCH_SIZE))
    n_epochs = MLP_NUM_EPOCHS
    has_val = X_val is not None

    # Ultralytics-style column header
    header = f"{Fore.CYAN}{'Epoch':>10}  {'GPU_mem':>8}  {'train_loss':>12}"
    if has_val:
        header += f"  {'val_loss':>10}  {'val_acc':>8}  {'precision':>10}  {'recall':>8}"
    header += f"{Style.RESET_ALL}"
    print(header)
    print(
        f"Training: {len(ds)} samples, BATCH_SIZE={BATCH_SIZE} -> "
        f"{iters_per_epoch} iterations/epoch, max {n_epochs} epochs."
    )

    # Ensure weights dir exists for best/last checkpoints
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    loss_curve = []
    val_loss_curve = []
    val_scores = []
    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    no_improve = 0
    # Patience is counted in EPOCHS (one full pass over the training set).
    patience = MLP_PATIENCE

    for epoch in range(n_epochs):
        model.train()
        running_loss, n_seen = 0.0, 0

        # GPU memory snapshot (ultralytics-style)
        gpu_mem = (
            torch.cuda.memory_reserved() / 1e9 if DEVICE.type == "cuda" else 0.0
        )
        epoch_label = f"{epoch + 1}/{n_epochs}"

        # Base description (epoch + GPU mem) — training loss is appended live
        # per mini-batch so it updates on the left side, ultralytics-style.
        base_desc = f"{epoch_label:>10}  {gpu_mem:>7.2f}G"

        pbar = tqdm(
            loader,
            desc=f"{base_desc}  loss=0.0000",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar:20}| "
                "{n_fmt}/{total_fmt} [{rate_fmt} {postfix}]"
            ),
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        )
        for xb, yb in pbar:
            # Normalize lazily and ship to device (overlaps with pinned copies)
            xb = _to_device_normalized(xb, mean, std)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            # Update loss in the DESCRIPTION (left of bar) every mini-batch
            pbar.set_description(
                f"{base_desc}  loss={running_loss / n_seen:.4f}"
            )

        epoch_loss = running_loss / max(n_seen, 1)
        loss_curve.append(epoch_loss)

        if has_val:
            # Run validation ONCE per epoch (NOT per iteration — too expensive)
            val_loss, val_acc, val_prec, val_rec = validation_metrics(
                model, X_val, y_val, mean, std, criterion
            )
            val_scores.append(val_acc)
            val_loss_curve.append(val_loss)
            # Put val metrics in the postfix (right of bar).
            pbar.set_description(f"{base_desc}  loss={epoch_loss:.4f}")
            pbar.set_postfix(
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
                prec=f"{val_prec:.4f}",
                rec=f"{val_rec:.4f}",
            )
        pbar.close()

        # Save "last" weights after EVERY epoch so they're always current,
        # even if training is killed/crashes mid-run.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_layers": MLP_HIDDEN_LAYERS,
            },
            os.path.join(WEIGHTS_DIR, "mlp_model_last.pt"),
        )

        # One-line per-epoch summary that persists after the tqdm bar is gone.
        if has_val:
            print(
                f"  Epoch {epoch + 1}/{n_epochs}  "
                f"train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  prec={val_prec:.4f}  rec={val_rec:.4f}"
            )
        else:
            print(f"  Epoch {epoch + 1}/{n_epochs}  train_loss={epoch_loss:.4f}")

        if has_val:
            # Track best weights by LOWEST VALIDATION LOSS and save best.pt.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "input_dim": input_dim,
                        "hidden_layers": MLP_HIDDEN_LAYERS,
                    },
                    os.path.join(WEIGHTS_DIR, "mlp_model_best.pt"),
                )
                no_improve = 0
            else:
                no_improve += 1

            # Early stopping (optional)
            if MLP_EARLY_STOPPING and no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})."
                )
                break

    # Restore best weights for evaluation (last weights already saved per-epoch).
    if best_state is not None:
        model.load_state_dict(best_state)

    # Ultralytics-style final summary
    if has_val and val_scores:
        print(
            f"\n{Fore.GREEN}Training complete. "
            f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.{Style.RESET_ALL}"
        )

    return model, loss_curve, val_loss_curve, val_scores


def extract_features(img_paths, return_paths=False):
    """Extract single- and double-compressed features for every image.

    Runs sequentially when NUM_FEATURE_WORKERS <= 1, otherwise distributes
    whole-image extraction across a ProcessPoolExecutor (approach D). Each
    worker runs the vectorised DCT + Numba histogram pipeline independently.

    Returns (X, y) with X of shape (2*N_ok, N_FEATURES) and y in {0, 1}.
    When `return_paths` is True, also returns a list `paths` (length 2*N_ok)
    holding the source image path of each row (added for main.py's patch-based
    image evaluation, which must map validation rows back to source files).
    """
    if ROTATION_AUGMENT:
        print(
            "Rotation augmentation: ON "
            "(random 90-degree multiples via albumentations)"
        )

    X, y, paths = [], [], []

    if NUM_FEATURE_WORKERS and NUM_FEATURE_WORKERS > 1:
        n_workers = NUM_FEATURE_WORKERS
        print(
            f"Parallel feature extraction with {n_workers} workers "
            f"(ProcessPoolExecutor)..."
        )
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_one_image, p): p for p in img_paths
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting",
                unit="img",
            ):
                res = fut.result()
                if res is None:
                    continue
                feats_single, lbl_s, feats_double, lbl_d = res
                p = futures[fut]
                X.append(feats_single)
                y.append(lbl_s)
                paths.append(p)
                X.append(feats_double)
                y.append(lbl_d)
                paths.append(p)
    else:
        for path in tqdm(img_paths, desc="Extracting", unit="img"):
            res = _process_one_image(path)
            if res is None:
                continue
            feats_single, lbl_s, feats_double, lbl_d = res
            X.append(feats_single)
            y.append(lbl_s)
            paths.append(path)
            X.append(feats_double)
            y.append(lbl_d)
            paths.append(path)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if return_paths:
        return X, y, paths
    return X, y


def main():
    # Discover images directly under DATA_DIR (on-the-fly; no .npy files)
    img_paths = list_images(DATA_DIR)
    if MAX_IMAGES and MAX_IMAGES > 0:
        img_paths = img_paths[:MAX_IMAGES]
    print(f"Using {len(img_paths)} images from {DATA_DIR}. Extracting features...")

    X, y = extract_features(img_paths)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features each.")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Always carve a stratified validation split for per-epoch validation logging
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1,
        random_state=RANDOM_STATE, stratify=y_train,
    )

    # Convert to tensors
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    X_val_t = torch.from_numpy(X_val) if X_val is not None else None
    y_val_t = torch.from_numpy(y_val) if y_val is not None else None

    # Z-score normalization statistics from TRAINING data only (replaces scaler)
    mean = X_tr_t.mean(dim=0, keepdim=True)
    std = X_tr_t.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)  # avoid div-by-zero

    # Build model
    input_dim = X.shape[1]
    model = MLP(input_dim, MLP_HIDDEN_LAYERS).to(DEVICE)
    print(
        f"Model: MLP(input={input_dim}, hidden={MLP_HIDDEN_LAYERS}) on {DEVICE}."
    )

    # Train
    model, loss_curve, val_loss_curve, val_scores = train_loop(
        model, X_tr_t, y_tr_t, X_val_t, y_val_t, mean, std, input_dim
    )

    # Evaluate on held-out test set
    y_pred, y_proba = evaluate(model, X_test_t, y_test_t, mean, std)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save normalization stats (model weights already saved during training)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(
        {"mean": mean.cpu(), "std": std.cpu()},
        os.path.join(WEIGHTS_DIR, "scaler_stats.pt"),
    )

    # Save diagnostic figures to the assets directory
    save_all_graphs(ASSETS_DIR, loss_curve, val_scores, y_test, y_pred, y_proba,
                    val_loss_curve)
    print(f"Saved best/last weights to '{WEIGHTS_DIR}/' and figures to '{ASSETS_DIR}/'.")


if __name__ == "__main__":
    main()