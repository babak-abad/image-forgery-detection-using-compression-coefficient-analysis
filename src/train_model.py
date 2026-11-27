# train_model.py
"""Train the MLP for single- vs multi-compressed JPEG detection.

Multi-source dataset (see dataset_sources.py):

  NO_COMPRESS_FOLDERS     (k=0)   -> uncompressed / synthetic single / synthetic multi
  SINGLE_COMPRESS_FOLDERS (k=1)   -> real single / derived multi (re-encoded)
  MULTI_COMPRESS_FOLDERS  (k>=2)  -> real multi (as-is)

A sample's label is 1 (multi-compressed) iff (source prior compressions +
applied rounds) >= 2. Every simulated compression round is a REAL JPEG
encode/decode. The first pass on pristine pixels draws a uniform quality from
COMPRESS_Q_RANGE; follow-up passes (and derived first passes) are drawn so
the double-quantisation comb is detectable -- a Q2_ASCEND_SHARE share of
finer re-saves (strong comb) and the rest descending within the
Q2_STEP_RATIO_RANGE band (still-detectable; much-coarser second grids are
statistically near-identical to single compression).

After training, the binary decision threshold is calibrated on the VALIDATION
split (max F1 of the double class) and persisted into both checkpoints; the
blind-test evaluation and predict.py then use that calibrated threshold.

The train/val/test split is at SOURCE-IMAGE level (70/10/20 by default): all
samples generated from one source image stay together, so no pixel content
leaks across splits. Features are raw DCT AC histograms (ANALYSIS_QF = None);
the scratch probe showed quantised-at-75 histograms lose separability
(0.625 vs 0.722 linear-probe accuracy).

There is NO "max iterations" knob: each epoch is a full pass over the training
set in BATCH_SIZE mini-batches, training runs up to MLP_NUM_EPOCHS with early
stopping (MLP_PATIENCE, counted in epochs, monitored on validation loss).
"""
import base64
import os
import time

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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)

from config import (
    ASSETS_DIR,
    WEIGHTS_DIR,
    DOC_DIR,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_FEATURE_WORKERS,
    MLP_LEARNING_RATE,
    MLP_NUM_EPOCHS,
    MLP_EARLY_STOPPING,
    MLP_PATIENCE,
    ANALYSIS_QF,
    NO_COMPRESS_FOLDERS,
    SINGLE_COMPRESS_FOLDERS,
    MULTI_COMPRESS_FOLDERS,
    COMPRESS_Q_RANGE,
    FORCE_DISTINCT_Q,
    Q2_ASCEND_SHARE,
    Q2_STEP_RATIO_RANGE,
    TRIPLE_COMPRESS_PROB,
    TRAIN_PATCH_SIZE,
    DQ_HIST_BINS,
    DQ_HIST_RANGE,
    RES_HIST_BINS,
    RES_HIST_RANGE,
    N_HIST_BINS,
    N_AC_COEFFS,
    SAMPLES_UNCOMPRESSED_PER_UNCOMPRESSED,
    SAMPLES_SINGLE_PER_UNCOMPRESSED,
    SAMPLES_DOUBLE_PER_UNCOMPRESSED,
    SAMPLES_ORIGINAL_PER_COMPRESSED,
    SAMPLES_DOUBLE_PER_COMPRESSED,
)
from dataset_sources import build_source_tasks, process_source

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

# YOLO (ultralytics)-style console progress for the training loop, built on
# tqdm: 11-char right-aligned columns and the bar_format constant used by
# ultralytics itself. No ultralytics dependency is introduced.
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
TRAIN_COLUMNS = ("Epoch", "GPU_mem", "bce_loss", "Samples", "Feat_dim")
VAL_METRIC_COLUMNS = ("val_loss", "val_acc", "prec", "recall")


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
        # Single logit output for BCEWithLogitsLoss (positive class = double)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
def evaluate(model, X, y, mean, std, device=DEVICE, batch_size=512, threshold=0.5):
    """Return (predictions, probabilities) for evaluation tensors X, y.

    Predictions cut the sigmoid probability at `threshold` (0.5 default;
    main() passes the validation-calibrated threshold).
    """
    model.eval()
    preds, probs = [], []
    for i in range(0, X.shape[0], batch_size):
        xb = _to_device_normalized(X_chunk=X[i:i + batch_size], mean=mean, std=std)
        logits = model(xb).cpu()
        p = torch.sigmoid(logits)
        preds.append((p >= threshold).long())
        probs.append(p)
    preds = torch.cat(preds).numpy()
    probs = torch.cat(probs).numpy()
    return preds, probs


def calibrate_threshold(y_true, proba, n_grid=199):
    """Sweep decision thresholds on VALIDATION probabilities; maximise F1(double).

    Returns (best_threshold, best_f1). The grid covers (0, 1) with `n_grid`
    points and always contains 0.5, so the default cut is a candidate.
    Calibration reads the validation split only; the test split stays blind.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, n_grid):
        pred = (p >= t).astype(np.int64)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


@torch.no_grad()
def validation_accuracy(model, X_val, y_val, mean, std, device=DEVICE):
    """Quick validation accuracy (val set is small)."""
    model.eval()
    xb = _to_device_normalized(X_val, mean, std)
    logits = model(xb).cpu()
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return (preds == y_val).float().mean().item()


@torch.no_grad()
def validation_metrics(
    model, X_val, y_val, mean, std, criterion, batch_size=512, progress=True
):
    """Return (val_loss, val_acc, precision, recall) on the validation set.

    The forward pass runs in `batch_size` chunks so a YOLO-style validation
    progress line (tqdm) can be shown; results are identical to a single
    full-batch pass because the chunk logits are concatenated before scoring.
    """
    model.eval()
    chunks = range(0, X_val.shape[0], batch_size)
    pbar = (
        tqdm(
            chunks,
            desc=" " * 11 + "".join(f"{c:>11}" for c in VAL_METRIC_COLUMNS),
            bar_format=TQDM_BAR_FORMAT,
            leave=True,
            dynamic_ncols=True,
        )
        if progress
        else chunks
    )
    logits_parts = []
    for start in pbar:
        xb = _to_device_normalized(
            X_chunk=X_val[start:start + batch_size], mean=mean, std=std
        )
        logits_parts.append(model(xb).cpu())
    logits = torch.cat(logits_parts)
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
    if progress:
        print(
            f"{'all':>11}"
            + "".join(f"{v:>11.4f}" for v in (val_loss, val_acc, precision, recall))
        )
    return val_loss, val_acc, precision, recall


def train_loop(model, X_tr, y_tr, X_val, y_val, mean, std, input_dim):
    """Train `model` epoch by epoch with manual early stopping.

    iterations_per_epoch = ceil(n_train / BATCH_SIZE). There is no global
    iteration cap; the outer loop runs up to MLP_NUM_EPOCHS and stops early
    (MLP_PATIENCE epochs without val-loss improvement) when
    MLP_EARLY_STOPPING is True.

    Data stays on the CPU (normalized lazily per batch); worker processes
    prefetch batches and pin them so GPU transfer overlaps with compute.

    Progress is rendered YOLO-style: a one-time column header, one live
    tqdm line per epoch (running-mean loss, GPU memory) that stays in the
    scrollback when complete, plus a validation progress line and metrics row.
    """
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

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

    print(
        f"Starting training for {n_epochs} epochs ({len(ds)} samples, "
        f"BATCH_SIZE={BATCH_SIZE} -> {iters_per_epoch} iterations/epoch)..."
    )
    print()
    header = f"{Fore.CYAN}" + "".join(f"{c:>11}" for c in TRAIN_COLUMNS) + f"{Style.RESET_ALL}"
    print(header)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    loss_curve = []
    val_loss_curve = []
    val_scores = []
    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    no_improve = 0
    patience = MLP_PATIENCE

    for epoch in range(n_epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        epoch_label = f"{epoch + 1}/{n_epochs}"

        pbar = tqdm(
            loader,
            total=iters_per_epoch,
            desc=f"{epoch_label:>11}" + " " * 44,
            bar_format=TQDM_BAR_FORMAT,
            leave=True,
            dynamic_ncols=True,
        )
        for xb, yb in pbar:
            xb = _to_device_normalized(X_chunk=xb, mean=mean, std=std)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            gpu_mem = (
                f"{torch.cuda.memory_reserved() / 1e9:.3g}G"
                if DEVICE.type == "cuda"
                else "0G"
            )
            pbar.set_description(
                f"{epoch_label:>11}"
                f"{gpu_mem:>11}"
                f"{running_loss / n_seen:>11.4f}"
                f"{n_seen:>11}"
                f"{input_dim:>11}"
            )
        pbar.close()

        epoch_loss = running_loss / max(n_seen, 1)
        loss_curve.append(epoch_loss)

        if has_val:
            val_loss, val_acc, val_prec, val_rec = validation_metrics(
                model=model,
                X_val=X_val,
                y_val=y_val,
                mean=mean,
                std=std,
                criterion=criterion,
            )
            val_scores.append(val_acc)
            val_loss_curve.append(val_loss)

        # Save "last" weights after EVERY epoch so they are always current,
        # even if training is killed/crashes mid-run.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_layers": MLP_HIDDEN_LAYERS,
            },
            os.path.join(WEIGHTS_DIR, "mlp_model_last.pt"),
        )

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

            if MLP_EARLY_STOPPING and no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if has_val and val_scores:
        print(
            f"\n{Fore.GREEN}Training complete. "
            f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.{Style.RESET_ALL}"
        )

    return model, loss_curve, val_loss_curve, val_scores



# ----------------------------------------------------------------------------
# Multi-source dataset assembly
# ----------------------------------------------------------------------------

def build_dataset_from_sources():
    """Extract all training samples from the configured source pools."""
    tasks = build_source_tasks()
    if not tasks:
        raise RuntimeError(
            "No source images found. Check NO_COMPRESS_FOLDERS / "
            "SINGLE_COMPRESS_FOLDERS / MULTI_COMPRESS_FOLDERS in config.py."
        )
    by_type = {}
    for t in tasks:
        by_type[t["source_type"]] = by_type.get(t["source_type"], 0) + 1
    print(
        "Source images: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
        + f" (analysis_qf={ANALYSIS_QF})"
    )

    samples = []
    t0 = time.time()
    if NUM_FEATURE_WORKERS and NUM_FEATURE_WORKERS > 1:
        print(
            f"Parallel feature extraction with {NUM_FEATURE_WORKERS} workers "
            f"(ProcessPoolExecutor)..."
        )
        with ProcessPoolExecutor(max_workers=NUM_FEATURE_WORKERS) as pool:
            futures = [pool.submit(process_source, t) for t in tasks]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting",
                unit="img",
            ):
                samples.extend(fut.result())
    else:
        for t in tqdm(tasks, desc="Extracting", unit="img"):
            samples.extend(process_source(t))

    # Deterministic sample order regardless of worker completion order.
    samples.sort(key=lambda s: (s["group_id"], s["sample_idx"]))
    print(f"Extracted {len(samples)} samples in {time.time() - t0:.1f}s")
    return samples


def split_samples(samples):
    """Group-level split: all samples from one source image stay together.

    Split fractions are applied per source type so every split keeps the same
    kind composition. Returns {"train": [...], "val": [...], "test": [...]}.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    groups_by_type = {}
    for s in samples:
        groups_by_type.setdefault(s["source_type"], set()).add(s["group_id"])

    split_of_group = {}
    for stype in sorted(groups_by_type):
        groups = np.array(sorted(groups_by_type[stype]))
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, int(round(n * TEST_SIZE))) if n >= 3 else 0
        n_val = max(1, int(round(n * VAL_SIZE))) if n - n_test >= 2 else 0
        for g in groups[:n_test]:
            split_of_group[g] = "test"
        for g in groups[n_test : n_test + n_val]:
            split_of_group[g] = "val"
        for g in groups[n_test + n_val :]:
            split_of_group[g] = "train"

    splits = {"train": [], "val": [], "test": []}
    for s in samples:
        splits[split_of_group[s["group_id"]]].append(s)
    return splits


def stack(samples):
    """Stack a list of sample dicts into (X, y) arrays."""
    X = np.stack([s["features"] for s in samples]).astype(np.float32)
    y = np.array([s["label"] for s in samples], dtype=np.float32)
    return X, y


def kind_table(y_true, y_pred, kinds):
    """Per-sample-kind accuracy rows: (kind, n, accuracy)."""
    rows = []
    for kind in sorted(set(kinds)):
        m = kinds == kind
        rows.append(
            [kind, int(m.sum()), round(float((y_true[m] == y_pred[m]).mean()), 4)]
        )
    return rows


def _chain_order_label(sample):
    """Ordering of the last applied pass vs the previous compression.

    'last < prev' = descending re-save (strong double-quantisation comb),
    'last > prev' = ascending re-save (the hard, weak-comb regime), 'unknown'
    when the previous quality is unavailable (e.g. estimate_qf failed).
    """
    chain = sample.get("q_chain") or []
    q_last = chain[-1] if chain else None
    q_prev = chain[-2] if len(chain) >= 2 else sample.get("q_prior")
    if q_last is None or q_prev is None:
        return "unknown"
    if q_last < q_prev:
        return "last < prev"
    return "last > prev"


def q_bucket_table(y_true, y_pred, samples):
    """Accuracy of positive samples bucketed by chain structure.

    Rows per positive kind: the ordering of the last pass vs the previous
    compression (descending re-saves are the realistic, strongly detectable
    regime; ascending ones are the hard regime) and the last-pass quality
    bucket (a high final quality weakens the comb artefacts).
    """
    kinds = np.array([s["kind"] for s in samples])
    order = np.array([_chain_order_label(s) for s in samples])
    q_last = np.array([s["q_chain"][-1] if s["q_chain"] else -1 for s in samples])
    rows = []
    for kind in ("synthetic_multi", "derived_multi"):
        base = kinds == kind
        if not base.any():
            continue
        for name, m in (
            ("re-save lower (q_last < q_prev)", base & (order == "last < prev")),
            ("re-save higher (q_last > q_prev)", base & (order == "last > prev")),
            ("prev quality unknown", base & (order == "unknown")),
            ("last q < 75", base & (q_last >= 0) & (q_last < 75)),
            ("last q >= 75", base & (q_last >= 75)),
        ):
            if m.sum():
                rows.append(
                    [
                        kind,
                        name,
                        int(m.sum()),
                        round(float((y_true[m] == y_pred[m]).mean()), 4),
                    ]
                )
    return rows


def _img_b64(path):
    """Read an image file and return it as a base64 data URI (or None)."""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")



# ----------------------------------------------------------------------------
# Self-contained HTML training report
# ----------------------------------------------------------------------------

def save_results_html(html_path, run):
    """Write a self-contained HTML training report (figures embedded base64)."""
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    css = (
        "body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;"
        "margin:24px auto;padding:0 16px;color:#222}"
        "h1{border-bottom:3px solid #4a76a8;padding-bottom:6px}"
        "h2{color:#4a76a8;margin-top:32px}"
        "table{border-collapse:collapse;margin:12px 0;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}"
        "th{background:#eef3f8}"
        "td.num{text-align:right}"
        ".cards{display:flex;gap:12px;flex-wrap:wrap}"
        ".card{border:1px solid #ccc;border-radius:8px;padding:12px 18px;"
        "min-width:150px;background:#f8fafc}"
        ".card b{font-size:1.5em;display:block}"
        "img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:6px 0}"
        "code{background:#f4f4f4;padding:2px 5px;border-radius:4px}"
        ".muted{color:#777}"
    )

    def table(headers, rows):
        h = "".join(f"<th>{c}</th>" for c in headers)
        body = ""
        for r in rows:
            body += "<tr>" + "".join(
                f'<td class="num">{c}</td>'
                if isinstance(c, (int, float))
                else f"<td>{c}</td>"
                for c in r
            ) + "</tr>"
        return f"<table><tr>{h}</tr>{body}</table>"

    def img(path, alt):
        data = _img_b64(path)
        return f'<p><img src="{data}" alt="{alt}"></p>' if data else ""

    sections = []
    sections.append("<h1>Double-compression detector &mdash; training results</h1>")
    sections.append(
        f"<p class='muted'>Generated {run['timestamp']} &middot; runtime "
        f"{run['elapsed_min']:.1f} min &middot; device {run['device']}</p>"
    )
    sections.append(
        "<div class='cards'>"
        f"<div class='card'>samples<b>{run['n_samples']}</b>total</div>"
        f"<div class='card'>test accuracy<b>{run['test_acc']:.4f}</b>blind split</div>"
        f"<div class='card'>test F1 (double)<b>{run['test_f1']:.4f}</b>positive class</div>"
        f"<div class='card'>ROC AUC<b>{run['roc_auc']:.4f}</b>test</div>"
        f"<div class='card'>threshold<b>{run['decision_threshold']:.2f}</b>val-calibrated</div>"
        f"<div class='card'>best epoch<b>{run['best_epoch']}</b>of {run['n_epochs_run']}</div>"
        "</div>"
    )
    sections.append("<h2>Dataset composition (samples by kind and split)</h2>")
    sections.append(
        table(["kind", "label", "train", "val", "test", "total"], run["comp_rows"])
    )
    sections.append("<h2>Training settings</h2>")
    sections.append(table(["setting", "value"], run["cfg_rows"]))
    sections.append("<h2>Blind test-set metrics</h2>")
    sections.append(
        table(
            ["class", "precision", "recall", "f1", "support"],
            run["report_rows"],
        )
    )
    sections.append("<h2>Confusion matrix (test)</h2>")
    sections.append(table(["", "pred single", "pred double"], run["cm_rows"]))
    sections.append("<h2>Accuracy by sample kind (test)</h2>")
    sections.append(table(["kind", "n", "accuracy"], run["kind_rows"]))
    sections.append(
        "<h2>Accuracy by simulated-QF bucket (positive samples)</h2>"
    )
    sections.append(
        table(
            ["kind", "bucket (last-pass ordering / quality)", "n", "accuracy"],
            run["q_rows"],
        )
    )
    sections.append("<h2>Figures</h2>")
    for p, alt in run["figs"]:
        sections.append(img(p, alt))
    sections.append("<h2>Artifacts</h2>")
    sections.append(
        "<ul>" + "".join(f"<li><code>{a}</code></li>" for a in run["artifacts"]) + "</ul>"
    )
    sections.append("<h2>Reproduce</h2>")
    sections.append(
        "<pre><code>python src/train_model.py    (from the project root)</code></pre>"
    )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Training results - compression detector</title>"
        f"<style>{css}</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Saved HTML report to {html_path}")



def main():
    t_start = time.time()

    samples = build_dataset_from_sources()
    if not samples:
        raise RuntimeError("No samples were generated from the source pools.")

    splits = split_samples(samples)
    for name in ("train", "val", "test"):
        n_pos = sum(1 for s in splits[name] if s["label"] == 1)
        print(
            f"{name}: {len(splits[name])} samples "
            f"({n_pos} positives / {len(splits[name]) - n_pos} negatives)"
        )

    X_tr, y_tr = stack(splits["train"])
    X_val, y_val = stack(splits["val"]) if splits["val"] else (None, None)
    X_te, y_te = stack(splits["test"])

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_val_t = torch.from_numpy(X_val) if X_val is not None else None
    y_val_t = torch.from_numpy(y_val) if X_val is not None else None
    X_te_t = torch.from_numpy(X_te)
    y_te_t = torch.from_numpy(y_te)

    # Z-score normalization statistics from TRAINING data only
    mean = X_tr_t.mean(dim=0, keepdim=True)
    std = X_tr_t.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    input_dim = X_tr.shape[1]
    model = MLP(input_dim, MLP_HIDDEN_LAYERS).to(DEVICE)
    print(f"Model: MLP(input={input_dim}, hidden={MLP_HIDDEN_LAYERS}) on {DEVICE}.")

    model, loss_curve, val_loss_curve, val_scores = train_loop(
        model, X_tr_t, y_tr_t, X_val_t, y_val_t, mean, std, input_dim
    )

    # Calibrate the decision threshold on the VALIDATION split (never on
    # test): sweep thresholds for the best F1 of the double class.
    if X_val_t is not None:
        _, proba_val = evaluate(model=model, X=X_val_t, y=y_val_t, mean=mean, std=std)
        decision_threshold, val_f1 = calibrate_threshold(y_true=y_val, proba=proba_val)
        print(
            f"Calibrated decision threshold on validation: "
            f"{decision_threshold:.3f} (val F1(double)={val_f1:.4f}; default 0.5)"
        )
    else:
        decision_threshold = 0.5

    # Evaluate on the blind (held-out source groups) test set with the
    # calibrated threshold.
    y_pred, y_proba = evaluate(
        model=model, X=X_te_t, y=y_te_t, mean=mean, std=std, threshold=decision_threshold
    )
    print("Classification report:\n", classification_report(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred)
    print("Confusion matrix:\n", cm)

    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_auc = float(auc(fpr, tpr))
    test_acc = float((y_te == y_pred).mean())
    rep = classification_report(
        y_te, y_pred, output_dict=True, target_names=["single (0)", "double (1)"]
    )
    test_f1 = float(rep["double (1)"]["f1-score"])

    # Save normalization stats (model weights already saved during training)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(
        {"mean": mean.cpu(), "std": std.cpu()},
        os.path.join(WEIGHTS_DIR, "scaler_stats.pt"),
    )

    # Persist the calibrated threshold in both checkpoints so predict.py
    # applies the same decision cut as this evaluation.
    for ckpt_name in ("mlp_model_best.pt", "mlp_model_last.pt"):
        ckpt_path = os.path.join(WEIGHTS_DIR, ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ckpt["decision_threshold"] = decision_threshold
            torch.save(ckpt, ckpt_path)

    # Diagnostic figures
    save_all_graphs(
        ASSETS_DIR, loss_curve, val_scores, y_te, y_pred, y_proba, val_loss_curve
    )
    print(
        f"Saved best/last weights to '{WEIGHTS_DIR}/' and figures to '{ASSETS_DIR}/'."
    )

    # ----- report tables -------------------------------------------------
    kinds_all = sorted({s["kind"] for s in samples})
    comp_rows = []
    for kind in kinds_all:
        lbl = next(s["label"] for s in samples if s["kind"] == kind)
        counts = [
            sum(1 for s in splits[sp] if s["kind"] == kind)
            for sp in ("train", "val", "test")
        ]
        comp_rows.append(
            [kind, "multi (1)" if lbl else "single (0)"] + counts + [sum(counts)]
        )

    kinds_te = np.array([s["kind"] for s in splits["test"]])
    kind_rows = kind_table(y_te, y_pred, kinds_te)
    q_rows = q_bucket_table(y_te, y_pred, splits["test"])

    report_rows = []
    for cls in ("single (0)", "double (1)"):
        r = rep[cls]
        report_rows.append(
            [
                cls,
                round(float(r["precision"]), 4),
                round(float(r["recall"]), 4),
                round(float(r["f1-score"]), 4),
                int(r["support"]),
            ]
        )
    cm_rows = [
        ["true single", int(cm[0, 0]), int(cm[0, 1])],
        ["true double", int(cm[1, 0]), int(cm[1, 1])],
    ]

    cfg_rows = [
        ["NO_COMPRESS_FOLDERS", ", ".join(NO_COMPRESS_FOLDERS) or "(none)"],
        ["SINGLE_COMPRESS_FOLDERS", ", ".join(SINGLE_COMPRESS_FOLDERS) or "(none)"],
        ["MULTI_COMPRESS_FOLDERS", ", ".join(MULTI_COMPRESS_FOLDERS) or "(none yet)"],
        ["COMPRESS_Q_RANGE", str(COMPRESS_Q_RANGE)],
        ["FORCE_DISTINCT_Q", str(FORCE_DISTINCT_Q)],
        ["Q2_ASCEND_SHARE / Q2_STEP_RATIO_RANGE", f"{Q2_ASCEND_SHARE} / {Q2_STEP_RATIO_RANGE}"],
        ["TRIPLE_COMPRESS_PROB", str(TRIPLE_COMPRESS_PROB)],
        [
            "samples per source image",
            f"uncompressed: 1 + {SAMPLES_SINGLE_PER_UNCOMPRESSED} single + "
            f"{SAMPLES_DOUBLE_PER_UNCOMPRESSED} double; compressed: "
            f"{SAMPLES_ORIGINAL_PER_COMPRESSED} original + "
            f"{SAMPLES_DOUBLE_PER_COMPRESSED} derived",
        ],
        ["TRAIN_PATCH_SIZE", str(TRAIN_PATCH_SIZE)],
        ["decision threshold (val-calibrated)", f"{decision_threshold:.3f}"],
        ["ANALYSIS_QF", "raw DCT (None)" if ANALYSIS_QF is None else str(ANALYSIS_QF)],
        [
            "feature set",
            f"raw hist {N_AC_COEFFS}x{N_HIST_BINS} + per-position stats 4x"
            f"{N_AC_COEFFS} + DQ hist {N_AC_COEFFS}x{DQ_HIST_BINS} over "
            f"{DQ_HIST_RANGE} + residue hist {N_AC_COEFFS}x{RES_HIST_BINS} "
            f"over {RES_HIST_RANGE}",
        ],
        [
            "source-group split",
            f"train {1 - TEST_SIZE - VAL_SIZE:.0%} / val {VAL_SIZE:.0%} "
            f"/ test {TEST_SIZE:.0%}",
        ],
        ["MLP hidden layers", str(MLP_HIDDEN_LAYERS)],
        [
            "batch / max epochs / patience",
            f"{BATCH_SIZE} / {MLP_NUM_EPOCHS} / {MLP_PATIENCE}",
        ],
        ["learning rate", str(MLP_LEARNING_RATE)],
        ["feature dim", str(input_dim)],
    ]

    figs = [
        (os.path.join(ASSETS_DIR, "loss_curve.png"), "Loss curve"),
        (os.path.join(ASSETS_DIR, "validation_scores.png"), "Validation accuracy"),
        (os.path.join(ASSETS_DIR, "confusion_matrix.png"), "Confusion matrix"),
        (os.path.join(ASSETS_DIR, "roc_curve.png"), "ROC curve"),
        (
            os.path.join(ASSETS_DIR, "precision_recall_bar.png"),
            "Per-class precision/recall/F1",
        ),
    ]

    run = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": (time.time() - t_start) / 60.0,
        "device": str(DEVICE),
        "n_samples": len(samples),
        "test_acc": test_acc,
        "test_f1": test_f1,
        "roc_auc": roc_auc,
        "decision_threshold": decision_threshold,
        "best_epoch": int(np.argmin(val_loss_curve)) + 1
        if val_loss_curve
        else len(loss_curve),
        "n_epochs_run": len(loss_curve),
        "comp_rows": comp_rows,
        "cfg_rows": cfg_rows,
        "report_rows": report_rows,
        "cm_rows": cm_rows,
        "kind_rows": kind_rows,
        "q_rows": q_rows,
        "figs": figs,
        "artifacts": [
            os.path.join(WEIGHTS_DIR, "mlp_model_best.pt"),
            os.path.join(WEIGHTS_DIR, "mlp_model_last.pt"),
            os.path.join(WEIGHTS_DIR, "scaler_stats.pt"),
            os.path.join(DOC_DIR, "training_results.html"),
        ],
    }
    save_results_html(os.path.join(DOC_DIR, "training_results.html"), run)
    print(f"Total runtime: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
