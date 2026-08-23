# main.py
"""Main pipeline for JPEG compression-coefficient forgery detection.

Orchestrates the full workflow:
  1. Create sample images (single + double compressed)
  2. Visualize DCT coefficient histograms (comb pattern) for patches
  3. Train the MLP model (MLP_NUM_EPOCHS from config)
  4. Save model diagnostic diagrams with "model_" prefix (.png + .jpg q=20)
  5. Patch-based real-image evaluation on validation set
  6. Save image-level diagrams with "image_" prefix (.png + .jpg q=20)

Usage (from project root):
    python src/main.py
"""
import os
import sys
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
import torch

# Ensure local imports resolve when launched from project root or src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import (
    DATA_DIR,
    ASSETS_DIR,
    WEIGHTS_DIR,
    QF_SINGLE,
    QF_FIRST,
    QF_SECOND,
    BLOCK_SIZE,
    N_HIST_BINS,
    TEST_SIZE,
    RANDOM_STATE,
    MLP_HIDDEN_LAYERS,
    PATCH_BLOCKS,
    PATCH_VOTE_THRESHOLD,
    HIST_COEFF_IDX,
)
from dct_utils import blockify, dct_2d_batch, quantise, dequantise
from quant_tables import get_quant_table
from build_dataset import extract_hist_features, HIST_LO, HIST_HI

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_PATCHES_TO_SHOW = 10  # number of patches visualised in the comb-pattern figure


# ===========================================================================
# Phase 1 -- Sample creation
# ===========================================================================

def create_sample_images():
    """Create ``assets/sample.jpg`` (single) and ``assets/sample_double.jpg``.

    If the first image in DATA_DIR is already JPEG, it is **copied** verbatim
    (preserving the existing first compression). Otherwise it is converted to
    JPEG at ``QF_SINGLE``. The double version is then produced by re-saving
    the single JPEG at ``QF_SECOND[0]``.
    """
    os.makedirs(ASSETS_DIR, exist_ok=True)

    files = sorted(
        f for f in os.listdir(DATA_DIR) if f.lower().endswith(IMG_EXTS)
    )
    if not files:
        raise FileNotFoundError(f"No images in {DATA_DIR}")

    src_path = os.path.join(DATA_DIR, files[0])
    sample_path = os.path.join(ASSETS_DIR, "sample.jpg")

    if src_path.lower().endswith((".jpg", ".jpeg")):
        shutil.copy2(src_path, sample_path)
        print(f"  Copied {files[0]} -> {sample_path}")
    else:
        # Convert only modes JPEG can't store (e.g. RGBA/P); keep L/RGB as-is.
        img = Image.open(src_path)
        if img.mode not in ("L", "RGB", "CMYK"):
            img = img.convert("RGB")
        img.save(sample_path, "JPEG", quality=QF_SINGLE)
        print(f"  Converted {files[0]} -> {sample_path}")

    # Double-compress: re-save at the second quality factor (preserve color mode)
    double_path = os.path.join(ASSETS_DIR, "sample_double.jpg")
    Image.open(sample_path).save(double_path, "JPEG", quality=QF_SECOND[0])
    print(f"  Double-compressed -> {double_path}")
    return sample_path, double_path


# ===========================================================================
# Phase 2 -- Comb-pattern visualisation
# ===========================================================================

def _compute_compressed_blocks(img_path, qf, qf2=None):
    """Load image and return compressed DCT blocks + grid dimensions.

    Mirrors the compression logic of ``build_dataset.process_image`` but also
    returns ``(n_h, n_w)`` so the caller can reshape blocks back to a spatial
    grid for patch extraction.

    Returns
    -------
    compressed_blocks : ndarray (n_h * n_w, 8, 8)
    n_h, n_w : int  -- number of 8x8 block-rows / block-cols
    """
    img = Image.open(img_path).convert("L")
    img = np.array(img)
    h, w = img.shape
    h = h - (h % BLOCK_SIZE)
    w = w - (w % BLOCK_SIZE)
    img = img[:h, :w]

    n_h = h // BLOCK_SIZE
    n_w = w // BLOCK_SIZE

    blocks = blockify(img)
    dct_blocks = dct_2d_batch(blocks.astype(np.float64))

    is_jpeg = img_path.lower().endswith((".jpg", ".jpeg"))
    if is_jpeg:
        if qf2 is None:
            compressed_blocks = dct_blocks
        else:
            compressed_blocks = quantise(dct_blocks, get_quant_table(qf2))
    else:
        q_table1 = get_quant_table(qf)
        if qf2 is None:
            compressed_blocks = quantise(dct_blocks, q_table1)
        else:
            q_table2 = get_quant_table(qf2)
            q_b = quantise(dct_blocks, q_table1)
            dct_rec = dequantise(q_b, q_table1)
            compressed_blocks = quantise(dct_rec, q_table2)

    return compressed_blocks, n_h, n_w


def _patch_coeff_values(compressed_blocks, n_h, n_w, coeff_idx,
                         patch_blocks, max_patches):
    """Collect raw AC-coefficient values per patch.

    ``coeff_idx`` is a flat raster index (1..63; 0 = DC, skipped).
    Returns a list of 1-D arrays (one per patch, up to ``max_patches``).
    """
    spatial = compressed_blocks.reshape(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    n_ph = n_h // patch_blocks
    n_pw = n_w // patch_blocks

    results = []
    for ph in range(n_ph):
        for pw in range(n_pw):
            if len(results) >= max_patches:
                return results
            tile = spatial[
                ph * patch_blocks:(ph + 1) * patch_blocks,
                pw * patch_blocks:(pw + 1) * patch_blocks,
            ]
            flat = tile.reshape(-1, BLOCK_SIZE * BLOCK_SIZE)
            ac = flat[:, 1:]                        # drop DC at index 0
            results.append(ac[:, coeff_idx - 1])    # coeff_idx is 1-based flat
    return results


def _save_figure(fig, name):
    """Save *fig* as ``assets/{name}.png`` and ``assets/{name}.jpg`` (q=20)."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    png_path = os.path.join(ASSETS_DIR, f"{name}.png")
    jpg_path = os.path.join(ASSETS_DIR, f"{name}.jpg")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    Image.open(png_path).convert("RGB").save(jpg_path, "JPEG", quality=20)
    print(f"  Saved {name}.png + {name}.jpg")


def visualize_comb_pattern(sample_path):
    """Plot single-vs-double patch histograms side by side.

    5 rows x 4 data columns, with extra space between columns 2 and 3:
      col 1: single patches 1-5     col 2: double patches 1-5
      col 3: single patches 6-10    col 4: double patches 6-10
    Single compression -> smooth histograms; double compression -> comb
    pattern (periodic gaps).
    """
    single_blocks, sn_h, sn_w = _compute_compressed_blocks(
        sample_path, qf=QF_SINGLE, qf2=None
    )
    double_blocks, dn_h, dn_w = _compute_compressed_blocks(
        sample_path, qf=QF_FIRST[0], qf2=QF_SECOND[0]
    )

    single_vals = _patch_coeff_values(
        single_blocks, sn_h, sn_w, HIST_COEFF_IDX, PATCH_BLOCKS, N_PATCHES_TO_SHOW
    )
    double_vals = _patch_coeff_values(
        double_blocks, dn_h, dn_w, HIST_COEFF_IDX, PATCH_BLOCKS, N_PATCHES_TO_SHOW
    )

    n_show = min(len(single_vals), len(double_vals), N_PATCHES_TO_SHOW)
    if n_show == 0:
        print("  Image too small for even one patch -- skipping comb visualisation.")
        return

    bins = np.linspace(HIST_LO, HIST_HI, N_HIST_BINS + 1)

    # Split the patches into two vertical groups: 1..half and half+1..n_show.
    half = (n_show + 1) // 2          # 5 for the default N_PATCHES_TO_SHOW = 10
    n_rows = half                     # -> 5 rows

    # 5 grid columns: [single | double | spacer | single | double]. The narrow
    # spacer column adds extra breathing room between column 2 and column 3.
    fig = plt.figure(figsize=(13, 3.1 * n_rows + 1.6))
    gs = fig.add_gridspec(
        n_rows, 5, width_ratios=[1, 1, 0.4, 1, 1], wspace=0.25, hspace=0.42
    )

    def _plot(grid_col, row, values, title, color):
        ax = fig.add_subplot(gs[row, grid_col])
        ax.hist(values, bins=bins, color=color, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(HIST_LO, HIST_HI)
        if grid_col in (0, 3):        # first column of each single/double pair
            ax.set_ylabel("count")
        if row == n_rows - 1:
            ax.set_xlabel("coefficient value")

    for row in range(n_rows):
        i_left = row                  # left pair: patches 1..half
        if i_left < n_show:
            _plot(0, row, single_vals[i_left],
                  f"Single patch {i_left + 1}", "steelblue")
            _plot(1, row, double_vals[i_left],
                  f"Double patch {i_left + 1}", "coral")

        i_right = half + row          # right pair: patches half+1..n_show
        if i_right < n_show:
            _plot(3, row, single_vals[i_right],
                  f"Single patch {i_right + 1}", "steelblue")
            _plot(4, row, double_vals[i_right],
                  f"Double patch {i_right + 1}", "coral")

    fig.suptitle(
        f"DCT coefficient histograms  (idx={HIST_COEFF_IDX}, "
        f"Q1~{QF_FIRST[0]}, Q2={QF_SECOND[0]})\n"
        "Single (left) vs Double (right) -- note the comb pattern in the "
        "double columns",
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, "patch_histograms")


# ===========================================================================
# Phase 3 -- Training
# ===========================================================================

def run_training():
    """Train the MLP and return model + validation image paths/labels.

    Uses the real config values: ``MLP_NUM_EPOCHS``, ``MAX_IMAGES``,
    ``NUM_FEATURE_WORKERS``, ``NUM_WORKERS``, ``MLP_PATIENCE``.
    """
    import train_model as tm

    # --- Discover images -----------------------------------------------
    img_paths = tm.list_images(DATA_DIR)
    max_imgs = (
        config.MAX_IMAGES
        if config.MAX_IMAGES and config.MAX_IMAGES > 0
        else len(img_paths)
    )
    img_paths = img_paths[:max_imgs]
    print(f"  Using {len(img_paths)} images from {DATA_DIR}")

    # --- Extract features WITH source paths -----------------------------
    X, y, paths = tm.extract_features(img_paths, return_paths=True)
    paths_arr = np.array(paths)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # --- Splits (identical to train_model.main, carrying paths) ---------
    (X_train, X_test, y_train, y_test,
     paths_train, paths_test) = train_test_split(
        X, y, paths_arr, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    X_tr, X_val, y_tr, y_val, paths_tr, paths_val = train_test_split(
        X_train, y_train, paths_train, test_size=0.1,
        random_state=RANDOM_STATE, stratify=y_train,
    )

    # --- Tensors + normalisation ---------------------------------------
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    mean = X_tr_t.mean(dim=0, keepdim=True)
    std = X_tr_t.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    # --- Build & train --------------------------------------------------
    input_dim = X.shape[1]
    model = tm.MLP(input_dim, MLP_HIDDEN_LAYERS).to(tm.DEVICE)
    model, loss_curve, val_loss_curve, val_scores = tm.train_loop(
        model, X_tr_t, y_tr_t, X_val_t, y_val_t, mean, std, input_dim
    )

    # --- Evaluate on test set & save graphs -----------------------------
    y_pred, y_proba = tm.evaluate(model, X_test_t, y_test_t, mean, std)
    print("\n  Test-set classification report:")
    print(classification_report(y_test, y_pred))

    tm.save_all_graphs(
        ASSETS_DIR, loss_curve, val_scores, y_test, y_pred, y_proba,
        val_loss_curve,
    )

    # Save normalisation stats
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(
        {"mean": mean.cpu(), "std": std.cpu()},
        os.path.join(WEIGHTS_DIR, "scaler_stats.pt"),
    )

    return {
        "model": model,
        "mean": mean,
        "std": std,
        "val_paths": paths_val,
        "val_labels": y_val.astype(int),
    }


# ===========================================================================
# Phase 4 -- Model diagram copies
# ===========================================================================

def save_model_diagram_copies():
    """Copy the 5 training diagrams with ``model_`` prefix (.png + .jpg q=20)."""
    names = [
        "confusion_matrix",
        "loss_curve",
        "precision_recall_bar",
        "roc_curve",
        "validation_scores",
    ]
    for name in names:
        src = os.path.join(ASSETS_DIR, f"{name}.png")
        if not os.path.exists(src):
            print(f"  Warning: {src} not found -- skipping model_{name}")
            continue
        dst_png = os.path.join(ASSETS_DIR, f"model_{name}.png")
        dst_jpg = os.path.join(ASSETS_DIR, f"model_{name}.jpg")
        shutil.copy2(src, dst_png)
        Image.open(src).convert("RGB").save(dst_jpg, "JPEG", quality=20)
        print(f"  Saved model_{name}.png + model_{name}.jpg")


# ===========================================================================
# Phase 5 -- Patch-based image evaluation
# ===========================================================================

def extract_patch_features(img_path, qf, qf2=None, patch_blocks=PATCH_BLOCKS):
    """Extract histogram feature vectors for every patch in an image.

    Computes compressed DCT blocks once (whole-image), reshapes to the spatial
    grid, then runs ``extract_hist_features`` on each ``patch_blocks x
    patch_blocks`` tile -- producing the same feature representation the model
    was trained on, but at patch granularity.

    Returns ndarray of shape ``(n_patches, N_FEATURES)``.
    """
    compressed_blocks, n_h, n_w = _compute_compressed_blocks(img_path, qf, qf2)
    spatial = compressed_blocks.reshape(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)

    n_ph = n_h // patch_blocks
    n_pw = n_w // patch_blocks
    analysis_qf = qf if qf2 is None else qf2

    feats = []
    for ph in range(n_ph):
        for pw in range(n_pw):
            tile = spatial[
                ph * patch_blocks:(ph + 1) * patch_blocks,
                pw * patch_blocks:(pw + 1) * patch_blocks,
            ]
            tile_flat = tile.reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
            feats.append(extract_hist_features(tile_flat, analysis_qf))

    if not feats:
        return np.empty((0, N_HIST_BINS * 63), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


@torch.no_grad()
def _classify_patches(model, patch_feats, mean, std):
    """Return fraction of patches classified as double-compressed."""
    if patch_feats.shape[0] == 0:
        return 0.0
    X = torch.from_numpy(patch_feats)
    X = ((X - mean.squeeze(0)) / std.squeeze(0)).to(DEVICE)
    logits = model(X).cpu()
    probs = torch.sigmoid(logits).numpy()
    return float((probs >= 0.5).mean())


def evaluate_images(model, mean, std, val_paths, val_labels):
    """Patch-voting evaluation on validation images.

    Each validation row is a (path, label) pair. For each row we extract
    patches with the compression settings implied by the label (single or
    double), classify every patch, and vote: if >= ``PATCH_VOTE_THRESHOLD``
    fraction of patches are double -> image is double.

    Returns ``(y_true, y_pred, y_proba)`` numpy arrays.
    """
    model.eval()
    y_true, y_pred, y_proba = [], [], []

    for idx, (path, label) in enumerate(zip(val_paths, val_labels)):
        try:
            if label == 0:
                feats = extract_patch_features(path, qf=QF_SINGLE, qf2=None)
            else:
                feats = extract_patch_features(
                    path, qf=QF_FIRST[0], qf2=QF_SECOND[0]
                )
            frac_double = _classify_patches(model, feats, mean, std)
            pred = 1 if frac_double >= PATCH_VOTE_THRESHOLD else 0
            y_true.append(int(label))
            y_pred.append(pred)
            y_proba.append(frac_double)
            print(
                f"  [{idx + 1}/{len(val_paths)}] {os.path.basename(path)}  "
                f"label={label}  pred={pred}  frac_double={frac_double:.2f}  "
                f"({feats.shape[0]} patches)"
            )
        except Exception as exc:
            print(f"  Skipping {path}: {exc}")

    return np.array(y_true), np.array(y_pred), np.array(y_proba)


# ===========================================================================
# Phase 6 -- Image-level diagrams
# ===========================================================================

def save_image_diagrams(y_true, y_pred, y_proba):
    """Generate confusion-matrix, ROC, and precision/recall diagrams.

    Uses the ``image_`` prefix and saves each in both .png and .jpg (q=20).
    """
    classes = ["single", "double"]

    # --- Confusion matrix -----------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Image-Level Confusion Matrix (patch voting)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(ax.images[0], ax=ax, label="count")
    plt.tight_layout()
    _save_figure(fig, "image_confusion_matrix")

    # --- ROC curve (needs both classes) ---------------------------------
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Image-Level ROC Curve")
        ax.legend()
        plt.tight_layout()
        _save_figure(fig, "image_roc_curve")
    else:
        print("  Skipping image_roc_curve (only one class present)")

    # --- Precision / recall / F1 bar ------------------------------------
    if len(np.unique(y_pred)) > 1:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1]
        )
    else:
        p = r = f = np.array([0.0, 0.0])

    x = np.arange(len(classes))
    w = 0.25
    fig, ax = plt.subplots()
    ax.bar(x - w, p, w, label="precision")
    ax.bar(x, r, w, label="recall")
    ax.bar(x + w, f, w, label="f1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_title("Image-Level Per-class Precision / Recall / F1")
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, "image_precision_recall_bar")


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    print("=" * 70)
    print("  JPEG COMPRESSION COEFFICIENT ANALYSIS -- MAIN PIPELINE")
    print(f"  Device: {DEVICE.type.upper()}")
    print("=" * 70)

    # Phase 1
    print("\n[1/6] Creating sample images ...")
    sample_path, _ = create_sample_images()

    # Phase 2
    print("\n[2/6] Visualising comb pattern ...")
    visualize_comb_pattern(sample_path)

    # Phase 3
    print("\n[3/6] Training model ...")
    results = run_training()

    # Phase 4
    print("\n[4/6] Saving model_ prefix diagrams ...")
    save_model_diagram_copies()

    # Phase 5
    print("\n[5/6] Patch-based evaluation on validation images ...")
    y_true, y_pred, y_proba = evaluate_images(
        results["model"],
        results["mean"],
        results["std"],
        results["val_paths"],
        results["val_labels"],
    )
    if len(y_true) > 0:
        print(f"\n  Image-level evaluation: {len(y_true)} images")
        print(classification_report(y_true, y_pred, zero_division=0))
    else:
        print("  No images evaluated.")

    # Phase 6
    print("\n[6/6] Saving image_ prefix diagrams ...")
    if len(y_true) > 0:
        save_image_diagrams(y_true, y_pred, y_proba)
    else:
        print("  Skipping (no evaluation data).")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE -- all outputs in assets/")
    print("=" * 70)


if __name__ == "__main__":
    if "--figures-only" in sys.argv[1:]:
        # Redraw figures only -- NO training, NO evaluation, NO weight changes.
        print("=" * 70)
        print("  FIGURES-ONLY MODE (no training / no evaluation)")
        print("=" * 70)
        sample_path = os.path.join(ASSETS_DIR, "sample.jpg")
        if not os.path.exists(sample_path):
            print("  sample.jpg missing -- creating sample images first")
            sample_path, _ = create_sample_images()
        visualize_comb_pattern(sample_path)  # overwrites patch_histograms.png/.jpg
        print("\n" + "=" * 70)
        print("  FIGURES COMPLETE -- outputs in assets/")
        print("=" * 70)
    else:
        main()