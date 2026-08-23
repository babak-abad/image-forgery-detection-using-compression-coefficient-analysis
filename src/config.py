# config.py

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

# Directory holding input images and saved artefacts (.npy, .joblib).
# Value: any path string. Expected to contain an "originals/" subfolder with source images.
DATA_DIR = "D:/projects/datasets/LV-MHP-v1/images"

# Maximum number of images to read from DATA_DIR for on-the-fly training.
# Range: non-negative integer. 0 = use ALL images in DATA_DIR.
# Each image produces 2 samples (single + double compressed), so 200 -> 400 samples.
# Keep small (e.g. 100-300) for quick experiments; set to 0 for a full run.
MAX_IMAGES = 0

# Hardcoded directory for generated figures (learning curve, confusion matrix, etc.).
# Relative path -> scripts must be run from the project root to land in /assets.
ASSETS_DIR = "assets"

# Hardcoded directory for trained model weights (scaler.joblib, mlp_model.joblib).
# Relative path -> scripts must be run from the project root to land in /weights.
WEIGHTS_DIR = "weights"

# ----------------------------------------------------------------------------
# JPEG quality factors for generating the dataset
# ----------------------------------------------------------------------------

# Quality factor used to generate single-compressed samples.
# Range: integer 1-100. Higher = better quality / less compression.
# Typical values: 70, 75, 80, 85, 90, 95. Current: 90.
QF_SINGLE = 90

# First (primary) quality factor used to generate double-compressed samples.
# Given as [min, max] inclusive; a random integer in this range is drawn per
# image at feature-extraction time. Range: integers 1-100.
# Typical values: [80, 95]. Current: [80, 95].
QF_FIRST = [50, 95]

# Second (re-compression) quality factor applied on top of the first one.
# Given as [min, max] inclusive; a random integer in this range is drawn per
# image at feature-extraction time. Range: integers 1-100.
# Typical values: [60, 80]. Current: [60, 80].
# Note: for meaningful double-compression artifacts, the drawn QF_SECOND
# should ideally differ from the drawn QF_FIRST.
QF_SECOND = [50, 95]

# ----------------------------------------------------------------------------
# DCT / histogram settings
# ----------------------------------------------------------------------------

# Side length (in pixels) of each square DCT block.
# Range: positive integer, normally a power of two (4, 8, 16).
# JPEG standard is 8 -> an 8x8 block contains 64 coefficients. Current: 8.
BLOCK_SIZE = 8

# Number of bins in the per-coefficient histogram feature.
# Range: positive integer. Typical values: 50, 100, 128, 256.
# More bins = finer feature resolution but larger feature vectors. Current: 100.
N_HIST_BINS = 100

# Number of AC coefficients kept per block (the DC coefficient at [0,0] is discarded).
# Range: depends on BLOCK_SIZE as (BLOCK_SIZE**2 - 1).
#   e.g. size 4 -> 15, size 8 -> 63, size 16 -> 255. Current (size 8): 63.
N_AC_COEFFS = 63

# ----------------------------------------------------------------------------
# Model / training settings
# ----------------------------------------------------------------------------

# Fraction of the dataset reserved for the test split (train_test_split).
# Range: float, 0.0 < TEST_SIZE < 1.0. Typical values: 0.1, 0.2, 0.25, 0.3. Current: 0.2.
TEST_SIZE = 0.2

# Seed used for reproducible train/test splitting and MLP initialization.
# Range: any non-negative integer. Typical values: 0, 1, 42, 2024. Current: 42.
RANDOM_STATE = 42

# Sizes of the hidden layers in the MLP classifier.
# Range: tuple of positive integers (or () for no hidden layer).
# Typical values: (64,), (128, 64), (256, 128, 64). Current: (128, 64).
MLP_HIDDEN_LAYERS = (128, 64)

# Mini-batch size used by the PyTorch DataLoader during training.
# iterations_per_epoch = ceil(n_train_samples / BATCH_SIZE). There is NO global
# "max iterations" knob anymore: training runs epoch by epoch (see
# MLP_NUM_EPOCHS) and stops early via MLP_PATIENCE when validation accuracy
# stops improving.
# Range: positive integer. Typical values: 16, 32, 64, 128, 256. Current: 64.
BATCH_SIZE = 1024

# Number of background DataLoader worker processes that prefetch batches ahead
# of time so the GPU is never starved waiting on data transfer.
# Range: non-negative integer. 0 = load in the main process. Typical: 2, 4, 8.
# Note: on Windows values >0 use process spawning; 2 is a safe default.
NUM_WORKERS = 16

# Number of parallel worker processes used for feature extraction in
# train_model.py (ProcessPoolExecutor). Each worker processes whole images
# independently. 0 or 1 = run in the main process (no pool).
# Range: non-negative integer. Typical: os.cpu_count() or half of it.
NUM_FEATURE_WORKERS = 8

# Learning rate for the Adam optimizer.
# Range: positive float, typically 1e-4 to 1e-2. Current: 1e-3.
MLP_LEARNING_RATE = 1e-3

# Upper bound on the number of training EPOCHS. Each epoch is one full pass
# over the training set (in BATCH_SIZE chunks). Early stopping
# (MLP_PATIENCE) usually ends training before this cap is reached.
# Range: positive integer. Typical values: 50, 100, 200. Current: 100.
MLP_NUM_EPOCHS = 200

# Number of EPOCHS with no validation-accuracy improvement to wait before
# stopping training early (only used when MLP_EARLY_STOPPING is True).
# Range: positive integer. Typical values: 5, 10, 20. Current: 10.
MLP_PATIENCE = 30

# Whether to stop training early when the validation score stops improving.
# Range: boolean True / False. Current: True.
MLP_EARLY_STOPPING = True

# ----------------------------------------------------------------------------
# main.py: patch-based image-evaluation settings
# ----------------------------------------------------------------------------

# Side length (in 8x8 blocks) of a square patch used for patch-based image
# voting in the final real-image evaluation. PATCH_BLOCKS=16 -> 128x128 px tile.
# Range: positive integer. Typical: 8, 16, 32. Current: 16.
PATCH_BLOCKS = 16

# Fraction (in [0,1]) of patches that must be classified as double-compressed
# for the whole image to be predicted double. Range: 0.0-1.0. Current: 0.5.
PATCH_VOTE_THRESHOLD = 0.5

# Flat AC-coefficient index (1..63, excluding DC at 0) whose histogram is
# plotted to show the double-compression "comb pattern".
# Range: 1..(BLOCK_SIZE**2 - 1). Typical: 1 (the (0,1) coefficient). Current: 1.
HIST_COEFF_IDX = 1
