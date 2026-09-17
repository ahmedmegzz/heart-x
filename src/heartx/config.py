"""Central configuration for the HEART-X ECG pipeline."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
MITBIH_DIR = DATA_ROOT / "mit-bih-arrhythmia-database-1.0.0"
PTBXL_DIR = (
    DATA_ROOT
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
MITBIH_FS = 360
PTBXL_FS = 100  # use low-resolution records by default (memory-friendly)
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
NOTCH_FREQ_HZ = 50.0

# Beat window: 0.4 s before / 0.6 s after R-peak  (~360 samples @ 360 Hz)
BEAT_PRE_S = 0.4
BEAT_POST_S = 0.6

# ---------------------------------------------------------------------------
# AAMI beat mapping (MIT-BIH)
# ---------------------------------------------------------------------------
AAMI_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_TO_IDX = {c: i for i, c in enumerate(AAMI_CLASSES)}

# PTB-XL diagnostic superclasses
PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
PTBXL_TO_IDX = {c: i for i, c in enumerate(PTBXL_CLASSES)}

# ---------------------------------------------------------------------------
# Dual-stream / spectrogram
# ---------------------------------------------------------------------------
STFT_NPERSEG = 64
STFT_NOVERLAP = 32
SPECTROGRAM_SIZE = (64, 64)  # (freq, time) after resize

# ---------------------------------------------------------------------------
# Clinical features
# ---------------------------------------------------------------------------
CLINICAL_FEATURE_NAMES = [
    "rr_pre",
    "rr_post",
    "rr_local_mean",
    "rr_local_std",
    "rr_ratio",
    "qrs_approx",
    "beat_rms",
    "beat_peak_to_peak",
]

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DEVICE = "cpu"  # override to "cuda" when available
VAL_RATIO = 0.15
TEST_RATIO = 0.15
USE_SMOTE = True
MAX_BEATS_PER_CLASS = None  # set e.g. 5000 for quick experiments

# Inter-patient split (DS1 / DS2) commonly used for MIT-BIH
MITBIH_DS1 = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
]
MITBIH_DS2 = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]