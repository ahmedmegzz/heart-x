"""PTB-XL Diagnostic ECG loader (superclass multi-label / single-label)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from heartx.config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    NOTCH_FREQ_HZ,
    PTBXL_CLASSES,
    PTBXL_DIR,
    PTBXL_FS,
    PTBXL_TO_IDX,
)
from heartx.data.preprocess import preprocess_ecg, zscore


def _load_statements(path: Path) -> pd.DataFrame:
    agg = pd.read_csv(path / "scp_statements.csv", index_col=0)
    return agg[agg.diagnostic == 1]


def aggregate_diagnostic(scp_codes: dict, agg_df: pd.DataFrame) -> list[str]:
    classes = []
    for key in scp_codes:
        if key in agg_df.index:
            classes.append(agg_df.loc[key].diagnostic_class)
    return sorted(set(classes))


def load_ptbxl_metadata(data_dir: Path | None = None) -> pd.DataFrame:
    root = Path(data_dir or PTBXL_DIR)
    y = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    y.scp_codes = y.scp_codes.apply(ast.literal_eval)
    agg = _load_statements(root)
    y["diagnostic_superclass"] = y.scp_codes.apply(
        lambda d: aggregate_diagnostic(d, agg)
    )
    return y


def load_ptbxl_signals(
    meta: pd.DataFrame,
    data_dir: Path | None = None,
    sampling_rate: int = PTBXL_FS,
    lead: int = 1,
    max_records: int | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load lead-wise 10 s ECG windows with single primary superclass labels.

    Returns
    -------
    X : (N, T) float32
    y : (N,) int64  primary superclass index
    folds : (N,) strat_fold
    multi_hot : (N, C) float32 multi-label targets
    """
    root = Path(data_dir or PTBXL_DIR)
    df = meta.copy()
    # Prefer records that have at least one diagnostic superclass
    df = df[df.diagnostic_superclass.map(len) > 0]
    if max_records is not None:
        df = df.iloc[:max_records]

    signals = []
    labels = []
    folds = []
    multi = []

    iterator = (
        tqdm(df.itertuples(), total=len(df), desc="PTB-XL records")
        if show_progress
        else df.itertuples()
    )
    fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    for row in iterator:
        fname = getattr(row, fname_col)
        signal, _ = wfdb.rdsamp(str(root / fname))
        lead_sig = signal[:, lead]
        clean = zscore(
            preprocess_ecg(
                lead_sig,
                fs=float(sampling_rate),
                lowcut=BANDPASS_LOW_HZ,
                highcut=BANDPASS_HIGH_HZ,
                notch_freq=NOTCH_FREQ_HZ,
            )
        ).astype(np.float32)

        classes = row.diagnostic_superclass
        # Primary label: NORM if present else first diagnostic class
        if "NORM" in classes and len(classes) == 1:
            primary = "NORM"
        else:
            non_norm = [c for c in classes if c != "NORM"]
            primary = non_norm[0] if non_norm else classes[0]
        if primary not in PTBXL_TO_IDX:
            continue

        mh = np.zeros(len(PTBXL_CLASSES), dtype=np.float32)
        for c in classes:
            if c in PTBXL_TO_IDX:
                mh[PTBXL_TO_IDX[c]] = 1.0

        signals.append(clean)
        labels.append(PTBXL_TO_IDX[primary])
        folds.append(int(row.strat_fold))
        multi.append(mh)

    x = np.stack(signals, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    fold_arr = np.asarray(folds, dtype=np.int64)
    multi_hot = np.stack(multi, axis=0)
    return x, y, fold_arr, multi_hot
