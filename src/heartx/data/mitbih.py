"""MIT-BIH Arrhythmia Database loader with AAMI beat labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import wfdb
from tqdm import tqdm

from heartx.config import (
    AAMI_MAP,
    AAMI_TO_IDX,
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BEAT_POST_S,
    BEAT_PRE_S,
    MITBIH_DIR,
    MITBIH_FS,
    NOTCH_FREQ_HZ,
)
from heartx.data.preprocess import preprocess_ecg, zscore


@dataclass
class BeatSample:
    signal_1d: np.ndarray
    label: str
    label_idx: int
    record_id: str
    r_peak: int
    rr_pre: float
    rr_post: float
    clinical: np.ndarray


def list_mitbih_records(data_dir: Path | None = None) -> list[str]:
    root = Path(data_dir or MITBIH_DIR)
    records = sorted({p.stem for p in root.glob("*.hea")})
    # Prefer MLII / lead 0 records that have .atr annotations
    return [r for r in records if (root / f"{r}.atr").exists()]


def _aami_label(symbol: str) -> str | None:
    return AAMI_MAP.get(symbol)


def _extract_window(
    signal: np.ndarray,
    center: int,
    pre: int,
    post: int,
) -> np.ndarray | None:
    start = center - pre
    end = center + post
    if start < 0 or end > len(signal):
        return None
    return signal[start:end].astype(np.float32)


def _clinical_features(
    beat: np.ndarray,
    rr_pre: float,
    rr_post: float,
    rr_local: list[float],
    fs: float,
) -> np.ndarray:
    local = np.asarray(rr_local, dtype=np.float64)
    rr_mean = float(local.mean()) if len(local) else rr_pre
    rr_std = float(local.std()) if len(local) > 1 else 0.0
    rr_ratio = rr_pre / (rr_post + 1e-8)

    # Approximate QRS width via samples above 50% of peak amplitude around R
    mid = len(beat) // 2
    half = int(0.12 * fs)
    segment = beat[max(0, mid - half) : mid + half]
    thr = 0.5 * np.max(np.abs(segment)) if len(segment) else 0.0
    above = np.where(np.abs(segment) >= thr)[0]
    qrs_approx = (above[-1] - above[0]) / fs if len(above) >= 2 else 0.08

    return np.array(
        [
            rr_pre,
            rr_post,
            rr_mean,
            rr_std,
            rr_ratio,
            qrs_approx,
            float(np.sqrt(np.mean(beat**2))),
            float(np.ptp(beat)),
        ],
        dtype=np.float32,
    )


def load_mitbih_beats(
    data_dir: Path | None = None,
    records: Iterable[str] | None = None,
    channel: int = 0,
    fs: float = MITBIH_FS,
    max_beats_per_record: int | None = None,
    show_progress: bool = True,
) -> list[BeatSample]:
    """Load filtered beat windows + AAMI labels from MIT-BIH."""
    root = Path(data_dir or MITBIH_DIR)
    record_ids = list(records) if records is not None else list_mitbih_records(root)
    pre = int(BEAT_PRE_S * fs)
    post = int(BEAT_POST_S * fs)
    samples: list[BeatSample] = []

    iterator = tqdm(record_ids, desc="MIT-BIH records") if show_progress else record_ids
    for rec_id in iterator:
        record = wfdb.rdrecord(str(root / rec_id))
        ann = wfdb.rdann(str(root / rec_id), "atr")
        raw = record.p_signal[:, channel]
        clean = zscore(
            preprocess_ecg(
                raw,
                fs=fs,
                lowcut=BANDPASS_LOW_HZ,
                highcut=BANDPASS_HIGH_HZ,
                notch_freq=NOTCH_FREQ_HZ,
            )
        )

        # Keep annotated beats that map to AAMI classes
        beats = []
        for sample, symbol in zip(ann.sample, ann.symbol):
            label = _aami_label(symbol)
            if label is None:
                continue
            beats.append((int(sample), label))

        if max_beats_per_record is not None:
            beats = beats[:max_beats_per_record]

        r_peaks = [b[0] for b in beats]
        for i, (r_peak, label) in enumerate(beats):
            window = _extract_window(clean, r_peak, pre, post)
            if window is None:
                continue

            rr_pre = (
                (r_peaks[i] - r_peaks[i - 1]) / fs if i > 0 else (pre + post) / fs
            )
            rr_post = (
                (r_peaks[i + 1] - r_peaks[i]) / fs
                if i < len(r_peaks) - 1
                else rr_pre
            )
            local_rr = []
            for j in range(max(1, i - 2), i + 1):
                if j < len(r_peaks):
                    local_rr.append((r_peaks[j] - r_peaks[j - 1]) / fs)

            clinical = _clinical_features(window, rr_pre, rr_post, local_rr, fs)
            samples.append(
                BeatSample(
                    signal_1d=window,
                    label=label,
                    label_idx=AAMI_TO_IDX[label],
                    record_id=rec_id,
                    r_peak=r_peak,
                    rr_pre=rr_pre,
                    rr_post=rr_post,
                    clinical=clinical,
                )
            )

    return samples


def beats_to_arrays(
    beats: list[BeatSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack beats into (X_1d, y, clinical, record_ids)."""
    if not beats:
        raise ValueError("No beats to convert.")
    x = np.stack([b.signal_1d for b in beats], axis=0)
    y = np.array([b.label_idx for b in beats], dtype=np.int64)
    c = np.stack([b.clinical for b in beats], axis=0)
    records = np.array([b.record_id for b in beats])
    return x, y, c, records
