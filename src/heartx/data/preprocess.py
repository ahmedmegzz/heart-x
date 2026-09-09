"""Shared ECG filtering helpers (baseline wander, band-pass, notch)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def butter_bandpass(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)
    if low >= high:
        return signal.astype(np.float64)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal.astype(np.float64))


def notch_filter(
    signal: np.ndarray,
    freq: float,
    fs: float,
    q: float = 30.0,
) -> np.ndarray:
    if freq >= 0.5 * fs:
        return signal.astype(np.float64)
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, signal.astype(np.float64))


def preprocess_ecg(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    notch_freq: float | None = 50.0,
) -> np.ndarray:
    """Baseline / EMG cleanup + optional powerline notch."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = butter_bandpass(x, lowcut, highcut, fs)
    if notch_freq is not None:
        x = notch_filter(x, notch_freq, fs)
    return x


def zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)
