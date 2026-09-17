"""Multi-modal representations: 1D signal + STFT / CWT spectrograms."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.ndimage import zoom

from heartx.config import SPECTROGRAM_SIZE, STFT_NOVERLAP, STFT_NPERSEG

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None


def _resize2d(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if img.shape == size:
        return img.astype(np.float32)
    factors = (size[0] / img.shape[0], size[1] / img.shape[1])
    return zoom(img, factors, order=1).astype(np.float32)


def stft_spectrogram(
    x: np.ndarray,
    fs: float,
    nperseg: int = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
    size: tuple[int, int] = SPECTROGRAM_SIZE,
) -> np.ndarray:
    """Return log-magnitude STFT resized to ``size`` (freq, time)."""
    nperseg = min(nperseg, len(x))
    noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0
    _, _, zxx = sps.stft(
        x.astype(np.float64),
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )
    mag = np.log1p(np.abs(zxx))
    mag = (mag - mag.mean()) / (mag.std() + 1e-8)
    return _resize2d(mag, size)


def cwt_scalogram(
    x: np.ndarray,
    fs: float,
    wavelet: str = "morl",
    num_scales: int = 64,
    size: tuple[int, int] = SPECTROGRAM_SIZE,
) -> np.ndarray:
    """Continuous wavelet transform magnitude (fallback to STFT if PyWavelets missing)."""
    if pywt is None:
        return stft_spectrogram(x, fs=fs, size=size)
    scales = np.arange(1, num_scales + 1)
    coeffs, _ = pywt.cwt(x.astype(np.float64), scales, wavelet, sampling_period=1.0 / fs)
    mag = np.log1p(np.abs(coeffs))
    mag = (mag - mag.mean()) / (mag.std() + 1e-8)
    return _resize2d(mag, size)


def make_dual_reps(
    signals: np.ndarray,
    fs: float,
    method: str = "stft",
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build dual-stream tensors from a batch of 1D windows.

    Parameters
    ----------
    signals : (N, T)
    method : 'stft' or 'cwt'

    Returns
    -------
    x_1d : (N, 1, T)
    x_2d : (N, 1, H, W)
    """
    from tqdm import tqdm

    n = len(signals)
    specs = []
    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Building {method.upper()} spectrograms")

    transform = stft_spectrogram if method == "stft" else cwt_scalogram
    for i in iterator:
        specs.append(transform(signals[i], fs=fs))

    x_1d = signals[:, None, :].astype(np.float32)
    x_2d = np.stack(specs, axis=0)[:, None, :, :].astype(np.float32)
    return x_1d, x_2d

# Backward-compatible alias
make_dual_representations = make_dual_reps
