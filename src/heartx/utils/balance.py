"""Balance helpers and deterministic seeding.

SMOTE policy (thesis-critical)
------------------------------
``method='smote'`` interpolates in the **1D waveform (+ clinical)** feature
space only. Spectrogram inputs for synthetic samples are **recomputed** from
the synthetic 1D signal via STFT (or CWT), so the 1D/2D pair stays physically
consistent. We do **not** run SMOTE independently in spectrogram pixel space,
which would break signal–spectrogram alignment.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def oversample_balance(
    x_1d: np.ndarray,
    x_2d: np.ndarray,
    clinical: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    method: str = "random",
    fs: float = 360.0,
    spectrogram_method: str = "stft",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Balance classes while keeping dual-stream modalities aligned.

    Parameters
    ----------
    method :
        - ``random`` / ``random_oversample``: duplicate minority indices
          (1D, 2D, clinical copied together).
        - ``smote``: SMOTE on flattened 1D (+ clinical). Synthetic 1D rows are
          taken from SMOTE output; spectrograms are regenerated from those
          synthetic waveforms (see module docstring).
    """
    method = method.lower()
    if method in {"random", "random_oversample", "ros"}:
        from imblearn.over_sampling import RandomOverSampler

        idx = np.arange(len(y)).reshape(-1, 1)
        ros = RandomOverSampler(random_state=random_state)
        idx_res, y_res = ros.fit_resample(idx, y)
        src = idx_res[:, 0]
        return x_1d[src], x_2d[src], clinical[src], y_res.astype(np.int64)

    if method != "smote":
        raise ValueError(f"Unknown oversample method: {method}")

    from imblearn.over_sampling import SMOTE

    from heartx.features.spectrogram import cwt_scalogram, stft_spectrogram

    n = len(y)
    counts = np.bincount(y)
    min_count = int(counts[counts > 0].min())
    k = max(1, min(5, min_count - 1))
    if k < 1:
        return x_1d, x_2d, clinical, y

    # --- SMOTE in 1D (+ clinical) space ---
    x1_flat = x_1d.reshape(n, -1)
    c_dim = clinical.shape[1]
    feat = np.concatenate([x1_flat, clinical], axis=1).astype(np.float32)
    sampler = SMOTE(random_state=random_state, k_neighbors=k)
    feat_res, y_res = sampler.fit_resample(feat, y)

    t = x1_flat.shape[1]
    x1_out = feat_res[:, :t].reshape(-1, *x_1d.shape[1:]).astype(np.float32)
    clin_out = feat_res[:, t : t + c_dim].astype(np.float32)

    # --- Regenerate spectrograms from (possibly synthetic) 1D ---
    transform = stft_spectrogram if spectrogram_method == "stft" else cwt_scalogram
    specs = []
    flat_1d = x1_out.reshape(len(x1_out), -1)
    for i in range(len(flat_1d)):
        specs.append(transform(flat_1d[i], fs=fs))
    x2_out = np.stack(specs, axis=0)[:, None, :, :].astype(np.float32)

    return x1_out, x2_out, clin_out, y_res.astype(np.int64)


def smote_balance(*args, **kwargs):
    """Backward-compatible alias; defaults to random oversampling. """
    kwargs.setdefault("method", "random")
    return oversample_balance(*args, **kwargs)
