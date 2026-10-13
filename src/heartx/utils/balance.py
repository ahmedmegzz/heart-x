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

import gc
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
    release_bag: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Balance classes while keeping dual-stream modalities aligned.

    Parameters
    ----------
    method :
        - ``random`` / ``random_oversample``: duplicate minority indices
          (1D, 2D, clinical copied together).
        - ``smote``: SMOTE on flattened 1D (+ clinical). Original rows keep
          their spectrograms; only **synthetic** 1D rows get a newly computed
          spectrogram (see module docstring).
    release_bag :
        Optional ``[x_1d, x_2d]`` list from the caller. Cleared after buffers
        are consumed so peak RAM drops on small hosts.
    """
    method = method.lower()
    if method in {"random", "random_oversample", "ros"}:
        from imblearn.over_sampling import RandomOverSampler

        idx = np.arange(len(y)).reshape(-1, 1)
        ros = RandomOverSampler(random_state=random_state)
        idx_res, y_res = ros.fit_resample(idx, y)
        src = idx_res[:, 0]
        out = x_1d[src], x_2d[src], clinical[src], y_res.astype(np.int64)
        if release_bag is not None:
            release_bag.clear()
            gc.collect()
        return out

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

    x1_tail_shape = x_1d.shape[1:]
    x2_tail_shape = x_2d.shape[1:]
    x1_flat = np.ascontiguousarray(x_1d.reshape(n, -1), dtype=np.float32)
    c_dim = clinical.shape[1]
    clin_f = np.ascontiguousarray(clinical, dtype=np.float32)
    feat = np.concatenate([x1_flat, clin_f], axis=1)
    del clin_f
    # 1D buffer no longer needed after feat is built
    if release_bag is not None and len(release_bag) > 0:
        release_bag[0] = None
    del x_1d
    gc.collect()

    sampler = SMOTE(random_state=random_state, k_neighbors=k)
    feat_res, y_res = sampler.fit_resample(feat, y)

    t = x1_flat.shape[1]
    # Exact float identity → original row (SMOTE copies rows as-is).
    orig_keys = {feat[i].tobytes(): i for i in range(n)}
    del feat, x1_flat
    gc.collect()

    n_out = int(len(y_res))
    x1_out = feat_res[:, :t].reshape(n_out, *x1_tail_shape).astype(np.float32, copy=False)
    clin_out = feat_res[:, t : t + c_dim].astype(np.float32, copy=False)

    src = np.full(n_out, -1, dtype=np.int64)
    is_original = np.zeros(n_out, dtype=bool)
    for i in range(n_out):
        j = orig_keys.get(feat_res[i].tobytes())
        if j is not None:
            is_original[i] = True
            src[i] = j
    del feat_res, orig_keys
    gc.collect()

    # float16 from the start — float32 buffer was ~3GB and OOM'd on 8GB hosts.
    x2_out = np.empty((n_out, *x2_tail_shape), dtype=np.float16)
    orig_pos = np.where(is_original)[0]
    x2_out[orig_pos] = np.asarray(x_2d[src[orig_pos]], dtype=np.float16)
    del orig_pos, x_2d
    if release_bag is not None:
        release_bag.clear()
    gc.collect()

    transform = stft_spectrogram if spectrogram_method == "stft" else cwt_scalogram
    synth_idx = np.where(~is_original)[0]
    print(
        f"SMOTE spectrograms: {int(is_original.sum())} reused, "
        f"{len(synth_idx)} regenerated",
        flush=True,
    )
    flat_1d = x1_out.reshape(n_out, -1)
    for bi, i in enumerate(synth_idx):
        spec = transform(flat_1d[i], fs=fs)
        x2_out[i, 0] = spec.astype(np.float16, copy=False)
        if (bi + 1) % 10000 == 0:
            print(f"  regenerated {bi + 1}/{len(synth_idx)}", flush=True)

    return x1_out, x2_out, clin_out, y_res.astype(np.int64)


def smote_balance(*args, **kwargs):
    """ Backward-compatible alias; defaults to random oversampling. """
    kwargs.setdefault("method", "random")
    return oversample_balance(*args, **kwargs)
