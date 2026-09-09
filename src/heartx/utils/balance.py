"""Balance helpers and deterministic seeding."""

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Balance classes by resampling indices (keeps 1D/2D pairs aligned).

    method='random' uses RandomOverSampler (fast, recommended).
    method='smote' runs SMOTE in a compact 1D+clinical space and copies
    nearest-neighbor spectrograms for synthetic rows.
    """
    if method == "random":
        from imblearn.over_sampling import RandomOverSampler

        idx = np.arange(len(y)).reshape(-1, 1)
        ros = RandomOverSampler(random_state=random_state)
        idx_res, y_res = ros.fit_resample(idx, y)
        src = idx_res[:, 0]
        return x_1d[src], x_2d[src], clinical[src], y_res.astype(np.int64)

    # Compact SMOTE path
    from imblearn.over_sampling import SMOTE
    from sklearn.neighbors import NearestNeighbors

    n = len(y)
    counts = np.bincount(y)
    min_count = int(counts[counts > 0].min())
    k = max(1, min(5, min_count - 1))
    if k < 1:
        return x_1d, x_2d, clinical, y

    x1_flat = x_1d.reshape(n, -1)
    compact = np.concatenate([x1_flat[:, ::4], clinical], axis=1).astype(np.float32)
    sampler = SMOTE(random_state=random_state, k_neighbors=k)
    compact_res, y_res = sampler.fit_resample(compact, y)

    nn = NearestNeighbors(n_neighbors=1).fit(compact)
    _, idxs = nn.kneighbors(compact_res)
    src = idxs[:, 0]
    c_dim = clinical.shape[1]
    clin_out = compact_res[:, -c_dim:].astype(np.float32)
    return (
        x_1d[src].astype(np.float32),
        x_2d[src].astype(np.float32),
        clin_out,
        y_res.astype(np.int64),
    )


# Backward-compatible alias
def smote_balance(*args, **kwargs):
    kwargs.setdefault("method", "random")
    return oversample_balance(*args, **kwargs)
