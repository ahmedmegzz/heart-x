"""PyTorch dataset wrapping dual-stream ECG tensors."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class DualStreamECGDataset(Dataset):
    """Lazy dual-stream dataset.

    Keeps modalities as NumPy (spectrogram preferably float16) and converts
    only the indexed sample in ``__getitem__`` to avoid materializing ~GB of
    float32 torch tensors on machines with limited RAM.
    """

    def __init__(
        self,
        x_1d: np.ndarray | None,
        x_2d: np.ndarray | None,
        y: np.ndarray,
        clinical: np.ndarray | None = None,
    ):
        self.x_1d = None if x_1d is None else np.asarray(x_1d)
        self.x_2d = None if x_2d is None else np.asarray(x_2d)
        self.y = np.asarray(y, dtype=np.int64)
        self.clinical = None if clinical is None else np.asarray(clinical)
        n = int(self.y.shape[0])
        for name, arr in (("x_1d", self.x_1d), ("x_2d", self.x_2d), ("clinical", self.clinical)):
            if arr is not None and int(arr.shape[0]) != n:
                raise ValueError(f"{name} length {arr.shape[0]} != y length {n}")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        item = {"y": torch.tensor(self.y[idx], dtype=torch.int64)}
        if self.x_1d is not None:
            item["x_1d"] = torch.from_numpy(np.asarray(self.x_1d[idx], dtype=np.float32))
        if self.x_2d is not None:
            item["x_2d"] = torch.from_numpy(np.asarray(self.x_2d[idx], dtype=np.float32))
        if self.clinical is not None:
            item["clinical"] = torch.from_numpy(np.asarray(self.clinical[idx], dtype=np.float32))
        return item
