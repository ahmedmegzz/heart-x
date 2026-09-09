"""PyTorch dataset wrapping dual-stream ECG tensors."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class DualStreamECGDataset(Dataset):
    def __init__(
        self,
        x_1d: np.ndarray,
        x_2d: np.ndarray,
        y: np.ndarray,
        clinical: np.ndarray | None = None,
    ):
        self.x_1d = torch.from_numpy(np.asarray(x_1d, dtype=np.float32))
        self.x_2d = torch.from_numpy(np.asarray(x_2d, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        if clinical is not None:
            self.clinical = torch.from_numpy(np.asarray(clinical, dtype=np.float32))
        else:
            self.clinical = None

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        item = {
            "x_1d": self.x_1d[idx],
            "x_2d": self.x_2d[idx],
            "y": self.y[idx],
        }
        if self.clinical is not None:
            item["clinical"] = self.clinical[idx]
        return item
