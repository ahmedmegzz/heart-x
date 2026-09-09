"""Dual-stream CNN-LSTM with optional clinical feature fusion (HEART-X)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SignalStream1D(nn.Module):
    """1D CNN + BiLSTM stream for raw ECG morphology / rhythm."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=out_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 1, T)
        feat = self.features(x)  # (B, C, T')
        self.last_conv = feat
        seq = feat.transpose(1, 2)  # (B, T', C)
        out, _ = self.lstm(seq)
        emb = out.mean(dim=1)
        return emb, feat


class SpectrogramStream2D(nn.Module):
    """2D CNN stream for STFT / CWT images."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        self.last_conv = feat
        emb = self.fc(feat.flatten(1))
        return emb, feat


class HeartXDualStream(nn.Module):
    """
    Dual-stream HEART-X classifier.

    Streams
    -------
    - 1D CNN-LSTM on raw beat / window
    - 2D CNN on spectrogram
    - optional clinical feature MLP fused before the classifier head
    """

    def __init__(
        self,
        num_classes: int = 5,
        emb_dim: int = 128,
        clinical_dim: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stream_1d = SignalStream1D(out_dim=emb_dim)
        self.stream_2d = SpectrogramStream2D(out_dim=emb_dim)
        self.clinical_dim = clinical_dim

        fused_dim = emb_dim * 2
        if clinical_dim > 0:
            self.clinical_mlp = nn.Sequential(
                nn.Linear(clinical_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, emb_dim),
                nn.ReLU(inplace=True),
            )
            fused_dim += emb_dim
        else:
            self.clinical_mlp = None

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e1, _ = self.stream_1d(x_1d)
        e2, _ = self.stream_2d(x_2d)
        parts = [e1, e2]
        if self.clinical_mlp is not None:
            if clinical is None:
                raise ValueError("clinical features required but not provided")
            parts.append(self.clinical_mlp(clinical))
        fused = torch.cat(parts, dim=1)
        return self.classifier(fused)

    def embeddings(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e1, _ = self.stream_1d(x_1d)
        e2, _ = self.stream_2d(x_2d)
        parts = [e1, e2]
        if self.clinical_mlp is not None and clinical is not None:
            parts.append(self.clinical_mlp(clinical))
        return torch.cat(parts, dim=1)


class SingleStream1D(nn.Module):
    """Baseline: 1D CNN-LSTM only."""

    def __init__(self, num_classes: int = 5, emb_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.stream = SignalStream1D(out_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_1d: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        emb, _ = self.stream(x_1d)
        return self.head(emb)


class SingleStream2D(nn.Module):
    """Baseline: spectrogram CNN only."""

    def __init__(self, num_classes: int = 5, emb_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.stream = SpectrogramStream2D(out_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        emb, _ = self.stream(x_2d)
        return self.head(emb)
