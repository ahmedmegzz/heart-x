"""Dual-stream CNN-LSTM with optional clinical feature fusion (HEART-X)."""

from __future__ import annotations

import torch
import torch.nn as nn

from heartx.models.fusion import build_fusion


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
        feat = self.features(x)
        self.last_conv = feat
        seq = feat.transpose(1, 2)
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


class ClinicalMLP(nn.Module):
    """MLP over handcrafted clinical features."""

    def __init__(self, clinical_dim: int, out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeartXDualStream(nn.Module):
    """
    Dual-stream HEART-X classifier with selectable fusion.

    Parameters
    ----------
    use_1d, use_2d, use_clinical : modality switches for ablation
    fusion_type : 'concat' | 'gated' | 'attention'
    """

    def __init__(
        self,
        num_classes: int = 5,
        emb_dim: int = 128,
        clinical_dim: int = 0,
        dropout: float = 0.3,
        fusion_type: str = "concat",
        use_1d: bool = True,
        use_2d: bool = True,
        use_clinical: bool = True,
    ):
        super().__init__()
        self.use_1d = use_1d
        self.use_2d = use_2d
        self.use_clinical = use_clinical and clinical_dim > 0
        self.clinical_dim = clinical_dim if self.use_clinical else 0
        self.fusion_type = fusion_type
        self.emb_dim = emb_dim

        self.stream_1d = SignalStream1D(out_dim=emb_dim) if self.use_1d else None
        self.stream_2d = SpectrogramStream2D(out_dim=emb_dim) if self.use_2d else None
        self.clinical_mlp = (
            ClinicalMLP(self.clinical_dim, emb_dim, dropout)
            if self.use_clinical
            else None
        )

        n_mod = int(self.use_1d) + int(self.use_2d) + int(self.use_clinical)
        if n_mod == 0:
            raise ValueError("At least one modality must be enabled")
        self.fusion = build_fusion(fusion_type, emb_dim, n_mod)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _parts(
        self,
        x_1d: torch.Tensor | None,
        x_2d: torch.Tensor | None,
        clinical: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        parts: list[torch.Tensor] = []
        if self.stream_1d is not None:
            if x_1d is None:
                raise ValueError("x_1d required")
            e1, _ = self.stream_1d(x_1d)
            parts.append(e1)
        if self.stream_2d is not None:
            if x_2d is None:
                raise ValueError("x_2d required")
            e2, _ = self.stream_2d(x_2d)
            parts.append(e2)
        if self.clinical_mlp is not None:
            if clinical is None:
                raise ValueError("clinical features required")
            parts.append(self.clinical_mlp(clinical))
        return parts

    def forward(
        self,
        x_1d: torch.Tensor | None = None,
        x_2d: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = self._parts(x_1d, x_2d, clinical)
        fused = self.fusion(parts)
        return self.classifier(fused)

    def embeddings(
        self,
        x_1d: torch.Tensor | None = None,
        x_2d: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = self._parts(x_1d, x_2d, clinical)
        return self.fusion(parts)

    def fusion_weights(self) -> torch.Tensor | None:
        return getattr(self.fusion, "last_weights", None)


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


class ClinicalOnlyMLP(nn.Module):
    """Baseline: clinical features MLP only."""

    def __init__(
        self,
        clinical_dim: int,
        num_classes: int = 5,
        emb_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = ClinicalMLP(clinical_dim, emb_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x_1d: torch.Tensor | None = None,
        x_2d: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clinical is None:
            raise ValueError("clinical required")
        return self.head(self.mlp(clinical))


def build_heartx_model(
    name: str,
    num_classes: int,
    clinical_dim: int,
    fusion_type: str = "concat",
) -> nn.Module:
    """Factory used by training / ablation scripts."""
    name = name.lower()
    if name in {"dual", "1d_2d_clinical", "full"}:
        return HeartXDualStream(
            num_classes=num_classes,
            clinical_dim=clinical_dim,
            fusion_type=fusion_type,
            use_1d=True,
            use_2d=True,
            use_clinical=True,
        )
    if name in {"dual_noclinical", "1d_2d"}:
        return HeartXDualStream(
            num_classes=num_classes,
            clinical_dim=0,
            fusion_type=fusion_type,
            use_1d=True,
            use_2d=True,
            use_clinical=False,
        )
    if name == "1d":
        return SingleStream1D(num_classes=num_classes)
    if name == "2d":
        return SingleStream2D(num_classes=num_classes)
    if name in {"clinical", "clin"}:
        return ClinicalOnlyMLP(clinical_dim=clinical_dim, num_classes=num_classes)
    if name == "1d_clinical":
        return HeartXDualStream(
            num_classes=num_classes,
            clinical_dim=clinical_dim,
            fusion_type=fusion_type,
            use_1d=True,
            use_2d=False,
            use_clinical=True,
        )
    if name == "2d_clinical":
        return HeartXDualStream(
            num_classes=num_classes,
            clinical_dim=clinical_dim,
            fusion_type=fusion_type,
            use_1d=False,
            use_2d=True,
            use_clinical=True,
        )
    raise ValueError(f"Unknown model: {name}")
