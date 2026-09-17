"""Loss functions for imbalanced ECG classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "weight", weight if weight is not None else torch.tensor([])
        )
        self._has_weight = weight is not None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target = target.long()
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -(1.0 - pt) ** self.gamma * log_pt
        if self._has_weight and self.weight.numel() > 0:
            w = self.weight.to(logits.device)[target]
            loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(
    loss_name: str,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    name = loss_name.lower()
    if name in {"ce", "cross_entropy", "crossentropy"}:
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    raise ValueError(f"Unknown loss: {loss_name}")
