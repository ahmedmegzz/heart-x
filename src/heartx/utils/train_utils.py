"""Training and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def _batch_inputs(batch: dict, device: torch.device):
    x_1d = batch["x_1d"].to(device)
    x_2d = batch["x_2d"].to(device)
    y = batch["y"].to(device)
    clinical = batch["clinical"].to(device) if "clinical" in batch else None
    return x_1d, x_2d, y, clinical


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    ys, preds = [], []
    for batch in loader:
        x_1d, x_2d, y, clinical = _batch_inputs(batch, device)
        logits = model(x_1d, x_2d, clinical)
        pred = logits.argmax(dim=1)
        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)

    # Specificity (macro): TN / (TN + FP) per class
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp + 1e-8))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "specificity_macro": float(np.mean(specificity)) if specificity else 0.0,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            digits=4,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x_1d, x_2d, y, clinical = _batch_inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_1d, x_2d, clinical)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total_loss / max(n, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_names: list[str],
    checkpoint_path: Path,
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, class_names)
        scheduler.step(val_metrics["f1_macro"])
        history["train_loss"].append(loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"loss={loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": best_f1,
                    "class_names": class_names,
                },
                checkpoint_path,
            )
            print(f"  saved checkpoint -> {checkpoint_path}")

    return history
