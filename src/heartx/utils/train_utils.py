"""Training and evaluation helpers (macro-F1 primary metric)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from heartx.utils.losses import build_criterion


def _batch_inputs(batch: dict, device: torch.device):
    x_1d = batch["x_1d"].to(device) if "x_1d" in batch else None
    x_2d = batch["x_2d"].to(device) if "x_2d" in batch else None
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
    labels = list(range(len(class_names))) if class_names else None

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(float(tn / (tn + fp + 1e-8)))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    if class_names is not None:
        for i, name in enumerate(class_names):
            per_class[name] = {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "specificity": specificity[i] if i < len(specificity) else 0.0,
            }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "specificity_macro": float(np.mean(specificity)) if specificity else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            digits=4,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return metrics


def save_eval_artifacts(
    metrics: dict[str, Any],
    class_names: list[str],
    out_dir: Path,
    prefix: str = "test",
) -> None:
    """Write confusion matrix CSV and per-class metrics CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics["confusion_matrix"])
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / f"{prefix}_confusion_matrix.csv"
    )
    if metrics.get("per_class"):
        rows = [{"class": k, **v} for k, v in metrics["per_class"].items()]
        pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_per_class_metrics.csv", index=False)


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
    loss_name: str = "ce",
    focal_gamma: float = 2.0,
) -> dict[str, Any]:
    """Train with **macro-F1** as the checkpoint / LR-scheduler criterion."""
    criterion = build_criterion(loss_name, class_weights, focal_gamma=focal_gamma)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)

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
            f"val_macroF1={val_metrics['f1_macro']:.4f}",
            flush=True,
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_f1_macro": best_f1,
                    "val_accuracy": val_metrics["accuracy"],
                    "class_names": class_names,
                    "primary_metric": "f1_macro",
                },
                checkpoint_path,
            )
            print(f"  saved best macro-F1={best_f1:.4f} -> {checkpoint_path}", flush=True)

    return history
