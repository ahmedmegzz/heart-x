#!/usr/bin/env python3
"""
Train HEART-X dual-stream model on prepared MIT-BIH data (Aşama 4–5, 7).

Example
-------
python scripts/train_heartx.py --epochs 10 --model dual
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import (  # noqa: E402
    AAMI_CLASSES,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    FIGURE_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    PROCESSED_DIR,
    SEED,
    USE_SMOTE,
    WEIGHT_DECAY,
)
from heartx.data.dataset import DualStreamECGDataset  # noqa: E402
from heartx.models.dual_stream import (  # noqa: E402
    HeartXDualStream,
    SingleStream1D,
    SingleStream2D,
)
from heartx.utils.balance import oversample_balance, set_seed  # noqa: E402
from heartx.utils.train_utils import evaluate, fit  # noqa: E402
from heartx.utils.viz import plot_confusion_matrix  # noqa: E402


def build_model(name: str, num_classes: int, clinical_dim: int):
    if name == "dual":
        return HeartXDualStream(
            num_classes=num_classes, clinical_dim=clinical_dim
        )
    if name == "dual_noclinical":
        return HeartXDualStream(num_classes=num_classes, clinical_dim=0)
    if name == "1d":
        return SingleStream1D(num_classes=num_classes)
    if name == "2d":
        return SingleStream2D(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--model", choices=["dual", "dual_noclinical", "1d", "2d"], default="dual")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-smote", action="store_true", help="Disable class oversampling")
    parser.add_argument(
        "--balance",
        choices=["random", "smote"],
        default="random",
        help="Oversampling strategy when balancing is enabled",
    )
    parser.add_argument("--max-train", type=int, default=None, help="Subsample train set for smoke tests")
    args = parser.parse_args()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    data = np.load(args.data, allow_pickle=True)
    x_1d = data["x_1d"]
    x_2d = data["x_2d"]
    y = data["y"]
    clinical = data["clinical"]
    train_mask = data["train_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)
    class_names = [str(c) for c in data["class_names"]]

    x1_tr, x2_tr, y_tr, c_tr = (
        x_1d[train_mask],
        x_2d[train_mask],
        y[train_mask],
        clinical[train_mask],
    )
    x1_te, x2_te, y_te, c_te = (
        x_1d[test_mask],
        x_2d[test_mask],
        y[test_mask],
        clinical[test_mask],
    )

    # Hold out validation from DS1 train patients
    idx = np.arange(len(y_tr))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.15, random_state=SEED, stratify=y_tr
    )
    if args.max_train is not None:
        tr_idx = tr_idx[: args.max_train]

    x1_train, x2_train, y_train, c_train = (
        x1_tr[tr_idx],
        x2_tr[tr_idx],
        y_tr[tr_idx],
        c_tr[tr_idx],
    )
    x1_val, x2_val, y_val, c_val = (
        x1_tr[va_idx],
        x2_tr[va_idx],
        y_tr[va_idx],
        c_tr[va_idx],
    )

    use_balance = USE_SMOTE and not args.no_smote
    if use_balance:
        print(f"Balancing training split ({args.balance})...", flush=True)
        x1_train, x2_train, c_train, y_train = oversample_balance(
            x1_train,
            x2_train,
            c_train,
            y_train,
            random_state=SEED,
            method=args.balance,
        )
        print(f"After balance: {len(y_train)} samples", flush=True)

    clinical_dim = c_train.shape[1] if args.model == "dual" else 0

    def make_loader(x1, x2, yy, cc, shuffle):
        clin = cc if clinical_dim > 0 else None
        ds = DualStreamECGDataset(x1, x2, yy, clin)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    train_loader = make_loader(x1_train, x2_train, y_train, c_train, True)
    val_loader = make_loader(x1_val, x2_val, y_val, c_val, False)
    test_loader = make_loader(x1_te, x2_te, y_te, c_te, False)

    # Class weights from (pre-SMOTE) training labels for stability
    counts = np.bincount(y_tr[tr_idx], minlength=len(class_names)).astype(np.float64)
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model = build_model(args.model, num_classes=len(class_names), clinical_dim=clinical_dim)
    model = model.to(device)
    ckpt = CHECKPOINT_DIR / f"heartx_{args.model}.pt"

    history = fit(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        class_names=class_names,
        checkpoint_path=ckpt,
        class_weights=class_weights,
    )

    # Load best checkpoint and evaluate on DS2
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_metrics = evaluate(model, test_loader, device, class_names)
    print("\n=== Test (DS2) ===")
    print(test_metrics["report"])
    print(
        f"acc={test_metrics['accuracy']:.4f}  "
        f"f1_macro={test_metrics['f1_macro']:.4f}  "
        f"recall={test_metrics['recall_macro']:.4f}  "
        f"specificity={test_metrics['specificity_macro']:.4f}"
    )

    fig_dir = FIGURE_DIR / args.model
    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        class_names,
        fig_dir / "confusion_matrix.png",
        title=f"HEART-X ({args.model}) — MIT-BIH DS2",
    )

    results = {
        "model": args.model,
        "epochs": args.epochs,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_specificity_macro": test_metrics["specificity_macro"],
        "history": history,
        "checkpoint": str(ckpt),
    }
    out_json = CHECKPOINT_DIR / f"results_{args.model}.json"
    # history values are plain floats already
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
