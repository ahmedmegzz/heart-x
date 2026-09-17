#!/usr/bin/env python3
"""
Train HEART-X dual-stream model (macro-F1 primary metric).

Examples
--------
python scripts/train_heartx.py --model dual --fusion gated --epochs 10
python scripts/train_heartx.py --model dual --loss focal --class-weight balanced --sampler none
python scripts/train_heartx.py --model 1d_clinical --fusion concat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    FIGURE_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    SEED,
    WEIGHT_DECAY,
)
from heartx.data.dataset import DualStreamECGDataset  # noqa: E402
from heartx.models.dual_stream import build_heartx_model  # noqa: E402
from heartx.utils.balance import oversample_balance, set_seed  # noqa: E402
from heartx.utils.train_utils import evaluate, fit, save_eval_artifacts  # noqa: E402
from heartx.utils.viz import plot_confusion_matrix  # noqa: E402

MODEL_CHOICES = [
    "dual",
    "dual_noclinical",
    "1d",
    "2d",
    "clinical",
    "1d_clinical",
    "2d_clinical",
    "1d_2d",
    "1d_2d_clinical",
]


def _needs_clinical(model_name: str) -> bool:
    return model_name in {
        "dual",
        "clinical",
        "1d_clinical",
        "2d_clinical",
        "1d_2d_clinical",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="dual")
    parser.add_argument("--fusion", choices=["concat", "gated", "attention"], default="concat")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Loss class weighting (computed on pre-oversample train labels)",
    )
    parser.add_argument(
        "--sampler",
        choices=["none", "random_oversample", "smote"],
        default="none",
        help="Train-set oversampling. smote regenerates spectrograms from synthetic 1D.",
    )
    # Backward-compatible aliases
    parser.add_argument("--no-smote", action="store_true", help="Deprecated: use --sampler none")
    parser.add_argument(
        "--balance",
        choices=["random", "smote"],
        default=None,
        help="Deprecated alias for --sampler",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    set_seed(args.seed)

    # Resolve sampler from legacy flags
    sampler = args.sampler
    if args.no_smote:
        sampler = "none"
    if args.balance is not None:
        sampler = "random_oversample" if args.balance == "random" else "smote"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    data = np.load(args.data, allow_pickle=True)
    x_1d = data["x_1d"]
    x_2d = data["x_2d"]
    y = data["y"]
    clinical = data["clinical"]
    train_mask = data["train_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)
    if "val_mask" in data.files:
        val_mask = data["val_mask"].astype(bool)
    else:
        # Backward compat: no patient-wise val in old npz — leave empty and error
        raise RuntimeError(
            "Dataset missing val_mask. Re-run scripts/prepare_mitbih.py "
            "(patient-wise split)."
        )
    class_names = [str(c) for c in data["class_names"]]
    fs = float(data["fs"]) if "fs" in data.files else 360.0
    method = str(data["method"]) if "method" in data.files else "stft"

    x1_train, x2_train, y_train, c_train = (
        x_1d[train_mask],
        x_2d[train_mask],
        y[train_mask],
        clinical[train_mask],
    )
    x1_val, x2_val, y_val, c_val = (
        x_1d[val_mask],
        x_2d[val_mask],
        y[val_mask],
        clinical[val_mask],
    )
    x1_te, x2_te, y_te, c_te = (
        x_1d[test_mask],
        x_2d[test_mask],
        y[test_mask],
        clinical[test_mask],
    )

    if args.max_train is not None:
        x1_train = x1_train[: args.max_train]
        x2_train = x2_train[: args.max_train]
        y_train = y_train[: args.max_train]
        c_train = c_train[: args.max_train]

    # Class weights from original train labels (before oversampling)
    class_weights = None
    if args.class_weight == "balanced":
        classes = np.arange(len(class_names))
        present = np.unique(y_train)
        w = compute_class_weight("balanced", classes=present, y=y_train)
        weights = np.ones(len(class_names), dtype=np.float64)
        for cls, wi in zip(present, w):
            weights[int(cls)] = wi
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights: {dict(zip(class_names, weights.round(3)))}", flush=True)

    if sampler != "none":
        print(f"Oversampling train set ({sampler})...", flush=True)
        x1_train, x2_train, c_train, y_train = oversample_balance(
            x1_train,
            x2_train,
            c_train,
            y_train,
            random_state=args.seed,
            method=sampler,
            fs=fs,
            spectrogram_method=method,
        )
        print(f"After oversample: {len(y_train)} samples", flush=True)

    use_clin = _needs_clinical(args.model)
    clinical_dim = int(c_train.shape[1]) if use_clin else 0

    def make_loader(x1, x2, yy, cc, shuffle):
        clin = cc if use_clin else None
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

    model = build_heartx_model(
        args.model,
        num_classes=len(class_names),
        clinical_dim=clinical_dim,
        fusion_type=args.fusion,
    ).to(device)

    run_name = args.run_name or f"{args.model}_{args.fusion}_{args.loss}_{sampler}"
    ckpt = CHECKPOINT_DIR / f"heartx_{run_name}.pt"
    results_dir = OUTPUT_DIR / "results" / run_name
    fig_dir = FIGURE_DIR / run_name

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
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
    )

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_metrics = evaluate(model, test_loader, device, class_names)
    print("\n=== Test (patient-wise) ===")
    print(f"PRIMARY metric macro-F1={test_metrics['f1_macro']:.4f}")
    print(f"accuracy={test_metrics['accuracy']:.4f} (secondary)")
    print(test_metrics["report"])

    # Highlight minority classes when present
    for rare in ("S", "F", "Q"):
        if rare in test_metrics["per_class"]:
            pc = test_metrics["per_class"][rare]
            print(
                f"  [{rare}] P={pc['precision']:.3f} R={pc['recall']:.3f} "
                f"F1={pc['f1']:.3f} n={pc['support']}",
                flush=True,
            )

    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        class_names,
        fig_dir / "confusion_matrix.png",
        title=f"HEART-X {run_name} (macro-F1={test_metrics['f1_macro']:.3f})",
    )
    save_eval_artifacts(test_metrics, class_names, results_dir, prefix="test")

    results = {
        "run_name": run_name,
        "model": args.model,
        "fusion": args.fusion,
        "loss": args.loss,
        "sampler": sampler,
        "class_weight": args.class_weight,
        "epochs": args.epochs,
        "seed": args.seed,
        "primary_metric": "f1_macro",
        "test_f1_macro": test_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_specificity_macro": test_metrics["specificity_macro"],
        "per_class": test_metrics["per_class"],
        "history": history,
        "checkpoint": str(ckpt),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    # also keep legacy path
    (CHECKPOINT_DIR / f"results_{run_name}.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
