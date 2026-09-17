#!/usr/bin/env python3
"""
External validation via binary normal/abnormal proxy mapping.

Trains (or loads) a MIT-BIH model, evaluates on the MIT-BIH patient-wise test
set and on PTB-XL (mapped to the same proxy labels). See
``heartx.data.label_mapping`` for taxonomy caveats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import CHECKPOINT_DIR, FIGURE_DIR, PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402
from heartx.data.dataset import DualStreamECGDataset  # noqa: E402
from heartx.data.label_mapping import (  # noqa: E402
    PROXY_CLASSES,
    aami_idx_to_proxy_idx,
    mapping_report,
    ptbxl_idx_to_proxy_idx,
)
from heartx.models.dual_stream import build_heartx_model  # noqa: E402
from heartx.utils.balance import set_seed  # noqa: E402
from heartx.utils.viz import plot_confusion_matrix  # noqa: E402


def _predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch["x_1d"].to(device)
            x2 = batch["x_2d"].to(device)
            clin = batch["clinical"].to(device) if "clinical" in batch else None
            logits = model(x1, x2, clin)
            # Collapse 5-way logits to binary via mapped training isn't available;
            # instead map argmax class indices through AAMI->proxy.
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(batch["y"].numpy())
            preds.append(pred)
    return np.concatenate(ys), np.concatenate(preds)


def _binary_metrics(y_true_proxy, y_pred_proxy):
    return {
        "f1_macro": float(f1_score(y_true_proxy, y_pred_proxy, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true_proxy, y_pred_proxy)),
        "report": classification_report(
            y_true_proxy,
            y_pred_proxy,
            target_names=PROXY_CLASSES,
            zero_division=0,
            digits=4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitbih-data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--ptbxl-data", type=Path, default=PROCESSED_DIR / "ptbxl" / "ptbxl_dual.npz")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", default="dual")
    parser.add_argument("--fusion", default="concat")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    set_seed(SEED)

    print(mapping_report())
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mit = np.load(args.mitbih_data, allow_pickle=True)
    ptb = np.load(args.ptbxl_data, allow_pickle=True)

    clinical_dim = int(mit["clinical"].shape[1])
    model = build_heartx_model(
        args.model,
        num_classes=5,
        clinical_dim=clinical_dim,
        fusion_type=args.fusion,
    ).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # Prefer a dual concat run if present
        candidates = sorted(CHECKPOINT_DIR.glob("heartx_*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found; train first or pass --checkpoint")
        ckpt_path = candidates[-1]
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    # Internal MIT-BIH test
    m_test = mit["test_mask"].astype(bool)
    mit_ds = DualStreamECGDataset(
        mit["x_1d"][m_test],
        mit["x_2d"][m_test],
        mit["y"][m_test],
        mit["clinical"][m_test],
    )
    mit_loader = DataLoader(mit_ds, batch_size=args.batch_size, shuffle=False)
    y_true_aami, y_pred_aami = _predict(model, mit_loader, device)
    y_true_mit_proxy = np.array(aami_idx_to_proxy_idx(y_true_aami))
    y_pred_mit_proxy = np.array(aami_idx_to_proxy_idx(y_pred_aami))
    mit_metrics = _binary_metrics(y_true_mit_proxy, y_pred_mit_proxy)

    # External PTB-XL (all or test fold)
    p_mask = ptb["test_mask"].astype(bool) if "test_mask" in ptb.files else np.ones(len(ptb["y"]), dtype=bool)
    # Model expects MIT-BIH window length / spectrogram size — shapes must match.
    if mit["x_1d"].shape[-1] != ptb["x_1d"].shape[-1] or mit["x_2d"].shape[-2:] != ptb["x_2d"].shape[-2:]:
        print(
            "WARNING: MIT-BIH and PTB-XL tensor shapes differ "
            f"(1D {mit['x_1d'].shape} vs {ptb['x_1d'].shape}, "
            f"2D {mit['x_2d'].shape} vs {ptb['x_2d'].shape}). "
            "External validation will interpolate PTB-XL windows to MIT-BIH length.",
            flush=True,
        )
        # Simple resample / pad-crop for 1D; resize already fixed for 2D specs
        target_t = mit["x_1d"].shape[-1]
        x1 = ptb["x_1d"][p_mask]
        # linear interpolate along time
        from scipy.ndimage import zoom

        x1_r = []
        for row in x1:
            sig = row[0] if row.ndim == 2 else row
            factor = target_t / len(sig)
            x1_r.append(zoom(sig, factor, order=1)[None, :].astype(np.float32))
        x1_ext = np.stack(x1_r, axis=0)
        x2_ext = ptb["x_2d"][p_mask]
        if x2_ext.shape[-2:] != mit["x_2d"].shape[-2:]:
            h, w = mit["x_2d"].shape[-2:]
            x2_r = []
            for spec in x2_ext:
                s = spec[0]
                zh, zw = h / s.shape[0], w / s.shape[1]
                x2_r.append(zoom(s, (zh, zw), order=1)[None, ...].astype(np.float32))
            x2_ext = np.stack(x2_r, axis=0)
    else:
        x1_ext = ptb["x_1d"][p_mask]
        x2_ext = ptb["x_2d"][p_mask]

    # PTB-XL labels -> proxy as ground truth; predictions still AAMI->proxy
    y_ptb = ptb["y"][p_mask]
    y_true_ptb_proxy = np.array(ptbxl_idx_to_proxy_idx(y_ptb))

    ptb_ds = DualStreamECGDataset(x1_ext, x2_ext, y_ptb, ptb["clinical"][p_mask])
    ptb_loader = DataLoader(ptb_ds, batch_size=args.batch_size, shuffle=False)
    _, y_pred_aami_ext = _predict(model, ptb_loader, device)
    y_pred_ptb_proxy = np.array(aami_idx_to_proxy_idx(y_pred_aami_ext))
    ptb_metrics = _binary_metrics(y_true_ptb_proxy, y_pred_ptb_proxy)

    print("=== Internal MIT-BIH test (binary proxy) ===")
    print(f"macro-F1={mit_metrics['f1_macro']:.4f}  acc={mit_metrics['accuracy']:.4f}")
    print(mit_metrics["report"])
    print("=== External PTB-XL (binary proxy) ===")
    print(f"macro-F1={ptb_metrics['f1_macro']:.4f}  acc={ptb_metrics['accuracy']:.4f}")
    print(ptb_metrics["report"])

    table = pd.DataFrame(
        [
            {
                "set": "MIT-BIH internal (proxy)",
                "f1_macro": mit_metrics["f1_macro"],
                "accuracy": mit_metrics["accuracy"],
            },
            {
                "set": "PTB-XL external (proxy)",
                "f1_macro": ptb_metrics["f1_macro"],
                "accuracy": ptb_metrics["accuracy"],
            },
        ]
    )
    out_dir = RESULTS_DIR / "external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "external_validation.csv", index=False)
    (out_dir / "mapping_notes.txt").write_text(mapping_report())
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "mitbih": mit_metrics,
                "ptbxl": ptb_metrics,
                "mapping": mapping_report(),
            },
            indent=2,
        )
    )

    from sklearn.metrics import confusion_matrix

    plot_confusion_matrix(
        confusion_matrix(y_true_mit_proxy, y_pred_mit_proxy),
        PROXY_CLASSES,
        FIGURE_DIR / "external_validation" / "mitbih_proxy_cm.png",
        title="MIT-BIH internal (proxy)",
    )
    plot_confusion_matrix(
        confusion_matrix(y_true_ptb_proxy, y_pred_ptb_proxy),
        PROXY_CLASSES,
        FIGURE_DIR / "external_validation" / "ptbxl_proxy_cm.png",
        title="PTB-XL external (proxy)",
    )
    print(table.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
