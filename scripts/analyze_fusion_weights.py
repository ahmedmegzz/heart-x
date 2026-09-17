#!/usr/bin/env python3
"""Visualize learned fusion weights (gated / attention) for HEART-X."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import CHECKPOINT_DIR, FIGURE_DIR, PROCESSED_DIR  # noqa: E402
from heartx.models.dual_stream import build_heartx_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="dual")
    parser.add_argument("--fusion", choices=["concat", "gated", "attention"], default="gated")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_DIR / "fusion_weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data, allow_pickle=True)
    mask = data["test_mask"].astype(bool)
    x_1d = data["x_1d"][mask][: args.num_samples]
    x_2d = data["x_2d"][mask][: args.num_samples]
    clinical = data["clinical"][mask][: args.num_samples]
    y = data["y"][mask][: args.num_samples]
    class_names = [str(c) for c in data["class_names"]]

    model = build_heartx_model(
        args.model,
        num_classes=len(class_names),
        clinical_dim=clinical.shape[1],
        fusion_type=args.fusion,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    modality_names = []
    if getattr(model, "use_1d", True):
        modality_names.append("1D")
    if getattr(model, "use_2d", True):
        modality_names.append("2D")
    if getattr(model, "use_clinical", False):
        modality_names.append("clinical")

    weights = []
    with torch.no_grad():
        for i in range(len(y)):
            t1 = torch.from_numpy(x_1d[i : i + 1]).to(device)
            t2 = torch.from_numpy(x_2d[i : i + 1]).to(device)
            tc = torch.from_numpy(clinical[i : i + 1]).to(device)
            _ = model(t1, t2, tc)
            w = model.fusion_weights()
            if w is None:
                raise RuntimeError("Model did not expose fusion weights")
            weights.append(w.detach().cpu().numpy()[0])

    W = np.stack(weights, axis=0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(W, columns=modality_names)
    df["label"] = [class_names[int(i)] for i in y]
    df.to_csv(args.out_dir / "fusion_weights.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(modality_names, W.mean(axis=0), yerr=W.std(axis=0), capsize=4)
    axes[0].set_title(f"Mean {args.fusion} fusion weights")
    axes[0].set_ylabel("Weight")

    for name in modality_names:
        for cls in class_names:
            subset = df[df["label"] == cls][name]
            if len(subset):
                axes[1].scatter(
                    [f"{cls}:{name}"] * len(subset),
                    subset,
                    alpha=0.4,
                    s=12,
                )
    axes[1].tick_params(axis="x", rotation=75)
    axes[1].set_title("Per-sample weights by class")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"fusion_weights_{args.fusion}.png", dpi=150)
    plt.close(fig)
    print(f"Saved -> {args.out_dir}")
    print(df.groupby("label")[modality_names].mean().round(3))


if __name__ == "__main__":
    main()
