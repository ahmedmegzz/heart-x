#!/usr/bin/env python3
"""Generate MM-GradCAM visualizations for a trained HEART-X checkpoint (Aşama 6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import AAMI_CLASSES, CHECKPOINT_DIR, FIGURE_DIR, PROCESSED_DIR  # noqa: E402
from heartx.models.dual_stream import build_heartx_model  # noqa: E402
from heartx.utils.viz import plot_mm_gradcam  # noqa: E402
from heartx.xai.mm_gradcam import MMGradCAMHooked  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", default="dual")
    parser.add_argument("--fusion", choices=["concat", "gated", "attention"], default="concat")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_DIR / "mm_gradcam")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data, allow_pickle=True)
    test_mask = data["test_mask"].astype(bool)
    x_1d = data["x_1d"][test_mask]
    x_2d = data["x_2d"][test_mask]
    y = data["y"][test_mask]
    clinical = data["clinical"][test_mask]

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = sorted(CHECKPOINT_DIR.glob("heartx_*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found")
        ckpt_path = candidates[-1]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_heartx_model(
        args.model,
        num_classes=len(AAMI_CLASSES),
        clinical_dim=clinical.shape[1],
        fusion_type=args.fusion,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    explainer = MMGradCAMHooked(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pick diverse classes when possible
    chosen = []
    for cls in range(len(AAMI_CLASSES)):
        idxs = np.where(y == cls)[0]
        if len(idxs):
            chosen.append(int(idxs[0]))
    while len(chosen) < args.num_samples and len(chosen) < len(y):
        i = len(chosen)
        if i not in chosen:
            chosen.append(i)
        else:
            break
    chosen = chosen[: args.num_samples]

    for i, idx in enumerate(chosen):
        t1 = torch.from_numpy(x_1d[idx : idx + 1]).to(device)
        t2 = torch.from_numpy(x_2d[idx : idx + 1]).to(device)
        tc = torch.from_numpy(clinical[idx : idx + 1]).to(device)
        out = explainer.generate(t1, t2, tc)
        true_name = AAMI_CLASSES[int(y[idx])]
        pred_name = AAMI_CLASSES[out["pred_class"]]
        plot_mm_gradcam(
            signal_1d=x_1d[idx, 0],
            spectrogram=x_2d[idx, 0],
            cam_1d=out["cam_1d"],
            cam_2d=out["cam_2d"],
            out_path=args.out_dir / f"sample_{i:02d}_true-{true_name}_pred-{pred_name}.png",
            title=f"true={true_name} pred={pred_name}",
        )
        print(f"[{i}] true={true_name} pred={pred_name} probs={out['probs'].round(3)}")

    explainer.close()
    print(f"Saved figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
