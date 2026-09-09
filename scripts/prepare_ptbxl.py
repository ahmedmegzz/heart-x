#!/usr/bin/env python3
"""Prepare a PTB-XL dual-stream subset (diagnostic superclasses)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import (  # noqa: E402
    PROCESSED_DIR,
    PTBXL_CLASSES,
    PTBXL_FS,
    SEED,
)
from heartx.data.ptbxl import load_ptbxl_metadata, load_ptbxl_signals  # noqa: E402
from heartx.features.spectrogram import make_dual_representations  # noqa: E402
from heartx.utils.balance import set_seed  # noqa: E402


def _window_clinical(sig: np.ndarray, fs: float) -> np.ndarray:
    """Lightweight clinical proxy features for full 10 s PTB-XL windows."""
    # Rough peak detection via simple threshold on absolute signal
    thr = 0.5 * np.max(np.abs(sig))
    peaks = np.where((np.abs(sig[1:-1]) > thr) & (np.abs(sig[1:-1]) >= np.abs(sig[:-2])) & (np.abs(sig[1:-1]) >= np.abs(sig[2:])))[0] + 1
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        rr_pre = float(rr[len(rr) // 2])
        rr_post = float(rr[min(len(rr) - 1, len(rr) // 2 + 1)])
        rr_mean = float(rr.mean())
        rr_std = float(rr.std())
    else:
        rr_pre = rr_post = rr_mean = 0.8
        rr_std = 0.0
    return np.array(
        [
            rr_pre,
            rr_post,
            rr_mean,
            rr_std,
            rr_pre / (rr_post + 1e-8),
            0.08,
            float(np.sqrt(np.mean(sig**2))),
            float(np.ptp(sig)),
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=2000)
    parser.add_argument("--method", choices=["stft", "cwt"], default="stft")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "ptbxl")
    args = parser.parse_args()
    set_seed(SEED)

    meta = load_ptbxl_metadata()
    print(f"PTB-XL metadata rows: {len(meta)}")
    x, y, folds, multi = load_ptbxl_signals(
        meta, sampling_rate=PTBXL_FS, max_records=args.max_records
    )
    print(f"Loaded signals: {x.shape}  labels={dict(zip(PTBXL_CLASSES, np.bincount(y, minlength=5)))}")

    x_1d, x_2d = make_dual_representations(x, fs=float(PTBXL_FS), method=args.method)
    clinical = np.stack([_window_clinical(s, float(PTBXL_FS)) for s in x], axis=0)

    # Official PTB-XL fold split: fold 10 = test
    test_mask = folds == 10
    train_mask = ~test_mask

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "ptbxl_dual.npz",
        x_1d=x_1d,
        x_2d=x_2d,
        y=y,
        clinical=clinical,
        folds=folds,
        multi_hot=multi,
        train_mask=train_mask,
        test_mask=test_mask,
        class_names=np.array(PTBXL_CLASSES),
        fs=PTBXL_FS,
        method=args.method,
    )
    meta_out = {
        "n": int(len(y)),
        "class_counts": {c: int(np.sum(y == i)) for i, c in enumerate(PTBXL_CLASSES)},
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "method": args.method,
        "fs": PTBXL_FS,
    }
    (out / "meta.json").write_text(json.dumps(meta_out, indent=2))
    print(json.dumps(meta_out, indent=2))


if __name__ == "__main__":
    main()
