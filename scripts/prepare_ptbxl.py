#!/usr/bin/env python3
"""Prepare a PTB-XL dual-stream subset (diagnostic superclasses).

Patient-wise split: official fold 10 = test; remaining folds split into
train/val by patient_id (GroupShuffleSplit).
"""

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
    VAL_RATIO,
)
from heartx.data.class_report import (  # noqa: E402
    assert_no_empty_test_classes,
    save_class_distribution,
    split_class_distribution,
)
from heartx.data.patient_split import (  # noqa: E402
    log_split_summary,
    masks_from_split,
    patient_wise_split,
)
from heartx.data.ptbxl import load_ptbxl_metadata, load_ptbxl_signals  # noqa: E402
from heartx.features.spectrogram import make_dual_reps  # noqa: E402
from heartx.utils.balance import set_seed  # noqa: E402


def _window_clinical(sig: np.ndarray, fs: float) -> np.ndarray:
    """Lightweight clinical proxy features for full 10 s PTB-XL windows."""
    thr = 0.5 * np.max(np.abs(sig))
    peaks = (
        np.where(
            (np.abs(sig[1:-1]) > thr)
            & (np.abs(sig[1:-1]) >= np.abs(sig[:-2]))
            & (np.abs(sig[1:-1]) >= np.abs(sig[2:]))
        )[0]
        + 1
    )
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
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()
    set_seed(SEED)

    meta = load_ptbxl_metadata()
    print(f"PTB-XL metadata rows: {len(meta)}")
    x, y, folds, multi, patients = load_ptbxl_signals(
        meta, sampling_rate=PTBXL_FS, max_records=args.max_records
    )
    print(
        f"Loaded signals: {x.shape}  "
        f"labels={dict(zip(PTBXL_CLASSES, np.bincount(y, minlength=5)))}"
    )

    x_1d, x_2d = make_dual_reps(x, fs=float(PTBXL_FS), method=args.method)
    clinical = np.stack([_window_clinical(s, float(PTBXL_FS)) for s in x], axis=0)

    # Official fold 10 records define test groups (via their patient_ids)
    test_patient_ids = set(patients[folds == 10].tolist())
    split = patient_wise_split(
        y=y,
        groups=patients,
        test_groups=test_patient_ids,
        val_ratio=args.val_ratio,
        seed=SEED,
    )
    log_split_summary("PTB-XL", split, patients)
    train_mask, val_mask, test_mask = masks_from_split(len(y), split)

    missing = assert_no_empty_test_classes(y, test_mask, PTBXL_CLASSES)
    if missing:
        print(
            f"  Decision: keep fold-10 patient groups fixed despite missing "
            f"test classes {missing}.",
            flush=True,
        )

    dist = split_class_distribution(
        y,
        {"train": train_mask, "val": val_mask, "test": test_mask},
        PTBXL_CLASSES,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_p, png_p = save_class_distribution(
        dist, out, title="PTB-XL class distribution (patient-wise)"
    )
    print(f"Class distribution -> {csv_p}, {png_p}", flush=True)

    np.savez_compressed(
        out / "ptbxl_dual.npz",
        x_1d=x_1d,
        x_2d=x_2d,
        y=y,
        clinical=clinical,
        folds=folds,
        multi_hot=multi,
        patients=patients,
        groups=patients,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        class_names=np.array(PTBXL_CLASSES),
        fs=PTBXL_FS,
        method=args.method,
    )
    meta_out = {
        "n": int(len(y)),
        "class_counts": {c: int(np.sum(y == i)) for i, c in enumerate(PTBXL_CLASSES)},
        **split.summary(),
        "missing_test_classes": missing,
        "method": args.method,
        "fs": PTBXL_FS,
        "split": "patient_wise_fold10_test",
    }
    (out / "meta.json").write_text(json.dumps(meta_out, indent=2))
    print(json.dumps(meta_out, indent=2))


if __name__ == "__main__":
    main()
