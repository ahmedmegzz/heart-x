#!/usr/bin/env python3
"""
Build processed MIT-BIH dual-stream dataset (Aşama 2–3 + clinical features).

Patient-wise split: Chazal DS1/DS2 for train+val / test; DS1 further split
into train/val via GroupShuffleSplit on record ID.

Example
-------
python scripts/prepare_mitbih.py --max-beats-per-record 800 --method stft
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
    AAMI_CLASSES,
    MITBIH_DS1,
    MITBIH_DS2,
    MITBIH_FS,
    PROCESSED_DIR,
    SEED,
    VAL_RATIO,
)
from heartx.data.class_report import (  # noqa: E402
    assert_no_empty_test_classes,
    save_class_distribution,
    split_class_distribution,
)
from heartx.data.mitbih import beats_to_arrays, load_mitbih_beats  # noqa: E402
from heartx.data.patient_split import (  # noqa: E402
    log_split_summary,
    masks_from_split,
    patient_wise_split,
)
from heartx.features.spectrogram import make_dual_reps  # noqa: E402
from heartx.utils.balance import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH dual-stream data")
    parser.add_argument("--max-beats-per-record", type=int, default=None)
    parser.add_argument("--method", choices=["stft", "cwt"], default="stft")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "mitbih")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()
    set_seed(SEED)

    print("Loading & preprocessing MIT-BIH beats...")
    beats = load_mitbih_beats(max_beats_per_record=args.max_beats_per_record)
    x, y, clinical, records = beats_to_arrays(beats)
    print(f"Beats: {len(y)}  class counts={dict(zip(AAMI_CLASSES, np.bincount(y, minlength=5)))}")

    print(f"Building dual representations ({args.method})...")
    x_1d, x_2d = make_dual_reps(x, fs=MITBIH_FS, method=args.method)

    # Patient-wise: DS2 = fixed test records; DS1 (+others) -> train/val
    split = patient_wise_split(
        y=y,
        groups=records,
        test_groups=MITBIH_DS2,
        val_ratio=args.val_ratio,
        seed=SEED,
    )
    log_split_summary("MIT-BIH", split, records)
    train_mask, val_mask, test_mask = masks_from_split(len(y), split)

    # Records not in DS1/DS2 (if any) already stay in train/val via GroupShuffleSplit
    # over non-DS2 groups. Log DS1 coverage for thesis reporting.
    ds1 = set(MITBIH_DS1)
    n_ds1_train = len({r for r in split.train_groups if str(r) in ds1})
    n_ds1_val = len({r for r in split.val_groups if str(r) in ds1})
    print(f"  DS1 records in train={n_ds1_train} val={n_ds1_val}", flush=True)

    missing = assert_no_empty_test_classes(y, test_mask, AAMI_CLASSES)
    if missing:
        print(
            f"  Decision: keep Chazal DS2 fixed despite missing test classes {missing}; "
            "do not move patients (would break inter-patient protocol).",
            flush=True,
        )

    dist = split_class_distribution(
        y,
        {"train": train_mask, "val": val_mask, "test": test_mask},
        AAMI_CLASSES,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_p, png_p = save_class_distribution(
        dist, out, title="MIT-BIH class distribution (patient-wise)"
    )
    print(f"Class distribution -> {csv_p}, {png_p}", flush=True)
    print(dist.pivot(index="class", columns="split", values="count").fillna(0).astype(int))

    np.savez_compressed(
        out / "mitbih_dual.npz",
        x_1d=x_1d,
        x_2d=x_2d,
        y=y,
        clinical=clinical,
        records=records,
        groups=records,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        class_names=np.array(AAMI_CLASSES),
        fs=MITBIH_FS,
        method=args.method,
    )
    meta = {
        "n_beats": int(len(y)),
        "class_counts": {c: int(np.sum(y == i)) for i, c in enumerate(AAMI_CLASSES)},
        **split.summary(),
        "missing_test_classes": missing,
        "method": args.method,
        "fs": MITBIH_FS,
        "signal_len": int(x_1d.shape[-1]),
        "spec_shape": list(x_2d.shape[-2:]),
        "clinical_dim": int(clinical.shape[1]),
        "split": "patient_wise_chazal_ds2_test",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved -> {out / 'mitbih_dual.npz'}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
