#!/usr/bin/env python3
"""
Build processed MIT-BIH dual-stream dataset under the Chazal protocol.

Protocol (advisor-aligned)
--------------------------
1. Exclude pacemaker recordings: 102, 104, 107, 217.
2. Keep only DS1 ∪ DS2 records.
3. DS2 = test only.
4. Train / validation are split **only within DS1** (patient-wise GroupShuffleSplit).
5. By default use **all** annotated AAMI beats (no per-record cap).

Example
-------
python scripts/prepare_mitbih.py --method stft
python scripts/prepare_mitbih.py --max-beats-per-record 500   # optional smoke only
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
    MITBIH_EXCLUDE_PACEMAKER,
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
    minority_aware_ds1_val_split,
    patient_wise_split,
)
from heartx.features.spectrogram import make_dual_reps  # noqa: E402
from heartx.utils.balance import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH (Chazal protocol)")
    parser.add_argument(
        "--max-beats-per-record",
        type=int,
        default=None,
        help="Optional smoke-test cap. Final experiments should omit this (use all beats).",
    )
    parser.add_argument("--method", choices=["stft", "cwt"], default="stft")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "mitbih")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument(
        "--min-val-minority",
        type=int,
        default=30,
        help="Minimum S and F samples required in DS1 validation (patient-wise search).",
    )
    parser.add_argument(
        "--drop-q-if-unevaluable",
        action="store_true",
        help="If Q has 0 samples in val or test after Chazal filtering, drop Q beats "
        "and reindex to 4-class (N,S,V,F). Common when paced records are excluded.",
    )
    args = parser.parse_args()
    set_seed(SEED)

    exclude = set(MITBIH_EXCLUDE_PACEMAKER)
    ds1 = set(MITBIH_DS1)
    ds2 = set(MITBIH_DS2)
    allowed = (ds1 | ds2) - exclude

    print("Loading & preprocessing MIT-BIH beats (Chazal protocol)...")
    print(f"  Excluding pacemaker records: {sorted(exclude)}")
    print(f"  Allowed records (DS1∪DS2 − paced): {len(allowed)}")
    if args.max_beats_per_record is None:
        print("  Using ALL annotated AAMI beats (no per-record cap).")
    else:
        print(
            f"  WARNING: capping at {args.max_beats_per_record} beats/record "
            "(smoke only; do not use for final advisor results)."
        )

    beats = load_mitbih_beats(
        records=sorted(allowed),
        max_beats_per_record=args.max_beats_per_record,
    )
    x, y, clinical, records = beats_to_arrays(beats)

    # Safety: drop any beat whose record is not allowed
    keep = np.array([str(r) in allowed for r in records])
    x, y, clinical, records = x[keep], y[keep], clinical[keep], records[keep]
    print(
        f"Beats after filter: {len(y)}  "
        f"class counts={dict(zip(AAMI_CLASSES, np.bincount(y, minlength=5)))}"
    )

    # Assert no excluded / out-of-protocol records remain
    uniq = set(map(str, np.unique(records)))
    assert uniq.isdisjoint(exclude), f"Pacemaker leak: {uniq & exclude}"
    assert uniq.issubset(allowed), f"Unexpected records: {uniq - allowed}"

    print(f"Building dual representations ({args.method})...")
    x_1d, x_2d = make_dual_reps(x, fs=MITBIH_FS, method=args.method)

    # DS2 = test; train/val from DS1 only
    # Restrict to DS1∪DS2 then fix test_groups=DS2 so remaining groups are DS1
    assert uniq.issubset(ds1 | ds2)
    non_ds1_in_trainval_candidates = uniq - ds2 - ds1
    if non_ds1_in_trainval_candidates:
        raise RuntimeError(f"Non-DS1/DS2 records present: {non_ds1_in_trainval_candidates}")

    # DS2 = test; train/val from DS1 with minority-aware (S/F) validation search
    split = minority_aware_ds1_val_split(
        y=y,
        groups=records,
        test_groups=ds2,
        val_ratio=args.val_ratio,
        seed=SEED,
        minority_class_indices=(1, 3),  # S, F
        min_minority_per_class=args.min_val_minority,
        n_trials=500,
    )
    # Enforce: every train/val group must be in DS1; every test group in DS2
    train_g = set(map(str, split.train_groups))
    val_g = set(map(str, split.val_groups))
    test_g = set(map(str, split.test_groups))
    assert train_g.issubset(ds1), f"Train groups outside DS1: {train_g - ds1}"
    assert val_g.issubset(ds1), f"Val groups outside DS1: {val_g - ds1}"
    assert test_g.issubset(ds2), f"Test groups outside DS2: {test_g - ds2}"
    assert "124" in (train_g | val_g), "Record 124 must be in DS1 train or val"
    assert train_g.isdisjoint(val_g)
    assert (train_g | val_g).isdisjoint(test_g)

    log_split_summary("MIT-BIH Chazal (+124, S/F-aware val)", split, records)
    print(
        f"  DS1 train records={len(train_g)}  DS1 val records={len(val_g)}  "
        f"DS2 test records={len(test_g)}",
        flush=True,
    )
    print(f"  train records: {sorted(train_g)}", flush=True)
    print(f"  val records:   {sorted(val_g)}", flush=True)
    train_mask, val_mask, test_mask = masks_from_split(len(y), split)

    class_names = list(AAMI_CLASSES)
    missing = assert_no_empty_test_classes(y, test_mask, class_names)
    missing_val = assert_no_empty_test_classes(y, val_mask, class_names)
    q_note = ""
    if "Q" in missing or "Q" in missing_val:
        q_note = (
            "Q has 0 samples in val and/or test under Chazal (paced records 102/104/107/217 "
            "excluded; remaining Q beats are rare and concentrated in DS1). "
        )
        print(f"  Q-class note: {q_note}", flush=True)
        if args.drop_q_if_unevaluable:
            q_idx = AAMI_CLASSES.index("Q")
            keep_q = y != q_idx
            print(f"  Dropping {int((~keep_q).sum())} Q beats → 4-class (N,S,V,F).", flush=True)
            x_1d, x_2d, y, clinical, records = (
                x_1d[keep_q],
                x_2d[keep_q],
                y[keep_q],
                clinical[keep_q],
                records[keep_q],
            )
            # Remap labels: N,S,V,F stay 0..3
            class_names = ["N", "S", "V", "F"]
            split = minority_aware_ds1_val_split(
                y=y,
                groups=records,
                test_groups=ds2,
                val_ratio=args.val_ratio,
                seed=SEED,
                minority_class_indices=(1, 3),
                min_minority_per_class=args.min_val_minority,
                n_trials=500,
            )
            train_mask, val_mask, test_mask = masks_from_split(len(y), split)
            missing = assert_no_empty_test_classes(y, test_mask, class_names)
            q_note += "Dropped Q for evaluable 4-class setup."
        else:
            print(
                "  Decision: keep 5-class labels but report Q as unevaluable on val/test. "
                "Pass --drop-q-if-unevaluable for 4-class N/S/V/F final tables.",
                flush=True,
            )

    # Refresh group sets after any optional Q drop / re-split
    train_g = set(map(str, split.train_groups))
    val_g = set(map(str, split.val_groups))
    test_g = set(map(str, split.test_groups))
    missing_val = assert_no_empty_test_classes(y, val_mask, class_names)

    dist = split_class_distribution(
        y,
        {"train": train_mask, "val": val_mask, "test": test_mask},
        class_names,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_p, png_p = save_class_distribution(
        dist, out, title="MIT-BIH class distribution (Chazal patient-wise)"
    )
    print(f"Class distribution -> {csv_p}, {png_p}", flush=True)
    print(dist.pivot(index="class", columns="split", values="count").fillna(0).astype(int))

    # Assert S/F not collapsed in validation
    val_counts = {
        class_names[i]: int(np.sum(y[val_mask] == i)) for i in range(len(class_names))
    }
    print(f"  Val class counts: {val_counts}", flush=True)
    if val_counts.get("S", 0) < args.min_val_minority or val_counts.get("F", 0) < args.min_val_minority:
        print(
            f"  WARNING: val S={val_counts.get('S', 0)} F={val_counts.get('F', 0)} "
            f"below target {args.min_val_minority}",
            flush=True,
        )

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
        class_names=np.array(class_names),
        fs=MITBIH_FS,
        method=args.method,
        excluded_records=np.array(sorted(exclude)),
        ds1=np.array(sorted(ds1)),
        ds2=np.array(sorted(ds2)),
    )
    vmin = getattr(split, "_val_minority_counts", {})
    vmin_named = {
        (class_names[i] if i < len(class_names) else str(i)): int(v)
        for i, v in vmin.items()
    }
    meta = {
        "n_beats": int(len(y)),
        "class_counts": {
            c: int(np.sum(y == i)) for i, c in enumerate(class_names)
        },
        **split.summary(),
        "missing_test_classes": missing,
        "missing_val_classes": missing_val,
        "q_note": q_note,
        "excluded_pacemaker": sorted(exclude),
        "method": args.method,
        "fs": MITBIH_FS,
        "signal_len": int(x_1d.shape[-1]),
        "spec_shape": list(x_2d.shape[-2:]),
        "clinical_dim": int(clinical.shape[1]),
        "split": "chazal_ds2_test_ds1_train_val_sf_aware",
        "max_beats_per_record": args.max_beats_per_record,
        "val_records": sorted(val_g),
        "train_records": sorted(train_g),
        "val_class_counts": val_counts,
        "val_minority_counts": vmin_named,
        "min_val_minority": args.min_val_minority,
        "protocol_notes": [
            "Excluded pacemaker records 102, 104, 107, 217",
            "DS2 used only as test",
            "Train/val split patient-wise within DS1 only (DS1 includes 124)",
            "Validation chosen to ensure adequate S and F representation",
            "All AAMI beats used unless --max-beats-per-record set",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved -> {out / 'mitbih_dual.npz'}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
