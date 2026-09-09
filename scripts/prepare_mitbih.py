#!/usr/bin/env python3
"""
Build processed MIT-BIH dual-stream dataset (Aşama 2–3 + clinical features).

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
)
from heartx.data.mitbih import beats_to_arrays, load_mitbih_beats  # noqa: E402
from heartx.features.spectrogram import make_dual_representations  # noqa: E402
from heartx.utils.balance import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH dual-stream data")
    parser.add_argument("--max-beats-per-record", type=int, default=None)
    parser.add_argument("--method", choices=["stft", "cwt"], default="stft")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "mitbih")
    args = parser.parse_args()
    set_seed(SEED)

    print("Loading & preprocessing MIT-BIH beats...")
    beats = load_mitbih_beats(max_beats_per_record=args.max_beats_per_record)
    x, y, clinical, records = beats_to_arrays(beats)
    print(f"Beats: {len(y)}  class counts={dict(zip(AAMI_CLASSES, np.bincount(y, minlength=5)))}")

    print(f"Building dual representations ({args.method})...")
    x_1d, x_2d = make_dual_representations(x, fs=MITBIH_FS, method=args.method)

    # Inter-patient split: DS1 train / DS2 test (Chazal-style)
    ds1 = set(MITBIH_DS1)
    ds2 = set(MITBIH_DS2)
    train_mask = np.array([r in ds1 for r in records])
    test_mask = np.array([r in ds2 for r in records])
    # Remaining records (if any) go to train
    other = ~(train_mask | test_mask)
    train_mask = train_mask | other

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "mitbih_dual.npz",
        x_1d=x_1d,
        x_2d=x_2d,
        y=y,
        clinical=clinical,
        records=records,
        train_mask=train_mask,
        test_mask=test_mask,
        class_names=np.array(AAMI_CLASSES),
        fs=MITBIH_FS,
        method=args.method,
    )
    meta = {
        "n_beats": int(len(y)),
        "class_counts": {
            c: int(np.sum(y == i)) for i, c in enumerate(AAMI_CLASSES)
        },
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "method": args.method,
        "fs": MITBIH_FS,
        "signal_len": int(x_1d.shape[-1]),
        "spec_shape": list(x_2d.shape[-2:]),
        "clinical_dim": int(clinical.shape[1]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved -> {out / 'mitbih_dual.npz'}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
