#!/usr/bin/env python3
"""
Imbalance-strategy experiment matrix.

SMOTE note
----------
When ``sampler=smote``, oversampling runs in 1D(+clinical) space and
spectrograms are regenerated from synthetic 1D waveforms (see
``heartx.utils.balance``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402

# (class_weight, loss, sampler)
EXPERIMENTS = [
    ("none", "ce", "none"),
    ("balanced", "ce", "none"),
    ("none", "focal", "none"),
    ("balanced", "focal", "none"),
    ("none", "ce", "random_oversample"),
    ("balanced", "ce", "random_oversample"),
    ("none", "ce", "smote"),
    ("balanced", "ce", "smote"),
    ("balanced", "focal", "random_oversample"),
    ("balanced", "focal", "smote"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="dual")
    parser.add_argument("--fusion", default="concat")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller subset of the matrix",
    )
    args = parser.parse_args()

    experiments = EXPERIMENTS[:4] if args.quick else EXPERIMENTS
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)

    rows = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    for class_weight, loss, sampler in experiments:
        run_name = f"imb_cw-{class_weight}_loss-{loss}_samp-{sampler}"
        cmd = [
            str(py),
            str(ROOT / "scripts" / "train_heartx.py"),
            "--data",
            str(args.data),
            "--model",
            args.model,
            "--fusion",
            args.fusion,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--class-weight",
            class_weight,
            "--loss",
            loss,
            "--sampler",
            sampler,
            "--seed",
            str(args.seed),
            "--run-name",
            run_name,
        ]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        res = json.loads((RESULTS_DIR / run_name / "results.json").read_text())
        row = {
            "class_weight": class_weight,
            "loss": loss,
            "sampler": sampler,
            "f1_macro": res["test_f1_macro"],
            "accuracy": res["test_accuracy"],
            "precision_macro": res["test_precision_macro"],
            "recall_macro": res["test_recall_macro"],
        }
        # Minority focus if available
        for cls in ("S", "F", "Q"):
            if cls in res.get("per_class", {}):
                row[f"f1_{cls}"] = res["per_class"][cls]["f1"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    out = RESULTS_DIR / "imbalance_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(df.to_string(index=False))
    print(
        "\nSMOTE policy: interpolates 1D(+clinical); spectrograms regenerated "
        "from synthetic 1D (not independent 2D SMOTE)."
    )


if __name__ == "__main__":
    main()
