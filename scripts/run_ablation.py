#!/usr/bin/env python3
"""
Run 7-way modality ablation with fixed seed / split / epochs.

Combinations
------------
1d, 2d, clinical, 1d_2d, 1d_clinical, 2d_clinical, 1d_2d_clinical
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import FIGURE_DIR, PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402

ABLATION_MODELS = [
    "1d",
    "2d",
    "clinical",
    "1d_2d",
    "1d_clinical",
    "2d_clinical",
    "1d_2d_clinical",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fusion", default="concat")
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--sampler", default="none")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models", nargs="+", default=ABLATION_MODELS)
    args = parser.parse_args()

    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)

    rows = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        run_name = f"ablation_{model}"
        cmd = [
            str(py),
            str(ROOT / "scripts" / "train_heartx.py"),
            "--data",
            str(args.data),
            "--model",
            model,
            "--fusion",
            args.fusion,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--loss",
            args.loss,
            "--class-weight",
            args.class_weight,
            "--sampler",
            args.sampler,
            "--seed",
            str(args.seed),
            "--run-name",
            run_name,
        ]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT), env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")})

        result_json = RESULTS_DIR / run_name / "results.json"
        import json

        res = json.loads(result_json.read_text())
        rows.append(
            {
                "model": model,
                "f1_macro": res["test_f1_macro"],
                "accuracy": res["test_accuracy"],
                "precision_macro": res["test_precision_macro"],
                "recall_macro": res["test_recall_macro"],
                "seed": args.seed,
                "epochs": args.epochs,
                "fusion": args.fusion,
            }
        )

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    csv_path = RESULTS_DIR / "ablation_results.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["f1_macro"], width=0.4, label="macro-F1 (primary)")
    ax.bar([i + 0.2 for i in x], df["accuracy"], width=0.4, label="accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("HEART-X modality ablation (same seed/split/epochs)")
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURE_DIR / "ablation_results.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {fig_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
