#!/usr/bin/env python3
"""Compare concat vs gated vs attention fusion under identical conditions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import FIGURE_DIR, PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402

FUSIONS = ["concat", "gated", "attention"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="1d_2d_clinical")
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--sampler", default="none")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fusions", nargs="+", default=FUSIONS)
    args = parser.parse_args()

    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for fusion in args.fusions:
        run_name = f"fusion_{fusion}_seed{args.seed}"
        cmd = [
            str(py),
            str(ROOT / "scripts" / "train_heartx.py"),
            "--data", str(args.data),
            "--model", args.model,
            "--fusion", fusion,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--loss", args.loss,
            "--class-weight", args.class_weight,
            "--sampler", args.sampler,
            "--seed", str(args.seed),
            "--run-name", run_name,
        ]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        res = json.loads((RESULTS_DIR / run_name / "results.json").read_text())
        row = {
            "fusion": fusion,
            "seed": args.seed,
            "f1_macro": res["test_f1_macro"],
            "accuracy": res["test_accuracy"],
            "precision_macro": res["test_precision_macro"],
            "recall_macro": res["test_recall_macro"],
        }
        for cls, stats in res.get("per_class", {}).items():
            row[f"f1_{cls}"] = stats["f1"]
            row[f"precision_{cls}"] = stats["precision"]
            row[f"recall_{cls}"] = stats["recall"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    csv_path = RESULTS_DIR / f"fusion_comparison_seed{args.seed}.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["fusion"], df["f1_macro"], label="macro-F1 (primary)")
    ax.set_ylabel("macro-F1")
    ax.set_title(f"Fusion comparison (seed={args.seed})")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_DIR / f"fusion_comparison_seed{args.seed}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {csv_path}\nWrote {fig_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
