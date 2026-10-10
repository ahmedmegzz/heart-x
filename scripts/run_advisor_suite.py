#!/usr/bin/env python3
"""
Advisor-requested experiment suite: ablation + fusion + imbalance, multi-seed.

Aggregates mean ± std of macro-F1 and per-class F1 (esp. S, F).

Example
-------
python scripts/run_advisor_suite.py --epochs 8 --seeds 42 43 44
python scripts/run_advisor_suite.py --epochs 5 --quick   # smaller matrix
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import (  # noqa: E402
    EXPERIMENT_SEEDS,
    FIGURE_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
)

ABLATION_MODELS = [
    "1d",
    "2d",
    "clinical",
    "1d_2d",
    "1d_clinical",
    "2d_clinical",
    "1d_2d_clinical",
]
FUSIONS = ["concat", "gated", "attention"]
IMBALANCE = [
    ("none", "ce", "none"),
    ("balanced", "ce", "none"),
    ("none", "focal", "none"),
    ("balanced", "focal", "none"),
    ("none", "ce", "random_oversample"),
    ("balanced", "ce", "random_oversample"),
    ("none", "ce", "smote"),
    ("balanced", "ce", "smote"),
    ("balanced", "focal", "random_oversample"),
]


def _py() -> Path:
    p = ROOT / ".venv" / "bin" / "python"
    return p if p.exists() else Path(sys.executable)


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}


def _train(run_name: str, data: Path, model: str, fusion: str, epochs: int,
           batch_size: int, loss: str, class_weight: str, sampler: str, seed: int) -> dict:
    cmd = [
        str(_py()), str(ROOT / "scripts" / "train_heartx.py"),
        "--data", str(data),
        "--model", model,
        "--fusion", fusion,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--loss", loss,
        "--class-weight", class_weight,
        "--sampler", sampler,
        "--seed", str(seed),
        "--run-name", run_name,
    ]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=_env())
    return json.loads((RESULTS_DIR / run_name / "results.json").read_text())


def _row_from_result(res: dict, **extra) -> dict:
    row = {
        **extra,
        "f1_macro": res["test_f1_macro"],
        "accuracy": res["test_accuracy"],
        "precision_macro": res["test_precision_macro"],
        "recall_macro": res["test_recall_macro"],
    }
    for cls, stats in res.get("per_class", {}).items():
        row[f"f1_{cls}"] = stats["f1"]
        row[f"precision_{cls}"] = stats["precision"]
        row[f"recall_{cls}"] = stats["recall"]
        row[f"support_{cls}"] = stats["support"]
    return row


def _aggregate(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = len(g)
        for m in metric_cols:
            if m not in g.columns:
                continue
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=EXPERIMENT_SEEDS)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--skip-imbalance", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Fewer imbalance configs")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Missing {args.data}. Run: python scripts/prepare_mitbih.py --drop-q-if-unevaluable"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / "advisor_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    imbalance_cfgs = IMBALANCE[:4] if args.quick else IMBALANCE
    all_ablation, all_fusion, all_imbalance = [], [], []

    # ---- Ablation ----
    if not args.skip_ablation:
        for seed in args.seeds:
            for model in ABLATION_MODELS:
                run = f"suite_ablation_{model}_seed{seed}"
                fusion = "concat" if model in {"1d", "2d", "clinical"} else "concat"
                res = _train(
                    run, args.data, model, fusion, args.epochs, args.batch_size,
                    "ce", "balanced", "none", seed,
                )
                all_ablation.append(_row_from_result(res, experiment="ablation", model=model, seed=seed))

        ab_df = pd.DataFrame(all_ablation)
        ab_df.to_csv(out_dir / "ablation_raw.csv", index=False)
        metric_cols = [c for c in ab_df.columns if c.startswith(("f1_", "precision_", "recall_")) or c in {
            "f1_macro", "accuracy", "precision_macro", "recall_macro"
        }]
        ab_agg = _aggregate(ab_df, ["model"], metric_cols)
        ab_agg = ab_agg.sort_values("f1_macro_mean", ascending=False)
        ab_agg.to_csv(out_dir / "ablation_mean_std.csv", index=False)
        # also legacy path
        ab_agg.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(ab_agg["model"], ab_agg["f1_macro_mean"], yerr=ab_agg["f1_macro_std"], capsize=3)
        ax.set_ylabel("macro-F1 (mean ± std)")
        ax.set_title(f"Ablation ({len(args.seeds)} seeds)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "ablation_results.png", dpi=150)
        plt.close(fig)

    # ---- Fusion ----
    if not args.skip_fusion:
        for seed in args.seeds:
            for fusion in FUSIONS:
                run = f"suite_fusion_{fusion}_seed{seed}"
                res = _train(
                    run, args.data, "1d_2d_clinical", fusion, args.epochs, args.batch_size,
                    "ce", "balanced", "none", seed,
                )
                all_fusion.append(_row_from_result(res, experiment="fusion", fusion=fusion, seed=seed))

        fu_df = pd.DataFrame(all_fusion)
        fu_df.to_csv(out_dir / "fusion_raw.csv", index=False)
        metric_cols = [c for c in fu_df.columns if c.startswith(("f1_", "precision_", "recall_")) or c in {
            "f1_macro", "accuracy", "precision_macro", "recall_macro"
        }]
        fu_agg = _aggregate(fu_df, ["fusion"], metric_cols).sort_values("f1_macro_mean", ascending=False)
        fu_agg.to_csv(out_dir / "fusion_mean_std.csv", index=False)
        fu_agg.to_csv(RESULTS_DIR / "fusion_comparison.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(fu_agg["fusion"], fu_agg["f1_macro_mean"], yerr=fu_agg["f1_macro_std"], capsize=4)
        ax.set_ylabel("macro-F1 (mean ± std)")
        ax.set_title(f"Fusion comparison ({len(args.seeds)} seeds)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fusion_comparison.png", dpi=150)
        plt.close(fig)

    # ---- Imbalance ----
    if not args.skip_imbalance:
        for seed in args.seeds:
            for class_weight, loss, sampler in imbalance_cfgs:
                run = f"suite_imb_cw-{class_weight}_loss-{loss}_samp-{sampler}_seed{seed}"
                res = _train(
                    run, args.data, "1d_2d_clinical", "concat", args.epochs, args.batch_size,
                    loss, class_weight, sampler, seed,
                )
                all_imbalance.append(
                    _row_from_result(
                        res,
                        experiment="imbalance",
                        class_weight=class_weight,
                        loss=loss,
                        sampler=sampler,
                        seed=seed,
                    )
                )

        im_df = pd.DataFrame(all_imbalance)
        im_df.to_csv(out_dir / "imbalance_raw.csv", index=False)
        metric_cols = [c for c in im_df.columns if c.startswith(("f1_", "precision_", "recall_")) or c in {
            "f1_macro", "accuracy", "precision_macro", "recall_macro"
        }]
        im_agg = _aggregate(im_df, ["class_weight", "loss", "sampler"], metric_cols)
        im_agg = im_agg.sort_values("f1_macro_mean", ascending=False)
        im_agg.to_csv(out_dir / "imbalance_mean_std.csv", index=False)
        im_agg.to_csv(RESULTS_DIR / "imbalance_comparison.csv", index=False)

    # ---- Text summary for advisor ----
    lines = [
        "HEART-X — Advisor Suite Results",
        "================================",
        f"Seeds: {args.seeds}",
        f"Epochs: {args.epochs}",
        f"Data: {args.data}",
        "",
        "Primary metric: macro-F1 (mean ± std over seeds).",
        "Accuracy reported only as secondary.",
        "",
    ]
    if (out_dir / "ablation_mean_std.csv").exists():
        lines.append("=== Ablation (macro-F1 mean ± std) ===")
        ab = pd.read_csv(out_dir / "ablation_mean_std.csv")
        for _, r in ab.iterrows():
            lines.append(
                f"  {r['model']:16s}  F1={r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}  "
                f"acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
            )
            for cls in ("N", "S", "V", "F", "Q"):
                if f"f1_{cls}_mean" in r and not np.isnan(r[f"f1_{cls}_mean"]):
                    lines.append(
                        f"      {cls}: P={r.get(f'precision_{cls}_mean', float('nan')):.3f} "
                        f"R={r.get(f'recall_{cls}_mean', float('nan')):.3f} "
                        f"F1={r[f'f1_{cls}_mean']:.3f}±{r[f'f1_{cls}_std']:.3f}"
                    )
        lines.append("")
    if (out_dir / "fusion_mean_std.csv").exists():
        lines.append("=== Fusion comparison ===")
        fu = pd.read_csv(out_dir / "fusion_mean_std.csv")
        for _, r in fu.iterrows():
            lines.append(
                f"  {r['fusion']:10s}  F1={r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}"
            )
        lines.append("")
    if (out_dir / "imbalance_mean_std.csv").exists():
        lines.append("=== Imbalance strategies ===")
        im = pd.read_csv(out_dir / "imbalance_mean_std.csv")
        for _, r in im.iterrows():
            sf = ""
            if "f1_S_mean" in r:
                sf += f"  F1_S={r['f1_S_mean']:.3f}±{r['f1_S_std']:.3f}"
            if "f1_F_mean" in r:
                sf += f"  F1_F={r['f1_F_mean']:.3f}±{r['f1_F_std']:.3f}"
            lines.append(
                f"  cw={r['class_weight']:8s} loss={r['loss']:5s} samp={r['sampler']:18s} "
                f"F1={r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}{sf}"
            )
        lines.append("")

    summary_path = out_dir / "advisor_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    # also copy to results root
    (RESULTS_DIR / "danisman_sonuc_ozeti.txt").write_text(summary_path.read_text())
    print("\n".join(lines))
    print(f"\nWrote {summary_path}")

    # ---- Pick final model & run external / XAI ----
    final_fusion = "concat"
    final_seed = args.seeds[0]
    if (out_dir / "fusion_mean_std.csv").exists():
        fu = pd.read_csv(out_dir / "fusion_mean_std.csv")
        final_fusion = str(fu.iloc[0]["fusion"])
    # Prefer seed with best raw fusion F1 for that fusion type
    if (out_dir / "fusion_raw.csv").exists():
        fr = pd.read_csv(out_dir / "fusion_raw.csv")
        sub = fr[fr["fusion"] == final_fusion]
        if len(sub):
            best_row = sub.loc[sub["f1_macro"].idxmax()]
            final_seed = int(best_row["seed"])
    ckpt = ROOT / "outputs" / "checkpoints" / f"heartx_suite_fusion_{final_fusion}_seed{final_seed}.pt"
    lines_final = [
        "",
        "=== Final model selection ===",
        f"fusion={final_fusion}  seed={final_seed}",
        f"checkpoint={ckpt}",
    ]
    print("\n".join(lines_final), flush=True)
    (out_dir / "final_model.txt").write_text("\n".join(lines_final) + "\n")

    if ckpt.exists():
        env = _env()
        # Fusion weight analysis
        cmd_fw = [
            str(_py()), str(ROOT / "scripts" / "analyze_fusion_weights.py"),
            "--data", str(args.data),
            "--checkpoint", str(ckpt),
            "--model", "1d_2d_clinical",
            "--fusion", final_fusion,
            "--num-samples", "64",
            "--out-dir", str(FIGURE_DIR / "fusion_weights_final"),
        ]
        print("\n>>>", " ".join(cmd_fw), flush=True)
        subprocess.run(cmd_fw, check=False, cwd=str(ROOT), env=env)

        # External validation (proxy / supplementary)
        cmd_ext = [
            str(_py()), str(ROOT / "scripts" / "external_validation.py"),
            "--mitbih-data", str(args.data),
            "--checkpoint", str(ckpt),
            "--model", "1d_2d_clinical",
            "--fusion", final_fusion,
        ]
        print("\n>>>", " ".join(cmd_ext), flush=True)
        subprocess.run(cmd_ext, check=False, cwd=str(ROOT), env=env)

        # MM-GradCAM
        cmd_xai = [
            str(_py()), str(ROOT / "scripts" / "explain_mm_gradcam.py"),
            "--data", str(args.data),
            "--checkpoint", str(ckpt),
            "--model", "1d_2d_clinical",
            "--fusion", final_fusion,
            "--num-samples", "8",
            "--out-dir", str(FIGURE_DIR / "mm_gradcam_final"),
        ]
        print("\n>>>", " ".join(cmd_xai), flush=True)
        subprocess.run(cmd_xai, check=False, cwd=str(ROOT), env=env)
    else:
        print(f"WARNING: final checkpoint missing: {ckpt}", flush=True)


if __name__ == "__main__":
    main()
