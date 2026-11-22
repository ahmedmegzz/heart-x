#!/usr/bin/env python3
"""
Lock the HEART-X final configuration from DS1 only, then test DS2 once.

Selection never reads DS2. Record 124 stays inside DS1, matching the locked
study protocol (Chazal DS1). It is not part of DS2.

Protocol, fixed before the DS2 test
------------------------------------
1. Patient-wise K-fold CV on MIT-BIH DS1 (``MITBIH_DS1``). Fold assignment
   seed is 42 and is shared by every candidate.
2. Imbalance recipes are the existing advisor grid. They are ranked on the
   attention model (1D+2D+clinical) by mean validation macro-F1 across folds
   at the pre-specified seed ``seeds[0]``. If the leading mean is within
   0.01 of another recipe, the lower standard deviation wins.
3. Three candidates are then compared on the same folds and all seeds,
   under the same epoch budget, batch size, and learning rate:
      - 2D + clinical, concat fusion, reference imbalance
        (balanced class weight, cross-entropy, no sampler)
      - 1D + 2D + clinical + attention, reference imbalance
      - 1D + 2D + clinical + attention + the DS1-CV imbalance winner
4. The same mean / stability rule picks the final candidate.
5. That candidate is retrained on DS1. A patient-wise 20% slice of DS1
   (seed 42) is used only to keep the best checkpoint. DS2 is evaluated
   once. External validation, fusion weights, and MM-GradCAM are regenerated
   from this checkpoint.

The epoch budget defaults to 8, the same budget as the completed advisor
suite. Every candidate in this script shares it.

F and Q are never dropped from the tables.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import (  # noqa: E402
    CHECKPOINT_DIR,
    FIGURE_DIR,
    LEARNING_RATE,
    MITBIH_DS1,
    MITBIH_DS2,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    WEIGHT_DECAY,
)
from heartx.data.dataset import DualStreamECGDataset  # noqa: E402
from heartx.data.patient_split import patient_wise_cv_folds  # noqa: E402
from heartx.models.dual_stream import build_heartx_model  # noqa: E402
from heartx.utils.balance import oversample_balance, set_seed  # noqa: E402
from heartx.utils.train_utils import evaluate, fit, save_eval_artifacts  # noqa: E402
from heartx.utils.viz import plot_confusion_matrix  # noqa: E402

# Same grid as scripts/run_advisor_suite.py. No new imbalance method.
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
REFERENCE = ("balanced", "ce", "none")
TIE_MARGIN = 0.01


def choose_winner(rows: list[dict], margin: float = TIE_MARGIN) -> dict:
    """Highest mean macro-F1. Within ``margin``, prefer the smaller std."""
    if not rows:
        raise ValueError("no candidates")
    best_mean = max(float(r["f1_mean"]) for r in rows)
    pool = [r for r in rows if best_mean - float(r["f1_mean"]) <= margin]
    pool.sort(key=lambda r: (float(r["f1_std"]), -float(r["f1_mean"]), str(r["name"])))
    return pool[0]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _fmt(mean: float, std: float) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.4f} ± {std:.4f}"


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    weights = np.ones(n_classes, dtype=np.float64)
    for cls, wi in zip(present, w):
        weights[int(cls)] = wi
    weights = np.sqrt(weights)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def _needs_clinical(model_name: str) -> bool:
    return model_name in {"2d_clinical", "1d_2d_clinical", "dual"}


def _loader(x1, x2, y, clinical, model_name: str, batch_size: int, shuffle: bool):
    clin = clinical if _needs_clinical(model_name) else None
    ds = DualStreamECGDataset(x1, x2, y, clin)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _slice(data, idx: np.ndarray):
    return (
        np.asarray(data["x_1d"][idx], dtype=np.float32),
        np.asarray(data["x_2d"][idx]),
        np.asarray(data["clinical"][idx], dtype=np.float32),
        np.asarray(data["y"][idx], dtype=np.int64),
    )


def train_val_once(
    data,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    class_names: list[str],
    model_name: str,
    fusion: str,
    class_weight: str,
    loss: str,
    sampler: str,
    seed: int,
    epochs: int,
    batch_size: int,
    checkpoint: Path,
) -> dict:
    set_seed(seed)
    device = _device()
    x1_tr, x2_tr, c_tr, y_tr = _slice(data, train_idx)
    x1_va, x2_va, c_va, y_va = _slice(data, val_idx)
    fs = float(data["fs"]) if "fs" in data.files else 360.0
    method = str(data["method"]) if "method" in data.files else "stft"

    weights = _class_weights(y_tr, len(class_names)) if class_weight == "balanced" else None
    if sampler != "none":
        x1_tr, x2_tr, c_tr, y_tr = oversample_balance(
            x1_tr, x2_tr, c_tr, y_tr,
            random_state=seed,
            method=sampler,
            fs=fs,
            spectrogram_method=method,
        )

    model = build_heartx_model(
        model_name,
        num_classes=len(class_names),
        clinical_dim=int(c_tr.shape[1]) if _needs_clinical(model_name) else 0,
        fusion_type=fusion,
    ).to(device)
    train_loader = _loader(x1_tr, x2_tr, y_tr, c_tr, model_name, batch_size, True)
    val_loader = _loader(x1_va, x2_va, y_va, c_va, model_name, batch_size, False)
    del x1_tr, x2_tr, c_tr
    history = fit(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        class_names=class_names,
        checkpoint_path=checkpoint,
        class_weights=weights,
        loss_name=loss,
        focal_gamma=2.0,
        extra_checkpoint={
            "model_name": model_name,
            "fusion": fusion,
            "class_weight": class_weight,
            "loss": loss,
            "sampler": sampler,
            "seed": seed,
            "epochs": epochs,
        },
    )
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    metrics = evaluate(model, val_loader, device, class_names)
    payload = {
        "model": model_name,
        "fusion": fusion,
        "class_weight": class_weight,
        "loss": loss,
        "sampler": sampler,
        "seed": seed,
        "epochs": epochs,
        "device": str(device),
        "best_epoch": int(state.get("epoch", -1)),
        "f1_macro": metrics["f1_macro"],
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "history_val_f1": history["val_f1"],
        "checkpoint_val_f1": float(state.get("val_f1_macro", metrics["f1_macro"])),
    }
    del model, train_loader, val_loader, state
    gc.collect()
    return payload


def _recipe_name(model: str, fusion: str, cw: str, loss: str, sampler: str) -> str:
    return f"{model}__{fusion}__cw-{cw}__loss-{loss}__samp-{sampler}"


def _summarize(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        c for c in raw.columns
        if c in {"f1_macro", "accuracy", "precision_macro", "recall_macro"}
        or c.startswith(("f1_", "precision_", "recall_"))
    ]
    rows = []
    for keys, g in raw.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(g))
        for col in metric_cols:
            mean, std = _mean_std(g[col].astype(float).tolist())
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_meanstd"] = _fmt(mean, std)
        # Stability across seeds: average folds first.
        if "seed" in g.columns and g["seed"].nunique() > 1:
            seed_means = g.groupby("seed")["f1_macro"].mean()
            sm, ss = _mean_std(seed_means.tolist())
            row["f1_macro_seed_mean"] = sm
            row["f1_macro_seed_std"] = ss
            row["f1_macro_seed_meanstd"] = _fmt(sm, ss)
        row["name"] = " | ".join(str(row[c]) for c in group_cols)
        row["f1_mean"] = row["f1_macro_mean"]
        row["f1_std"] = row["f1_macro_std"]
        rows.append(row)
    return pd.DataFrame(rows)


def _flatten_run(payload: dict, **extra) -> dict:
    row = {
        **extra,
        "model": payload["model"],
        "fusion": payload["fusion"],
        "class_weight": payload["class_weight"],
        "loss": payload["loss"],
        "sampler": payload["sampler"],
        "seed": payload["seed"],
        "best_epoch": payload["best_epoch"],
        "f1_macro": payload["f1_macro"],
        "accuracy": payload["accuracy"],
        "precision_macro": payload["precision_macro"],
        "recall_macro": payload["recall_macro"],
    }
    for cls, stats in payload["per_class"].items():
        row[f"precision_{cls}"] = stats["precision"]
        row[f"recall_{cls}"] = stats["recall"]
        row[f"f1_{cls}"] = stats["f1"]
        row[f"support_{cls}"] = stats["support"]
    return row


def _sum_cms(payloads: list[dict], class_names: list[str]) -> np.ndarray:
    total = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for p in payloads:
        total += np.asarray(p["confusion_matrix"], dtype=np.int64)
    return total


def _write_cm(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(path)
    plot_confusion_matrix(cm, class_names, path.with_suffix(".png"), title=title)


def _payload_from_checkpoint(path: Path, epochs: int) -> dict | None:
    """Rebuild a run record from the best-epoch metrics stored in a checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Older checkpoints omitted the budget and were fit on folds that dropped
    # record 124. Do not reuse them.
    if int(state.get("epochs", -1)) != int(epochs):
        return None
    if "val_f1_macro" not in state or "val_per_class" not in state:
        return None
    per_in = state["val_per_class"]
    class_names = [str(c) for c in state.get("class_names", list(per_in))]
    per = {}
    for name in class_names:
        stats = per_in[name]
        per[name] = {
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1": float(stats["f1"]),
            "support": int(stats["support"]),
            "specificity": float(stats.get("specificity", 0.0)),
        }
    cm = np.asarray(state["val_confusion_matrix"], dtype=np.int64).tolist()
    prec = [per[c]["precision"] for c in class_names]
    rec = [per[c]["recall"] for c in class_names]
    return {
        "model": str(state["model_name"]),
        "fusion": str(state["fusion"]),
        "class_weight": str(state["class_weight"]),
        "loss": str(state["loss"]),
        "sampler": str(state["sampler"]),
        "seed": int(state["seed"]),
        "epochs": int(epochs),
        "device": "resumed-checkpoint",
        "best_epoch": int(state.get("epoch", -1)),
        "f1_macro": float(state["val_f1_macro"]),
        "accuracy": float(state["val_accuracy"]),
        "precision_macro": float(np.mean(prec)) if prec else float("nan"),
        "recall_macro": float(np.mean(rec)) if rec else float("nan"),
        "per_class": per,
        "confusion_matrix": cm,
        "history_val_f1": [],
        "checkpoint_val_f1": float(state["val_f1_macro"]),
        "resumed_from_checkpoint": True,
        "run": path.stem,
    }


def _py() -> Path:
    for rel in (
        Path(".venv") / "Scripts" / "python.exe",
        Path(".venv") / "bin" / "python",
    ):
        candidate = ROOT / rel
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_logged(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "mitbih" / "mitbih_dual.npz")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fold-seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(8)
    print(f"Device: {_device()}", flush=True)
    out_dir = RESULTS_DIR / "final_selection"
    run_dir = out_dir / "runs"
    fig_dir = FIGURE_DIR / "final_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, mmap_mode="r")
    groups = np.asarray(data["groups"]).astype(str)
    y_all = np.asarray(data["y"])
    class_names = [str(c) for c in data["class_names"]]
    ds1 = {str(g) for g in MITBIH_DS1}
    ds2 = {str(g) for g in MITBIH_DS2}
    if ds1 & ds2:
        raise RuntimeError(f"DS1/DS2 overlap: {ds1 & ds2}")
    present = set(np.unique(groups))
    if not ds1.issubset(present):
        raise RuntimeError(f"DS1 records missing from npz: {sorted(ds1 - present)}")
    outside = present - ds1
    # Anything outside DS1 (DS2, and any other record) stays out of selection.
    # Record 124 is inside DS1 and therefore stays in.
    folds = patient_wise_cv_folds(
        y_all, groups, n_splits=args.folds, seed=args.fold_seed, exclude_groups=sorted(outside)
    )
    fold_meta = []
    for k, (tr, va, tr_g, va_g) in enumerate(folds):
        tr_set, va_set = set(map(str, tr_g)), set(map(str, va_g))
        if not tr_set.issubset(ds1) or not va_set.issubset(ds1):
            raise RuntimeError(f"Fold {k} left DS1")
        if tr_set & ds2 or va_set & ds2:
            raise RuntimeError(f"Fold {k} contains DS2")
        fold_meta.append({
            "fold": k,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "train_records": sorted(tr_set),
            "val_records": sorted(va_set),
            "val_class_counts": {
                class_names[i]: int(np.sum(y_all[va] == i)) for i in range(len(class_names))
            },
        })
        print(
            f"Fold {k}: train {len(tr_set)} records / {len(tr)} beats, "
            f"val {sorted(va_set)} / {len(va)} beats",
            flush=True,
        )
    covered = set()
    for meta in fold_meta:
        covered.update(meta["val_records"])
    if covered != ds1:
        raise RuntimeError(f"CV did not cover DS1 exactly: missing {sorted(ds1 - covered)}")
    (out_dir / "protocol.json").write_text(json.dumps({
        "epochs": args.epochs,
        "folds": args.folds,
        "seeds": list(args.seeds),
        "batch_size": args.batch_size,
        "fold_seed": args.fold_seed,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "tie_margin": TIE_MARGIN,
        "reference_imbalance": {
            "class_weight": REFERENCE[0],
            "loss": REFERENCE[1],
            "sampler": REFERENCE[2],
        },
        "ds2_used_for_selection": False,
        "record_124_in_ds1": True,
        "epoch_budget_note": (
            "Eight epochs matches the completed advisor-suite budget. "
            "All candidates in this script share it."
        ),
        "record_124": "included in DS1; excluded from DS2",
    }, indent=2))
    (out_dir / "folds.json").write_text(json.dumps({
        "ds1": sorted(ds1),
        "excluded_from_selection": sorted(outside),
        "folds": fold_meta,
        "note": (
            "Record 124 is part of this study's DS1 and is included in "
            "patient-wise selection and in the final DS1 fit. DS2 is excluded."
        ),
    }, indent=2))

    screen_seed = int(args.seeds[0])

    def cached(name: str) -> dict | None:
        path = run_dir / f"{name}.json"
        if path.exists():
            hit = json.loads(path.read_text())
            if int(hit.get("epochs", -1)) != int(args.epochs):
                print(
                    f"ignore {name}: cached epochs={hit.get('epochs')} "
                    f"!= requested {args.epochs}",
                    flush=True,
                )
            else:
                return hit
        # A .pt file is written after every improved epoch, including runs that
        # were interrupted. Only the JSON written after the last epoch counts.
        return None

    def execute(name: str, train_idx, val_idx, model, fusion, cw, loss, sampler, seed) -> dict:
        hit = cached(name)
        if hit is not None:
            print(f"resume {name}  val_macroF1={hit['f1_macro']:.4f}", flush=True)
            return hit
        print(f"\n=== {name} ===", flush=True)
        ckpt = CHECKPOINT_DIR / "final_selection" / f"{name}.pt"
        payload = train_val_once(
            data, train_idx, val_idx, class_names, model, fusion, cw, loss, sampler,
            seed, args.epochs, args.batch_size, ckpt,
        )
        payload["run"] = name
        (run_dir / f"{name}.json").write_text(json.dumps(payload))
        print(
            f"done {name}  val_macroF1={payload['f1_macro']:.4f}  "
            f"acc={payload['accuracy']:.4f}  epoch={payload['best_epoch']}",
            flush=True,
        )
        return payload

    # ----- Phase 1: imbalance method on the attention model, DS1 CV, one seed -----
    screen_payloads = []
    for cw, loss, sampler in IMBALANCE:
        for k, (tr, va, _, _) in enumerate(folds):
            name = _recipe_name("1d_2d_clinical", "attention", cw, loss, sampler)
            name = f"{name}__fold{k}__seed{screen_seed}"
            screen_payloads.append(execute(
                name, tr, va, "1d_2d_clinical", "attention", cw, loss, sampler, screen_seed,
            ) | {"fold": k})

    screen_rows = [
        _flatten_run(p, fold=p["fold"], phase="imbalance_screen") for p in screen_payloads
    ]
    screen_df = pd.DataFrame(screen_rows)
    screen_df.to_csv(out_dir / "imbalance_cv_raw.csv", index=False)
    screen_sum = _summarize(
        screen_df, ["model", "fusion", "class_weight", "loss", "sampler"]
    )
    screen_sum.to_csv(out_dir / "imbalance_cv_mean_std.csv", index=False)
    winner_imb = choose_winner(screen_sum.to_dict(orient="records"))
    imb_cw = str(winner_imb["class_weight"])
    imb_loss = str(winner_imb["loss"])
    imb_sampler = str(winner_imb["sampler"])
    (out_dir / "imbalance_decision.json").write_text(json.dumps({
        "rule": "max mean validation macro-F1 across DS1 folds; within 0.01, lower std",
        "seed": screen_seed,
        "folds": args.folds,
        "chosen": {
            "class_weight": imb_cw,
            "loss": imb_loss,
            "sampler": imb_sampler,
            "f1_mean": float(winner_imb["f1_mean"]),
            "f1_std": float(winner_imb["f1_std"]),
        },
        "ds2_used": False,
    }, indent=2))
    print(
        f"\nImbalance winner (DS1 CV, seed {screen_seed}): "
        f"class_weight={imb_cw} loss={imb_loss} sampler={imb_sampler} "
        f"macro-F1 {_fmt(winner_imb['f1_mean'], winner_imb['f1_std'])}",
        flush=True,
    )

    # ----- Phase 2: three candidates, same folds, all seeds -----
    ref_cw, ref_loss, ref_sampler = REFERENCE
    candidates = [
        ("2d_clinical", "2d_clinical", "concat", ref_cw, ref_loss, ref_sampler),
        ("attention", "1d_2d_clinical", "attention", ref_cw, ref_loss, ref_sampler),
        ("attention_imbalance", "1d_2d_clinical", "attention", imb_cw, imb_loss, imb_sampler),
    ]
    # Identical recipe: do not train it twice. The comparison table still
    # lists the third candidate, with the same DS1 CV numbers.
    identical_to_reference = (imb_cw, imb_loss, imb_sampler) == REFERENCE
    if identical_to_reference:
        candidates = candidates[:2]

    cmp_payloads = []
    for cand_name, model, fusion, cw, loss, sampler in candidates:
        for seed in args.seeds:
            for k, (tr, va, _, _) in enumerate(folds):
                name = f"{_recipe_name(model, fusion, cw, loss, sampler)}__fold{k}__seed{seed}"
                payload = execute(name, tr, va, model, fusion, cw, loss, sampler, seed)
                cmp_payloads.append(payload | {"fold": k, "candidate": cand_name})

    if identical_to_reference:
        for payload in list(cmp_payloads):
            if payload["candidate"] == "attention":
                cmp_payloads.append(dict(payload, candidate="attention_imbalance"))

    cmp_rows = [
        _flatten_run(p, fold=p["fold"], candidate=p["candidate"], phase="candidate_cv")
        for p in cmp_payloads
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(out_dir / "candidates_cv_raw.csv", index=False)
    cmp_sum = _summarize(cmp_df, ["candidate", "model", "fusion", "class_weight", "loss", "sampler"])
    cmp_sum.to_csv(out_dir / "candidates_cv_mean_std.csv", index=False)

    for cand, g in cmp_df.groupby("candidate"):
        matched = [p for p in cmp_payloads if p["candidate"] == cand]
        cm = _sum_cms(matched, class_names)
        _write_cm(
            cm, class_names,
            fig_dir / f"cv_{cand}_confusion_sum.csv",
            title=f"DS1 CV summed CM ({cand})",
        )
        # Mean of per-seed fold-sums, so each DS1 beat is counted once per seed.
        seed_cms = []
        for seed, sg in g.groupby("seed"):
            seed_payloads = [p for p in matched if int(p["seed"]) == int(seed)]
            seed_cms.append(_sum_cms(seed_payloads, class_names).astype(np.float64))
        mean_cm = np.mean(np.stack(seed_cms, axis=0), axis=0)
        pd.DataFrame(mean_cm, index=class_names, columns=class_names).to_csv(
            fig_dir / f"cv_{cand}_confusion_mean_over_seeds.csv"
        )

    final = choose_winner(cmp_sum.to_dict(orient="records"))
    decision = {
        "rule": "max mean DS1 CV macro-F1 over folds and seeds; within 0.01, lower std",
        "tie_margin": TIE_MARGIN,
        "chosen_candidate": final["candidate"],
        "model": final["model"],
        "fusion": final["fusion"],
        "class_weight": final["class_weight"],
        "loss": final["loss"],
        "sampler": final["sampler"],
        "f1_mean": float(final["f1_mean"]),
        "f1_std": float(final["f1_std"]),
        "ds2_used_for_selection": False,
        "record_124_in_ds1": True,
    }
    (out_dir / "model_decision.json").write_text(json.dumps(decision, indent=2))
    print(
        f"\nFinal candidate: {final['candidate']}  "
        f"macro-F1 {_fmt(final['f1_mean'], final['f1_std'])}",
        flush=True,
    )

    # ----- Phase 3: refit on DS1, one DS2 test -----
    final_json = out_dir / "final_ds2_test.json"
    if final_json.exists():
        final_metrics = json.loads(final_json.read_text())
        print("resume final DS2 test", flush=True)
    else:
        ds1_idx = np.where(np.isin(groups, list(ds1)))[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.fold_seed)
        tr_rel, va_rel = next(gss.split(ds1_idx, y_all[ds1_idx], groups=groups[ds1_idx]))
        tr_idx, va_idx = ds1_idx[tr_rel], ds1_idx[va_rel]
        if set(groups[tr_idx]) & set(groups[va_idx]):
            raise RuntimeError("final DS1 split leaked a record")
        if set(groups[tr_idx]) & ds2 or set(groups[va_idx]) & ds2:
            raise RuntimeError("final DS1 split included DS2")
        ckpt = CHECKPOINT_DIR / "heartx_final.pt"
        print("\n=== final DS1 refit ===", flush=True)
        _ = train_val_once(
            data, tr_idx, va_idx, class_names,
            str(final["model"]), str(final["fusion"]),
            str(final["class_weight"]), str(final["loss"]), str(final["sampler"]),
            screen_seed, args.epochs, args.batch_size, ckpt,
        )
        # One DS2 evaluation.
        ds2_idx = np.where(np.isin(groups, list(ds2)))[0]
        if set(groups[ds2_idx]) - ds2:
            raise RuntimeError("DS2 index set is not pure")
        device = _device()
        x1, x2, clin, yy = _slice(data, ds2_idx)
        loader = _loader(x1, x2, yy, clin, str(final["model"]), args.batch_size, False)
        model = build_heartx_model(
            str(final["model"]),
            num_classes=len(class_names),
            clinical_dim=int(clin.shape[1]) if _needs_clinical(str(final["model"])) else 0,
            fusion_type=str(final["fusion"]),
        ).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        test_metrics = evaluate(model, loader, device, class_names)
        save_eval_artifacts(test_metrics, class_names, out_dir, prefix="ds2")
        _write_cm(
            np.asarray(test_metrics["confusion_matrix"]),
            class_names,
            fig_dir / "ds2_confusion_matrix.csv",
            title=f"DS2 once (macro-F1={test_metrics['f1_macro']:.3f})",
        )
        final_metrics = {
            "checkpoint": str(ckpt),
            "seed": screen_seed,
            "model": final["model"],
            "fusion": final["fusion"],
            "class_weight": final["class_weight"],
            "loss": final["loss"],
            "sampler": final["sampler"],
            "ds1_train_records": sorted(set(groups[tr_idx])),
            "ds1_val_records_for_checkpoint_only": sorted(set(groups[va_idx])),
            "n_ds2": int(len(ds2_idx)),
            "f1_macro": test_metrics["f1_macro"],
            "accuracy": test_metrics["accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "report": test_metrics["report"],
            "ds2_evaluations": 1,
        }
        final_json.write_text(json.dumps(final_metrics, indent=2))
        print("\n=== DS2 (single evaluation) ===", flush=True)
        print(test_metrics["report"], flush=True)
        del model, loader, x1, x2, clin
        gc.collect()

    # ----- Phase 4: external validation + XAI from the final checkpoint -----
    ckpt = CHECKPOINT_DIR / "heartx_final.pt"
    marker = out_dir / "posthoc_done.json"
    if not marker.exists():
        env_note = os.environ.copy()
        env_note["PYTHONPATH"] = str(ROOT / "src")
        env_note["PYTHONUNBUFFERED"] = "1"
        commands = [
            [
                str(_py()), str(ROOT / "scripts" / "external_validation.py"),
                "--checkpoint", str(ckpt),
                "--model", str(final_metrics["model"]),
                "--fusion", str(final_metrics["fusion"]),
                "--batch-size", str(args.batch_size),
            ],
            [
                str(_py()), str(ROOT / "scripts" / "analyze_fusion_weights.py"),
                "--checkpoint", str(ckpt),
                "--model", str(final_metrics["model"]),
                "--fusion", str(final_metrics["fusion"]),
                "--out-dir", str(FIGURE_DIR / "fusion_weights"),
            ],
            [
                str(_py()), str(ROOT / "scripts" / "explain_mm_gradcam.py"),
                "--checkpoint", str(ckpt),
                "--model", str(final_metrics["model"]),
                "--fusion", str(final_metrics["fusion"]),
                "--num-samples", "8",
                "--out-dir", str(FIGURE_DIR / "mm_gradcam"),
            ],
        ]
        ptbxl_path = PROCESSED_DIR / "ptbxl" / "ptbxl_dual.npz"
        skipped = []
        for cmd in commands:
            if cmd[1].endswith("external_validation.py") and not ptbxl_path.exists():
                note = (
                    "External validation was not run because "
                    f"{ptbxl_path} is not on disk. "
                    "The DS2 test above is unaffected."
                )
                (out_dir / "external_validation_skipped.txt").write_text(
                    note + "\n", encoding="utf-8"
                )
                skipped.append("external_validation")
                print("\nskip external validation:", note, flush=True)
                continue
            print("\n>>>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=str(ROOT), env=env_note)
        marker.write_text(json.dumps({
            "checkpoint": str(ckpt),
            "skipped": skipped,
        }, indent=2))

    _write_thesis_report(
        out_dir, fig_dir, class_names, screen_sum, cmp_sum, final, final_metrics,
        epochs=args.epochs,
    )
    print(f"\nWrote {out_dir / 'final_selection_report.txt'}", flush=True)


def _write_thesis_report(
    out_dir, fig_dir, class_names, screen_sum, cmp_sum, final, test, epochs: int = 8,
) -> None:
    lines = []
    lines.append("HEART-X final model selection")
    lines.append("==============================")
    lines.append("")
    lines.append("DS2 was not used to choose the model, fusion, imbalance method, or seed.")
    lines.append("Record 124 is part of DS1 and was included in selection and in the DS1 refit.")
    lines.append("Primary metric: macro-F1. Accuracy is secondary. F and Q stay in every average.")
    lines.append(
        f"Epoch budget: {epochs} (the completed advisor suite used 8). "
        "Batch size, learning rate, folds, and seeds are shared by every candidate."
    )
    lines.append(
        "Binary-proxy MIT-BIH numbers written by external validation use the same DS2 records. "
        "They are produced after the checkpoint is locked and are not used for selection."
    )
    lines.append("Multiple runs are reported as mean ± sample standard deviation (ddof=1).")
    lines.append("")
    lines.append("Imbalance screen (1D+2D+clinical+attention, DS1 patient-wise CV, seed 42)")
    lines.append("-------------------------------------------------------------------------------")
    keep = [
        "class_weight", "loss", "sampler", "n_runs",
        "f1_macro_meanstd", "accuracy_meanstd",
    ]
    for cls in class_names:
        for kind in ("precision", "recall", "f1"):
            col = f"{kind}_{cls}_meanstd"
            if col in screen_sum.columns:
                keep.append(col)
    show = screen_sum.sort_values("f1_mean", ascending=False)
    lines.append(show[[c for c in keep if c in show.columns]].to_string(index=False))
    lines.append("")
    lines.append("Head-to-head (same DS1 folds and seeds 42, 43, 44)")
    lines.append("--------------------------------------------------")
    keep2 = ["candidate", "model", "fusion", "class_weight", "loss", "sampler", "n_runs",
             "f1_macro_meanstd", "accuracy_meanstd"]
    if "f1_macro_seed_meanstd" in cmp_sum.columns:
        keep2.append("f1_macro_seed_meanstd")
    for cls in class_names:
        for kind in ("precision", "recall", "f1"):
            col = f"{kind}_{cls}_meanstd"
            if col in cmp_sum.columns:
                keep2.append(col)
    lines.append(cmp_sum[[c for c in keep2 if c in cmp_sum.columns]].to_string(index=False))
    lines.append("")
    lines.append(
        f"Selected: {final['candidate']} | {final['model']} | fusion={final['fusion']} | "
        f"class_weight={final['class_weight']} | loss={final['loss']} | sampler={final['sampler']}"
    )
    lines.append(f"DS1 CV macro-F1: {_fmt(float(final['f1_mean']), float(final['f1_std']))}")
    lines.append("")
    lines.append("DS2 test (one evaluation of the refit checkpoint)")
    lines.append("-------------------------------------------------")
    lines.append(f"macro-F1: {test['f1_macro']:.4f}")
    lines.append(f"accuracy: {test['accuracy']:.4f}")
    lines.append("class, precision, recall, f1, support")
    for cls, stats in test["per_class"].items():
        lines.append(
            f"{cls}, {stats['precision']:.4f}, {stats['recall']:.4f}, "
            f"{stats['f1']:.4f}, {stats['support']}"
        )
    lines.append("")
    lines.append(test.get("report", ""))
    lines.append("")
    lines.append("Limitation — classes F and Q")
    lines.append("----------------------------")
    f_stats = test["per_class"].get("F", {})
    q_stats = test["per_class"].get("Q", {})
    lines.append(
        "F and Q are reported in the macro-F1 and in the confusion matrix. "
        "Their low scores are a limitation of this final model, not a filtered result."
    )
    if f_stats:
        lines.append(
            f"DS2 F: precision={f_stats['precision']:.4f}, recall={f_stats['recall']:.4f}, "
            f"F1={f_stats['f1']:.4f}, support={f_stats['support']}. "
            "Fusion beats are rare and morphologically overlap N and V, so the model "
            "does not separate them reliably under the patient-wise protocol."
        )
    if q_stats:
        lines.append(
            f"DS2 Q: precision={q_stats['precision']:.4f}, recall={q_stats['recall']:.4f}, "
            f"F1={q_stats['f1']:.4f}, support={q_stats['support']}. "
            "After pacemaker records are removed, Q is extremely scarce and is concentrated "
            "in few recordings. A patient-wise split therefore gives the classifier almost "
            "no stable Q examples, and a low Q score should be discussed as a data limitation."
        )
    lines.append("")
    lines.append("Confusion matrices:")
    lines.append(f"  DS2: {fig_dir / 'ds2_confusion_matrix.csv'}")
    lines.append(f"  CV sums: {fig_dir}")
    lines.append(f"Checkpoint: {test['checkpoint']}")
    text = "\n".join(lines) + "\n"
    (out_dir / "final_selection_report.txt").write_text(text, encoding="utf-8")
    # Thesis-facing numeric table of the single DS2 test.
    pd.DataFrame([
        {"metric": "macro_f1", "value": test["f1_macro"]},
        {"metric": "accuracy", "value": test["accuracy"]},
        {"metric": "precision_macro", "value": test["precision_macro"]},
        {"metric": "recall_macro", "value": test["recall_macro"]},
    ]).to_csv(out_dir / "ds2_overall.csv", index=False)


if __name__ == "__main__":
    main()
