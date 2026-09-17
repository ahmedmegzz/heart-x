"""Class-distribution helpers and reporting for split sets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def class_counts(
    y: np.ndarray,
    class_names: list[str],
) -> dict[str, int]:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=len(class_names))
    return {name: int(counts[i]) for i, name in enumerate(class_names)}


def split_class_distribution(
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    class_names: list[str],
) -> pd.DataFrame:
    """
    Build a tidy table of per-split class counts.

    Parameters
    ----------
    masks : mapping split_name -> boolean mask aligned with ``y``
    """
    rows = []
    for split_name, mask in masks.items():
        ys = y[mask]
        counts = class_counts(ys, class_names)
        for cls, n in counts.items():
            rows.append({"split": split_name, "class": cls, "count": n})
    return pd.DataFrame(rows)


def save_class_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    title: str = "Class distribution by split",
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "class_distribution.csv"
    png_path = out_dir / "class_distribution.png"
    df.to_csv(csv_path, index=False)

    pivot = df.pivot(index="class", columns="split", values="count").fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return csv_path, png_path


def assert_no_empty_test_classes(
    y: np.ndarray,
    test_mask: np.ndarray,
    class_names: list[str],
    raise_error: bool = False,
) -> list[str]:
    """Return names of classes absent from the test set."""
    present = set(np.unique(y[test_mask]).tolist())
    missing = [c for i, c in enumerate(class_names) if i not in present]
    if missing and raise_error:
        raise AssertionError(f"Test set missing classes: {missing}")
    return missing
