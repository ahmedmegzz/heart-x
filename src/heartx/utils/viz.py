"""Plotting helpers for metrics and MM-GradCAM overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mm_gradcam(
    signal_1d: np.ndarray,
    spectrogram: np.ndarray,
    cam_1d: np.ndarray,
    cam_2d: np.ndarray,
    out_path: Path,
    title: str = "MM-GradCAM",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    t = np.arange(len(signal_1d))
    axes[0].plot(t, signal_1d, color="black", linewidth=0.8)
    axes[0].imshow(
        cam_1d[None, :],
        aspect="auto",
        extent=[0, len(signal_1d), signal_1d.min(), signal_1d.max()],
        cmap="jet",
        alpha=0.35,
        interpolation="bilinear",
    )
    axes[0].set_title(f"{title} — 1D stream")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")

    axes[1].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    axes[1].imshow(cam_2d, aspect="auto", origin="lower", cmap="jet", alpha=0.4)
    axes[1].set_title(f"{title} — 2D spectrogram stream")
    axes[1].set_xlabel("Time bins")
    axes[1].set_ylabel("Frequency bins")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
