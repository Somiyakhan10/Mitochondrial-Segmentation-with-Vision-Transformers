"""Publication-quality figures: overlays, box/bar plots, correlation scatter (FR-35-FR-37)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    save_path: str | Path | None = None,
    overlay_color: tuple[float, float, float] = (1.0, 0.25, 0.35),
    alpha: float = 0.45,
) -> plt.Figure:
    """Overlay a segmentation mask on the original image (FR-35)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [*overlay_color, alpha]
    ax.imshow(overlay)
    ax.axis("off")
    fig.tight_layout(pad=0)
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def plot_condition_comparison(
    data: pd.DataFrame,
    feature: str,
    groupby: str = "condition",
    kind: str = "box",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Box/bar plot comparing a morphometric feature across experimental conditions
    (e.g. Naive vs SCI vs PTEN-KO vs PGC1a) (FR-36).
    """
    if feature not in data.columns or groupby not in data.columns:
        raise ValueError(f"data must contain columns '{feature}' and '{groupby}'")

    grouped = data.groupby(groupby)
    labels = [str(name) for name, _ in grouped]
    groups = [g[feature].dropna().values for _, g in grouped]

    fig, ax = plt.subplots(figsize=(6, 4))
    if kind == "box":
        ax.boxplot(groups, tick_labels=labels)
    elif kind == "bar":
        means = [g.mean() if len(g) else 0.0 for g in groups]
        ax.bar(labels, means)
    else:
        raise ValueError(f"Unknown kind '{kind}', expected 'box' or 'bar'")
    ax.set_ylabel(feature)
    ax.set_xlabel(groupby)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def plot_feature_correlation(
    data: pd.DataFrame, x_feature: str, y_feature: str, save_path: str | Path | None = None
) -> plt.Figure:
    """Scatter plot of the correlation between two morphometric features (FR-37)."""
    if x_feature not in data.columns or y_feature not in data.columns:
        raise ValueError(f"data must contain columns '{x_feature}' and '{y_feature}'")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(data[x_feature], data[y_feature], alpha=0.7, edgecolors="none")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig
