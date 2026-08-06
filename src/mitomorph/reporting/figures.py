"""Publication-quality figures: overlays, box/bar/scatter plots (FR-35–FR-37)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def plot_segmentation_overlay(image: np.ndarray, mask: np.ndarray, save_path: str | Path | None = None) -> Any:
    """Overlay a segmentation mask on the original image (FR-35)."""
    raise NotImplementedError("Overlay rendering pending real segmentation output (FR-35)")


def plot_condition_comparison(
    data: Any, feature: str, groupby: str = "condition", kind: str = "box", save_path: str | Path | None = None
) -> Any:
    """Box/bar plot comparing a morphometric feature across experimental conditions
    (e.g. Naïve vs SCI vs PTEN-KO vs PGC1α) (FR-36).
    """
    raise NotImplementedError("Comparison plots pending multi-condition analysis results (FR-36)")


def plot_feature_correlation(data: Any, x_feature: str, y_feature: str, save_path: str | Path | None = None) -> Any:
    """Scatter plot of the correlation between two morphometric features (FR-37)."""
    raise NotImplementedError("Correlation plots pending analysis results with multiple features (FR-37)")
