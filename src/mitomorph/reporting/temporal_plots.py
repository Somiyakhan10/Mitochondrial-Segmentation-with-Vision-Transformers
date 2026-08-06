"""Longitudinal visualizations of mitochondrial change over time (SRS §1.3, FR-38)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mitomorph.data.temporal import TimeSeriesData


def timecourse_plot(
    series: list[TimeSeriesData], feature: str = "health_score", save_path: str | Path | None = None
) -> Any:
    """Line plot of a feature's trajectory over time, one line per animal/condition."""
    raise NotImplementedError("Timecourse plotting pending multi-timepoint analysis results (FR-38)")


def longitudinal_heatmap(series: list[TimeSeriesData], save_path: str | Path | None = None) -> Any:
    """Heatmap of mitochondrial network connectivity change across time points (FR-38)."""
    raise NotImplementedError("Heatmap pending network feature extraction (FR-21, FR-38)")
