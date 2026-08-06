"""Comprehensive PDF report generation (FR-39)."""

from __future__ import annotations

from pathlib import Path

from mitomorph.data.schema import AnalysisResult


def build_pdf_report(
    results: list[AnalysisResult], output_path: str | Path, experiment_metadata: dict | None = None
) -> Path:
    """Build a PDF report with experimental details, analysis parameters, results, and statistics (FR-39).

    Intended to compose figures from :mod:`mitomorph.reporting.figures` and
    :mod:`mitomorph.reporting.temporal_plots` via ``reportlab``.
    """
    raise NotImplementedError("PDF layout pending reporting/figures.py implementation (FR-39)")
