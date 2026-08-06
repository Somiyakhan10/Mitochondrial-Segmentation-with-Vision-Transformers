"""Longitudinal data model for tracking mitochondrial changes over time (SRS §1.3).

Feeds :mod:`mitomorph.reporting.temporal_plots`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mitomorph.data.schema import AnalysisResult


def _time_point_sort_key(time_point: str) -> tuple[int, float | str]:
    """Sort time points numerically when possible (e.g. '6 weeks' before '21 weeks')."""
    match = re.search(r"\d+(\.\d+)?", time_point)
    return (0, float(match.group())) if match else (1, time_point)


@dataclass
class TimeSeriesData:
    """Per-animal analysis results indexed by time point."""

    animal_id: str
    condition: str
    measurements: dict[str, AnalysisResult] = field(default_factory=dict)

    def add_measurement(self, time_point: str, result: AnalysisResult) -> None:
        self.measurements[time_point] = result

    def time_points(self) -> list[str]:
        return sorted(self.measurements.keys(), key=_time_point_sort_key)

    def health_score_trend(self) -> dict[str, float]:
        """Health score at each time point, in chronological order."""
        return {tp: self.measurements[tp].classification.health_score for tp in self.time_points()}
