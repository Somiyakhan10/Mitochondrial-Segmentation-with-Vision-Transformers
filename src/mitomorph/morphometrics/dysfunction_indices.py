"""Dysfunction indicator scores derived from single-mitochondrion features (FR-22–FR-24)."""

from __future__ import annotations

import numpy as np

from mitomorph.morphometrics.single_features import SingleMitoFeatures


def fragmentation_index(features: list[SingleMitoFeatures], total_area: float) -> float:
    """Number of discrete mitochondria per unit area; higher means more fragmented (FR-22)."""
    if total_area <= 0:
        return 0.0
    return len(features) / total_area


def swelling_score(features: list[SingleMitoFeatures], healthy_baseline_ratio: float) -> float:
    """Fractional deviation of the mean area/perimeter ratio from a healthy baseline (FR-23).

    Positive values indicate swelling relative to baseline.
    """
    if healthy_baseline_ratio == 0:
        raise ValueError("healthy_baseline_ratio must be nonzero")
    ratios = [f.area / f.perimeter for f in features if f.perimeter > 0]
    if not ratios:
        return 0.0
    mean_ratio = float(np.mean(ratios))
    return (mean_ratio - healthy_baseline_ratio) / healthy_baseline_ratio


def mitochondrial_density(features: list[SingleMitoFeatures], cell_area: float) -> float:
    """Fraction of cell area occupied by mitochondria (FR-24)."""
    if cell_area <= 0:
        return 0.0
    return sum(f.area for f in features) / cell_area
