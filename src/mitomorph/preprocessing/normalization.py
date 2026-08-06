"""Intensity normalization (FR-04)."""

from __future__ import annotations

import numpy as np


def zscore_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize intensities to zero mean, unit variance."""
    image = image.astype(np.float64)
    return (image - image.mean()) / (image.std() + eps)


def percentile_normalize(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Rescale intensities to [0, 1] using percentile clipping.

    Percentiles are computed on the whole array; ``low``/``high`` are
    percentages in [0, 100].
    """
    image = image.astype(np.float64)
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)
