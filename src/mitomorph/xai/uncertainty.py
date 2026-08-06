"""Prediction uncertainty estimation (FR-34)."""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_uncertainty(pipeline: Any, features: np.ndarray) -> np.ndarray:
    """Estimate per-sample prediction uncertainty (e.g. via Random Forest vote variance).

    Returns:
        Array of uncertainty scores in [0, 1], one per sample.
    """
    raise NotImplementedError("Requires a trained classifier ensemble (FR-34)")
