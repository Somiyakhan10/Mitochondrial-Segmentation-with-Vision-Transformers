"""Illumination correction (FR-03).

Stub: real implementations require either a captured flat-field reference
image or tuned rolling-ball radius parameters validated against lab
imaging conditions, neither of which exist yet.
"""

from __future__ import annotations

import numpy as np


def flat_field_correct(image: np.ndarray, flat_field: np.ndarray) -> np.ndarray:
    """Correct uneven illumination using a captured flat-field reference image.

    Args:
        image: raw 2D intensity image.
        flat_field: reference image of the illumination pattern, same shape as ``image``.

    Returns:
        Illumination-corrected image.
    """
    raise NotImplementedError("flat_field_correct requires a validated flat-field reference (FR-03)")


def rolling_ball_subtract(image: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """Subtract background using a rolling-ball algorithm.

    Args:
        image: raw 2D intensity image.
        radius: rolling ball radius in pixels.

    Returns:
        Background-subtracted image.
    """
    raise NotImplementedError("rolling_ball_subtract needs a radius tuned to lab imaging conditions (FR-03)")
