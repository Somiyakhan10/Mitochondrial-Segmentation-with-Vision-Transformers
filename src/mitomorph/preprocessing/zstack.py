"""Z-stack collapsing to a single 2D image (FR-06)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace


def max_intensity_projection(zstack: np.ndarray, z_axis: int = 0) -> np.ndarray:
    """Collapse a Z-stack to 2D by taking the per-pixel maximum across Z."""
    return np.max(zstack, axis=z_axis)


def focus_stack(zstack: np.ndarray, z_axis: int = 0) -> np.ndarray:
    """Collapse a Z-stack to 2D by picking, per pixel, the value from the
    slice with the highest local Laplacian magnitude (a simple sharpness proxy).
    """
    zstack = np.moveaxis(zstack, z_axis, 0)
    sharpness = np.stack([np.abs(laplace(sl.astype(np.float64))) for sl in zstack])
    best_slice = np.argmax(sharpness, axis=0)
    return np.take_along_axis(zstack, best_slice[np.newaxis, ...], axis=0)[0]
