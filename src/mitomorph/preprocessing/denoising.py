"""Denoising (FR-05).

Stub: non-local means and BM3D parameters (patch size, search window,
noise sigma) need to be tuned against real lab images before this is
worth implementing for real.
"""

from __future__ import annotations

import numpy as np


def non_local_means(image: np.ndarray, patch_size: int = 5, patch_distance: int = 6, h: float = 0.1) -> np.ndarray:
    """Denoise via non-local means (skimage.restoration.denoise_nl_means).

    Args:
        image: 2D intensity image.
        patch_size: size of patches used for denoising.
        patch_distance: maximal search distance for similar patches.
        h: cut-off distance / filter strength.
    """
    raise NotImplementedError("non_local_means parameters need tuning against real lab images (FR-05)")


def bm3d_denoise(image: np.ndarray, sigma_psd: float = 0.1) -> np.ndarray:
    """Denoise via BM3D.

    Args:
        image: 2D intensity image.
        sigma_psd: estimated noise standard deviation.
    """
    raise NotImplementedError("bm3d_denoise requires a noise sigma estimated from real lab images (FR-05)")
