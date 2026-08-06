"""Neuronal vs. non-neuronal mitochondria classification (FR-15–FR-17)."""

from __future__ import annotations

import numpy as np

from mitomorph.preprocessing.io import MicroscopyImage


def classify_neuronal(
    mito_mask: np.ndarray, image: MicroscopyImage, overlap_threshold: float = 0.5
) -> np.ndarray:
    """Label each connected mitochondrial region as neuronal or non-neuronal
    based on spatial overlap with the neuronal marker channel.

    Args:
        mito_mask: labeled mitochondrial segmentation mask.
        image: source image, must include a neuronal marker channel
            (see :mod:`mitomorph.preprocessing.channel_utils`).
        overlap_threshold: minimum fraction of a mitochondrion's area
            overlapping neuronal marker signal to be classified as neuronal.

    Returns:
        Integer array, same labels as ``mito_mask``, with 1 = neuronal, 0 = non-neuronal.
    """
    raise NotImplementedError(
        "Overlap classification needs a threshold tuned against annotated images (FR-15-FR-17)"
    )
