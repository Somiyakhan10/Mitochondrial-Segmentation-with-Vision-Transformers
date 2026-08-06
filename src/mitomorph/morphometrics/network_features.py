"""Mitochondrial network/ensemble features (FR-21).

Stub: branch/end point counting requires skeletonization tuned to real
mitochondrial network images (thin, potentially noisy structures) to
avoid spurious branches; deferred until sample data is available to
validate against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkFeatures:
    mitochondrial_footprint: float
    network_size: int
    branch_points: int
    end_points: int


def compute_network_features(mask: np.ndarray) -> NetworkFeatures:
    """Compute network-level morphology features from a binary mitochondrial mask.

    Args:
        mask: binary mask where True/1 marks mitochondrial pixels.
    """
    raise NotImplementedError(
        "Network feature extraction (skeletonization + branch/end point detection) needs "
        "validation against real mitochondrial network images (FR-21)"
    )
