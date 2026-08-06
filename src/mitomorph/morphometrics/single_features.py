"""Per-mitochondrion morphometric feature extraction (FR-19, FR-20)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label, regionprops


@dataclass
class SingleMitoFeatures:
    label: int
    area: float
    perimeter: float
    major_axis_length: float
    minor_axis_length: float
    aspect_ratio: float
    circularity: float
    solidity: float
    eccentricity: float
    feret_diameter_max: float
    extent: float
    equivalent_diameter: float


def extract_single_features(mask: np.ndarray) -> list[SingleMitoFeatures]:
    """Extract per-mitochondrion shape features from a segmentation mask.

    Args:
        mask: either a boolean/binary mask (connected components are
            labeled automatically) or an already-labeled integer mask.
    """
    labeled = mask if np.issubdtype(mask.dtype, np.integer) and mask.max() > 1 else label(mask)

    features: list[SingleMitoFeatures] = []
    for region in regionprops(labeled):
        perimeter = region.perimeter if region.perimeter > 0 else 1e-8
        circularity = 4 * np.pi * region.area / (perimeter**2)
        aspect_ratio = (
            region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else float("nan")
        )
        features.append(
            SingleMitoFeatures(
                label=region.label,
                area=float(region.area),
                perimeter=float(region.perimeter),
                major_axis_length=float(region.major_axis_length),
                minor_axis_length=float(region.minor_axis_length),
                aspect_ratio=float(aspect_ratio),
                circularity=float(circularity),
                solidity=float(region.solidity),
                eccentricity=float(region.eccentricity),
                feret_diameter_max=float(region.feret_diameter_max),
                extent=float(region.extent),
                equivalent_diameter=float(region.equivalent_diameter),
            )
        )
    return features
