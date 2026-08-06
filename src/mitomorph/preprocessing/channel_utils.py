"""Channel identification and extraction for multi-channel images (FR-02, FR-15).

Cell-type-specific analysis (FR-15–FR-17) needs to know which loaded
channel is the mitochondrial marker and which is the neuronal marker.
Channel names come from image metadata (OME-XML / CZI channel names) or
user-supplied config; this module matches them against known marker
name substrings (SRS §5.1: Tom20, COX IV, MitoTracker, VDAC / NeuN, NFH, MAP2).
"""

from __future__ import annotations

import numpy as np

from mitomorph.exceptions import ImageValidationError
from mitomorph.preprocessing.io import MicroscopyImage

DEFAULT_MITOCHONDRIAL_MARKERS = ("tom20", "cox iv", "coxiv", "mitotracker", "vdac", "porin")
DEFAULT_NEURONAL_MARKERS = ("neun", "nfh", "map2")


def identify_channel_type(
    channel_name: str,
    mitochondrial_markers: tuple[str, ...] = DEFAULT_MITOCHONDRIAL_MARKERS,
    neuronal_markers: tuple[str, ...] = DEFAULT_NEURONAL_MARKERS,
) -> str:
    """Classify a channel name as ``"mitochondrial"``, ``"neuronal"``, or ``"unknown"``."""
    name = channel_name.strip().lower()
    if any(marker in name for marker in mitochondrial_markers):
        return "mitochondrial"
    if any(marker in name for marker in neuronal_markers):
        return "neuronal"
    return "unknown"


def extract_mitochondrial_channel(image: MicroscopyImage, **marker_kwargs) -> np.ndarray:
    """Return the array slice for the mitochondrial marker channel (FR-02).

    Raises:
        ImageValidationError: if no channel name matches a known mitochondrial marker.
    """
    return _select_channel(image, _find_channel_index(image, "mitochondrial", **marker_kwargs))


def extract_neuronal_channel(image: MicroscopyImage, **marker_kwargs) -> np.ndarray:
    """Return the array slice for the neuronal marker channel (FR-15).

    Raises:
        ImageValidationError: if no channel name matches a known neuronal marker.
    """
    return _select_channel(image, _find_channel_index(image, "neuronal", **marker_kwargs))


def _find_channel_index(image: MicroscopyImage, channel_type: str, **marker_kwargs) -> int:
    for idx, name in enumerate(image.channel_names):
        if identify_channel_type(name, **marker_kwargs) == channel_type:
            return idx
    raise ImageValidationError(
        f"No {channel_type} channel found among {image.channel_names} for {image.source_path!r}"
    )


def _select_channel(image: MicroscopyImage, channel_index: int) -> np.ndarray:
    if "C" not in image.axes:
        return image.data
    return np.take(image.data, channel_index, axis=image.axes.index("C"))
