"""Microscopy image loading (FR-01, FR-02).

Supports TIFF/OME-TIFF natively via ``tifffile``, and CZI via the optional
``aicsimageio`` dependency. LIF support is intentionally deferred until a
real LIF sample is available to validate against.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from mitomorph.exceptions import ImageLoadError
from mitomorph.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".czi")


@dataclass
class MicroscopyImage:
    """In-memory representation of a loaded microscopy image."""

    data: np.ndarray
    channel_names: list[str]
    axes: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @property
    def n_channels(self) -> int:
        return self.data.shape[self.axes.index("C")] if "C" in self.axes else 1


def load_image(path: str | Path) -> MicroscopyImage:
    """Load a TIFF/OME-TIFF or CZI microscopy image from disk.

    Raises:
        ImageLoadError: if the file doesn't exist, has an unsupported
            extension, or fails to parse.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"Image file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return _load_tiff(path)
    if suffix == ".czi":
        return _load_czi(path)
    raise ImageLoadError(
        f"Unsupported image format '{suffix}' for {path}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def _load_tiff(path: Path) -> MicroscopyImage:
    try:
        with tifffile.TiffFile(str(path)) as tif:
            data = tif.asarray()
            axes = tif.series[0].axes if tif.series else "YX"
            ome_xml = tif.ome_metadata
    except Exception as exc:
        raise ImageLoadError(f"Failed to read TIFF file {path}: {exc}") from exc

    channel_names = _channel_names_from_ome(ome_xml, axes, data.shape) or _default_channel_names(
        axes, data.shape
    )
    metadata = {"ome_xml": ome_xml} if ome_xml else {}
    logger.debug("Loaded TIFF %s with axes=%s shape=%s", path, axes, data.shape)
    return MicroscopyImage(
        data=data, channel_names=channel_names, axes=axes, metadata=metadata, source_path=str(path)
    )


def _load_czi(path: Path) -> MicroscopyImage:
    try:
        from aicsimageio import AICSImage
    except ImportError as exc:
        raise ImageLoadError(
            "Reading CZI files needs the optional 'aicsimageio' dependency: pip install aicsimageio==4.14.0"
        ) from exc

    try:
        img = AICSImage(str(path))
        axes = "CZYX" if img.dims.Z > 1 else "CYX"
        data = img.get_image_data(axes)
        channel_names = (
            list(img.channel_names) if img.channel_names else _default_channel_names(axes, data.shape)
        )
    except Exception as exc:
        raise ImageLoadError(f"Failed to read CZI file {path}: {exc}") from exc

    logger.debug("Loaded CZI %s with axes=%s shape=%s", path, axes, data.shape)
    return MicroscopyImage(
        data=data, channel_names=channel_names, axes=axes, metadata={}, source_path=str(path)
    )


def _default_channel_names(axes: str, shape: tuple[int, ...]) -> list[str]:
    if "C" not in axes:
        return ["channel_0"]
    n_channels = shape[axes.index("C")]
    return [f"channel_{i}" for i in range(n_channels)]


def _channel_names_from_ome(ome_xml: str | None, axes: str, shape: tuple[int, ...]) -> list[str] | None:
    """Extract <Channel Name="..."> values from OME-XML metadata, if present and complete."""
    if not ome_xml or "C" not in axes:
        return None
    n_channels = shape[axes.index("C")]
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None
    names = [
        elem.get("Name") for elem in root.iter() if elem.tag.endswith("}Channel") or elem.tag == "Channel"
    ]
    names = [n for n in names if n]
    return names if len(names) == n_channels else None
