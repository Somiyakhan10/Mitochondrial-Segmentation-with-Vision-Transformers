"""Input validation for loaded images and their metadata.

``min_resolution`` defaults to 0 (disabled) here so unit tests and small
development images aren't blocked; production runs should set it via
``config/default_config.yaml`` to match SRS §5.1 (1024px minimum).
"""

from __future__ import annotations

from collections.abc import Iterable

from mitomorph.exceptions import ImageValidationError
from mitomorph.preprocessing.io import MicroscopyImage

REQUIRED_METADATA_FIELDS = ("experimental_condition", "time_point", "animal_id")


def validate_image(image: MicroscopyImage, min_channels: int = 2, min_resolution: int = 0) -> None:
    """Validate a loaded image against the pipeline's minimum requirements (SRS §5.1, FR-02).

    Raises:
        ImageValidationError: if the image is empty, has too few channels,
            or is below the minimum resolution.
    """
    if image.data.size == 0:
        raise ImageValidationError(f"Image {image.source_path} is empty")

    n_channels = image.n_channels
    if n_channels < min_channels:
        raise ImageValidationError(
            f"Image {image.source_path} has {n_channels} channel(s); at least {min_channels} required "
            "(mitochondrial marker + nuclear/neuronal marker, FR-02)"
        )

    y_idx = image.axes.index("Y") if "Y" in image.axes else len(image.data.shape) - 2
    x_idx = image.axes.index("X") if "X" in image.axes else len(image.data.shape) - 1
    height, width = image.data.shape[y_idx], image.data.shape[x_idx]
    if height < min_resolution or width < min_resolution:
        raise ImageValidationError(
            f"Image {image.source_path} resolution {width}x{height} is below the minimum "
            f"{min_resolution}x{min_resolution} (SRS §5.1)"
        )


def validate_metadata(metadata: dict, required_fields: Iterable[str] = REQUIRED_METADATA_FIELDS) -> None:
    """Validate that required experimental metadata fields are present (SRS §5.1).

    Raises:
        ImageValidationError: if any required field is missing or empty.
    """
    missing = [f for f in required_fields if f not in metadata or metadata[f] in (None, "")]
    if missing:
        raise ImageValidationError(f"Metadata missing required field(s): {', '.join(missing)}")
