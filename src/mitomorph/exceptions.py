"""Custom exception hierarchy for the mitomorph pipeline."""


class MitoMorphError(Exception):
    """Base class for all mitomorph errors."""


class ImageLoadError(MitoMorphError):
    """Raised when a microscopy image cannot be read or is malformed."""


class ImageValidationError(MitoMorphError):
    """Raised when an image or its metadata fails validation checks."""


class ConfigError(MitoMorphError):
    """Raised when a configuration file is missing required keys or has invalid values."""


class SegmentationError(MitoMorphError):
    """Raised when the segmentation stage fails to produce a usable mask."""


class ClassificationError(MitoMorphError):
    """Raised when the health/cell-type classification stage fails."""


class DatabaseError(MitoMorphError):
    """Raised for failures reading from or writing to the metadata database."""
