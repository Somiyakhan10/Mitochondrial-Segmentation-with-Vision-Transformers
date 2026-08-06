"""Training-time data augmentation for segmentation fine-tuning (FR-10, FR-11).

Stub: augmentation strength (rotation range, elastic deformation
parameters, noise level) should be tuned once real training images are
available, to avoid over- or under-augmenting relative to true imaging
variability.
"""

from __future__ import annotations

from typing import Any


def get_training_transforms(config: dict[str, Any] | None = None):
    """Return a MONAI ``Compose`` of training-time augmentations
    (RandRotate90, RandFlip, Rand2DElastic, RandGaussianNoise, etc.).
    """
    raise NotImplementedError("Augmentation strength needs tuning against real training images (FR-10, FR-11)")


def get_validation_transforms(config: dict[str, Any] | None = None):
    """Return a MONAI ``Compose`` of deterministic validation-time transforms (resize/normalize only)."""
    raise NotImplementedError("Validation transforms pending finalized input size (FR-10, FR-11)")
