"""Segmentation inference (FR-12)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.nn as nn


@dataclass
class SegmentationResult:
    mask: np.ndarray
    confidence: np.ndarray


def segment(model: nn.Module, image: np.ndarray, confidence_threshold: float = 0.5) -> SegmentationResult:
    """Run inference on a preprocessed image and return a binary mask + confidence map.

    Args:
        model: a trained segmentation model (e.g. :class:`UNetResNet34`).
        image: preprocessed 2D intensity image.
        confidence_threshold: probability threshold for the binary mask.
    """
    raise NotImplementedError("Inference pending a trained segmentation checkpoint (FR-12)")
