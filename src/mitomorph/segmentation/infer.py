"""Segmentation inference (FR-12)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mitomorph.exceptions import SegmentationError
from mitomorph.logger import get_logger
from mitomorph.segmentation.checkpoint import load_checkpoint
from mitomorph.segmentation.models.unet import UNetResNet34

logger = get_logger(__name__)


@dataclass
class SegmentationResult:
    mask: np.ndarray
    confidence: np.ndarray


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> UNetResNet34:
    """Instantiate a :class:`UNetResNet34` and load trained weights from a checkpoint.

    ``pretrained=False`` since the checkpoint fully overwrites the backbone
    weights anyway — downloading ImageNet weights first would be wasted work.
    """
    model = UNetResNet34(in_channels=1, out_channels=1, pretrained=False)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()
    return model


def _pad_to_multiple(image: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (h, w)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, (h, w)


def segment(model: UNetResNet34, image: np.ndarray, confidence_threshold: float = 0.5) -> SegmentationResult:
    """Run inference on a preprocessed, single-channel 2D image.

    Args:
        model: a trained :class:`UNetResNet34` (see :func:`load_model`).
        image: preprocessed 2D intensity image (mitochondrial marker
            channel only — see :mod:`mitomorph.preprocessing.channel_utils`).
        confidence_threshold: probability threshold for the binary mask.

    Returns:
        A :class:`SegmentationResult` with ``mask``/``confidence`` cropped
        back to the input's original shape.
    """
    if model is None:
        raise SegmentationError("No segmentation model loaded (see mitomorph.segmentation.infer.load_model)")
    if image.ndim != 2:
        raise SegmentationError(f"segment() expects a single-channel 2D image, got shape {image.shape}")

    model.eval()
    padded, (orig_h, orig_w) = _pad_to_multiple(image, multiple=32)

    device = next(model.parameters()).device
    tensor = torch.from_numpy(padded.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    probs = probs[:orig_h, :orig_w]
    mask = probs > confidence_threshold
    logger.debug("Segmented image shape=%s -> %d positive pixels", image.shape, int(mask.sum()))
    return SegmentationResult(mask=mask, confidence=probs)
