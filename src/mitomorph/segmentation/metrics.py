"""Segmentation quality metrics (FR-14, NFR-05)."""

from __future__ import annotations

import numpy as np


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks."""
    pred, true = pred_mask.astype(bool), true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return 1.0 if union == 0 else float(intersection / union)


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Dice coefficient between two binary masks. Target: >= 0.85 (NFR-05)."""
    pred, true = pred_mask.astype(bool), true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    denom = pred.sum() + true.sum()
    return 1.0 if denom == 0 else float(2 * intersection / denom)
