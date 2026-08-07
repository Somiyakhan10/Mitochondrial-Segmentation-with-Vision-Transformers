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


def confusion_matrix(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, int]:
    """Pixel-level confusion matrix between a predicted and ground-truth mask (NFR-05, NFR-06)."""
    pred, true = pred_mask.astype(bool), true_mask.astype(bool)
    return {
        "tp": int(np.logical_and(pred, true).sum()),
        "fp": int(np.logical_and(pred, ~true).sum()),
        "fn": int(np.logical_and(~pred, true).sum()),
        "tn": int(np.logical_and(~pred, ~true).sum()),
    }


def precision_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Fraction of predicted-positive pixels that are actually positive."""
    cm = confusion_matrix(pred_mask, true_mask)
    denom = cm["tp"] + cm["fp"]
    return cm["tp"] / denom if denom else 0.0


def recall_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Fraction of actually-positive pixels that were predicted positive."""
    cm = confusion_matrix(pred_mask, true_mask)
    denom = cm["tp"] + cm["fn"]
    return cm["tp"] / denom if denom else 0.0


def f1_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Harmonic mean of precision and recall."""
    precision = precision_score(pred_mask, true_mask)
    recall = recall_score(pred_mask, true_mask)
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0
