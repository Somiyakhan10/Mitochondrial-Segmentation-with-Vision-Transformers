from __future__ import annotations

import numpy as np
import pytest

from mitomorph.segmentation.metrics import (
    confusion_matrix,
    dice_score,
    f1_score,
    iou_score,
    precision_score,
    recall_score,
)


def test_iou_perfect_overlap():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    assert iou_score(mask, mask) == 1.0


def test_iou_no_overlap():
    a = np.zeros((10, 10), dtype=bool)
    a[0:2, 0:2] = True
    b = np.zeros((10, 10), dtype=bool)
    b[5:7, 5:7] = True
    assert iou_score(a, b) == 0.0


def test_dice_perfect_overlap():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    assert dice_score(mask, mask) == 1.0


def test_dice_partial_overlap():
    a = np.zeros((4, 4), dtype=bool)
    a[0:2, 0:2] = True
    b = np.zeros((4, 4), dtype=bool)
    b[1:3, 1:3] = True
    assert dice_score(a, b) == pytest.approx(0.25)


def test_iou_and_dice_empty_masks_return_one():
    empty = np.zeros((5, 5), dtype=bool)
    assert iou_score(empty, empty) == 1.0
    assert dice_score(empty, empty) == 1.0


def _pred_and_true():
    # 4x4 grid: pred is top-left 2x2, true is a 1-pixel-shifted 2x2 overlapping in 1 cell.
    pred = np.zeros((4, 4), dtype=bool)
    pred[0:2, 0:2] = True
    true = np.zeros((4, 4), dtype=bool)
    true[1:3, 1:3] = True
    return pred, true


def test_confusion_matrix_values():
    pred, true = _pred_and_true()
    cm = confusion_matrix(pred, true)
    # overlap is exactly the single cell (1,1)
    assert cm["tp"] == 1
    assert cm["fp"] == 3  # predicted-positive cells not in true
    assert cm["fn"] == 3  # true-positive cells not predicted
    assert cm["tn"] == 16 - 1 - 3 - 3


def test_confusion_matrix_perfect_match():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True
    cm = confusion_matrix(mask, mask)
    assert cm["fp"] == 0
    assert cm["fn"] == 0
    assert cm["tp"] == 4
    assert cm["tn"] == 21


def test_precision_recall_f1_partial_overlap():
    pred, true = _pred_and_true()
    assert precision_score(pred, true) == pytest.approx(1 / 4)
    assert recall_score(pred, true) == pytest.approx(1 / 4)
    assert f1_score(pred, true) == pytest.approx(1 / 4)


def test_precision_recall_f1_perfect_match():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True
    assert precision_score(mask, mask) == 1.0
    assert recall_score(mask, mask) == 1.0
    assert f1_score(mask, mask) == 1.0


def test_precision_zero_when_no_predictions():
    empty = np.zeros((5, 5), dtype=bool)
    true = np.zeros((5, 5), dtype=bool)
    true[0:2, 0:2] = True
    assert precision_score(empty, true) == 0.0
    assert recall_score(empty, true) == 0.0
    assert f1_score(empty, true) == 0.0
