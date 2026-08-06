from __future__ import annotations

import numpy as np
import pytest

from mitomorph.segmentation.metrics import dice_score, iou_score


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
