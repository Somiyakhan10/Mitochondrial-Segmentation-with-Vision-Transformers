from __future__ import annotations

import numpy as np
from skimage.draw import disk

from mitomorph.morphometrics.dysfunction_indices import (
    fragmentation_index,
    mitochondrial_density,
    swelling_score,
)
from mitomorph.morphometrics.quality_control import validate_features
from mitomorph.morphometrics.single_features import extract_single_features


def _make_two_square_mask() -> np.ndarray:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True
    mask[10:13, 10:15] = True
    return mask


def _make_circular_mask() -> np.ndarray:
    mask = np.zeros((40, 40), dtype=bool)
    rr, cc = disk((20, 20), 10, shape=mask.shape)
    mask[rr, cc] = True
    return mask


def test_extract_single_features_counts_regions():
    features = extract_single_features(_make_two_square_mask())
    assert len(features) == 2
    for f in features:
        assert f.area > 0
        assert f.perimeter > 0


def test_fragmentation_index():
    features = extract_single_features(_make_two_square_mask())
    assert fragmentation_index(features, total_area=400.0) == len(features) / 400.0


def test_fragmentation_index_zero_area():
    features = extract_single_features(_make_two_square_mask())
    assert fragmentation_index(features, total_area=0.0) == 0.0


def test_mitochondrial_density():
    features = extract_single_features(_make_two_square_mask())
    expected = sum(f.area for f in features) / 400.0
    assert abs(mitochondrial_density(features, cell_area=400.0) - expected) < 1e-9


def test_swelling_score_zero_at_baseline():
    features = extract_single_features(_make_two_square_mask())
    baseline = float(np.mean([f.area / f.perimeter for f in features]))
    assert abs(swelling_score(features, healthy_baseline_ratio=baseline)) < 1e-9


def test_validate_features_flags_low_area():
    features = extract_single_features(_make_two_square_mask())
    results = validate_features(features, min_area=1000.0)
    assert all(not r.is_valid for r in results)


def test_validate_features_passes_normal_ranges():
    features = extract_single_features(_make_circular_mask())
    results = validate_features(features, min_area=1.0)
    assert all(r.is_valid for r in results)
