from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mitomorph.reporting.figures import (
    plot_condition_comparison,
    plot_confusion_matrix,
    plot_feature_correlation,
    plot_feature_histogram,
    plot_segmentation_overlay,
)


def test_plot_segmentation_overlay_returns_figure():
    image = np.random.rand(32, 32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 10:20] = True
    fig = plot_segmentation_overlay(image, mask)
    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_segmentation_overlay_saves_file(tmp_path):
    image = np.random.rand(16, 16)
    mask = np.zeros((16, 16), dtype=bool)
    path = tmp_path / "overlay.png"
    plot_segmentation_overlay(image, mask, save_path=path)
    assert path.exists()


def _comparison_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": ["Naive", "Naive", "SCI", "SCI", "SCI"],
            "fragmentation_index": [0.01, 0.012, 0.03, 0.028, 0.031],
        }
    )


def test_plot_condition_comparison_box():
    fig = plot_condition_comparison(_comparison_df(), "fragmentation_index", kind="box")
    assert len(fig.axes) == 1


def test_plot_condition_comparison_bar():
    fig = plot_condition_comparison(_comparison_df(), "fragmentation_index", kind="bar")
    assert len(fig.axes) == 1


def test_plot_condition_comparison_missing_column_raises():
    with pytest.raises(ValueError):
        plot_condition_comparison(_comparison_df(), "does_not_exist", kind="box")


def test_plot_condition_comparison_invalid_kind_raises():
    with pytest.raises(ValueError):
        plot_condition_comparison(_comparison_df(), "fragmentation_index", kind="pie")


def test_plot_feature_correlation_returns_figure():
    data = pd.DataFrame({"area": [1.0, 2.0, 3.0], "circularity": [0.5, 0.6, 0.7]})
    fig = plot_feature_correlation(data, "area", "circularity")
    assert len(fig.axes) == 1


def test_plot_feature_correlation_missing_column_raises():
    data = pd.DataFrame({"area": [1.0, 2.0]})
    with pytest.raises(ValueError):
        plot_feature_correlation(data, "area", "circularity")


def test_plot_feature_histogram_returns_figure():
    fig = plot_feature_histogram([1.0, 2.0, 2.5, 3.0, 3.2, 5.0], xlabel="area")
    assert len(fig.axes) == 1


def test_plot_feature_histogram_saves_file(tmp_path):
    path = tmp_path / "hist.png"
    plot_feature_histogram([1.0, 2.0, 3.0], xlabel="area", save_path=path)
    assert path.exists()


def test_plot_confusion_matrix_returns_figure():
    fig = plot_confusion_matrix({"tp": 10, "fp": 2, "fn": 3, "tn": 100})
    assert len(fig.axes) == 2  # heatmap axis + colorbar axis


def test_plot_confusion_matrix_saves_file(tmp_path):
    path = tmp_path / "cm.png"
    plot_confusion_matrix({"tp": 5, "fp": 1, "fn": 1, "tn": 20}, save_path=path)
    assert path.exists()
