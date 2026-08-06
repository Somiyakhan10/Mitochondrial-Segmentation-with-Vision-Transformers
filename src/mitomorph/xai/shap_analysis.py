"""SHAP-based feature importance for morphometric classification (FR-32, FR-33)."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_shap_values(model: Any, features: np.ndarray) -> np.ndarray:
    """Compute SHAP values for each morphometric feature's contribution to a classification.

    Args:
        model: a trained scikit-learn classifier (e.g. from
            :func:`mitomorph.classification.health_classifier.build_classifier_pipeline`).
        features: feature matrix, shape ``(n_samples, n_features)``.
    """
    raise NotImplementedError("Requires a trained classifier (FR-32)")


def feature_importance_summary(shap_values: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Summarize SHAP values into a mean-absolute-importance ranking per feature (FR-33)."""
    raise NotImplementedError("Requires computed SHAP values (FR-33)")
