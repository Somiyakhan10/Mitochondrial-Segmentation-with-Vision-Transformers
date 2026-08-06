"""Mitochondrial health classification: Healthy / Fragmented / Swollen / Dysfunctional (FR-26–FR-28)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HEALTH_CATEGORIES = ("healthy", "fragmented", "swollen", "dysfunctional")


def build_classifier_pipeline(n_estimators: int = 200, random_state: int = 42) -> Pipeline:
    """Construct the (untrained) scikit-learn pipeline: StandardScaler + RandomForestClassifier (FR-27)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
        ]
    )


def train_classifier(pipeline: Pipeline, features: np.ndarray, labels: np.ndarray) -> Pipeline:
    """Fit the classifier on expert-annotated morphometric features."""
    raise NotImplementedError("Training requires expert-annotated mitochondria labels (FR-27)")


def predict_health_score(pipeline: Pipeline, features: np.ndarray) -> np.ndarray:
    """Predict a 0-100 mitochondrial health score from morphometric features (FR-28)."""
    raise NotImplementedError("Prediction requires a trained classifier pipeline (FR-26–FR-28)")
