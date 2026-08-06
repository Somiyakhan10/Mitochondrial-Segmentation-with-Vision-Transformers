"""K-fold cross-validation splitting for segmentation/classification models (NFR-05, NFR-06)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold


def make_folds(n_samples: int, k: int = 5, random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``k`` (train_indices, val_indices) splits over ``n_samples`` items."""
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))
