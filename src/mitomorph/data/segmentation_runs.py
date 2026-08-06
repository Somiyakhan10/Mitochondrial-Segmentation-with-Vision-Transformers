"""Storage helpers for segmentation preview runs.

Persists the image + labeled mask for a single dashboard analysis run
to disk (as a compressed .npz), so the Results and Mask Correction tabs
can reload and re-render them without recomputing segmentation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_run_artifacts(
    run_dir: str | Path, run_uuid: str, image: np.ndarray, labeled_mask: np.ndarray
) -> Path:
    """Save an image + labeled mask pair to ``<run_dir>/<run_uuid>.npz``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{run_uuid}.npz"
    np.savez_compressed(path, image=image, labeled_mask=labeled_mask)
    return path


def load_run_artifacts(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an ``(image, labeled_mask)`` pair saved by :func:`save_run_artifacts`."""
    data = np.load(path)
    return data["image"], data["labeled_mask"]
