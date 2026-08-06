"""Compare model checkpoints on a shared validation set (SRS §10 success criteria)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compare_models(checkpoint_paths: list[str | Path], val_data_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Evaluate each checkpoint on the validation set and return per-model metrics."""
    raise NotImplementedError("Requires trained checkpoints and an annotated validation set")
