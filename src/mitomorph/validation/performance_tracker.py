"""MLflow-backed experiment tracking against SRS accuracy targets (NFR-05, NFR-06)."""

from __future__ import annotations

from typing import Any

TARGET_DICE = 0.85  # NFR-05: segmentation Dice score >= 0.85
TARGET_ACCURACY = 0.80  # NFR-06: classification accuracy >= 0.80


def log_run(run_name: str, params: dict[str, Any], metrics: dict[str, float]) -> None:
    """Log a training/evaluation run's parameters and metrics to MLflow.

    Wraps ``mlflow.start_run()`` / ``log_param`` / ``log_metric``; deferred
    until a tracking URI is configured for the lab's environment.
    """
    raise NotImplementedError(
        "Wire up a configured MLflow tracking URI before logging real runs (NFR-05, NFR-06)"
    )


def meets_targets(metrics: dict[str, float]) -> bool:
    """Check whether a run's metrics meet the SRS success criteria (Dice >= 0.85, accuracy >= 0.80)."""
    return metrics.get("dice", 0.0) >= TARGET_DICE and metrics.get("accuracy", 0.0) >= TARGET_ACCURACY
