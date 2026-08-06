"""Model checkpoint save/load/resume for training runs (NFR-14: reproducible training)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mitomorph.exceptions import MitoMorphError
from mitomorph.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model/optimizer/epoch state to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(state, str(path))
    logger.info("Saved checkpoint to %s (epoch=%d)", path, epoch)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load model/optimizer state from a checkpoint file.

    Returns:
        The full checkpoint dict (including ``epoch`` and ``extra``).
    """
    path = Path(path)
    if not path.exists():
        raise MitoMorphError(f"Checkpoint not found: {path}")
    state = torch.load(str(path), map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch=%d)", path, state.get("epoch", 0))
    return state


def resume_training(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Load a checkpoint and return the epoch training should resume from."""
    state = load_checkpoint(path, model, optimizer)
    return state.get("epoch", 0) + 1
