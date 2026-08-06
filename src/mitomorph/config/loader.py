"""Configuration loading, validation, and CLI-override merging (FR-42).

FR-42 requires version tracking for analysis parameters. Every config
loaded through :func:`load_config` is stamped with a content hash
(``_meta.config_hash``) so a given analysis run can be traced back to the
exact parameter set that produced it.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from mitomorph.exceptions import ConfigError
from mitomorph.logger import get_logger

logger = get_logger(__name__)

REQUIRED_TOP_LEVEL_KEYS = (
    "version",
    "preprocessing",
    "segmentation",
    "celltype",
    "classification",
    "paths",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, validate it, and stamp it with a content hash.

    Raises:
        ConfigError: if the file is missing, unparsable, or fails validation.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        config = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse config file {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"Config file {config_path} must contain a top-level mapping")

    validate_config(config)

    config["_meta"] = {
        "source_path": str(config_path),
        "config_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12],
    }
    logger.info("Loaded config from %s (hash=%s)", config_path, config["_meta"]["config_hash"])
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate that required top-level sections are present.

    Raises:
        ConfigError: if any required key is missing.
    """
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ConfigError(f"Config is missing required section(s): {', '.join(missing)}")


def merge_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge CLI-supplied overrides on top of a loaded config.

    ``overrides`` uses dotted keys, e.g. ``{"segmentation.confidence_threshold": 0.7}``.
    Returns a new dict; ``config`` is not mutated.
    """
    merged = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = merged
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return merged
