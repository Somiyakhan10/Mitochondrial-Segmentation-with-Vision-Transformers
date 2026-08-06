from __future__ import annotations

from pathlib import Path

import pytest

from mitomorph.config.loader import load_config, merge_overrides, validate_config
from mitomorph.exceptions import ConfigError

MINIMAL_CONFIG = """
version: "0.1.0"
preprocessing:
  normalization: zscore
segmentation:
  architecture: unet_resnet34
celltype:
  neuronal_markers: [NeuN]
classification:
  model: random_forest
paths:
  raw_data_dir: data/raw
"""

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_config_success(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_CONFIG)
    config = load_config(path)
    assert config["version"] == "0.1.0"
    assert "_meta" in config
    assert len(config["_meta"]["config_hash"]) == 12


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_missing_required_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("version: '0.1.0'\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_validate_config_raises_on_missing_keys():
    with pytest.raises(ConfigError):
        validate_config({"version": "0.1.0"})


def test_merge_overrides_dotted_keys():
    config = {"segmentation": {"confidence_threshold": 0.5}}
    merged = merge_overrides(config, {"segmentation.confidence_threshold": 0.7})
    assert merged["segmentation"]["confidence_threshold"] == 0.7
    assert config["segmentation"]["confidence_threshold"] == 0.5


def test_merge_overrides_creates_missing_nodes():
    merged = merge_overrides({}, {"a.b.c": 42})
    assert merged["a"]["b"]["c"] == 42


def test_load_real_default_config():
    config = load_config(REPO_ROOT / "config" / "default_config.yaml")
    assert config["validation"]["target_dice"] == 0.85
    assert config["validation"]["target_accuracy"] == 0.80
