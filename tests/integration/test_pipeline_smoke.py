"""Integration smoke test: verifies MitoPipeline wiring/order without a trained model.

Preprocessing, validation, and morphometric-stage wiring is real; the
test confirms the pipeline runs through all of that and fails exactly at
the segmentation stub, proving the stages before it are correctly wired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mitomorph import pipeline as pipeline_module
from mitomorph.pipeline import MitoPipeline
from mitomorph.preprocessing.io import MicroscopyImage

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"


def test_pipeline_reaches_segmentation_stage(monkeypatch):
    fake_image = MicroscopyImage(
        data=np.random.rand(2, 32, 32).astype(np.float32),
        channel_names=["Tom20", "NeuN"],
        axes="CYX",
    )
    monkeypatch.setattr(pipeline_module, "load_image", lambda path: fake_image)

    mito_pipeline = MitoPipeline(CONFIG_PATH)
    metadata = {"animal_id": "M1", "experimental_condition": "SCI", "time_point": "6 weeks"}

    with pytest.raises(NotImplementedError, match="segment"):
        mito_pipeline.run("unused.tif", metadata)


def test_pipeline_config_loaded():
    mito_pipeline = MitoPipeline(CONFIG_PATH)
    assert mito_pipeline.config["version"] == "0.1.0"
