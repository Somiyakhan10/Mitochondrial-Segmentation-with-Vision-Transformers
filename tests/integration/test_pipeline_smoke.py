"""Integration smoke test: verifies MitoPipeline wiring/order.

Preprocessing, segmentation, and morphometric-stage wiring is real. The
test uses a small, untrained UNetResNet34 (pretrained=False, so no
network access needed) rather than the real trained checkpoint, which
isn't committed to git — this proves the pipeline runs through
preprocessing and segmentation and fails exactly at the cell-type
classification stub, without depending on any external file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mitomorph import pipeline as pipeline_module
from mitomorph.pipeline import MitoPipeline
from mitomorph.preprocessing.io import MicroscopyImage
from mitomorph.segmentation.models.unet import UNetResNet34

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"


def test_pipeline_reaches_celltype_stage(monkeypatch):
    fake_image = MicroscopyImage(
        data=np.random.rand(2, 64, 64).astype(np.float32),
        channel_names=["Tom20", "NeuN"],
        axes="CYX",
    )
    monkeypatch.setattr(pipeline_module, "load_image", lambda path: fake_image)

    untrained_model = UNetResNet34(in_channels=1, out_channels=1, pretrained=False)
    mito_pipeline = MitoPipeline(CONFIG_PATH, segmentation_model=untrained_model)
    metadata = {"animal_id": "M1", "experimental_condition": "SCI", "time_point": "6 weeks"}

    with pytest.raises(NotImplementedError, match="classif"):
        mito_pipeline.run("unused.tif", metadata)


def test_pipeline_config_loaded():
    mito_pipeline = MitoPipeline(CONFIG_PATH)
    assert mito_pipeline.config["version"] == "0.1.0"
