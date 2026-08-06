from __future__ import annotations

import numpy as np
import pytest

from mitomorph.exceptions import SegmentationError
from mitomorph.segmentation.checkpoint import save_checkpoint
from mitomorph.segmentation.infer import load_model, segment
from mitomorph.segmentation.models.unet import UNetResNet34


@pytest.fixture(scope="module")
def tiny_model() -> UNetResNet34:
    model = UNetResNet34(in_channels=1, out_channels=1, pretrained=False)
    model.eval()
    return model


def test_segment_output_shape_matches_input(tiny_model):
    image = np.random.rand(64, 64).astype(np.float32)
    result = segment(tiny_model, image)
    assert result.mask.shape == (64, 64)
    assert result.confidence.shape == (64, 64)
    assert result.mask.dtype == bool


def test_segment_handles_size_not_a_multiple_of_32(tiny_model):
    image = np.random.rand(50, 70).astype(np.float32)
    result = segment(tiny_model, image)
    assert result.mask.shape == (50, 70)
    assert result.confidence.shape == (50, 70)


def test_segment_rejects_multichannel_input(tiny_model):
    image = np.random.rand(2, 64, 64).astype(np.float32)
    with pytest.raises(SegmentationError):
        segment(tiny_model, image)


def test_segment_requires_a_model():
    image = np.random.rand(64, 64).astype(np.float32)
    with pytest.raises(SegmentationError):
        segment(None, image)


def test_load_model_roundtrip(tmp_path, tiny_model):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, tiny_model, epoch=1)

    loaded = load_model(path)
    result = segment(loaded, np.random.rand(64, 64).astype(np.float32))
    assert result.mask.shape == (64, 64)
