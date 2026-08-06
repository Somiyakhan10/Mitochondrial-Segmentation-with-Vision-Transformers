from __future__ import annotations

import numpy as np
import pytest
import tifffile

from mitomorph.exceptions import ImageLoadError, ImageValidationError
from mitomorph.preprocessing import channel_utils, normalization, zstack
from mitomorph.preprocessing.io import MicroscopyImage, load_image
from mitomorph.preprocessing.validators import validate_image, validate_metadata


def test_load_image_tiff(tmp_path):
    data = (np.random.rand(2, 32, 32) * 255).astype(np.uint8)
    path = tmp_path / "sample.tif"
    tifffile.imwrite(str(path), data, metadata={"axes": "CYX"})

    image = load_image(path)
    assert image.data.shape == (2, 32, 32)
    assert image.axes == "CYX"
    assert image.n_channels == 2


def test_load_image_missing_file(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "does_not_exist.tif")


def test_load_image_unsupported_extension(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"not a real image")
    with pytest.raises(ImageLoadError):
        load_image(path)


def test_zscore_normalize():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    normalized = normalization.zscore_normalize(arr)
    assert abs(normalized.mean()) < 1e-8
    assert abs(normalized.std() - 1.0) < 1e-6


def test_percentile_normalize_range():
    arr = np.array([0, 10, 20, 30, 100], dtype=float)
    normalized = normalization.percentile_normalize(arr, low=0, high=100)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_max_intensity_projection():
    zstack_data = np.array([[[1, 2], [3, 4]], [[5, 0], [2, 9]]])
    projected = zstack.max_intensity_projection(zstack_data, z_axis=0)
    np.testing.assert_array_equal(projected, np.array([[5, 2], [3, 9]]))


def test_focus_stack_shape():
    zstack_data = np.random.rand(3, 16, 16)
    result = zstack.focus_stack(zstack_data, z_axis=0)
    assert result.shape == (16, 16)


def test_identify_channel_type():
    assert channel_utils.identify_channel_type("Tom20-AF488") == "mitochondrial"
    assert channel_utils.identify_channel_type("NeuN") == "neuronal"
    assert channel_utils.identify_channel_type("DAPI") == "unknown"


def test_extract_mitochondrial_and_neuronal_channels():
    data = np.stack([np.full((8, 8), 1), np.full((8, 8), 2)])
    image = MicroscopyImage(data=data, channel_names=["Tom20", "NeuN"], axes="CYX")

    assert (channel_utils.extract_mitochondrial_channel(image) == 1).all()
    assert (channel_utils.extract_neuronal_channel(image) == 2).all()


def test_extract_channel_raises_when_missing():
    data = np.stack([np.zeros((8, 8)), np.zeros((8, 8))])
    image = MicroscopyImage(data=data, channel_names=["DAPI", "GFP"], axes="CYX")
    with pytest.raises(ImageValidationError):
        channel_utils.extract_mitochondrial_channel(image)


def test_validate_image_channel_count():
    image = MicroscopyImage(data=np.zeros((1, 32, 32)), channel_names=["channel_0"], axes="CYX")
    with pytest.raises(ImageValidationError):
        validate_image(image, min_channels=2)


def test_validate_image_passes_with_enough_channels():
    image = MicroscopyImage(data=np.ones((2, 32, 32)), channel_names=["Tom20", "NeuN"], axes="CYX")
    validate_image(image, min_channels=2)  # should not raise


def test_validate_metadata_missing_fields():
    with pytest.raises(ImageValidationError):
        validate_metadata({"animal_id": "M1"})


def test_validate_metadata_ok():
    validate_metadata({"experimental_condition": "SCI", "time_point": "6 weeks", "animal_id": "M1"})
