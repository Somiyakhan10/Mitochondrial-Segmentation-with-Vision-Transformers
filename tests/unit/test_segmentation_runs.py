from __future__ import annotations

import numpy as np

from mitomorph.data.segmentation_runs import load_run_artifacts, save_run_artifacts


def test_save_and_load_run_artifacts_roundtrip(tmp_path):
    image = np.random.rand(32, 32).astype(np.float32)
    labeled_mask = np.zeros((32, 32), dtype=np.int32)
    labeled_mask[2:6, 2:6] = 1
    labeled_mask[10:14, 10:14] = 2

    path = save_run_artifacts(tmp_path, "run-abc123", image, labeled_mask)
    assert path.name == "run-abc123.npz"
    assert path.exists()

    loaded_image, loaded_mask = load_run_artifacts(path)
    np.testing.assert_array_equal(loaded_image, image)
    np.testing.assert_array_equal(loaded_mask, labeled_mask)


def test_save_run_artifacts_creates_directory(tmp_path):
    run_dir = tmp_path / "nested" / "runs"
    image = np.zeros((8, 8), dtype=np.float32)
    labeled_mask = np.zeros((8, 8), dtype=np.int32)

    path = save_run_artifacts(run_dir, "run-1", image, labeled_mask)
    assert path.exists()
    assert run_dir.exists()
