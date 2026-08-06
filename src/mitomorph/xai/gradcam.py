"""Grad-CAM heatmaps for segmentation/classification model decisions (FR-31)."""

from __future__ import annotations

import numpy as np
import torch.nn as nn


def generate_gradcam_heatmap(model: nn.Module, image: np.ndarray, target_layer: str) -> np.ndarray:
    """Generate a Grad-CAM heatmap highlighting regions most important to the model's prediction.

    Intended to use Captum's ``LayerGradCam`` once a trained model and its
    target layer name are available.
    """
    raise NotImplementedError("Requires a trained model to compute gradients against (FR-31)")
