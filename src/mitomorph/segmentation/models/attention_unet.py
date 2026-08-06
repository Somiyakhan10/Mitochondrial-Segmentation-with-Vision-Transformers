"""Attention U-Net for segmenting overlapping/dense mitochondrial networks (FR-09).

Stub: attention gate configuration is deferred until real images with
overlapping mitochondria are available to validate against.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionUNet(nn.Module):
    """Attention U-Net: improves segmentation of overlapping/densely packed
    mitochondria compared to a plain U-Net (FR-09).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AttentionUNet forward pass pending architecture selection (FR-09)")
