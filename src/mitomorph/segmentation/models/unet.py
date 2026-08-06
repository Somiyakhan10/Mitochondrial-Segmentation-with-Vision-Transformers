"""U-Net with ResNet34 encoder for mitochondrial segmentation (FR-08).

Trained on the EPFL/Lucchi hippocampus EM mitochondria dataset (Val Dice
0.803, Val IoU 0.726 on the held-out test volume) — see
``data/models/segmentation_unet.pt`` and
:func:`mitomorph.segmentation.infer.load_model`. This is an EM-trained
model; running it on fluorescence microscopy images (the pipeline's
primary input per FR-01/FR-02) is a domain-mismatched demonstration of
the inference path, not a validated mitochondria detector for that
modality — see docs/user_guide.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetResNet34(nn.Module):
    """U-Net segmentation model with a ResNet34 encoder (FR-08, FR-10, FR-11).

    Downsamples by a total stride of 32 (5 encoder stages); inputs must
    have height/width divisible by 32 — see
    :func:`mitomorph.segmentation.infer.segment`, which pads/crops
    automatically.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, pretrained: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrained = pretrained

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2, 64ch
        self.pool0 = resnet.maxpool  # /4
        self.encoder1 = resnet.layer1  # /4, 64ch
        self.encoder2 = resnet.layer2  # /8, 128ch
        self.encoder3 = resnet.layer3  # /16, 256ch
        self.encoder4 = resnet.layer4  # /32, 512ch

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder4 = DecoderBlock(512 + 256, 256)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        self.decoder2 = DecoderBlock(128 + 64, 64)
        self.decoder1 = DecoderBlock(64 + 64, 32)
        self.decoder0 = DecoderBlock(32, 16)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.encoder0(x)
        p0 = self.pool0(e0)
        e1 = self.encoder1(p0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(torch.cat([self.up(e4), e3], dim=1))
        d3 = self.decoder3(torch.cat([self.up(d4), e2], dim=1))
        d2 = self.decoder2(torch.cat([self.up(d3), e1], dim=1))
        d1 = self.decoder1(torch.cat([self.up(d2), e0], dim=1))
        d0 = self.decoder0(self.up(d1))
        return self.final(d0)
