"""U-Net model for multi-class Kuro Siwo segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolution layers used by each U-Net stage."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class UNet(nn.Module):
    """U-Net with logits for each segmentation class."""

    def __init__(self, in_channels, num_classes, base_channels=32):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.decoder4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.decoder3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    @staticmethod
    def _decode(upsampled, skip, block):
        # Odd patch dimensions can leave the upsampled tensor one pixel off.
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return block(torch.cat((skip, upsampled), dim=1))

    def forward(self, inputs):
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))
        features = self.bottleneck(self.pool(skip4))

        features = self._decode(self.up4(features), skip4, self.decoder4)
        features = self._decode(self.up3(features), skip3, self.decoder3)
        features = self._decode(self.up2(features), skip2, self.decoder2)
        features = self._decode(self.up1(features), skip1, self.decoder1)
        return self.classifier(features)