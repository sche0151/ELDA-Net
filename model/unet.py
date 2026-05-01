"""
ELDA-Net Lightweight U-Net Architecture
========================================
Thesis Chapter 3, Section 3.6 and Figure 3.1.

Architecture:
  - 4 encoder levels + bottleneck + 4 decoder levels
  - All standard 3x3 convolutions replaced by depth-wise separable equivalents
    (thesis Section 3.6.1) -- reduces params from 31M -> 2.1M
  - Skip connections concatenate encoder feature maps to decoder at each level
  - Multi-output head (thesis Section 3.6.2):
      (1) binary segmentation mask  [B, 1, H, W]
      (2) scalar confidence score   [B, 1]
      (3) polynomial coefficients   [B, 3]  -- (a, b, c) for x=ay^2+by+c
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWSepConvBlock(nn.Module):
    """Depth-wise separable conv block: DW -> BN -> ReLU -> PW -> BN -> ReLU.
    Achieves ~8-9x reduction in compute vs. standard Conv2d for K=3 (thesis Figure 3.4).
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.dw   = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.bn1  = nn.BatchNorm2d(in_ch)
        self.pw   = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return self.drop(x)


class EncoderBlock(nn.Module):
    """Two DWSepConvBlocks then MaxPool2d(2). Returns (skip_feat, pooled)."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            DWSepConvBlock(in_ch, out_ch, dropout),
            DWSepConvBlock(out_ch, out_ch, dropout),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)


class DecoderBlock(nn.Module):
    """ConvTranspose2d up + skip concat + two DWSepConvBlocks."""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            DWSepConvBlock(out_ch + skip_ch, out_ch, dropout),
            DWSepConvBlock(out_ch, out_ch, dropout),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """
    ELDA-Net Lightweight Depth-wise Separable U-Net (thesis Figure 3.1).

    Channel progression:
      Input  H x W x 3
      Enc1   H    x W    x 64
      Enc2   H/2  x W/2  x 128
      Enc3   H/4  x W/4  x 256
      Enc4   H/8  x W/8  x 512  (bottleneck)

    Three output heads: segmentation logits, confidence score, poly coefficients.
    """

    def __init__(self, in_channels=3, out_channels=1, dropout=0.2):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64,  dropout)
        self.enc2 = EncoderBlock(64,          128, dropout)
        self.enc3 = EncoderBlock(128,         256, dropout)
        self.enc4 = EncoderBlock(256,         512, dropout)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DWSepConvBlock(512, 512, dropout),
            DWSepConvBlock(512, 512, dropout),
        )

        # Decoder
        self.dec4 = DecoderBlock(512, 512, 256, dropout)
        self.dec3 = DecoderBlock(256, 256, 128, dropout)
        self.dec2 = DecoderBlock(128, 128,  64, dropout)
        self.dec1 = DecoderBlock( 64,  64,  32, dropout)

        # Head 1: binary segmentation mask
        self.seg_head = nn.Conv2d(32, out_channels, kernel_size=1)

        # Head 2: confidence score -- GAP -> 2-layer MLP -> sigmoid scalar
        self.conf_gap = nn.AdaptiveAvgPool2d(1)
        self.conf_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1),  nn.Sigmoid(),
        )

        # Head 3: polynomial coefficients (a, b, c) -- GAP -> 3-layer MLP
        self.poly_gap = nn.AdaptiveAvgPool2d(1)
        self.poly_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        seg  = self.seg_head(x)
        conf = self.conf_mlp(self.conf_gap(x))
        poly = self.poly_mlp(self.poly_gap(x))
        return seg, conf, poly
