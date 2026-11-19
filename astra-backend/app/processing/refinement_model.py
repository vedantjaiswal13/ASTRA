"""
ASTRA Neural Optical Flow Refinement Model
==========================================
File: app/processing/refinement_model.py

Purpose:
--------
Defines a compact U-Net convolutional neural network designed to take a
motion-compensated (RAFT-warped) intermediate satellite patch as input
and output an artifact-corrected, visually sharp refined patch.

Mathematical Formulation & Architecture:
----------------------------------------
Optical flow warping models physical advection:
    I_warped(x) = I_0(x - t * v(x))
However, optical flow estimates often suffer from:
1. Occlusion boundaries and disocclusions where motion vectors are ambiguous.
2. Fast-evolving convective clouds where cloud tops cool/expand non-rigidly.
3. Sub-pixel bilinear interpolation blurring.

The PatchRefinementUNet learns a residual correction field:
    I_refined(x) = clamp(I_warped(x) + R(I_warped(x)), 0, 1)
where R(.) is the compact U-Net operator.

DATASET SPECIFICATION & RESEARCH PROTOTYPE NOTE:
------------------------------------------------
- Current Dataset Size: 2 triplets (2024-01-01_18, 2024-01-15_00),
  16 patches total of size 256x256 (8 train patches / 8 validation patches).
- NOTE: This is a small-scale research prototype demonstrating the architectural
  pipeline (RAFT motion warping + deep residual neural refinement), not a
  production-scale operational weather model.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> LeakyReLU) * 2 block with residual option."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PatchRefinementUNet(nn.Module):
    """
    Compact 4-level U-Net with skip connections and residual output head.
    Takes a RAFT-warped grayscale patch [B, 1, H, W] in range [0, 1]
    and outputs a refined patch [B, 1, H, W] of the same size.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_features, base_features * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_features * 2, base_features * 4)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_features * 4, base_features * 8)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)

        self.up3 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features * 2, base_features)

        # Output projection head (predicts residual correction map)
        self.out_head = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual learning:
        Refined = clamp(x[:, :1] + residual, 0.0, 1.0)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        b = self.bottleneck(x3)

        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.dec3(d3)

        # Residual correction
        residual = self.out_head(d3)
        refined = torch.clamp(x[:, :1] + residual, 0.0, 1.0)
        return refined


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss for 2D images.
    Returns 1.0 - SSIM (range: 0 to 2, where 0 is perfect structural match).
    """

    def __init__(self, window_size: int = 11, in_channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels

        # Construct 2D normalized Gaussian window
        sigma = 1.5
        coords = np.arange(window_size) - (window_size - 1) / 2.0
        gauss = np.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / np.sum(gauss)
        _1d = torch.from_numpy(gauss).float().unsqueeze(1)
        _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
        self.register_buffer(
            "window",
            _2d.expand(in_channels, 1, window_size, window_size).contiguous()
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        window = self.window.to(dtype=img1.dtype, device=img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.in_channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.in_channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.in_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.in_channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.in_channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()


class CombinedL1SSIMLoss(nn.Module):
    """
    Weighted combination of photometric L1 loss and perceptual SSIM loss:
        Loss = alpha * L1 + (1 - alpha) * (1 - SSIM)
    """

    def __init__(self, alpha: float = 0.8, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=window_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1_val = self.l1(pred, target)
        ssim_val = self.ssim(pred, target)
        total = self.alpha * l1_val + (1.0 - self.alpha) * ssim_val
        return total, l1_val, ssim_val
