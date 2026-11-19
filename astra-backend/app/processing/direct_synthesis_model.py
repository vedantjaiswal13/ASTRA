"""
ASTRA Direct Synthesis Generative Model
======================================
File: app/processing/direct_synthesis_model.py

# This is a lightweight proof-of-concept direct-synthesis model, explicitly
# not the full diffusion architecture described in the project proposal,
# trained on a small dataset.

Mathematical Formulation:
-------------------------
Unlike motion-compensated warping pipelines which remap pixels along flow trajectories:
    I_warp(x) = I_0(x - 0.5 * flow(x))
this conditional U-Net takes a 4-channel tensor:
    X = concat([I_0, I_1, flow_u, flow_v], dim=channel)
and directly synthesizes the intermediate frame at t=0.5:
    I_synth(x) = DirectSynthesisUNet(X)

Direct synthesis generates intermediate cloud states without spatial remapping (cv2.remap),
learning non-rigid cloud formation, convective growth, and boundary evolution.
"""

import torch
import torch.nn as nn
from app.processing.refinement_model import DoubleConv


class DirectSynthesisUNet(nn.Module):
    """
    Compact conditional U-Net for direct intermediate frame synthesis.
    Input: [B, 4, H, W] containing:
        channel 0: T0 frame in [0.0, 1.0]
        channel 1: T1 frame in [0.0, 1.0]
        channel 2: Farneback flow horizontal displacement (u) normalized
        channel 3: Farneback flow vertical displacement (v) normalized
    Output: [B, 1, H, W] directly synthesized frame in [0.0, 1.0].
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_features: int = 32
    ):
        super().__init__()
        # DirectSynthesisUNet: lightweight proof-of-concept direct-synthesis model,
        # explicitly not the full diffusion architecture described in the project proposal,
        # trained on a small dataset.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features

        # Contracting Path (Encoder)
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

        # Expanding Path (Decoder with Skip Connections)
        self.up1 = nn.ConvTranspose2d(
            base_features * 8, base_features * 4, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(
            base_features * 4, base_features * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)

        self.up3 = nn.ConvTranspose2d(
            base_features * 2, base_features, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_features * 2, base_features)

        # Output Projection Head: Direct Synthesis
        self.out_head = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing directly synthesized frame.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        b = self.bottleneck(x3)

        d1 = self.up1(b)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.dec3(d3)

        out = self.out_head(d3)
        return torch.clamp(out, 0.0, 1.0)
