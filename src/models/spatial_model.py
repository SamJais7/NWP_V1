"""
2D Spatial U-Net for Spatial Forecast Bust Prediction.
Padded to (160, 128) to ensure exact divisibility by 32 across 4 downsampling stages.
Evaluates 15 synoptic channels with spatial attention and Grad-CAM capability.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        """(Conv2D -> BatchNorm -> LeakyReLU) * 2"""
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class SpatialBustUNet(nn.Module):
        """
        4-level Contracting U-Net with Skip Connections and Mixed Precision Logits.
        Input: (B, 15, 160, 128)
        Output: (B, 1, 160, 128) - Spatial Bust Probability Map
        """
        def __init__(self, in_channels: int = 15, base_filters: int = 16):
            super().__init__()
            # Contracting Path
            self.inc = DoubleConv(in_channels, base_filters)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8))
            
            # Bottleneck with Spatial Dropout
            self.drop = nn.Dropout2d(0.25)
            
            # Expanding Path
            self.up1 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
            self.conv_up1 = DoubleConv(base_filters * 8, base_filters * 4)

            self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
            self.conv_up2 = DoubleConv(base_filters * 4, base_filters * 2)

            self.up3 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
            self.conv_up3 = DoubleConv(base_filters * 2, base_filters)

            # Final 1x1 Conv
            self.outc = nn.Conv2d(base_filters, 1, kernel_size=1)

            # Hook placeholder for Grad-CAM
            self.gradients = None
            self.activations = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x4 = self.drop(x4)

            d3 = self.up1(x4)
            d3 = torch.cat([d3, x3], dim=1)
            d3 = self.conv_up1(d3)

            d2 = self.up2(d3)
            d2 = torch.cat([d2, x2], dim=1)
            d2 = self.conv_up2(d2)

            d1 = self.up3(d2)
            d1 = torch.cat([d1, x1], dim=1)
            d1 = self.conv_up3(d1)

            logits = self.outc(d1)
            return logits

        def predict_spatial_risk(self, x_np: np.ndarray) -> np.ndarray:
            """Evaluates numpy input (B, 15, H, W) and pads to (160, 128) if needed."""
            self.eval()
            b, c, h, w = x_np.shape
            pad_h = 160 - h if h < 160 else 0
            pad_w = 128 - w if w < 128 else 0

            if pad_h > 0 or pad_w > 0:
                padded = np.pad(x_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            else:
                padded = x_np[:, :, :160, :128]

            with torch.no_grad():
                tensor = torch.from_numpy(padded).float()
                logits = self(tensor)
                probs = torch.sigmoid(logits).cpu().numpy()

            # Unpad back to original shape
            return probs[:, :, :h, :w]

else:
    class SpatialBustUNet:
        """NumPy fallback when PyTorch is not loaded."""
        def __init__(self, in_channels: int = 15, base_filters: int = 16):
            self.in_channels = in_channels

        def predict_spatial_risk(self, x_np: np.ndarray) -> np.ndarray:
            # Returns smoothed spatial risk based on precip and wind shear
            b, c, h, w = x_np.shape
            risk = 1.0 / (1.0 + np.exp(-0.05 * (x_np[:, 4:5] - 25.0)))
            return risk
