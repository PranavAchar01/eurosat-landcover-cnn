"""Compact CNN for 64x64 RGB land-cover patch classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class LandCoverCNN(nn.Module):
    """A small VGG-style CNN: 3 conv blocks + a classifier head.

    Designed to train quickly on CPU while remaining a genuine convolutional
    classifier (not an MLP). ~0.5M parameters at the default width.
    """

    def __init__(self, num_classes: int = 10, in_ch: int = 3, width: int = 32):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, width),         # 64 -> 32
            block(width, width * 2),     # 32 -> 16
            block(width * 2, width * 4), # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x).argmax(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
