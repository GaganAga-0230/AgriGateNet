"""
baseline_cnn.py
───────────────
Model 0 — Basic CNN Baseline.

Reproduces the style of the WeedWatch reference repository
(simple stacked Conv → Pool → FC layers, no attention, no pretrained weights).

This acts as the "floor" benchmark to demonstrate why a novel architecture
is necessary.

Architecture:
    Conv(3,32,3) → BN → ReLU → MaxPool(2)
    Conv(32,64,3) → BN → ReLU → MaxPool(2)
    Conv(64,128,3) → BN → ReLU → MaxPool(2)
    Conv(128,256,3) → BN → ReLU → AdaptiveAvgPool
    Flatten → Dropout(0.5) → FC(256, num_classes)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Simple 4-block CNN baseline — Model 0.

    Parameters
    ----------
    num_classes : int — number of weed species to classify (default 9)
    dropout     : float — dropout probability before the classifier
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  3, pool=True),   # 224 → 112
            ConvBlock(32,  64,  3, pool=True),   # 112 → 56
            ConvBlock(64,  128, 3, pool=True),   # 56  → 28
            ConvBlock(128, 256, 3, pool=True),   # 28  → 14
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → 1×1
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 256, 14, 14)
        x = self.pool(x)               # (B, 256, 1, 1)
        x = torch.flatten(x, 1)        # (B, 256)
        return self.classifier(x)      # (B, num_classes)


def build_baseline_cnn(num_classes: int = 9) -> BaselineCNN:
    """Factory function — returns an untrained BaselineCNN."""
    return BaselineCNN(num_classes=num_classes)


if __name__ == "__main__":
    from torchinfo import summary
    model = build_baseline_cnn(num_classes=9)
    print("=" * 60)
    print("Model 0 — Baseline CNN")
    print("=" * 60)
    summary(model, input_size=(1, 3, 224, 224), device="cpu")
