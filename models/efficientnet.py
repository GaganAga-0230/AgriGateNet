"""
efficientnet.py
───────────────
Model 2 — EfficientNet-B0 (Transfer Learning Baseline).

Uses torchvision's pretrained EfficientNet-B0 with a custom
classification head tuned for DeepWeeds' 9 weed classes.

EfficientNet-B0 applies compound scaling of depth, width, and resolution
— providing the best accuracy-per-parameter ratio among non-novel baselines.

Key stats (ImageNet-pretrained):
    Parameters  : ~5.29 M
    FLOPs       : ~390 MFLOPs @ 224×224
    Latency     : ~18 ms (CPU, batch=1)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_efficientnet(
    num_classes: int = 9,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Build a fine-tunable EfficientNet-B0 model.

    Parameters
    ----------
    num_classes     : number of output classes
    pretrained      : load ImageNet-pretrained weights
    freeze_backbone : freeze all layers except the classifier head
    dropout         : dropout rate in the custom classification head
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier
    # Original head: Dropout → Linear(1280 → 1000)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    # Always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    from torchinfo import summary
    model = build_efficientnet(num_classes=9, pretrained=False)
    print("=" * 60)
    print("Model 2 — EfficientNet-B0")
    print("=" * 60)
    summary(model, input_size=(1, 3, 224, 224), device="cpu")
