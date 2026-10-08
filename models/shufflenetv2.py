"""
shufflenetv2.py
───────────────
Model 3 — ShuffleNetV2-0.5x (Transfer Learning Baseline).

ShuffleNetV2 uses channel splitting and channel shuffling to enable
parallel branches with minimal cross-channel computation cost.
The 0.5x variant is the most extreme in size/speed trade-off.

Key stats (ImageNet-pretrained):
    Parameters  : ~1.37 M
    FLOPs       : ~41 MFLOPs @ 224×224
    Latency     : ~6 ms (CPU, batch=1)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_shufflenetv2(
    num_classes: int = 9,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a fine-tunable ShuffleNetV2 x0.5 model.

    Parameters
    ----------
    num_classes     : number of output classes
    pretrained      : load ImageNet-pretrained weights
    freeze_backbone : freeze all layers except the classifier head
    """
    weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.shufflenet_v2_x0_5(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer
    in_features = model.fc.in_features  # 1024
    model.fc = nn.Linear(in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    from torchinfo import summary
    model = build_shufflenetv2(num_classes=9, pretrained=False)
    print("=" * 60)
    print("Model 3 — ShuffleNetV2 x0.5")
    print("=" * 60)
    summary(model, input_size=(1, 3, 224, 224), device="cpu")
