"""
mobilenetv3.py
──────────────
Model 1 — MobileNetV3-Small (Transfer Learning Baseline).

Uses torchvision's pretrained MobileNetV3-Small with a custom
classification head tuned for DeepWeeds' 9 weed classes.

Key stats (ImageNet-pretrained):
    Parameters  : ~2.54 M
    FLOPs       : ~56.5 MFLOPs @ 224×224
    Latency     : ~8 ms (CPU, batch=1)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_mobilenetv3(
    num_classes: int = 9,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a fine-tunable MobileNetV3-Small model.

    Parameters
    ----------
    num_classes     : number of output classes
    pretrained      : load ImageNet-pretrained weights
    freeze_backbone : if True, freeze all layers except the classifier
                      (useful for feature-extraction-only fine-tuning)
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head
    # Original: Linear(576 → 1024) → HardSwish → Dropout → Linear(1024 → 1000)
    in_features = model.classifier[3].in_features  # 1024
    model.classifier[3] = nn.Linear(in_features, num_classes)

    # Always ensure the new head is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    from torchinfo import summary
    model = build_mobilenetv3(num_classes=9, pretrained=False)
    print("=" * 60)
    print("Model 1 — MobileNetV3-Small")
    print("=" * 60)
    summary(model, input_size=(1, 3, 224, 224), device="cpu")
