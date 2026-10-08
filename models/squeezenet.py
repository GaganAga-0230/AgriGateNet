"""
squeezenet.py
─────────────
Model 4 — SqueezeNet1.1 (Compression Baseline).

SqueezeNet uses "Fire modules" (squeeze + expand) to achieve
AlexNet-level accuracy at 50× fewer parameters and <0.5 MB model size.
SqueezeNet1.1 is a reduced-computation update over the original.

Key stats (ImageNet-pretrained):
    Parameters  : ~1.24 M
    FLOPs       : ~349 MFLOPs @ 224×224
    Model size  : ~4.8 MB
    Latency     : ~15 ms (CPU, batch=1)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_squeezenet(
    num_classes: int = 9,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Build a fine-tunable SqueezeNet1.1 model.

    Parameters
    ----------
    num_classes     : number of output classes
    pretrained      : load ImageNet-pretrained weights
    freeze_backbone : freeze all layers except the classifier head
    dropout         : dropout probability in the custom classifier
    """
    weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.squeezenet1_1(weights=weights)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # SqueezeNet's classifier is a convolutional head, not a Linear layer.
    # We replace the final conv to change the number of output channels.
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Conv2d(512, num_classes, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    model.num_classes = num_classes

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    from torchinfo import summary
    model = build_squeezenet(num_classes=9, pretrained=False)
    print("=" * 60)
    print("Model 4 — SqueezeNet1.1")
    print("=" * 60)
    summary(model, input_size=(1, 3, 224, 224), device="cpu")
