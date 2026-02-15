"""
Model utilities for the Transfer Learning MVP.
Provides a frozen ResNet18 feature extractor with a trainable classification head.
"""

import torch.nn as nn
import torchvision.models as models


def create_feature_extractor(num_classes, weights="IMAGENET1K_V1"):
    """
    Create a ResNet18-based feature extractor: convolutional base frozen,
    new linear classification head trainable.

    Args:
        num_classes: Number of target classes (e.g., 2 for ants vs bees).
        weights: Pre-trained weights identifier (default ImageNet).

    Returns:
        model_conv: PyTorch model with frozen backbone and new fc layer.
    """
    model_conv = models.resnet18(weights=weights)

    # Freeze all parameters in the convolutional base
    for param in model_conv.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer with a new head for num_classes
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)

    return model_conv
