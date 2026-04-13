"""
ResNet-50 Feature Extraction Module.

Extracts deep features from preprocessed fabric images using a pre-trained
ResNet-50 backbone, as described in Section 3.2 of the paper.

Architecture (Table 1):
- 7x7 conv, 64 filters, stride 2 → BN + ReLU → 3x3 max-pool, stride 2
- Residual Block 1: (1x1, 64 → 3x3, 64 → 1x1, 256) x 3
- Transition 1: 1x1 conv + 2x2 avg pool, stride 2
- Residual Block 2: (1x1, 128 → 3x3, 128 → 1x1, 512) x 4
- Transition 2: 1x1 conv + 2x2 avg pool, stride 2
- Residual Block 3: (1x1, 256 → 3x3, 256 → 1x1, 1024) x 6
- Transition 3: 1x1 conv + 2x2 avg pool, stride 2
- Residual Block 4: (1x1, 512 → 3x3, 512 → 1x1, 2048) x 3
- Global Average Pooling → 2048-dim feature vector
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class ResNet50FeatureExtractor(nn.Module):
    """ResNet-50 backbone for feature extraction from fabric images."""

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Remove the final FC layer — keep everything up to avg pool
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Batch of images [B, 3, 224, 224]

        Returns:
            Feature vectors [B, 2048]
        """
        features = self.features(x)
        return features.flatten(1)


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 with a classification head for fabric defect classification.
    Used as a baseline comparison model.
    """

    def __init__(
        self,
        num_classes: int = 12,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = ResNet50FeatureExtractor(pretrained, freeze_backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def get_resnet_transforms(image_size: int = 224) -> transforms.Compose:
    """Get the standard transforms for ResNet-50 input."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
