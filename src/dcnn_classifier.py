"""
Custom Dual-Track Deep Convolutional Neural Network (DCNN) for
Fabric Defect Classification.

Architecture as described in Section 3.4 and Figure 6 of the paper:

Track 1 (Convolutional):
  Conv(32, 3x3) → ReLU → MaxPool(2x2)
  Conv(64, 3x3) → ReLU → MaxPool(2x2)
  Conv(128, 3x3) → ReLU → MaxPool(2x2)

Track 2 (Dense):
  Flatten → FC(256) → ReLU → Dropout
  FC(128) → ReLU → Dropout
  FC(64) → ReLU → Dropout

Output:
  FC(num_classes) → Softmax

Input: 150 x 150 x 3 (RGB, preprocessed with grayscale + bilateral filter)
Output: 12 fabric defect classes

The dual-track combines a profound CNN with max-pooling (Track 1) and
fully connected layers (Track 2) for comprehensive feature extraction.
Uses 3x1 filters to preserve spatial pixel correlation while extracting
intrinsic image attributes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTrackDCNN(nn.Module):
    """
    Proposed DCNN architecture for fabric defect classification.
    Achieves 96.29% accuracy as reported in the paper.
    """

    def __init__(
        self,
        num_classes: int = 12,
        input_size: int = 150,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_size = input_size

        # Track 1: Convolutional Feature Extraction
        # 3 convolutional-pooling layers as described in the paper
        self.conv_track = nn.Sequential(
            # Layer 1: 32 feature maps, 3x3 kernel
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 150 → 75

            # Layer 2: 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 75 → 37

            # Layer 3: 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 37 → 18
        )

        # Compute flattened size
        self._flat_size = 128 * (input_size // 8) * (input_size // 8)

        # Track 2: Dense Classification Head
        # 4 hidden layers: densely interconnected
        self.dense_track = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.4),
        )

        # Output layer: softmax over defect classes
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dual-track DCNN.

        Args:
            x: Input images [B, 3, 150, 150]

        Returns:
            Class logits [B, num_classes]
        """
        # Track 1: Extract convolutional features
        conv_features = self.conv_track(x)

        # Track 2: Dense classification
        dense_out = self.dense_track(conv_features)

        # Classification output
        logits = self.classifier(dense_out)
        return logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted class indices and probabilities."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


class EnhancedDCNN(nn.Module):
    """
    Enhanced version with residual connections and attention,
    building on the paper's DCNN architecture.
    """

    def __init__(self, num_classes: int = 12, dropout: float = 0.5):
        super().__init__()

        # Convolutional blocks with residual connections
        self.block1 = self._make_block(3, 32)
        self.block2 = self._make_block(32, 64)
        self.block3 = self._make_block(64, 128)
        self.block4 = self._make_block(128, 256)

        # Channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.Sigmoid(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, num_classes),
        )

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Channel attention
        att = self.attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * att

        x = self.global_pool(x)
        return self.classifier(x)
