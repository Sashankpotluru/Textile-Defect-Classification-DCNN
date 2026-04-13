"""Tests for model architectures."""

import pytest
import torch

from src.dcnn_classifier import DualTrackDCNN, EnhancedDCNN
from src.feature_extraction import ResNet50FeatureExtractor, ResNet50Classifier


class TestDualTrackDCNN:
    def test_forward_shape(self):
        model = DualTrackDCNN(num_classes=12, input_size=150)
        x = torch.randn(2, 3, 150, 150)
        out = model(x)
        assert out.shape == (2, 12)

    def test_predict(self):
        model = DualTrackDCNN(num_classes=12)
        x = torch.randn(4, 3, 150, 150)
        preds, probs = model.predict(x)
        assert preds.shape == (4,)
        assert probs.shape == (4, 12)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_different_num_classes(self):
        model = DualTrackDCNN(num_classes=5)
        x = torch.randn(1, 3, 150, 150)
        assert model(x).shape == (1, 5)


class TestEnhancedDCNN:
    def test_forward_shape(self):
        model = EnhancedDCNN(num_classes=12)
        x = torch.randn(2, 3, 150, 150)
        out = model(x)
        assert out.shape == (2, 12)


class TestResNet50:
    def test_feature_extractor(self):
        model = ResNet50FeatureExtractor(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        features = model(x)
        assert features.shape == (2, 2048)

    def test_classifier(self):
        model = ResNet50Classifier(num_classes=12, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 12)
