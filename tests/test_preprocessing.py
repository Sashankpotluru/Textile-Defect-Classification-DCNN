"""Tests for preprocessing module."""

import numpy as np
import pytest

from src.preprocessing import (
    rgb_to_grayscale,
    apply_bilateral_filter,
    preprocess_image,
    preprocess_for_dcnn,
)


class TestGrayscale:
    def test_rgb_to_gray(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = rgb_to_grayscale(img)
        assert gray.shape == (100, 100)
        assert gray.dtype == np.uint8

    def test_already_gray(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray = rgb_to_grayscale(img)
        assert gray.shape == (100, 100)


class TestBilateralFilter:
    def test_output_shape(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        filtered = apply_bilateral_filter(img)
        assert filtered.shape == img.shape

    def test_reduces_noise(self):
        # Noisy image should have lower variance after filtering
        base = np.full((100, 100), 128, dtype=np.uint8)
        noise = np.random.randint(-30, 30, (100, 100), dtype=np.int16)
        noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        filtered = apply_bilateral_filter(noisy)
        assert np.std(filtered) <= np.std(noisy)


class TestPreprocessImage:
    def test_output_shape(self):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = preprocess_image(img, target_size=150)
        assert result.shape == (150, 150)

    def test_preprocess_for_dcnn_shape(self):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = preprocess_for_dcnn(img, target_size=150)
        assert result.shape == (150, 150, 3)
