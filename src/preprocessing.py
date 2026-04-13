"""
Preprocessing module for textile defect images.

Pipeline: RGB → Grayscale → Bilateral Filter (noise reduction)
As described in Section 3.1 of the paper.
"""

import cv2
import numpy as np
from pathlib import Path


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using the standard luminance formula.
    G(x,y) = 0.299 * R(x,y) + 0.587 * G(x,y) + 0.114 * B(x,y)
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Apply bilateral filter for noise reduction while preserving edges.

    BF(I)(i) = (1/Wp) * sum_j { Gs(||i-j||) * Gr(|I(i)-I(j)|) * I(j) }

    Args:
        image: Input image (grayscale or color)
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def preprocess_image(
    image: np.ndarray,
    target_size: int = 150,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Full preprocessing pipeline as described in the paper:
    1. Convert to grayscale
    2. Apply bilateral filter for noise reduction
    3. Resize to target dimensions

    Returns grayscale denoised image.
    """
    gray = rgb_to_grayscale(image)
    denoised = apply_bilateral_filter(gray, d, sigma_color, sigma_space)
    resized = cv2.resize(denoised, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def preprocess_for_resnet(
    image: np.ndarray,
    target_size: int = 224,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Preprocess image for ResNet-50 feature extraction.
    Keeps 3 channels (grayscale replicated) for ResNet compatibility.
    """
    gray = rgb_to_grayscale(image)
    denoised = apply_bilateral_filter(gray, d, sigma_color, sigma_space)
    resized = cv2.resize(denoised, (target_size, target_size), interpolation=cv2.INTER_AREA)
    # Convert single channel to 3 channels for ResNet
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return rgb


def preprocess_for_dcnn(
    image: np.ndarray,
    target_size: int = 150,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Preprocess for the custom DCNN classifier.
    Returns 3-channel image at 150x150.
    """
    gray = rgb_to_grayscale(image)
    denoised = apply_bilateral_filter(gray, d, sigma_color, sigma_space)
    resized = cv2.resize(denoised, (target_size, target_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return rgb


def load_and_preprocess(image_path: str, target_size: int = 150) -> np.ndarray:
    """Load an image from disk and apply full preprocessing."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return preprocess_for_dcnn(image, target_size)
