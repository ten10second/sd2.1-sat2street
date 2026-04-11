"""
Structural Similarity Index (SSIM) metric.
"""

import cv2
import numpy as np
import torch


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute SSIM between two RGB images.

    Args:
        image1: (H, W, 3) RGB image in [0, 255]
        image2: (H, W, 3) RGB image in [0, 255]

    Returns:
        ssim: SSIM value
    """
    # Convert to grayscale
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    else:
        image1_gray = image1
        image2_gray = image2

    # Compute SSIM using OpenCV
    ssim = cv2.quality.QualitySSIM_compute([image1_gray], [image2_gray])[0][0]
    return float(ssim)


def compute_ssim_tensor(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """
    Compute SSIM between two tensor images.

    Args:
        image1: (B, C, H, W) tensor in [0, 1]
        image2: (B, C, H, W) tensor in [0, 1]

    Returns:
        ssim: (B,) SSIM values
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        return ssim(image1, image2)
    except ImportError:
        raise ImportError("Please install torchmetrics: pip install torchmetrics")
