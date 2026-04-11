"""
Peak Signal-to-Noise Ratio (PSNR) metric.
"""

import cv2
import numpy as np
import torch


def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute PSNR between two RGB images.

    Args:
        image1: (H, W, 3) RGB image in [0, 255]
        image2: (H, W, 3) RGB image in [0, 255]

    Returns:
        psnr: PSNR value
    """
    # Convert to grayscale if needed
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    else:
        image1_gray = image1
        image2_gray = image2

    # Compute MSE
    mse = np.mean((image1_gray - image2_gray) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_psnr_tensor(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """
    Compute PSNR between two tensor images.

    Args:
        image1: (B, C, H, W) tensor in [0, 1]
        image2: (B, C, H, W) tensor in [0, 1]

    Returns:
        psnr: (B,) PSNR values
    """
    # Convert to grayscale if RGB
    if image1.shape[1] == 3:
        image1_gray = 0.2989 * image1[:, 0] + 0.5870 * image1[:, 1] + 0.1140 * image1[:, 2]
        image2_gray = 0.2989 * image2[:, 0] + 0.5870 * image2[:, 1] + 0.1140 * image2[:, 2]
    else:
        image1_gray = image1.squeeze(1)
        image2_gray = image2.squeeze(1)

    # Compute MSE
    mse = ((image1_gray - image2_gray) ** 2).mean(dim=(1, 2))
    max_pixel = 1.0
    psnr = 10 * torch.log10((max_pixel ** 2) / mse)
    return psnr
