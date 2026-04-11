"""
Learned Perceptual Image Patch Similarity (LPIPS) metric.
"""

import torch


def compute_lpips(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """
    Compute LPIPS between two images.

    Args:
        image1: (1, 3, H, W) tensor in [0, 1]
        image2: (1, 3, H, W) tensor in [0, 1]

    Returns:
        lpips: LPIPS value
    """
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
        score = lpips(image1, image2)
        return float(score.item())
    except ImportError:
        raise ImportError("Please install torchmetrics: pip install torchmetrics")
