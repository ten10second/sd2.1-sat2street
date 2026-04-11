"""
Geometric Consistency Loss for Stable Diffusion.

Combines traditional Stable Diffusion loss with geometric consistency loss
to ensure generated frontviews align with satellite image geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from utils.geometry.differentiable_projection import differentiable_camera_to_sat_warp


class GeometricLoss(nn.Module):
    """
    Geometric consistency loss for satellite-to-frontview generation.

    Args:
        loss_type: Type of loss to use ('l1', 'l2')
        resolution: Satellite resolution in m/pixel
        sat_size: Satellite image size in pixels
        warmup_epochs: Number of epochs to warm up before applying loss
        transition_epochs: Number of epochs to transition to full loss weight
        lambda_geometric: Weight of geometric loss
    """

    def __init__(
        self,
        loss_type: str = 'l1',
        resolution: float = 0.2,
        sat_size: int = 512,
        warmup_epochs: int = 5,
        transition_epochs: int = 15,
        lambda_geometric: float = 0.1,
    ):
        super().__init__()

        self.loss_type = loss_type.lower()
        self.resolution = resolution
        self.sat_size = sat_size
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        self.lambda_geometric = lambda_geometric

        # Loss function
        if self.loss_type == 'l1':
            self.loss_func = F.l1_loss
        elif self.loss_type == 'l2':
            self.loss_func = F.mse_loss
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        print(f"[GeometricLoss] Initialization:")
        print(f"  Loss type: {loss_type}")
        print(f"  Resolution: {resolution} m/px")
        print(f"  Satellite size: {sat_size}x{sat_size}")
        print(f"  Warmup epochs: {warmup_epochs}")
        print(f"  Transition epochs: {transition_epochs}")
        print(f"  Lambda geometric: {lambda_geometric}")

    def get_temperature(self, epoch: int) -> float:
        """Get temperature for annealing."""
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.transition_epochs:
            progress = (epoch - self.warmup_epochs) / self.transition_epochs
            return progress
        else:
            return 1.0

    def get_lambda_weight(self, epoch: int, ce_loss: float, geometric_loss: float) -> float:
        """Adaptive lambda weight based on loss ratio."""
        base_weight = self.lambda_geometric * self.get_temperature(epoch)

        if epoch < self.warmup_epochs:
            return 0.0

        # Adjust based on loss ratio
        if geometric_loss > 0 and ce_loss > 0:
            ratio = geometric_loss / ce_loss
            if ratio > 10:
                base_weight *= 0.1
            elif ratio > 5:
                base_weight *= 0.5
            elif ratio < 0.1:
                base_weight *= 2.0

        return min(base_weight, 0.2)  # Upper limit

    def forward(
        self,
        generated_img: torch.Tensor,
        real_img: torch.Tensor,
        K: torch.Tensor,
        T_cam_to_world: torch.Tensor,
        sat_image_gt: torch.Tensor,
        epoch: int,
        ce_loss: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass to compute combined loss.

        Args:
            generated_img: (B, 3, H, W) - Generated frontview image
            real_img: (B, 3, H, W) - Real frontview image
            K: (3, 3) or (B, 3, 3) - Camera intrinsics
            T_cam_to_world: (4, 4) or (B, 4, 4) - Camera to world transform
            sat_image_gt: (B, 3, sat_size, sat_size) - Real satellite image
            epoch: Current epoch
            ce_loss: Cross entropy loss for weight adjustment

        Returns:
            total_loss: Combined loss
            info: Dictionary with loss breakdown
        """
        device = generated_img.device

        # Compute reconstruction loss
        recon_loss = self.loss_func(generated_img, real_img)

        # Compute geometric consistency loss
        if epoch < self.warmup_epochs:
            geometric_loss = torch.tensor(0.0, device=device)
        else:
            # Project generated image to satellite view
            warped_gen, valid_mask = differentiable_camera_to_sat_warp(
                generated_img,
                K,
                T_cam_to_world,
                sat_size=self.sat_size,
                resolution=self.resolution,
            )

            # Compute loss on valid pixels
            valid_pixel_count = valid_mask.sum().clamp(min=1e-6)
            loss_per_pixel = self.loss_func(warped_gen * valid_mask, sat_image_gt * valid_mask, reduction='none')
            geometric_loss = loss_per_pixel.sum() / valid_pixel_count

        # Compute total loss
        lambda_weight = self.get_lambda_weight(epoch, ce_loss, geometric_loss.item() if epoch >= self.warmup_epochs else 0)
        total_loss = recon_loss + lambda_weight * geometric_loss

        info = {
            'recon_loss': recon_loss.item(),
            'geometric_loss': geometric_loss.item() if epoch >= self.warmup_epochs else 0.0,
            'lambda_weight': lambda_weight,
            'temperature': self.get_temperature(epoch),
        }

        if epoch >= self.warmup_epochs:
            valid_ratio = valid_mask.mean().item()
            info['valid_ratio'] = valid_ratio

        return total_loss, info
