"""
Differentiable camera to satellite projection using grid_sample.

This module implements fully differentiable inverse projection from camera view
to satellite (BEV) view using PyTorch's grid_sample for gradient flow.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def differentiable_camera_to_sat_warp(
    cam_image: torch.Tensor,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution: float = 0.2,
    ground_height: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable camera → satellite projection using grid_sample.

    For each satellite pixel:
    1. Convert to world coordinates (assuming ground plane at z=ground_height)
    2. Transform to camera coordinates
    3. Project to camera image plane
    4. Sample from camera image using grid_sample (differentiable)

    Args:
        cam_image: Camera image tensor (B, C, H_cam, W_cam) or (C, H_cam, W_cam)
        K: Camera intrinsic matrix (3, 3) or (B, 3, 3)
        T_cam_to_world: Camera to world transform (4, 4) or (B, 4, 4)
        sat_size: Satellite image size in pixels (default: 512)
        resolution: Satellite resolution in m/pixel (default: 0.2)
        ground_height: Ground plane height in world coordinates (default: 0.0)

    Returns:
        sat_image_warped: Warped satellite image (B, C, sat_size, sat_size)
        valid_mask: Valid pixel mask (B, 1, sat_size, sat_size)
    """
    device = cam_image.device

    # Handle batch dimension
    if cam_image.ndim == 3:
        cam_image = cam_image.unsqueeze(0)  # (1, C, H, W)

    B, C, H_cam, W_cam = cam_image.shape

    # Ensure K and T_cam_to_world have batch dimension
    if K.ndim == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)
    if T_cam_to_world.ndim == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 4)

    # Compute world to camera transform (inverse)
    R_cam_to_world = T_cam_to_world[:, :3, :3]  # (B, 3, 3)
    t_cam_to_world = T_cam_to_world[:, :3, 3:4]  # (B, 3, 1)

    # T_world_to_cam = [R^T | -R^T * t]
    R_world_to_cam = R_cam_to_world.transpose(1, 2)  # (B, 3, 3)
    t_world_to_cam = -R_world_to_cam @ t_cam_to_world  # (B, 3, 1)

    # Create satellite pixel grid
    half_size = sat_size // 2
    u = torch.arange(sat_size, device=device, dtype=torch.float32) - half_size
    v = torch.arange(sat_size, device=device, dtype=torch.float32) - half_size
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # (sat_size, sat_size)

    # Convert satellite pixels to world coordinates
    # Satellite: u=right (East), v=down (South), origin at center
    # World: X=East, Y=North, Z=up
    x_world = uu * resolution  # (sat_size, sat_size)
    y_world = -vv * resolution  # (sat_size, sat_size) - flip v to get North
    z_world = torch.full_like(x_world, ground_height)  # (sat_size, sat_size)

    # Stack to (sat_size, sat_size, 3)
    world_coords = torch.stack([x_world, y_world, z_world], dim=-1)  # (sat_size, sat_size, 3)
    world_coords_flat = world_coords.reshape(-1, 3).T  # (3, N) where N = sat_size^2

    # Expand for batch
    world_coords_flat = world_coords_flat.unsqueeze(0).expand(B, -1, -1)  # (B, 3, N)

    # Transform to camera coordinates
    cam_coords = R_world_to_cam @ world_coords_flat + t_world_to_cam  # (B, 3, N)

    # Project to camera image plane
    cam_pixels = K @ cam_coords  # (B, 3, N)

    # Normalize by depth
    depth = cam_pixels[:, 2:3, :]  # (B, 1, N)
    cam_pixels_norm = cam_pixels[:, :2, :] / (depth + 1e-8)  # (B, 2, N)

    # Reshape to (B, sat_size, sat_size, 2)
    cam_pixels_2d = cam_pixels_norm.permute(0, 2, 1).reshape(B, sat_size, sat_size, 2)  # (B, H, W, 2)

    # Normalize to [-1, 1] for grid_sample
    # grid_sample expects: x in [-1, 1] maps to [0, W-1], y in [-1, 1] maps to [0, H-1]
    grid_x = 2.0 * cam_pixels_2d[..., 0] / (W_cam - 1) - 1.0  # (B, sat_size, sat_size)
    grid_y = 2.0 * cam_pixels_2d[..., 1] / (H_cam - 1) - 1.0  # (B, sat_size, sat_size)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, sat_size, sat_size, 2)

    # Create valid mask: depth > 0 and within image bounds
    valid_depth = (depth.squeeze(1) > 0.1).reshape(B, sat_size, sat_size)  # (B, sat_size, sat_size)
    valid_x = (grid_x >= -1) & (grid_x <= 1)  # (B, sat_size, sat_size)
    valid_y = (grid_y >= -1) & (grid_y <= 1)  # (B, sat_size, sat_size)
    valid_mask = (valid_depth & valid_x & valid_y).unsqueeze(1).float()  # (B, 1, sat_size, sat_size)

    # Sample from camera image using grid_sample (differentiable!)
    sat_image_warped = F.grid_sample(
        cam_image,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # (B, C, sat_size, sat_size)

    # Apply valid mask
    sat_image_warped = sat_image_warped * valid_mask

    return sat_image_warped, valid_mask
