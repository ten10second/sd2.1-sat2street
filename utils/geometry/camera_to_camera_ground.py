#!/usr/bin/env python3
"""
相机到相机的地平面投影（通过地平面中介）

用于生成目标视角的伪GT：
1. 对每个目标像素，反投影到射线
2. 与地平面相交得到世界坐标
3. 投回源相机坐标系采样

使用 pull-based 方法，避免空洞。
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def camera_to_camera_groundplane_pull(
    src_rgb: torch.Tensor,
    K: torch.Tensor,
    T_src: torch.Tensor,
    T_tgt: torch.Tensor,
    ground_height: float = 0.0,
    padding_mode: str = 'zeros',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pull-based camera-to-camera warp via ground plane.

    For each target pixel, compute its corresponding source pixel by:
    1. Backproject target pixel to ray (in target camera frame)
    2. Intersect ray with ground plane (Z=ground_height) → world point
    3. Project world point to source camera frame → source pixel
    4. Sample from source image using grid_sample

    Args:
        src_rgb: (B, C, H, W) or (C, H, W) - source camera image
        K: (3, 3) or (B, 3, 3) - camera intrinsics (same for src and tgt)
        T_src: (4, 4) or (B, 4, 4) - source camera to world transform
        T_tgt: (4, 4) or (B, 4, 4) - target camera to world transform
        ground_height: float - Z coordinate of ground plane in world frame
        padding_mode: str - padding mode for grid_sample ('zeros', 'border', 'reflection')

    Returns:
        tgt_rgb: (B, C, H, W) - RGB in target view
        valid_mask: (B, 1, H, W) - validity mask (1.0 = valid, 0.0 = invalid)
                    Valid = ray intersects ground plane (t > 0) AND
                            reprojection falls within source image bounds
    """
    # Handle batch dimension
    if src_rgb.dim() == 3:
        src_rgb = src_rgb.unsqueeze(0)

    B, C, H, W = src_rgb.shape
    device = src_rgb.device

    # Ensure batch dimension for K and transforms
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, 3, 3)
    if T_src.dim() == 2:
        T_src = T_src.unsqueeze(0).expand(B, 4, 4)
    if T_tgt.dim() == 2:
        T_tgt = T_tgt.unsqueeze(0).expand(B, 4, 4)

    # Extract rotation and translation
    R_tgt = T_tgt[:, :3, :3]      # (B, 3, 3) target camera to world
    t_tgt = T_tgt[:, :3, 3:4]     # (B, 3, 1)

    R_src = T_src[:, :3, :3]      # (B, 3, 3) source camera to world
    t_src = T_src[:, :3, 3:4]     # (B, 3, 1)

    # Inverse of K for backprojection
    K_inv = torch.inverse(K)      # (B, 3, 3)

    # Build target image pixel grid
    v_tgt = torch.arange(H, device=device, dtype=torch.float32)
    u_tgt = torch.arange(W, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v_tgt, u_tgt, indexing='ij')  # (H, W)

    # Homogeneous pixel coordinates (B, 3, H*W)
    pixels_homo = torch.stack([
        uu.reshape(-1),
        vv.reshape(-1),
        torch.ones_like(uu.reshape(-1)),
    ], dim=0).unsqueeze(0).expand(B, 3, -1)  # (B, 3, H*W)

    # Backproject to rays in target camera frame
    rays_tgt = torch.bmm(K_inv, pixels_homo)  # (B, 3, H*W)

    # Transform rays to world frame
    rays_world = torch.bmm(R_tgt, rays_tgt)   # (B, 3, H*W)

    # Intersect with ground plane: P_world = t_tgt + t * rays_world
    # Z = ground_height => t = (ground_height - t_tgt[2]) / rays_world[2]
    rays_z = rays_world[:, 2:3, :]            # (B, 1, H*W)
    t = torch.where(
        rays_z.abs() > 1e-6,
        (ground_height - t_tgt[:, 2:3, :]) / rays_z,
        torch.full_like(rays_z, -1.0),  # Invalid: ray parallel to ground
    )  # (B, 1, H*W)

    # Compute intersection points in world frame
    points_world = t_tgt + t * rays_world    # (B, 3, H*W)

    # Transform to source camera frame: P_src = R_src^T (P_world - t_src)
    R_src_inv = R_src.transpose(1, 2)        # (B, 3, 3)
    points_src = torch.bmm(R_src_inv, (points_world - t_src))  # (B, 3, H*W)

    z_src = points_src[:, 2:3, :]            # (B, 1, H*W)

    # Project to source image plane
    points_src_norm = points_src / torch.clamp(z_src, min=1e-6)
    uv_src = torch.bmm(K, points_src_norm)   # (B, 3, H*W)
    u_src = uv_src[:, 0, :]                  # (B, H*W)
    v_src = uv_src[:, 1, :]                  # (B, H*W)

    # Validity mask: ray intersects ground (t > 0) AND reprojection in bounds
    valid = (t[:, 0, :] > 0) & \
            (z_src[:, 0, :] > 0) & \
            (u_src >= 0) & (u_src < W) & \
            (v_src >= 0) & (v_src < H)  # (B, H*W)

    # Normalize to [-1, 1] for grid_sample
    u_src_norm = 2.0 * u_src / (W - 1) - 1.0
    v_src_norm = 2.0 * v_src / (H - 1) - 1.0

    # Push invalid samples out of range
    u_src_norm = torch.where(valid, u_src_norm, torch.full_like(u_src_norm, -2.0))
    v_src_norm = torch.where(valid, v_src_norm, torch.full_like(v_src_norm, -2.0))

    # Build sampling grid
    grid = torch.stack([u_src_norm, v_src_norm], dim=-1).reshape(B, H, W, 2)

    # Sample from source image
    tgt_rgb = F.grid_sample(
        src_rgb, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True
    )  # (B, C, H, W)

    # Reshape validity mask
    valid_mask = valid.reshape(B, 1, H, W).float()  # (B, 1, H, W)

    return tgt_rgb, valid_mask


def apply_yaw_rotation_to_pose(T_cam_to_world: torch.Tensor, yaw_angle: float) -> torch.Tensor:
    """
    Apply a yaw rotation to a camera pose.

    Rotates the camera around the Z-axis (world frame) by yaw_angle.

    Args:
        T_cam_to_world: (4, 4) or (B, 4, 4) - camera to world transform
        yaw_angle: float - rotation angle in radians (positive = counter-clockwise when viewed from above)

    Returns:
        T_cam_to_world_rotated: (4, 4) or (B, 4, 4) - rotated pose
    """
    device = T_cam_to_world.device
    is_batched = T_cam_to_world.dim() == 3

    if not is_batched:
        T_cam_to_world = T_cam_to_world.unsqueeze(0)

    B = T_cam_to_world.shape[0]

    # Rotation matrix around Z-axis
    cos_y = torch.cos(torch.tensor(yaw_angle, device=device, dtype=torch.float32))
    sin_y = torch.sin(torch.tensor(yaw_angle, device=device, dtype=torch.float32))

    R_z = torch.tensor([
        [cos_y, -sin_y, 0.0],
        [sin_y,  cos_y, 0.0],
        [0.0,    0.0,   1.0],
    ], device=device, dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)

    # Extract current rotation and translation
    R = T_cam_to_world[:, :3, :3]
    t = T_cam_to_world[:, :3, 3:4]

    # Apply rotation: R_new = R_z @ R
    # This rotates the camera's orientation in world frame
    R_new = torch.bmm(R_z, R)

    # Build new transform
    T_new = torch.zeros_like(T_cam_to_world)
    T_new[:, :3, :3] = R_new
    T_new[:, :3, 3:4] = t
    T_new[:, 3, 3] = 1.0

    if not is_batched:
        T_new = T_new.squeeze(0)

    return T_new
