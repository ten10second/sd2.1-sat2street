"""
Perspective Position Encoding for satellite tokens.

Projects BEV patch centers into the perspective view and encodes the resulting
pixel coordinates as a learnable positional encoding that is added to satellite
tokens before they enter the UNet cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


def compute_sat_patch_perspective_uv(
    bev_coords: torch.Tensor,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    image_w: int,
    image_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project BEV patch centres (meters) into perspective pixel coordinates.

    This is the inverse of ``compute_camera_bev_xy`` in the dataset module.

    Args:
        bev_coords: (B, N, 2) patch centres in meters relative to satellite/IMU centre.
        K: (B, 3, 3) camera intrinsics.
        T_cam_to_world: (B, 4, 4) camera-to-world extrinsics.
        T_imu_to_world: (B, 4, 4) IMU-to-world extrinsics (defines sat centre).
        image_w: perspective image width in pixels.
        image_h: perspective image height in pixels.

    Returns:
        uv_norm: (B, N, 2) normalised pixel coords in [-1, 1].
        valid: (B, N) boolean validity mask.
    """
    B, N, _ = bev_coords.shape
    device = bev_coords.device
    dtype = bev_coords.dtype

    # Satellite centre in world XY (the IMU position)
    sat_center_xy = T_imu_to_world[:, :2, 3]  # (B, 2)

    # World point on ground plane (z = 0)
    world_xy = sat_center_xy.unsqueeze(1) + bev_coords  # (B, N, 2)
    world_xyz = torch.cat(
        [world_xy, torch.zeros(B, N, 1, device=device, dtype=dtype)], dim=-1
    )  # (B, N, 3)
    world_h = torch.cat(
        [world_xyz, torch.ones(B, N, 1, device=device, dtype=dtype)], dim=-1
    )  # (B, N, 4)

    # World → camera
    T_world_to_cam = torch.inverse(T_cam_to_world.float()).to(dtype)  # (B, 4, 4)
    cam_h = torch.bmm(
        T_world_to_cam.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, 4, 4),
        world_h.reshape(B * N, 4, 1),
    ).reshape(B, N, 4)  # (B, N, 4)
    cam_xyz = cam_h[..., :3]  # (B, N, 3)

    # Camera → pixel
    pixel_h = torch.bmm(
        K.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, 3, 3),
        cam_xyz.reshape(B * N, 3, 1),
    ).reshape(B, N, 3)  # (B, N, 3)

    u = pixel_h[..., 0] / (pixel_h[..., 2] + 1e-8)  # (B, N)
    v = pixel_h[..., 1] / (pixel_h[..., 2] + 1e-8)  # (B, N)

    # Normalise to [-1, 1]
    u_norm = 2.0 * u / max(image_w, 1) - 1.0
    v_norm = 2.0 * v / max(image_h, 1) - 1.0
    uv_norm = torch.stack([u_norm, v_norm], dim=-1)  # (B, N, 2)

    # Validity: in front of camera + within image bounds
    valid = (
        (cam_xyz[..., 2] > 0.01)
        & (u >= 0) & (u < image_w)
        & (v >= 0) & (v < image_h)
    )  # (B, N)

    return uv_norm, valid


class PerspectivePositionEncoder(nn.Module):
    """
    Learnable MLP that encodes normalised perspective pixel coordinates.

    Architecture matches ``coord_encoder`` in SatelliteConditionEncoder:
        Linear(2 → dim) → LayerNorm → GELU → Linear(dim → dim) → LayerNorm
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, uv_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv_norm: (B, N, 2) normalised pixel coords in [-1, 1].

        Returns:
            pe: (B, N, dim) perspective position encoding.
        """
        return self.mlp(uv_norm)