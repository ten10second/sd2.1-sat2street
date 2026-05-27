"""Projection helpers for satellite patch geometry."""

from __future__ import annotations

from typing import Tuple, Union

import torch


def compute_sat_patch_perspective_uv(
    bev_coords: torch.Tensor,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    camera_height_m: Union[torch.Tensor, float],
    image_w: int,
    image_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project BEV patch centres (meters) into perspective pixel coordinates.

    This is the inverse of ``compute_camera_bev_xy`` in the dataset module.

    All heavy geometry is computed in fp32 in an IMU-relative local frame,
    avoiding UTM-scale (~1e6 m) values that lose precision in bf16.

    Args:
        bev_coords: (B, N, 2) patch centres in meters relative to satellite/IMU centre.
        K: (B, 3, 3) camera intrinsics.
        T_cam_to_world: (B, 4, 4) camera-to-world extrinsics.
        T_imu_to_world: (B, 4, 4) IMU-to-world extrinsics (defines sat centre).
        camera_height_m: scalar or (B,) camera height above the local ground plane.
        image_w: perspective image width in pixels.
        image_h: perspective image height in pixels.

    Returns:
        uv_norm: (B, N, 2) normalised pixel coords in [-1, 1].
        valid: (B, N) boolean validity mask.
    """
    B, N, _ = bev_coords.shape
    device = bev_coords.device
    output_dtype = bev_coords.dtype

    # --- all geometry held in fp32 to avoid bf16 UTM-scale truncation ---
    calc_dtype = torch.float32

    camera_height = _as_batch_vector(
        camera_height_m,
        batch_size=B,
        device=device,
        dtype=calc_dtype,
        name="camera_height_m",
    )

    T_cam_f32 = T_cam_to_world.to(dtype=calc_dtype)   # (B, 4, 4)
    T_imu_f32 = T_imu_to_world.to(dtype=calc_dtype)   # (B, 4, 4)

    R_cam_to_world = T_cam_f32[:, :3, :3]              # (B, 3, 3)
    t_cam_to_world = T_cam_f32[:, :3, 3]               # (B, 3)
    t_imu_world    = T_imu_f32[:, :3, 3]               # (B, 3)

    # analytic world→camera (R^T, -R^T·t) — no torch.inverse needed
    R_world_to_cam = R_cam_to_world.transpose(1, 2)    # (B, 3, 3)
    t_world_to_cam = -torch.bmm(
        R_world_to_cam, t_cam_to_world.unsqueeze(-1)
    ).squeeze(-1)  # (B, 3)

    # IMU position in camera frame (small, ~10 m)
    t_imu_to_cam = t_world_to_cam + torch.bmm(
        R_world_to_cam, t_imu_world.unsqueeze(-1)
    ).squeeze(-1)  # (B, 3)

    # Ground-plane Z relative to IMU (metres, avoids UTM Z ~1e5)
    ground_z_relative = (
        t_cam_to_world[:, 2] - camera_height
    ) - t_imu_world[:, 2]  # (B,)

    # 3-d local offset from IMU: all components ≤ O(50 m)
    bev_coords_f32 = bev_coords.to(dtype=calc_dtype)
    bev_offset_3d = torch.cat(
        [
            bev_coords_f32,                                    # (B, N, 2)
            ground_z_relative.view(B, 1, 1).expand(B, N, 1),  # (B, N, 1)
        ],
        dim=-1,
    )  # (B, N, 3)

    # cam_xyz = t_imu_to_cam + R_world_to_cam @ bev_offset
    cam_xyz = t_imu_to_cam.unsqueeze(1) + torch.bmm(
        bev_offset_3d, R_world_to_cam.transpose(1, 2)
    )  # (B, N, 3)

    # ensure K is batched and in calc_dtype
    K_f32 = K.to(dtype=calc_dtype)
    if K_f32.dim() == 2:
        K_f32 = K_f32.unsqueeze(0).expand(B, -1, -1)

    # Camera → pixel
    pixel_h = torch.bmm(
        K_f32.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, 3, 3),
        cam_xyz.reshape(B * N, 3, 1),
    ).reshape(B, N, 3)  # (B, N, 3)

    u = pixel_h[..., 0] / (pixel_h[..., 2] + 1e-8)  # (B, N)
    v = pixel_h[..., 1] / (pixel_h[..., 2] + 1e-8)  # (B, N)

    # Normalise to [-1, 1] — cast to output dtype
    u_norm = (2.0 * u / max(image_w, 1) - 1.0).to(dtype=output_dtype)
    v_norm = (2.0 * v / max(image_h, 1) - 1.0).to(dtype=output_dtype)
    uv_norm = torch.stack([u_norm, v_norm], dim=-1)  # (B, N, 2)

    # Validity: in front of camera + within image bounds
    valid = (
        (cam_xyz[..., 2] > 0.01)
        & (u >= 0) & (u < image_w)
        & (v >= 0) & (v < image_h)
    )  # (B, N)

    return uv_norm, valid


def _as_batch_vector(
    value: Union[torch.Tensor, float],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(value, device=device, dtype=dtype)

    if tensor.ndim == 0:
        tensor = tensor.reshape(1)

    if tensor.numel() == 1:
        return tensor.reshape(1).expand(batch_size)
    if tensor.numel() == batch_size:
        return tensor.reshape(batch_size)
    raise ValueError(
        f"{name} must be a scalar or contain {batch_size} values, got shape {tuple(tensor.shape)}"
    )
