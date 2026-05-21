#!/usr/bin/env python3
"""KITTI-360 geometry utilities.

This module keeps only the geometry helpers used by the current branch.
The conventions assumed here are:

- image_00 camera: x=right, y=down, z=forward
- IMU / GPS: x=forward, y=right, z=down
- world: X=east, Y=north, Z=up
- satellite crop: north-up, east-right, centered on the world/IMU origin
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


PathLike = Union[str, Path]


def _read_key_value_matrices(path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value_str = line.split(":", 1)
            values = value_str.strip().split()
            if not values:
                continue
            try:
                data[key.strip()] = np.asarray([float(v) for v in values], dtype=np.float64)
            except ValueError:
                continue
    return data


def load_kitti360_cam_to_pose_calib(calib_path: PathLike) -> Dict[str, torch.Tensor]:
    """Load ``calib_cam_to_pose.txt``.

    Returns a mapping like ``{"image_00": T_pose_cam}`` where each matrix is
    4x4 and maps camera coordinates into the KITTI-360 pose / IMU frame.
    """
    calib_path = Path(calib_path)
    if not calib_path.is_file():
        raise FileNotFoundError(f"Missing KITTI-360 camera calibration: {calib_path}")

    raw = _read_key_value_matrices(calib_path)
    out: Dict[str, torch.Tensor] = {}
    for key, values in raw.items():
        if values.size != 12:
            continue
        T = np.eye(4, dtype=np.float64)
        T[:3, :] = values.reshape(3, 4)
        out[key] = torch.from_numpy(T.astype(np.float32))

    if not out:
        raise ValueError(f"No valid camera-to-pose matrices found in {calib_path}")
    return out


def load_indexed_pose_txt(path: PathLike, mat_size: int) -> Dict[int, torch.Tensor]:
    """Load KITTI-360 indexed pose text files.

    Supported formats:
    - ``poses.txt``: ``frame_id`` + 12 floats -> 3x4 ``imu->world``
    - ``cam0_to_world.txt``: ``frame_id`` + 16 floats -> 4x4 ``cam0->world``
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing pose file: {path}")

    out: Dict[int, torch.Tensor] = {}
    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 1 + mat_size:
                continue

            frame_id = int(parts[0])
            values = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            if mat_size == 12:
                T[:3, :] = values.reshape(3, 4)
            elif mat_size == 16:
                T[:, :] = values.reshape(4, 4)
            else:
                raise ValueError(f"Unsupported mat_size={mat_size}; expected 12 or 16")
            out[frame_id] = torch.from_numpy(T.astype(np.float32))

    if not out:
        raise ValueError(f"No valid pose entries found in {path}")
    return out


def nearest_pose(frame_id: int, pose_dict: Dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
    """Return the exact pose for ``frame_id`` or the nearest previous pose."""
    if frame_id in pose_dict:
        return pose_dict[frame_id]
    eligible = [key for key in pose_dict.keys() if key <= frame_id]
    if not eligible:
        return None
    return pose_dict[max(eligible)]


def load_imu_to_world_pose(
    poses_path: PathLike,
    frame_id: Optional[int] = None,
    *,
    nearest_previous: bool = True,
    strict: bool = False,
) -> Union[Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    """Load KITTI-360 ``imu->world`` poses from ``poses.txt``."""
    pose_dict = load_indexed_pose_txt(poses_path, mat_size=12)
    if frame_id is None:
        return pose_dict

    pose = pose_dict.get(frame_id)
    if pose is None and nearest_previous:
        pose = nearest_pose(frame_id, pose_dict)
    if pose is None and strict:
        raise KeyError(f"No pose available for frame_id={frame_id}")
    return pose


def load_cam0_to_world_pose(
    cam0_to_world_path: PathLike,
    frame_id: Optional[int] = None,
    *,
    nearest_previous: bool = True,
    strict: bool = False,
) -> Union[Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    """Load rectified ``cam0->world`` poses from ``cam0_to_world.txt``."""
    pose_dict = load_indexed_pose_txt(cam0_to_world_path, mat_size=16)
    if frame_id is None:
        return pose_dict

    pose = pose_dict.get(frame_id)
    if pose is None and nearest_previous:
        pose = nearest_pose(frame_id, pose_dict)
    if pose is None and strict:
        raise KeyError(f"No cam0->world pose available for frame_id={frame_id}")
    return pose


def compose_camera_to_world_transform(
    T_cam_to_pose: torch.Tensor,
    T_imu_to_world: torch.Tensor,
) -> torch.Tensor:
    """Compose ``camera->pose`` with ``pose/IMU->world``."""
    return T_imu_to_world @ T_cam_to_pose


def get_world_to_satellite_transform(
    sat_size: int = 512,
    resolution_m_per_px: float = 0.2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the north-up world-to-satellite transform.

    World coordinates are interpreted as east/right and north/forward. Satellite
    coordinates use image indexing with the origin at the center of the crop.
    """
    scale = 1.0 / float(resolution_m_per_px)
    T = torch.eye(4, dtype=dtype, device=device)
    T[0, 0] = scale
    T[1, 1] = -scale
    T[0, 3] = float(sat_size) / 2.0
    T[1, 3] = float(sat_size) / 2.0
    return T


def compose_camera_to_satellite_transform(
    T_cam_to_pose: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution_m_per_px: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose camera->satellite and return the intermediate world->satellite matrix."""
    if T_cam_to_pose.dim() != 2 or T_cam_to_pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 camera pose matrix, got {tuple(T_cam_to_pose.shape)}")
    if T_imu_to_world.dim() != 2 or T_imu_to_world.shape != (4, 4):
        raise ValueError(f"Expected 4x4 IMU/world matrix, got {tuple(T_imu_to_world.shape)}")

    T_world_to_sat = get_world_to_satellite_transform(
        sat_size=sat_size,
        resolution_m_per_px=resolution_m_per_px,
        device=T_cam_to_pose.device,
        dtype=T_cam_to_pose.dtype,
    )
    T_cam_to_world = compose_camera_to_world_transform(T_cam_to_pose, T_imu_to_world)
    T_cam_to_sat = T_world_to_sat @ T_cam_to_world
    return T_cam_to_sat, T_world_to_sat


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) transform."""
    if T.dim() != 2 or T.shape != (4, 4):
        raise ValueError(f"Expected a single 4x4 matrix, got {tuple(T.shape)}")

    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.transpose(0, 1)
    t_inv = -(R_inv @ t)

    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv
