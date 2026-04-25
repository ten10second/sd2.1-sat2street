"""KITTI-360D dataloader.

This repository currently uses ad-hoc list_txt parsing inside training scripts.
This module adds a proper torch.utils.data.Dataset for KITTI-360 style folders.

Folder example:
  2013_05_28_drive_0003_sync/
    calibration/
      perspective.txt
      image_02.yaml
      image_03.yaml
    image_00/data_rgb/0000000000.png        (front perspective)
    image_02/data_rgb/0000000000.png        (left fisheye raw)
    image_03/data_rgb/0000000000.png        (right fisheye raw)
    oxts/data/0000000000.txt

This dataset can optionally generate a *virtual* perspective view from fisheye,
with a desired viewing yaw relative to the vehicle heading.
The core fisheye->perspective logic follows utils/integrated_fisheye_visualizer.py.

NOTE:
- For fisheye we assume MEI omnidirectional model in YAML files (xi, K, D).
- For vehicle yaw we read oxts yaw (same as integrated_fisheye_visualizer.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2  # type: ignore
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml


# Mounting yaw (deg) of fisheye cameras relative to vehicle front (CW+ convention)
MOUNT_ANGLES = {
    "image_02": -90.0,
    "image_03": +90.0,
}

CAMERA_HEIGHT_M = {
    "image_00": 1.55,
    "image_02": 1.95,
    "image_03": 1.95,
}
DEFAULT_CAMERA_HEIGHT_M = CAMERA_HEIGHT_M["image_00"]


def _load_mei_yaml(path: Path) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (xi, K, D) from KITTI-360 MEI YAML."""
    raw = path.read_text()
    if raw.lstrip().startswith("%YAML"):
        raw = "\n".join(raw.splitlines()[1:])
    y = yaml.safe_load(raw)

    xi = float(y["mirror_parameters"]["xi"])
    k1 = float(y["distortion_parameters"]["k1"])
    k2 = float(y["distortion_parameters"]["k2"])
    p1 = float(y["distortion_parameters"]["p1"])
    p2 = float(y["distortion_parameters"]["p2"])

    gamma1 = float(y["projection_parameters"]["gamma1"])
    gamma2 = float(y["projection_parameters"]["gamma2"])
    u0 = float(y["projection_parameters"]["u0"])
    v0 = float(y["projection_parameters"]["v0"])

    K = np.array([[gamma1, 0, u0], [0, gamma2, v0], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, p1, p2], dtype=np.float64)
    return xi, K, D


def _rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix around the X axis."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=np.float64)

def _rot_z(deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix around the Z axis."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def _make_newK(out_w: int, out_h: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(hfov_deg)
    fx = (out_w * 0.5) / math.tan(hfov * 0.5)
    vfov = 2.0 * math.atan((out_h / out_w) * math.tan(hfov * 0.5))
    fy = (out_h * 0.5) / math.tan(vfov * 0.5)
    cx, cy = out_w * 0.5, out_h * 0.5
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _read_oxts_yaw(oxts_path: Path) -> float:
    """Read yaw from oxts file. Matches utils/integrated_fisheye_visualizer.py."""
    vals = oxts_path.read_text().strip().split()
    if len(vals) <= 5:
        raise ValueError(f"Oxts file format error: {oxts_path}")
    return float(vals[5])


def _load_cam_to_pose(calib_path: Path) -> Dict[str, np.ndarray]:
    """Load camera->pose extrinsics from calib_cam_to_pose.txt

    Returns:
        Dict mapping camera name (e.g. "image_00") to its 4x4 pose matrix T_pose_cam.
    """
    calib_data = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) != 2:
                continue

            cam_name = parts[0].strip()
            values = [float(v) for v in parts[1].strip().split()]
            if len(values) != 12:
                continue

            T_pose_cam = np.eye(4, dtype=np.float64)
            T_pose_cam[:3, :] = np.array(values).reshape(3, 4)
            calib_data[cam_name] = T_pose_cam
    return calib_data

def _load_indexed_poses_txt(path: Path, mat_size: int) -> Dict[int, np.ndarray]:
    """Load KITTI-360 pose text file with leading frame_id.

    poses.txt:  frame_id + 12 floats -> 3x4 (imu->world)
    cam0_to_world.txt: frame_id + 16 floats -> 4x4 (cam0->world)

    Args:
        mat_size: 12 or 16

    Returns:
        dict: frame_id -> 4x4 transform
    """
    out: Dict[int, np.ndarray] = {}
    if not path.exists():
        return out
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 1 + mat_size:
                continue
            fid = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            T = np.eye(4, dtype=np.float64)
            if mat_size == 12:
                T[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
            elif mat_size == 16:
                T[:, :] = np.array(vals, dtype=np.float64).reshape(4, 4)
            else:
                raise ValueError(f"Unsupported mat_size={mat_size}")
            out[fid] = T
    return out


def _nearest_pose(frame_id: int, pose_dict: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """Return pose for frame_id if exists; otherwise nearest previous frame.

    KITTI-360 poses.txt may be sparse.
    """
    if frame_id in pose_dict:
        return pose_dict[frame_id]
    # nearest previous
    keys = [k for k in pose_dict.keys() if k <= frame_id]
    if not keys:
        return None
    return pose_dict[max(keys)]

def _load_perspective_calib(calib_path: Path) -> Dict[str, Any]:
    """Load calibration from perspective.txt file.

    Returns:
        Dict containing K_00, D_00, R_00, T_00, P_rect_00, etc.
    """
    calib_data = {}

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('calib_time:'):
                continue

            parts = line.split(':')
            if len(parts) != 2:
                continue

            key = parts[0].strip()
            values_str = parts[1].strip()

            if key.startswith('K_') or key.startswith('D_') or key.startswith('R_') or key.startswith('T_'):
                values = [float(v) for v in values_str.split()]
                if key.startswith('K_'):
                    # K matrix: 3x3
                    calib_data[key] = np.array(values).reshape(3, 3)
                elif key.startswith('D_'):
                    # distortion coefficients
                    calib_data[key] = np.array(values)
                elif key.startswith('R_') or key.startswith('T_'):
                    # rotation/translation vectors
                    calib_data[key] = np.array(values)
            elif key.startswith('P_rect_'):
                # projection matrix: 3x4
                values = [float(v) for v in values_str.split()]
                calib_data[key] = np.array(values).reshape(3, 4)
            elif key.startswith('S_') or key.startswith('S_rect_'):
                # image size
                values = [float(v) for v in values_str.split()]
                calib_data[key] = np.array(values)

    return calib_data


FISHEYE_CAMERAS = ("image_02", "image_03")

FIXED_FIVE_VIEW_SPECS = (
    {
        "view_name": "front",
        "mode_override": "front",
    },
    {
        "view_name": "left_forward_45",
        "mode_override": "fisheye_virtual",
        "fisheye_camera_override": "image_02",
        "fisheye_relative_yaw_deg_override": 45.0,
        "vehicle_relative_yaw_deg_override": -45.0,
    },
    {
        "view_name": "left_side",
        "mode_override": "fisheye_virtual",
        "fisheye_camera_override": "image_02",
        "fisheye_relative_yaw_deg_override": 0.0,
        "vehicle_relative_yaw_deg_override": -90.0,
    },
    {
        "view_name": "right_forward_45",
        "mode_override": "fisheye_virtual",
        "fisheye_camera_override": "image_03",
        "fisheye_relative_yaw_deg_override": -45.0,
        "vehicle_relative_yaw_deg_override": 45.0,
    },
    {
        "view_name": "right_side",
        "mode_override": "fisheye_virtual",
        "fisheye_camera_override": "image_03",
        "fisheye_relative_yaw_deg_override": 0.0,
        "vehicle_relative_yaw_deg_override": 90.0,
    },
)


def _get_camera_height_m(camera_name: Optional[str]) -> float:
    if camera_name is None:
        return DEFAULT_CAMERA_HEIGHT_M
    return float(CAMERA_HEIGHT_M.get(camera_name, DEFAULT_CAMERA_HEIGHT_M))


def _wrap_angle_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return float(wrapped)


def _angular_distance_deg(a_deg: float, b_deg: float) -> float:
    return float(abs(_wrap_angle_deg(float(a_deg) - float(b_deg))))


def _choose_nearest_fisheye_camera(vehicle_relative_yaw_deg: float) -> str:
    return min(
        FISHEYE_CAMERAS,
        key=lambda cam: (_angular_distance_deg(vehicle_relative_yaw_deg, MOUNT_ANGLES[cam]), cam),
    )


def compute_camera_bev_xy(
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    height: int,
    width: int,
    T_imu_to_world: Optional[np.ndarray] = None,
    sat_m_per_px: float = 0.2,
    sat_w: int = 512,
    sat_h: int = 512,
    cam_height: float = DEFAULT_CAMERA_HEIGHT_M,
) -> torch.Tensor:
    """Project camera rays to the ground plane and return normalized BEV xy.

    If ``T_imu_to_world`` is provided, the BEV coordinates are centered on the IMU
    position so they align with the stored satellite crop. Otherwise the BEV grid is
    centered on the camera position.
    """
    K_inv = np.linalg.inv(K.astype(np.float64))
    R = T_cam_to_world[:3, :3].astype(np.float64)
    cam_center = T_cam_to_world[:3, 3].astype(np.float64)
    if T_imu_to_world is not None:
        sat_center_xy = T_imu_to_world[:2, 3].astype(np.float64)
    else:
        sat_center_xy = cam_center[:2]

    bev_range_x = float(sat_w) * float(sat_m_per_px) / 2.0
    bev_range_y = float(sat_h) * float(sat_m_per_px) / 2.0

    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    vv, uu = np.meshgrid(v, u, indexing='ij')
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=0)

    pixels_flat = pixels.reshape(3, -1)
    dirs_cam = K_inv @ pixels_flat
    s = float(cam_height) / (dirs_cam[1, :] + 1e-8)

    hit_cam = s[np.newaxis, :] * dirs_cam
    hit_world = cam_center[:, np.newaxis] + (R @ hit_cam)

    bev_x_norm = (hit_world[0, :] - sat_center_xy[0]) / (bev_range_x + 1e-8)
    bev_y_norm = (hit_world[1, :] - sat_center_xy[1]) / (bev_range_y + 1e-8)

    invalid = (
        (dirs_cam[1, :] < 0.01) |
        (s < 0.0) |
        (np.abs(bev_x_norm) > 1.0) |
        (np.abs(bev_y_norm) > 1.0)
    )
    bev_x_norm[invalid] = 0.0
    bev_y_norm[invalid] = 0.0

    bev_xy = np.stack([bev_x_norm, bev_y_norm], axis=0).reshape(2, height, width)
    return torch.from_numpy(bev_xy).to(torch.float32).contiguous()


def compute_plucker_map(
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    height: int,
    width: int,
    T_imu_to_world: Optional[np.ndarray] = None,
    sat_m_per_px: float = 0.2,
    sat_w: int = 512,
) -> torch.Tensor:
    """Return a per-pixel Plucker ray map as (6, H, W)."""
    K_inv = np.linalg.inv(K.astype(np.float64))
    R = T_cam_to_world[:3, :3].astype(np.float64)
    cam_center = T_cam_to_world[:3, 3].astype(np.float64)

    if T_imu_to_world is not None:
        local_origin = np.array(
            [T_imu_to_world[0, 3], T_imu_to_world[1, 3], 0.0],
            dtype=np.float64,
        )
    else:
        local_origin = np.array([cam_center[0], cam_center[1], 0.0], dtype=np.float64)

    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    vv, uu = np.meshgrid(v, u, indexing='ij')
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=0).reshape(3, -1)

    dirs_cam = K_inv @ pixels
    dirs_world = R @ dirs_cam
    dirs_world = dirs_world / np.clip(np.linalg.norm(dirs_world, axis=0, keepdims=True), a_min=1e-8, a_max=None)

    c_local = (cam_center - local_origin).astype(np.float64)
    moments = np.cross(
        np.broadcast_to(c_local[None, :], (dirs_world.shape[1], 3)),
        dirs_world.T,
    ).T

    bev_extent_m = float(sat_w) * float(sat_m_per_px) / 2.0
    plucker = np.concatenate(
        [dirs_world, moments / (bev_extent_m + 1e-8)],
        axis=0,
    ).reshape(6, height, width)
    return torch.from_numpy(plucker).to(torch.float32).contiguous()


def _sample_random_signed_yaw(
    yaw_min_abs: float,
    yaw_max_abs: float,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Sample yaw from [-yaw_max_abs, -yaw_min_abs] U [yaw_min_abs, yaw_max_abs]."""
    yaw_min_abs = float(abs(yaw_min_abs))
    yaw_max_abs = float(abs(yaw_max_abs))
    if yaw_max_abs < yaw_min_abs:
        yaw_min_abs, yaw_max_abs = yaw_max_abs, yaw_min_abs
    rng = rng or np.random
    mag = rng.uniform(yaw_min_abs, yaw_max_abs)
    sign = 1.0 if rng.rand() > 0.5 else -1.0
    return float(mag * sign)


def _sample_uniform_yaw(
    yaw_min_deg: float,
    yaw_max_deg: float,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    low = float(min(yaw_min_deg, yaw_max_deg))
    high = float(max(yaw_min_deg, yaw_max_deg))
    rng = rng or np.random
    return float(rng.uniform(low, high))




def fisheye_to_virtual_perspective(
    img_bgr: np.ndarray,
    calib_yaml: Path,
    fisheye_relative_yaw_deg: float = 0.0,
    hfov_deg: float = 100.0,
    out_w: int = 640,
    out_h: int = 256,
    R_rectify: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate a virtual perspective view from fisheye.

    Yaw is defined relative to the *physical fisheye optical axis*, and rotation is applied
    around the physical fisheye camera Y axis (same convention as debug_ipm_alignment.py).

    If R_rectify is provided, it is used directly (phys->virt). Otherwise we compute it as
    rot_y(-fisheye_relative_yaw_deg).

    Returns:
        perspective BGR image (out_h, out_w, 3)
    """
    xi, K, D = _load_mei_yaml(calib_yaml)
    newK = _make_newK(out_w, out_h, hfov_deg)

    if R_rectify is None:
        R_rectify = _rot_y(-float(fisheye_relative_yaw_deg))

    map1, map2 = cv2.omnidir.initUndistortRectifyMap(
        K,
        D,
        np.array([xi]),
        R_rectify,
        newK,
        (out_w, out_h),
        cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE,
    )
    view = cv2.remap(img_bgr, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return view


@dataclass
class SampleIndex:
    drive_dir: Path
    frame_id: int
    meta: Optional[Dict[str, Any]] = None


class Kitti360dDataset(Dataset):
    """KITTI-360 drive-folder dataset.

    It scans the given drive folder(s) and produces samples (drive, frame).

    Modes:
      - front: read image_00 perspective directly.
      - fisheye_virtual: read image_02 or image_03 fisheye and synthesize a virtual perspective.

    View sets:
      - single: one sample per frame using the requested mode.
      - fixed5: expand each frame to five fixed views:
        front, left_forward_45, left_side, right_forward_45, right_side.
      - front_plus_random: expand each frame to front plus one random virtual fisheye view.

    Returned dict keys:
      - image: torch.float32 (3,H,W) in [0,1]
      - frame_id: int
      - drive: str
      - meta: dict (contains K after any resize/crop)

    Notes about intrinsics:
      - For mode="front", we parse calibration/perspective.txt to get original intrinsics and
        update them according to the image resizing/cropping applied in the dataloader.
      - For mode="fisheye_virtual", we return the virtual camera K (newK).
    """

    def __init__(
        self,
        drives: Union[str, Path, List[Union[str, Path]]],
        frames: Optional[Union[List[int], List[List[int]]]] = None,
        exclude_frames: Optional[List[int]] = None,
        require_exact_pose: bool = False,
        mode: str = "front",
        yaw_mode: str = "fisheye_relative",
        # Camera selection: if None, will randomly pick between image_02/image_03
        fisheye_camera: Optional[str] = None,
        # Yaw is now relative to the selected fisheye's optical axis
        fisheye_relative_yaw_deg: float = 0.0,
        vehicle_relative_yaw_deg: Optional[float] = None,
        virtual_hfov_deg: float = 100.0,
        # Random yaw sampling (relative to fisheye optical axis)
        random_fisheye_relative_yaw: bool = False,
        yaw_min_abs: float = 0.0,  # Now relative to fisheye optical axis
        yaw_max_abs: float = 90.0,  # Reasonable max yaw relative to fisheye
        random_vehicle_relative_yaw: bool = False,
        vehicle_yaw_min_deg: float = 60.0,
        vehicle_yaw_max_deg: float = 120.0,
        # IPM correction angles (degrees)
        roll_deg: float = 0.0,  # Roll correction for IPM
        pitch_deg: float = 0.0,  # Pitch correction for IPM

        # Reproducible per-item randomness (e.g. yaw sampling)
        seed: Optional[int] = None,
        view_set: str = "single",
        virtual_size: Tuple[int, int] = (640, 256),
        front_resize: Optional[Tuple[int, int]] = (640, 256),
        front_center_crop: Optional[Tuple[int, int]] = None,
        return_bgr: bool = False,
    ):
        self.drives: List[Path] = [Path(drives)] if not isinstance(drives, list) else [Path(x) for x in drives]
        self.mode = str(mode)
        self.yaw_mode = str(yaw_mode)
        self.exclude_frames = set(exclude_frames) if exclude_frames else set()

        # Camera selection: None means randomly pick between image_02/image_03
        self.fisheye_camera = fisheye_camera

        # Yaw is now relative to the selected fisheye's optical axis
        self.fisheye_relative_yaw_deg = float(fisheye_relative_yaw_deg)
        self.vehicle_relative_yaw_deg = None if vehicle_relative_yaw_deg is None else float(vehicle_relative_yaw_deg)
        self.virtual_hfov_deg = float(virtual_hfov_deg)
        self.virtual_w = int(virtual_size[0])
        self.virtual_h = int(virtual_size[1])

        # Random yaw settings (relative to fisheye optical axis)
        self.random_fisheye_relative_yaw = bool(random_fisheye_relative_yaw)
        self.yaw_min_abs = float(yaw_min_abs)
        self.yaw_max_abs = float(yaw_max_abs)
        self.random_vehicle_relative_yaw = bool(random_vehicle_relative_yaw)
        self.vehicle_yaw_min_deg = float(vehicle_yaw_min_deg)
        self.vehicle_yaw_max_deg = float(vehicle_yaw_max_deg)
        self.roll_deg = float(roll_deg)
        self.pitch_deg = float(pitch_deg)

        # For reproducible randomness
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None
        self.epoch = 0  # For DDP safety
        self.view_set = str(view_set)

        self.front_resize = front_resize  # (W,H) if not None
        self.front_center_crop = front_center_crop  # (W,H) if not None

        # Satellite image properties
        self.sat_m_per_px: float = 0.2  # meters per pixel in satellite image
        self.sat_size: Tuple[int, int] = (512, 512)  # (W,H), as stored on disk
        self.require_exact_pose = bool(require_exact_pose)

        self.return_bgr = return_bgr

        # cache cam->pose extrinsics per drive_dir
        self._cam_to_pose_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Optional cache for poses (frame-indexed)
        # - poses.txt: imu->world (may be sparse)
        # - cam0_to_world.txt: rectified cam0->world
        self._imu_to_world_cache: Dict[str, Dict[int, np.ndarray]] = {}
        self._cam0_to_world_cache: Dict[str, Dict[int, np.ndarray]] = {}
        self._front_bev_xy_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        if self.mode not in {"front", "fisheye_virtual"}:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.yaw_mode not in {"fisheye_relative", "vehicle_relative"}:
            raise ValueError(f"Unknown yaw_mode: {self.yaw_mode}")
        if self.view_set not in {"single", "fixed5", "front_plus_random"}:
            raise ValueError(f"Unknown view_set: {self.view_set}")
        if self.view_set in {"fixed5", "front_plus_random"} and self.mode != "fisheye_virtual":
            raise ValueError(f"view_set='{self.view_set}' requires mode='fisheye_virtual'")
        # fisheye_camera can be auto-selected based on yaw; only validate when explicitly set
        if self.fisheye_camera is not None and self.fisheye_camera not in {"image_02", "image_03"}:
            raise ValueError(f"fisheye_camera must be image_02 or image_03, got {self.fisheye_camera}")

        # Support both single frame list and per-drive frame lists
        if frames is not None and len(frames) > 0 and isinstance(frames[0], list):
            # frames is List[List[int]] - one list per drive
            frames_per_drive = frames
        elif frames is not None:
            # frames is List[int] - same frames for all drives (backward compatibility)
            frames_per_drive = [frames] * len(self.drives)
        else:
            # No frames specified - will auto-discover
            frames_per_drive = [None] * len(self.drives)

        self.samples: List[SampleIndex] = []
        for d, frame_list in zip(self.drives, frames_per_drive):
            if frame_list is None:
                # discover frames from image_00 by default, fall back to fisheye
                probe = d / "image_00" / "data_rgb"
                if not probe.exists():
                    probe_cam = self.fisheye_camera or "image_02"
                    probe = d / probe_cam / "data_rgb"
                ids = sorted([int(p.stem) for p in probe.glob("*.png")])
            else:
                ids = list(frame_list)
            # Filter out excluded frames
            if self.exclude_frames:
                ids = [fid for fid in ids if fid not in self.exclude_frames]
            for fid in ids:
                if self.view_set == "fixed5":
                    for spec in FIXED_FIVE_VIEW_SPECS:
                        self.samples.append(
                            SampleIndex(
                                drive_dir=d,
                                frame_id=int(fid),
                                meta=dict(spec),
                            )
                        )
                elif self.view_set == "front_plus_random":
                    self.samples.append(
                        SampleIndex(
                            drive_dir=d,
                            frame_id=int(fid),
                            meta={
                                "view_name": "front",
                                "mode_override": "front",
                            },
                        )
                    )
                    self.samples.append(
                        SampleIndex(
                            drive_dir=d,
                            frame_id=int(fid),
                            meta={
                                "view_name": "random_side",
                                "mode_override": "fisheye_virtual",
                            },
                        )
                    )
                else:
                    self.samples.append(SampleIndex(drive_dir=d, frame_id=int(fid)))

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_sample_mode(self, sample: SampleIndex) -> str:
        mode = self.mode
        if isinstance(sample.meta, dict):
            mode = str(sample.meta.get("mode_override", mode))
        if mode not in {"front", "fisheye_virtual"}:
            raise ValueError(f"Unknown sample mode: {mode}")
        return mode

    @staticmethod
    def _resolve_sample_view_name(sample: SampleIndex) -> Optional[str]:
        if isinstance(sample.meta, dict):
            view_name = sample.meta.get("view_name")
            if view_name is not None:
                return str(view_name)
        return None

    def _read_front(self, drive_dir: Path, frame_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read front image_00 and return (img_bgr, K_original).

        For KITTI-360 "rectified" front images (e.g. 1408x376), the correct intrinsics
        correspond to P_rect_00 (not K_00).

        We convert P_rect_00 (3x4) to a 3x3 K by taking the left 3x3 block.
        If the file is missing, returns None.
        """
        # KITTI-360 front rectified images can be under data_rect or data_rgb
        p_rect = drive_dir / "image_00" / "data_rect" / f"{frame_id:010d}.png"
        p_rgb = drive_dir / "image_00" / "data_rgb" / f"{frame_id:010d}.png"

        img = cv2.imread(str(p_rect), cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to data_rgb if data_rect is not found
            img = cv2.imread(str(p_rgb), cv2.IMREAD_COLOR)

        K0 = None
        calib_path = drive_dir / "calibration" / "perspective.txt"
        if calib_path.exists():
            try:
                calib = _load_perspective_calib(calib_path)

                # Prefer rectified projection (matches 1408x376)
                if "P_rect_00" in calib:
                    P = calib["P_rect_00"].astype(np.float64)
                    K0 = P[:, :3].copy()
                elif "K_00" in calib:
                    # fallback
                    K0 = calib["K_00"].astype(np.float64)
            except Exception:
                K0 = None

        return img, K0

    def _apply_resize_center_crop(
        self,
        img_bgr: np.ndarray,
        K: Optional[np.ndarray],
        resize_wh: Optional[Tuple[int, int]],
        crop_wh: Optional[Tuple[int, int]],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply optional resize and center crop, and update intrinsics.

        resize_wh/crop_wh are (W,H).
        """
        meta: Dict[str, Any] = {
            "orig_size": (int(img_bgr.shape[1]), int(img_bgr.shape[0])),
            "resize": None,
            "crop": None,
        }

        out = img_bgr
        K_new = None if K is None else K.copy()

        # Resize
        if resize_wh is not None:
            rw, rh = int(resize_wh[0]), int(resize_wh[1])
            h0, w0 = out.shape[0], out.shape[1]
            sx, sy = rw / float(w0), rh / float(h0)
            out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_LINEAR)
            meta["resize"] = (rw, rh)

            if K_new is not None:
                K_new[0, 0] *= sx  # fx
                K_new[1, 1] *= sy  # fy
                K_new[0, 2] *= sx  # cx
                K_new[1, 2] *= sy  # cy

        # Center crop
        if crop_wh is not None:
            cw, ch = int(crop_wh[0]), int(crop_wh[1])
            h, w = out.shape[0], out.shape[1]
            if cw > w or ch > h:
                raise ValueError(f"center_crop {crop_wh} larger than image {(w,h)}")
            x0 = int(round((w - cw) * 0.5))
            y0 = int(round((h - ch) * 0.5))
            out = out[y0 : y0 + ch, x0 : x0 + cw]
            meta["crop"] = {
                "type": "center",
                "xy0": (x0, y0),
                "size": (cw, ch),
            }

            if K_new is not None:
                K_new[0, 2] -= float(x0)
                K_new[1, 2] -= float(y0)

        meta["final_size"] = (int(out.shape[1]), int(out.shape[0]))
        return out, K_new, meta

    def _read_fisheye_virtual(
        self,
        drive_dir: Path,
        frame_id: int,
        *,
        fisheye_camera: str,
        fisheye_relative_yaw_deg: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[str]]:
        """Return (virtual_view_bgr, K_virtual, fisheye_relative_yaw_deg, camera_name_used)."""
        cam = fisheye_camera

        img_path_rgb = drive_dir / cam / "data_rgb" / f"{frame_id:010d}.png"
        img = cv2.imread(str(img_path_rgb), cv2.IMREAD_COLOR)
        if img is None:
            return None, None, None, None

        calib_yaml = drive_dir / "calibration" / f"{cam}.yaml"
        if not calib_yaml.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_yaml}")

        K_virtual = _make_newK(self.virtual_w, self.virtual_h, self.virtual_hfov_deg)

        view = fisheye_to_virtual_perspective(
            img_bgr=img,
            calib_yaml=calib_yaml,
            fisheye_relative_yaw_deg=float(fisheye_relative_yaw_deg),
            hfov_deg=self.virtual_hfov_deg,
            out_w=self.virtual_w,
            out_h=self.virtual_h,
        )
        return view, K_virtual, float(fisheye_relative_yaw_deg), cam

    def _read_satellite(self, drive_dir: Path, frame_id: int) -> Tuple[np.ndarray, bool]:
        """Read satellite BEV image (north-up, centered at vehicle position).

        Returns:
            - RGB image as numpy array (H, W, 3) in [0, 255]
            - Boolean indicating if the satellite image was found and loaded.
        """
        # Try both .png and .jpg extensions
        sat_dir = drive_dir / "satellite"
        sat_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = sat_dir / f"{frame_id:010d}{ext}"
            if candidate.is_file():
                sat_path = candidate
                break

        if sat_path is None:
            # Silently fail if file doesn't exist, as this is common.
            return np.zeros((self.sat_size[1], self.sat_size[0], 3), dtype=np.uint8), False

        try:
            img = Image.open(str(sat_path)).convert('RGB')
            return np.array(img), True
        except PermissionError:
            print(f"[Error] PermissionError: Cannot access {sat_path.resolve()}")
            return np.zeros((self.sat_size[1], self.sat_size[0], 3), dtype=np.uint8), False
        except Exception as e:
            print(f"[Error] Unexpected error reading {sat_path.resolve()}: {e}")
            return np.zeros((self.sat_size[1], self.sat_size[0], 3), dtype=np.uint8), False

    def _get_dummy_sample(self, s) -> Dict:
        """Return a zeroed-out sample dict for frames that failed to load."""
        sample_mode = self._resolve_sample_mode(s)
        view_name = self._resolve_sample_view_name(s)
        meta = s.meta if isinstance(s.meta, dict) else {}
        fisheye_camera_used = meta.get("fisheye_camera_override")
        if fisheye_camera_used not in FISHEYE_CAMERAS:
            fisheye_camera_used = self.fisheye_camera
        fisheye_relative_yaw_used = meta.get("fisheye_relative_yaw_deg_override", self.fisheye_relative_yaw_deg)
        vehicle_yaw_used = meta.get("vehicle_relative_yaw_deg_override", self.vehicle_relative_yaw_deg)
        if sample_mode == "front":
            if self.front_resize is not None:
                w, h = self.front_resize
            else:
                w, h = (1408, 376)
            physical_camera = "image_00"
        else:
            h, w = self.virtual_h, self.virtual_w
            physical_camera = fisheye_camera_used
        camera_height_m = _get_camera_height_m(physical_camera)
        front_bev_xy = self._get_front_bev_xy(int(h), int(w))
        return {
            "image": torch.zeros(3, h, w, dtype=torch.float32),
            "sat": torch.zeros(3, self.sat_size[1], self.sat_size[0], dtype=torch.float32),
            "sat_available": False,
            "sat_m_per_px": self.sat_m_per_px,
            "camera_height_m": camera_height_m,
            "front_bev_xy": front_bev_xy,
            "coords_map": front_bev_xy,
            "plucker_map": torch.zeros(6, h, w, dtype=torch.float32),
            "K": torch.eye(3, dtype=torch.float32),
            "T_pose_cam": torch.eye(4, dtype=torch.float32),
            "T_imu_to_world": torch.eye(4, dtype=torch.float32),
            "T_cam0_to_world": torch.eye(4, dtype=torch.float32),
            "T_cam_to_world": torch.eye(4, dtype=torch.float32),
            "frame_id": s.frame_id,
            "drive": s.drive_dir.name,
            "meta": {
                "mode": sample_mode,
                "requested_mode": self.mode,
                "view_set": self.view_set,
                "view_name": view_name,
                "yaw_mode": self.yaw_mode,
                "fisheye_camera": fisheye_camera_used,
                "fisheye_camera_used": fisheye_camera_used if sample_mode != "front" else None,
                "fisheye_relative_yaw_deg": fisheye_relative_yaw_used if sample_mode != "front" else None,
                "fisheye_relative_yaw_deg_used": fisheye_relative_yaw_used if sample_mode != "front" else None,
                "vehicle_relative_yaw_deg": vehicle_yaw_used if sample_mode != "front" else None,
                "vehicle_yaw_deg_used": vehicle_yaw_used if sample_mode != "front" else None,
                "virtual_hfov_deg": self.virtual_hfov_deg,
                "physical_camera": physical_camera,
                "camera_height_m": camera_height_m,
                "oxts_yaw": None,
                "drive_dir": str(s.drive_dir),
                "aug": {},
                "dummy": True,
            },
        }

    def _get_front_bev_xy(self, height: int, width: int) -> torch.Tensor:
        """Return normalized front BEV XY grid as (2, H, W)."""
        key = (int(height), int(width))
        cached = self._front_bev_xy_cache.get(key)
        if cached is not None:
            return cached

        # Pixel-center grid, normalized to [-1, 1] with ego-centric convention.
        # x: right-positive, y: up-positive.
        u = torch.arange(width, dtype=torch.float32) + 0.5
        v = torch.arange(height, dtype=torch.float32) + 0.5
        vv, uu = torch.meshgrid(v, u, indexing='ij')

        x = (uu - (width / 2.0)) / (width / 2.0)
        y = ((height / 2.0) - vv) / (height / 2.0)
        front_bev_xy = torch.stack([x, y], dim=0).contiguous()

        self._front_bev_xy_cache[key] = front_bev_xy
        return front_bev_xy

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        drive_dir, frame_id = s.drive_dir, s.frame_id
        sample_mode = self._resolve_sample_mode(s)
        view_name = self._resolve_sample_view_name(s)

        # Optional deterministic overrides (provided by an outer wrapper dataset).
        # This avoids relying on internal RNG, and is DDP/worker safe.
        fisheye_relative_yaw_override_deg = None
        vehicle_relative_yaw_override_deg = None
        fisheye_camera_override = None
        if hasattr(s, "meta") and isinstance(getattr(s, "meta"), dict):
            fisheye_relative_yaw_override_deg = s.meta.get("fisheye_relative_yaw_deg_override")
            vehicle_relative_yaw_override_deg = s.meta.get("vehicle_relative_yaw_deg_override")
            fisheye_camera_override = s.meta.get("fisheye_camera_override")

        fisheye_camera_item: Optional[str] = None
        vehicle_yaw_deg_item: Optional[float] = None
        fisheye_relative_yaw_deg_item: Optional[float] = None

        if sample_mode != "front":
            explicit_fisheye_camera = None
            if fisheye_camera_override in FISHEYE_CAMERAS:
                explicit_fisheye_camera = str(fisheye_camera_override)
            elif self.fisheye_camera in FISHEYE_CAMERAS:
                explicit_fisheye_camera = str(self.fisheye_camera)

            if self.yaw_mode == "vehicle_relative":
                if vehicle_relative_yaw_override_deg is not None:
                    vehicle_yaw_deg_item = float(vehicle_relative_yaw_override_deg)
                elif self.random_vehicle_relative_yaw:
                    rng = self.rng if self.rng is not None else None
                    vehicle_yaw_deg_item = _sample_random_signed_yaw(
                        self.vehicle_yaw_min_deg,
                        self.vehicle_yaw_max_deg,
                        rng=rng,
                    )
                elif self.vehicle_relative_yaw_deg is not None:
                    vehicle_yaw_deg_item = float(self.vehicle_relative_yaw_deg)
                else:
                    base_camera = explicit_fisheye_camera or _choose_nearest_fisheye_camera(0.0)
                    vehicle_yaw_deg_item = _wrap_angle_deg(MOUNT_ANGLES[base_camera] + self.fisheye_relative_yaw_deg)

                fisheye_camera_item = explicit_fisheye_camera or _choose_nearest_fisheye_camera(vehicle_yaw_deg_item)
                fisheye_relative_yaw_deg_item = _wrap_angle_deg(vehicle_yaw_deg_item - MOUNT_ANGLES[fisheye_camera_item])
            else:
                fisheye_camera_item = explicit_fisheye_camera
                if fisheye_camera_item is None:
                    if self.rng is not None:
                        fisheye_camera_item = str(self.rng.choice(FISHEYE_CAMERAS))
                    else:
                        fisheye_camera_item = str(np.random.choice(FISHEYE_CAMERAS))

                sampled_fisheye_relative_yaw_deg = self.fisheye_relative_yaw_deg
                if fisheye_relative_yaw_override_deg is not None:
                    sampled_fisheye_relative_yaw_deg = float(fisheye_relative_yaw_override_deg)
                elif self.random_fisheye_relative_yaw:
                    rng = self.rng if self.rng is not None else None
                    sampled_fisheye_relative_yaw_deg = _sample_random_signed_yaw(
                        self.yaw_min_abs,
                        self.yaw_max_abs,
                        rng=rng,
                    )

                fisheye_relative_yaw_deg_item = float(sampled_fisheye_relative_yaw_deg)
                vehicle_yaw_deg_item = _wrap_angle_deg(
                    MOUNT_ANGLES[fisheye_camera_item] + fisheye_relative_yaw_deg_item
                )

        # Load satellite BEV image
        sat_rgb, sat_available = self._read_satellite(drive_dir, frame_id)
        sat_t = torch.from_numpy(sat_rgb).to(torch.float32).permute(2, 0, 1) / 255.0

        # Load extrinsics (T_pose_cam) and cache them
        drive_dir_str = str(drive_dir)
        if drive_dir_str not in self._cam_to_pose_cache:
            calib_path = drive_dir / "calibration" / "calib_cam_to_pose.txt"
            if not calib_path.exists():
                raise FileNotFoundError(f"Extrinsics file not found: {calib_path}")
            self._cam_to_pose_cache[drive_dir_str] = _load_cam_to_pose(calib_path)

        drive_dir_str = str(drive_dir)
        if drive_dir_str not in self._imu_to_world_cache:
            self._imu_to_world_cache[drive_dir_str] = _load_indexed_poses_txt(drive_dir / "poses.txt", mat_size=12)
        if drive_dir_str not in self._cam0_to_world_cache:
            self._cam0_to_world_cache[drive_dir_str] = _load_indexed_poses_txt(drive_dir / "cam0_to_world.txt", mat_size=16)

        if self.require_exact_pose:
            T_imu_to_world = self._imu_to_world_cache[drive_dir_str].get(frame_id)
            T_cam0_to_world = self._cam0_to_world_cache[drive_dir_str].get(frame_id)
        else:
            T_imu_to_world = _nearest_pose(frame_id, self._imu_to_world_cache[drive_dir_str])
            T_cam0_to_world = _nearest_pose(frame_id, self._cam0_to_world_cache[drive_dir_str])

        T_imu_to_world_t = None if T_imu_to_world is None else torch.from_numpy(T_imu_to_world).to(torch.float32)
        T_cam0_to_world_t = None if T_cam0_to_world is None else torch.from_numpy(T_cam0_to_world).to(torch.float32)
        all_extrinsics = self._cam_to_pose_cache[drive_dir_str]

        K: Optional[np.ndarray] = None
        T_pose_cam: Optional[np.ndarray] = None
        aug_meta: Dict[str, Any] = {}

        if sample_mode == "front":
            img_bgr, K0 = self._read_front(drive_dir, frame_id)
            img_bgr, K, aug_meta = self._apply_resize_center_crop(
                img_bgr,
                K0,
                resize_wh=self.front_resize,
                crop_wh=self.front_center_crop,
            )
            T_pose_cam = all_extrinsics.get("image_00")

            # Prefer user-provided per-frame cam0->world if available
            if T_cam0_to_world is not None:
                T_pose_cam = T_cam0_to_world

        else:
            cam_used = fisheye_camera_item
            img_bgr, K_virtual, fisheye_yaw_used, cam_used = self._read_fisheye_virtual(
                drive_dir,
                frame_id,
                fisheye_camera=cam_used,
                fisheye_relative_yaw_deg=fisheye_relative_yaw_deg_item,
            )

            if img_bgr is None:
                return self._get_dummy_sample(s)

            K = K_virtual
            aug_meta = {
                "orig_size": (self.virtual_w, self.virtual_h),
                "resize": None,
                "crop": None,
                "final_size": (self.virtual_w, self.virtual_h),
                "fisheye_relative_yaw_deg": float(fisheye_yaw_used),
                "vehicle_yaw_deg": float(vehicle_yaw_deg_item) if vehicle_yaw_deg_item is not None else None,
            }

            # remember which camera was used for this sample
            aug_meta["fisheye_camera_used"] = cam_used

            T_pose_cam = all_extrinsics.get(cam_used)

        # BGR->RGB unless requested
        if img_bgr is not None:
            if self.return_bgr:
                img = img_bgr
            else:
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Handle cases where image loading failed
            h, w = self.front_resize if sample_mode == 'front' else (self.virtual_h, self.virtual_w)
            img = np.zeros((h, w, 3), dtype=np.uint8)

        img_t = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1) / 255.0

        # Keep oxts yaw available if needed later
        oxts_path = drive_dir / "oxts" / "data" / f"{frame_id:010d}.txt"
        yaw = None
        if oxts_path.exists():
            try:
                yaw = _read_oxts_yaw(oxts_path)
            except Exception:
                yaw = None

        K_t = None if K is None else torch.from_numpy(K).to(torch.float32)
        T_cam_to_world: Optional[np.ndarray] = None
        physical_camera = "image_00" if sample_mode == "front" else aug_meta.get("fisheye_camera_used", self.fisheye_camera)
        camera_height_m = _get_camera_height_m(physical_camera)

        # Compose per-frame camera->world when possible.
        # calib_cam_to_pose.txt provides T_imu_cam (camera->IMU/GPS).
        # poses.txt provides T_world_imu (IMU->world).
        # Therefore for the *physical camera*: T_world_cam = T_world_imu @ T_imu_cam.
        if T_imu_to_world is not None and T_pose_cam is not None:
            if T_cam0_to_world is not None and T_pose_cam.shape == (4, 4) and np.allclose(T_pose_cam, T_cam0_to_world):
                T_cam_to_world = T_pose_cam
            else:
                T_fisheye_to_world = T_imu_to_world @ T_pose_cam

                # If we are generating a virtual perspective from fisheye, we must rotate the
                # *camera frame* by the same R used in fisheye_to_virtual_perspective.
                if sample_mode == "fisheye_virtual":
                    cam_used = aug_meta.get("fisheye_camera_used")
                    fisheye_yaw_used = float(aug_meta.get("fisheye_relative_yaw_deg", 0.0))

                    # This logic now matches debug_ipm_alignment.py (without gravity alignment)
                    # R_rectify is the rotation from physical fisheye to virtual camera.
                    R_rectify = _rot_y(-fisheye_yaw_used)

                    # The virtual camera's pose in the world is the physical camera's pose
                    # followed by the local rotation.
                    T_phys_to_world = T_fisheye_to_world
                    R_phys_to_world = T_phys_to_world[:3, :3]
                    R_virt_to_phys = R_rectify.T # virt -> phys
                    R_virt_to_world = R_phys_to_world @ R_virt_to_phys

                    # Apply roll and pitch corrections if specified
                    if self.roll_deg != 0.0 or self.pitch_deg != 0.0:
                        R_roll = _rot_z(self.roll_deg)
                        R_pitch = _rot_x(self.pitch_deg)
                        R_correction = R_pitch @ R_roll  # Applied in the virtual camera's local frame
                        R_virt_to_world = R_virt_to_world @ R_correction

                    T_cam_to_world = np.eye(4, dtype=np.float64)
                    T_cam_to_world[:3, :3] = R_virt_to_world
                    T_cam_to_world[:3, 3] = T_phys_to_world[:3, 3]

                else:
                    T_cam_to_world = T_fisheye_to_world

        elif T_cam0_to_world is not None and sample_mode == "front":
            # Fallback: front camera can still use cam0_to_world if provided
            T_cam_to_world = T_cam0_to_world
        T_pose_cam_t = None if T_pose_cam is None else torch.from_numpy(T_pose_cam).to(torch.float32)
        T_cam_to_world_t = None if T_cam_to_world is None else torch.from_numpy(T_cam_to_world).to(torch.float32)

        # If T_cam_to_world could not be computed, the sample is invalid.
        if T_cam_to_world_t is None:
            return self._get_dummy_sample(s)

        if K is not None and T_cam_to_world is not None:
            front_bev_xy = compute_camera_bev_xy(
                K=K,
                T_cam_to_world=T_cam_to_world,
                T_imu_to_world=T_imu_to_world,
                height=int(img.shape[0]),
                width=int(img.shape[1]),
                sat_m_per_px=self.sat_m_per_px,
                sat_w=self.sat_size[0],
                sat_h=self.sat_size[1],
                cam_height=camera_height_m,
            )
            plucker_map = compute_plucker_map(
                K=K,
                T_cam_to_world=T_cam_to_world,
                T_imu_to_world=T_imu_to_world,
                height=int(img.shape[0]),
                width=int(img.shape[1]),
                sat_m_per_px=self.sat_m_per_px,
                sat_w=self.sat_size[0],
            )
        else:
            front_bev_xy = self._get_front_bev_xy(int(img.shape[0]), int(img.shape[1]))
            plucker_map = torch.zeros(6, int(img.shape[0]), int(img.shape[1]), dtype=torch.float32)

        return {
            "image": img_t,
            "sat": sat_t,  # (3,512,512) north-up satellite BEV in [0,1]
            "sat_available": sat_available,
            "sat_m_per_px": self.sat_m_per_px,
            "camera_height_m": camera_height_m,
            "front_bev_xy": front_bev_xy,
            "coords_map": front_bev_xy,
            "plucker_map": plucker_map,
            "K": K_t,  # (3,3) after resize/crop, or virtual K in fisheye mode
            "T_pose_cam": T_pose_cam_t,  # (4,4) camera pose (prefer cam->world if available)
            "T_imu_to_world": T_imu_to_world_t,  # (4,4) imu->world, may be None if missing
            "T_cam0_to_world": T_cam0_to_world_t,  # (4,4) cam0(rectified)->world, may be None
            "T_cam_to_world": T_cam_to_world_t,
            "frame_id": frame_id,
            "drive": drive_dir.name,
            "meta": {
                "mode": sample_mode,
                "requested_mode": self.mode,
                "view_set": self.view_set,
                "view_name": view_name,
                "yaw_mode": self.yaw_mode,
                "fisheye_camera": aug_meta.get("fisheye_camera_used", self.fisheye_camera),
                "fisheye_camera_used": aug_meta.get("fisheye_camera_used", self.fisheye_camera),
                "physical_camera": physical_camera,
                "fisheye_relative_yaw_deg": aug_meta.get("fisheye_relative_yaw_deg"),
                "fisheye_relative_yaw_deg_used": aug_meta.get("fisheye_relative_yaw_deg"),
                "vehicle_relative_yaw_deg": aug_meta.get("vehicle_yaw_deg"),
                "vehicle_yaw_deg_used": aug_meta.get("vehicle_yaw_deg"),
                "virtual_hfov_deg": self.virtual_hfov_deg,
                "camera_height_m": camera_height_m,
                "oxts_yaw": yaw,
                "drive_dir": str(drive_dir),
                "aug": aug_meta,
            },
        }
