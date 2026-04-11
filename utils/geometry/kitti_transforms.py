"""
KITTI coordinate transformation utilities.

This module provides functions for loading KITTI calibration files and computing
coordinate transformations between different reference frames:
- Camera (C): X=right, Y=down, Z=forward
- Lidar/Velodyne (V): X=forward, Y=left, Z=up
- IMU (I): X=forward, Y=left, Z=up
- World/UTM (W): X=east, Y=north, Z=up
- Satellite Image (W'): col=east, row=north (north-up)
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional


def _read_calib_file(path: str) -> dict:
    """Read KITTI calibration file and parse key-value pairs."""
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.strip().split(':', 1)
            v = v.strip()
            try:
                nums = [float(x) for x in v.split()] if v else []
            except Exception:
                nums = []
            data[k] = np.array(nums, dtype=np.float64)
    return data


def load_kitti_calib(calib_dir: str, cam: str = 'P2') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load KITTI calibration files with proper R_rect correction.

    Args:
        calib_dir: Path to calibration directory (e.g., '2011_09_26_calib')
        cam: Camera projection matrix to load (default: 'P2' for left color camera)

    Returns:
        K: 3x3 camera intrinsic matrix
        T_velo_to_cam: 4x4 transformation matrix from Lidar to Camera (with R_rect applied)

    Note:
        The transform chain is: T_velo_to_cam = R_rect @ T_velo_to_cam0
        where R_rect is the rectification matrix for the target camera.
    """
    cam2cam = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
    velo2cam = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
    if not os.path.isfile(cam2cam) or not os.path.isfile(velo2cam):
        raise FileNotFoundError(f"Expected KITTI calib files at {calib_dir}")

    c = _read_calib_file(cam2cam)
    v = _read_calib_file(velo2cam)

    # Determine camera index and get intrinsics
    cam_idx = None
    key = cam
    if key not in c:
        # Try P_rect_x format
        if key == 'P2' and 'P_rect_02' in c:
            key = 'P_rect_02'
            cam_idx = 2
        elif key == 'P0' and 'P_rect_00' in c:
            key = 'P_rect_00'
            cam_idx = 0
        elif key == 'P1' and 'P_rect_01' in c:
            key = 'P_rect_01'
            cam_idx = 1
        elif key == 'P3' and 'P_rect_03' in c:
            key = 'P_rect_03'
            cam_idx = 3
    else:
        # Extract camera index from key (e.g., 'P2' -> 2)
        if key.startswith('P') and len(key) == 2:
            cam_idx = int(key[1])

    P = c[key].reshape(3, 4)
    K = P[:3, :3]

    # Get Lidar to Camera0 (unrectified) transform
    if 'Tr_velo_to_cam' in v:
        Tr = v['Tr_velo_to_cam'].reshape(3, 4)
        T_velo_to_cam0 = np.eye(4, dtype=np.float64)
        T_velo_to_cam0[:3, :4] = Tr
    else:
        # Some variants store R and T separately
        R = v.get('R', np.eye(9, dtype=np.float64)).reshape(3, 3)
        t = v.get('T', np.zeros(3, dtype=np.float64)).reshape(3, 1)
        T_velo_to_cam0 = np.eye(4, dtype=np.float64)
        T_velo_to_cam0[:3, :3] = R
        T_velo_to_cam0[:3, 3:4] = t

    # Get R_rect_00 - the rectification matrix
    # CRITICAL: According to KITTI documentation, R_rect_00 is used for ALL cameras
    # The complete transform chain for Cam2 is:
    #   T_velo_to_cam2 = T_cam0_to_cam2 @ R_rect_00 @ T_velo_to_cam0
    # where T_cam0_to_cam2 accounts for the baseline between cameras
    if 'R_rect_00' in c:
        R_rect = c['R_rect_00'].reshape(3, 3)
        # Build 4x4 rectification matrix
        T_rect = np.eye(4, dtype=np.float64)
        T_rect[:3, :3] = R_rect

        # Get camera baseline offset from P_rect matrices
        # P_rect = [K | K @ t], so t = K^-1 @ P[:,3]
        P_rect_00_key = 'P_rect_00'
        P_rect_cam_key = f'P_rect_{cam_idx:02d}' if cam_idx is not None else None

        T_cam0_to_camX = np.eye(4, dtype=np.float64)
        if P_rect_00_key in c and P_rect_cam_key and P_rect_cam_key in c:
            P_rect_00 = c[P_rect_00_key].reshape(3, 4)
            P_rect_cam = c[P_rect_cam_key].reshape(3, 4)
            K_rect = P_rect_cam[:3, :3]
            K_rect_inv = np.linalg.inv(K_rect)
            t_cam0 = K_rect_inv @ P_rect_00[:, 3]
            t_camX = K_rect_inv @ P_rect_cam[:, 3]
            T_cam0_to_camX[:3, 3] = t_camX - t_cam0

        # Apply full transform: T_velo_to_cam = T_cam0_to_camX @ T_rect @ T_velo_to_cam0
        T_velo_to_cam = T_cam0_to_camX @ T_rect @ T_velo_to_cam0
    else:
        print(f"Warning: R_rect_00 not found in calib, using unrectified transform")
        T_velo_to_cam = T_velo_to_cam0

    K_t = torch.from_numpy(K.astype(np.float32))
    T_t = torch.from_numpy(T_velo_to_cam.astype(np.float32))
    return K_t, T_t


def load_imu_to_velo_calib(calib_dir: str) -> torch.Tensor:
    """
    Load KITTI IMU to Velodyne calibration.

    Args:
        calib_dir: Path to calibration directory (e.g., '2011_09_26_calib')

    Returns:
        T_imu_to_velo: 4x4 transformation matrix from IMU to Lidar
    """
    imu2velo = os.path.join(calib_dir, 'calib_imu_to_velo.txt')
    if not os.path.isfile(imu2velo):
        raise FileNotFoundError(f"Expected calib_imu_to_velo.txt at {calib_dir}")

    data = _read_calib_file(imu2velo)

    # Read R (3x3) and T (3x1)
    R = data.get('R', np.eye(9, dtype=np.float64)).reshape(3, 3)
    t = data.get('T', np.zeros(3, dtype=np.float64)).reshape(3, 1)

    # Build 4x4 transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t

    T_t = torch.from_numpy(T.astype(np.float32))
    return T_t


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
    """
    Convert latitude/longitude to UTM coordinates.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        easting: UTM easting in meters
        northing: UTM northing in meters
        zone_number: UTM zone number
        zone_letter: UTM zone letter
    """
    try:
        import utm
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return easting, northing, zone_number, zone_letter
    except ImportError:
        raise ImportError("Please install utm package: pip install utm")


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.

    KITTI convention:
    - Roll (α): rotation around x-axis (forward)
    - Pitch (β): rotation around y-axis (left)
    - Yaw (γ): rotation around z-axis (up)
    - Rotation order: Rz(yaw) * Ry(pitch) * Rx(roll)

    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians

    Returns:
        R: 3x3 rotation matrix (IMU → World frame)
    """
    # Roll (rotation around x-axis)
    cr, sr = np.cos(roll), np.sin(roll)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)

    # Pitch (rotation around y-axis)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)

    # Yaw (rotation around z-axis)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def parse_oxts_line(line: str) -> dict:
    """
    Parse a single line from KITTI OXTS file.

    OXTS format (30 values):
    0-2: lat, lon, alt (GPS position)
    3-5: roll, pitch, yaw (IMU orientation in radians)
    6-29: velocities, accelerations, angular rates, accuracy metrics

    Args:
        line: Single line from OXTS file

    Returns:
        dict with parsed GPS and orientation values
    """
    values = [float(x) for x in line.strip().split()]
    return {
        'lat': values[0],
        'lon': values[1],
        'alt': values[2],
        'roll': values[3],
        'pitch': values[4],
        'yaw': values[5],
    }


def load_oxts_pose(
    oxts_file: str,
    origin: Optional[Tuple[float, float, float]] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Load OXTS pose and compute T_imu_to_world transformation.

    Args:
        oxts_file: Path to OXTS .txt file
        origin: Optional (easting_0, northing_0, alt_0) origin for world frame.
                If None, uses the current frame as origin.

    Returns:
        T_imu_to_world: 4x4 transformation matrix (IMU → World)
        oxts_data: dict with parsed OXTS data including UTM coordinates
    """
    with open(oxts_file, 'r') as f:
        line = f.readline()

    oxts_data = parse_oxts_line(line)

    # Convert GPS to UTM
    easting, northing, zone_num, zone_letter = latlon_to_utm(
        oxts_data['lat'], oxts_data['lon']
    )

    # Set origin if not provided
    if origin is None:
        origin = (easting, northing, oxts_data['alt'])

    # Compute translation in world frame
    tx = easting - origin[0]
    ty = northing - origin[1]
    tz = oxts_data['alt'] - origin[2]

    # Compute rotation matrix from Euler angles
    R = euler_to_rotation_matrix(
        oxts_data['roll'], oxts_data['pitch'], oxts_data['yaw']
    )

    # Build 4x4 transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    T_t = torch.from_numpy(T.astype(np.float32))

    # Add UTM info to oxts_data
    oxts_data['easting'] = easting
    oxts_data['northing'] = northing
    oxts_data['zone_num'] = zone_num
    oxts_data['zone_letter'] = zone_letter
    oxts_data['origin'] = origin

    return T_t, oxts_data


def get_world_to_satellite_transform(
    sat_size: int = 512,
    resolution_m_per_px: float = 0.2,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Get transformation from World coordinates (UTM) to Satellite image coordinates.

    Satellite image coordinate system (north-up):
    - Origin at image center (sat_size/2, sat_size/2)
    - Row 0 = north edge, Row 511 = south edge
    - Col 0 = west edge, Col 511 = east edge

    World coordinate system (UTM):
    - X-axis points east (Easting)
    - Y-axis points north (Northing)
    - Origin at first frame GPS position

    Coordinate mapping:
    - World +X (east) → Satellite +col (right)
    - World +Y (north) → Satellite -row (up, because row increases downward)

    Args:
        sat_size: Satellite image size in pixels (default: 512)
        resolution_m_per_px: Satellite resolution in meters per pixel (default: 0.2)
        device: torch device

    Returns:
        T_world_to_sat: 4x4 transformation matrix (World → Satellite image coords)
    """
    # Scale factor: meters to pixels
    scale = 1.0 / resolution_m_per_px  # 5 pixels/meter

    # Translation: move origin to image center
    tx = sat_size / 2.0  # 256 pixels
    ty = sat_size / 2.0  # 256 pixels

    # Build transformation matrix
    # sat_col = scale * world_x + tx
    # sat_row = -scale * world_y + ty  (negative because row increases downward)
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[0, 0] = scale      # X: East → columns
    T[1, 1] = -scale     # Y: North → rows (negative)
    T[0, 3] = tx         # Center X
    T[1, 3] = ty         # Center Y

    return T


def compose_camera_to_satellite_transform(
    T_cam_to_velo: torch.Tensor,
    T_velo_to_imu: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    sat_size: int = 512,
    resolution_m_per_px: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compose full transformation from Camera to Satellite image coordinates.

    Transformation chain:
    T_cam_to_sat = T_world_to_sat @ T_imu_to_world @ T_velo_to_imu @ T_cam_to_velo

    Args:
        T_cam_to_velo: 4x4 Camera → Lidar transform
        T_velo_to_imu: 4x4 Lidar → IMU transform
        T_imu_to_world: 4x4 IMU → World transform
        sat_size: Satellite image size in pixels
        resolution_m_per_px: Satellite resolution in m/pixel

    Returns:
        T_cam_to_sat: 4x4 Camera → Satellite image transform
        T_world_to_sat: 4x4 World → Satellite image transform
    """
    device = T_cam_to_velo.device

    # Get World → Satellite transform
    T_world_to_sat = get_world_to_satellite_transform(
        sat_size, resolution_m_per_px, device
    )

    # Compose full chain: Camera → Lidar → IMU → World → Satellite
    T_cam_to_sat = T_world_to_sat @ T_imu_to_world @ T_velo_to_imu @ T_cam_to_velo

    return T_cam_to_sat, T_world_to_sat


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """
    Invert a 4x4 SE(3) transformation matrix.

    For SE(3) matrix T = [R t; 0 1], the inverse is:
    T^(-1) = [R^T -R^T*t; 0 1]

    Args:
        T: 4x4 transformation matrix

    Returns:
        T_inv: 4x4 inverted transformation matrix
    """
    assert T.shape[-2:] == (4, 4), f"Expected 4x4 matrix, got {T.shape}"

    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.t()
    t_inv = -R_inv @ t

    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv
