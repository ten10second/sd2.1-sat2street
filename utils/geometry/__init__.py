"""Geometry utilities for KITTI-360 coordinate transformations and encoding."""

from .kitti360_transforms import (
    compose_camera_to_satellite_transform,
    compose_camera_to_world_transform,
    get_world_to_satellite_transform,
    invert_se3,
    load_cam0_to_world_pose,
    load_indexed_pose_txt,
    load_imu_to_world_pose,
    load_kitti360_cam_to_pose_calib,
    nearest_pose,
)

from .pose_encoding import (
    PoseEncoder,
    rotation_matrix_to_6d,
    rotation_matrix_to_quaternion,
)

from .homography import (
    compute_homography_ground_plane,
    compute_homography_from_transform,
    extract_R_T_from_transform,
)

try:
    from .camera_to_sat_projection import (
        project_camera_to_satellite,
    )
except ModuleNotFoundError:
    project_camera_to_satellite = None

from .differentiable_projection import (
    differentiable_camera_to_sat_warp,
)

# New: camera-to-camera ground-plane warp for pseudo-GT
from .camera_to_camera_ground import (
    camera_to_camera_groundplane_pull,
    apply_yaw_rotation_to_pose,
)

__all__ = [
    # KITTI-360 transforms
    'compose_camera_to_satellite_transform',
    'compose_camera_to_world_transform',
    'get_world_to_satellite_transform',
    'invert_se3',
    'load_cam0_to_world_pose',
    'load_indexed_pose_txt',
    'load_imu_to_world_pose',
    'load_kitti360_cam_to_pose_calib',
    'nearest_pose',
    # Pose encoding
    'PoseEncoder',
    'rotation_matrix_to_6d',
    'rotation_matrix_to_quaternion',
    # Homography
    'compute_homography_ground_plane',
    'compute_homography_from_transform',
    'extract_R_T_from_transform',
    # Camera to satellite projection
    'differentiable_camera_to_sat_warp',
    # Camera-to-camera ground-plane
    'camera_to_camera_groundplane_pull',
    'apply_yaw_rotation_to_pose',
]

if project_camera_to_satellite is not None:
    __all__.append('project_camera_to_satellite')
