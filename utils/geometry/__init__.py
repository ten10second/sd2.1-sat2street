"""
Geometry utilities for coordinate transformations and calibration loading.
"""

from .kitti_transforms import (
    load_kitti_calib,
    load_imu_to_velo_calib,
    load_oxts_pose,
    get_world_to_satellite_transform,
    compose_camera_to_satellite_transform,
    invert_se3,
    latlon_to_utm,
    euler_to_rotation_matrix,
    parse_oxts_line,
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

from .bev_to_camera_warp import (
    warp_bev_to_camera,
    warp_bev_to_camera_with_coords,
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

from .pose_loss import (
    PoseLoss,
)

# New: camera-to-camera ground-plane warp for pseudo-GT
from .camera_to_camera_ground import (
    camera_to_camera_groundplane_pull,
    apply_yaw_rotation_to_pose,
)

__all__ = [
    # KITTI transforms
    'load_kitti_calib',
    'load_imu_to_velo_calib',
    'load_oxts_pose',
    'get_world_to_satellite_transform',
    'compose_camera_to_satellite_transform',
    'invert_se3',
    'latlon_to_utm',
    'euler_to_rotation_matrix',
    'parse_oxts_line',
    # Pose encoding
    'PoseEncoder',
    'rotation_matrix_to_6d',
    'rotation_matrix_to_quaternion',
    # Homography
    'compute_homography_ground_plane',
    'compute_homography_from_transform',
    'extract_R_T_from_transform',
    # BEV to camera warp
    'warp_bev_to_camera',
    'warp_bev_to_camera_with_coords',
    # Camera to satellite projection
    'differentiable_camera_to_sat_warp',
    # Pose loss
    'PoseLoss',
    # Camera-to-camera ground-plane
    'camera_to_camera_groundplane_pull',
    'apply_yaw_rotation_to_pose',
]

if project_camera_to_satellite is not None:
    __all__.append('project_camera_to_satellite')
