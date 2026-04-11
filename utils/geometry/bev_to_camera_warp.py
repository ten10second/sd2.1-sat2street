#!/usr/bin/env python3
"""
BEV 卫星图到相机图的 Warp

将卫星图（BEV）投影到相机图像平面。

流程：
1. 卫星图像素 (u_sat, v_sat) → 世界坐标 (X, Y, Z=0)
2. 世界坐标 → 相机坐标
3. 相机坐标 → 相机图像素 (u_cam, v_cam)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
IMU_TO_GROUND_HEIGHT = 0.93
CAMERA_TO_GROUND_HEIGHT = 1.65


def ipm_valid_mask(
    u_sat: torch.Tensor,          # (B,H,W)
    v_sat: torch.Tensor,          # (B,H,W)
    t_img: torch.Tensor,          # (B,H,W)
    rays_z: torch.Tensor,         # (B,1,H*W) or (B,H,W)
    vv_cam: torch.Tensor,         # (H,W)
    K: torch.Tensor,              # (B,3,3)
    W_sat: int,
    H_sat: int,
    *,
    rays_z_down_eps: float = 1e-3,
    t_max: float = 120.0,
    cy_margin_px: float = 0.0,
) -> torch.Tensor:
    """
    Build a robust validity mask for world-Z-plane IPM.
    Returns:
        valid_mask: (B,H,W) bool
    """
    B = K.shape[0]

    # rays_z -> (B,H,W)
    if rays_z.dim() == 3 and rays_z.shape[1] == 1:
        rays_z_img = rays_z.reshape(B, 1, t_img.shape[1], t_img.shape[2]).squeeze(1)
    else:
        rays_z_img = rays_z

    # 1) ray must point downward
    ray_down = rays_z_img < (-rays_z_down_eps)

    # 2) reasonable intersection distance
    t_ok = (t_img > 0) & (t_img < t_max)

    # 3) sky / horizon reject (pixel below principal point)
    cy = K[:, 1, 2].view(B, 1, 1)
    ground_pixel = vv_cam.unsqueeze(0) > (cy + cy_margin_px)
    # ground_pixel = torch.ones_like(u_sat, dtype=torch.bool)  # keep all pixels; rely on ray_down/t_ok


    # 4) BEV bounds
    in_bounds = (u_sat >= 0) & (u_sat < W_sat) & (v_sat >= 0) & (v_sat < H_sat)

    return ray_down & t_ok & ground_pixel & in_bounds


def warp_bev_to_camera(
    sat_image: torch.Tensor,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    cam_height: int,
    cam_width: int,
    sat_size: int = 512,
    resolution: float = 0.2,
    ground_height: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 BEV 卫星图 warp 到相机图像平面

    流程：
    1. 对于相机图的每个像素 (u_cam, v_cam)
    2. 反投影到 3D 射线
    3. 与地面平面 (Z=ground_height) 相交
    4. 转换为卫星图像素 (u_sat, v_sat)
    5. 从卫星图采样

    Args:
        sat_image: (B, C, H_sat, W_sat) - 卫星图像
        K: (3, 3) 或 (B, 3, 3) - 相机内参
        T_cam_to_world: (4, 4) 或 (B, 4, 4) - 相机到世界的变换
        T_imu_to_world: (4, 4) 或 (B, 4, 4) - IMU 到世界的变换（用于 BEV 中心对齐）
        cam_height: int - 相机图像高度
        cam_width: int - 相机图像宽度
        sat_size: int - 卫星图尺寸（默认 512）
        resolution: float - 卫星图分辨率 (m/pixel, 默认 0.2)
        ground_height: float - 地面高度（世界坐标 Z，默认 0.0）

    Returns:
        warped_sat: (B, C, cam_height, cam_width) - Warp 后的卫星图
        valid_mask: (B, 1, cam_height, cam_width) - 有效像素掩码
    """
    B, C, H_sat, W_sat = sat_image.shape
    device = sat_image.device

    # 处理批次维度
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, 3, 3)
    if T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0).expand(B, 4, 4)
    if T_imu_to_world.dim() == 2:
        T_imu_to_world = T_imu_to_world.unsqueeze(0).expand(B, 4, 4)

    # 提取 R 和 t
    R_cam_to_world = T_cam_to_world[:, :3, :3]  # (B, 3, 3)
    t_cam_to_world = T_cam_to_world[:, :3, 3:4]  # (B, 3, 1)

    # 卫星图中心 = IMU/车辆位置 (X, Y)
    # 注意：与 compute_camera_to_sat_grid_norm 保持一致
    t_imu = T_imu_to_world[:, :3, 3]  # (B, 3)
    sat_center = t_imu[:, :2]  # (B, 2)

    # 创建相机图的像素网格
    v_cam = torch.arange(cam_height, dtype=torch.float32, device=device)
    u_cam = torch.arange(cam_width, dtype=torch.float32, device=device)
    vv_cam, uu_cam = torch.meshgrid(v_cam, u_cam, indexing='ij')  # (cam_height, cam_width)

    # 像素坐标 → 归一化坐标
    K_inv = torch.inverse(K)  # (B, 3, 3)

    # 组合为齐次坐标 (B, 3, cam_height*cam_width)
    pixels_homo = torch.stack([
        uu_cam.reshape(-1),
        vv_cam.reshape(-1),
        torch.ones_like(uu_cam.reshape(-1)),
    ], dim=0).unsqueeze(0).expand(B, 3, -1)  # (B, 3, H*W)

    # 相机坐标系中的射线方向
    rays_cam = torch.bmm(K_inv, pixels_homo)  # (B, 3, H*W)

    # 转换到世界坐标系
    rays_world = torch.bmm(R_cam_to_world, rays_cam)  # (B, 3, H*W)

    # 与地面平面相交 (Z=ground_height)
    rays_z = rays_world[:, 2:3, :]
    t = torch.where(
        rays_z.abs() > 1e-6,
        (ground_height - t_cam_to_world[:, 2:3, :]) / rays_z,
        torch.full_like(rays_z, -1.0),  # 平行于地面的射线，设置为无效
    )  # (B, 1, H*W)

    # 计算交点
    points_world = t_cam_to_world + t * rays_world  # (B, 3, H*W)

    # 世界坐标 → 卫星图像素
    X_world = points_world[:, 0, :]  # (B, H*W)
    Y_world = points_world[:, 1, :]

    # 相对于卫星图中心的偏移
    offset_x = X_world - sat_center[:, 0:1]  # (B, H*W)
    offset_y = Y_world - sat_center[:, 1:2]

    # 转换为卫星图像素
    u_sat = W_sat / 2 + offset_x / resolution
    v_sat = H_sat / 2 - offset_y / resolution  # 注意负号

    # 重塑为 (B, cam_height, cam_width)
    u_sat = u_sat.reshape(B, cam_height, cam_width)
    v_sat = v_sat.reshape(B, cam_height, cam_width)
    t = t.reshape(B, cam_height, cam_width)

    # 创建有效性掩码：只保留在相机前方且在卫星图范围内的点
    valid_mask = (t > 0) & \
                 (u_sat >= 0) & (u_sat < W_sat) & \
                 (v_sat >= 0) & (v_sat < H_sat)

    # 归一化到 [-1, 1] 范围 (grid_sample 的要求)
    u_normalized = 2.0 * u_sat / (W_sat - 1) - 1.0
    v_normalized = 2.0 * v_sat / (H_sat - 1) - 1.0

    # 对于无效点，设置为超出范围的值（grid_sample 会填充 0）
    u_normalized = torch.where(valid_mask, u_normalized, torch.full_like(u_normalized, -2.0))
    v_normalized = torch.where(valid_mask, v_normalized, torch.full_like(v_normalized, -2.0))

    # 组合为 grid (B, cam_height, cam_width, 2)
    grid = torch.stack([u_normalized, v_normalized], dim=-1)

    # 使用 grid_sample 从卫星图采样
    warped_sat = F.grid_sample(
        sat_image,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )

    # 返回 warp 后的图像和有效性掩码
    valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, cam_height, cam_width)

    return warped_sat, valid_mask


def warp_bev_to_camera_with_coords(
    sat_image: torch.Tensor,
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: Optional[torch.Tensor] = None,
    cam_height: int = 256,
    cam_width: int = 640,
    sat_size: int = 512,
    resolution: float = 0.2,
    ground_height: Optional[float] = None,
):
    """
    与 warp_bev_to_camera 相同，但额外返回每个相机像素对应的 BEV 坐标 (u_sat, v_sat) 的归一化坐标图。
    无效像素的坐标设为 -2（grid_sample 约定外值）。

    Args:
        sat_image: (B, C, H_sat, W_sat) 卫星/BEV 图像
        K: (B, 3, 3) 或 (3, 3) 相机内参矩阵
        T_cam_to_world: (B, 4, 4) 或 (4, 4) 相机到世界坐标系的变换矩阵
        cam_height: 输出相机图像的高度
        cam_width: 输出相机图像的宽度
        sat_size: 卫星/BEV 图像的尺寸（假设为正方形）
        resolution: 卫星图像的分辨率（米/像素）
        ground_height: 地平面在世界坐标系中的 Z 值

    Returns:
        warped_sat: (B, C, cam_height, cam_width) 变形后的卫星/BEV 图像
        valid_mask: (B, 1, cam_height, cam_width) 有效像素掩码
        coords_map: (B, 2, cam_height, cam_width) 归一化到 [-1,1] 的坐标图，无效像素为 -2
    """
    B, C, H_sat, W_sat = sat_image.shape
    device = sat_image.device

    # 处理批次维度
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, 3, 3)
    if T_cam_to_world.dim() == 2:
        T_cam_to_world = T_cam_to_world.unsqueeze(0).expand(B, 4, 4)

    R_cam_to_world = T_cam_to_world[:, :3, :3]
    t_cam_to_world = T_cam_to_world[:, :3, 3:4]

    # If ground_height is not provided, estimate it from IMU height (local ground plane).
    # This avoids using a global Z=0 plane when world coordinates carry absolute altitude.
    if ground_height is None:
        if T_imu_to_world is None:
            ground_height = 0.0
        else:
            # Normalize to (B,4,4)
            if T_imu_to_world.dim() == 2:
                T_imu_to_world_b = T_imu_to_world.unsqueeze(0)
            else:
                T_imu_to_world_b = T_imu_to_world
            ground_height = float(T_imu_to_world_b[0, 2, 3].detach().cpu().item() - IMU_TO_GROUND_HEIGHT)


    # 卫星中心：优先使用 IMU 位置，否则使用相机位置
    if T_imu_to_world is not None:
        # Ensure T_imu_to_world is batched
        T_imu_b = T_imu_to_world if T_imu_to_world.dim() == 3 else T_imu_to_world.unsqueeze(0)
        sat_center = T_imu_b[:, :2, 3] # (B, 2)
    else:
        sat_center = t_cam_to_world[:, :2, 0]  # (B, 2)

    # 相机像素网格
    v_cam = torch.arange(cam_height, dtype=torch.float32, device=device)
    u_cam = torch.arange(cam_width, dtype=torch.float32, device=device)
    vv_cam, uu_cam = torch.meshgrid(v_cam, u_cam, indexing='ij')

    # 像素 → 射线 (保持 OpenCV pinhole 约定)
    # 这里不做额外的 v 翻转/轴转换；与 cv2 的虚拟透视相机模型保持一致。
    K_inv = torch.inverse(K)
    pixels_homo = torch.stack(
        [
            uu_cam.reshape(-1),
            vv_cam.reshape(-1),
            torch.ones_like(uu_cam.reshape(-1)),
        ],
        dim=0,
    ).unsqueeze(0).expand(B, 3, -1)
    rays_cam = torch.bmm(K_inv, pixels_homo)
    rays_world = torch.bmm(R_cam_to_world, rays_cam)

    # Intersect with ground plane in WORLD coordinates: Z = ground_height
    # This matches warp_bev_to_camera() behavior.
    rays_z = rays_world[:, 2:3, :]
    t = torch.where(
        rays_z.abs() > 1e-6,
        (ground_height - t_cam_to_world[:, 2:3, :]) / rays_z,
        torch.full_like(rays_z, -1.0),  # rays parallel to the ground plane
    )  # (B,1,N)

    # World coordinates of intersection points
    points_world = t_cam_to_world + t * rays_world  # (B,3,N)

    # For ipm_valid_mask interface: use rays_z as "rays_z" (downward test is handled there)
    ray_dot_up = rays_z  # (B,1,N)

    points_world = t_cam_to_world + t * rays_world


    X_world = points_world[:, 0, :]
    Y_world = points_world[:, 1, :]

    # 相对卫星中心的偏移
    X_off = X_world - sat_center[:, 0:1]
    Y_off = Y_world - sat_center[:, 1:2]

    # 转为卫星像素
    u_sat = W_sat / 2 + X_off / resolution
    v_sat = H_sat / 2 - Y_off / resolution  # 注意负号

    # 重塑
    u_sat = u_sat.reshape(B, cam_height, cam_width)
    v_sat = v_sat.reshape(B, cam_height, cam_width)
    t_img = t.reshape(B, cam_height, cam_width)

    # 有效性
    valid_mask = ipm_valid_mask(
        u_sat=u_sat,
        v_sat=v_sat,
        t_img=t_img,
        rays_z=ray_dot_up,
        vv_cam=vv_cam,
        K=K,
        W_sat=W_sat,
        H_sat=H_sat,
        t_max=120.0,
        cy_margin_px=0.0,   # 如果天空还残留，可调到 10~40
    )

    # 归一化坐标到 [-1,1]
    u_norm = 2.0 * u_sat / (W_sat - 1) - 1.0
    v_norm = 2.0 * v_sat / (H_sat - 1) - 1.0

    # grid_sample 网格
    u_grid = torch.where(valid_mask, u_norm, torch.full_like(u_norm, -2.0))
    v_grid = torch.where(valid_mask, v_norm, torch.full_like(v_norm, -2.0))
    grid = torch.stack([u_grid, v_grid], dim=-1)

    # 采样
    warped_sat = F.grid_sample(
        sat_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    # coords_map：为方便，直接返回归一化后的 (u_norm,v_norm)，无效为 -2
    coords_u = torch.where(valid_mask, u_norm, torch.full_like(u_norm, -2.0))
    coords_v = torch.where(valid_mask, v_norm, torch.full_like(v_norm, -2.0))
    coords_map = torch.stack([coords_u, coords_v], dim=1)  # (B,2,H,W)

    return warped_sat, valid_mask.unsqueeze(1).float(), coords_map
