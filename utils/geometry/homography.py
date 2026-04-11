#!/usr/bin/env python3
"""
单应性矩阵 (Homography) 计算和几何损失

用于计算从 BEV 卫星图到相机图像的单应性变换，并计算几何一致性损失。

参考：
- Ground Plane Assumption: 假设地面为平面 (Z = 0)
- Homography: H = K · (R - T·n^T / d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_homography_ground_plane(
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    n: Optional[torch.Tensor] = None,
    d: float = 1.0,
) -> torch.Tensor:
    """
    计算基于地面平面假设的单应性矩阵 H

    公式: H = K · (R - T·n^T / d)

    其中:
    - K: 3×3 相机内参矩阵 (来自 calib_cam_to_cam.txt 的 K_02 或 K_03)
    - R: 3×3 旋转矩阵 (从 T_C→W' 中提取)
    - T: 3×1 平移向量 (从 T_C→W' 中提取)
    - n: 3×1 地面平面法向量 (在相机坐标系中，通常假设 n = [0, -1, 0]^T，即 Y 轴向下)
    - d: 地面平面距离 (在相机坐标系中，通常假设 d = 地面到相机的距离，或归一化为 1)

    Args:
        K: (3, 3) 或 (B, 3, 3) - 相机内参矩阵
        R: (3, 3) 或 (B, 3, 3) - 旋转矩阵 (从 T_C→W' 提取)
        T: (3, 1) 或 (B, 3, 1) - 平移向量 (从 T_C→W' 提取)
        n: (3, 1) 或 (B, 3, 1) - 地面平面法向量 (默认: [0, -1, 0]^T)
        d: float - 地面平面距离 (默认: 1.0)

    Returns:
        H: (3, 3) 或 (B, 3, 3) - 单应性矩阵

    注意:
    - KITTI Cam 02 坐标系: X=右, Y=下, Z=前
    - 地面平面假设: 通常假设地面在相机下方，法向量 n = [0, -1, 0]^T (Y 轴向下)
    - 距离 d: 可以设置为相机到地面的实际距离，或归一化为 1
    """
    # 处理批次维度
    if K.dim() == 2:
        K = K.unsqueeze(0)  # (1, 3, 3)
    if R.dim() == 2:
        R = R.unsqueeze(0)  # (1, 3, 3)
    if T.dim() == 2:
        T = T.unsqueeze(0)  # (1, 3, 1)

    B = K.shape[0]
    device = K.device

    # 默认地面平面法向量: n = [0, -1, 0]^T (KITTI Cam 坐标系，Y 轴向下)
    if n is None:
        n = torch.tensor([[0.0], [-1.0], [0.0]], device=device, dtype=K.dtype)
        n = n.unsqueeze(0).expand(B, 3, 1)  # (B, 3, 1)
    elif n.dim() == 2:
        n = n.unsqueeze(0).expand(B, 3, 1)  # (B, 3, 1)

    # 计算 T·n^T: (B, 3, 1) @ (B, 1, 3) = (B, 3, 3)
    T_nT = torch.bmm(T, n.transpose(1, 2))  # (B, 3, 3)

    # 计算 R - T·n^T / d
    R_modified = R - T_nT / d  # (B, 3, 3)

    # 计算 H = K · (R - T·n^T / d)
    H = torch.bmm(K, R_modified)  # (B, 3, 3)

    # 如果输入是单个矩阵，返回单个矩阵
    if H.shape[0] == 1:
        H = H.squeeze(0)  # (3, 3)

    return H


def extract_R_T_from_transform(T_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 4×4 变换矩阵中提取旋转矩阵 R 和平移向量 T

    Args:
        T_matrix: (4, 4) 或 (B, 4, 4) - SE(3) 变换矩阵

    Returns:
        R: (3, 3) 或 (B, 3, 3) - 旋转矩阵
        T: (3, 1) 或 (B, 3, 1) - 平移向量
    """
    if T_matrix.dim() == 2:
        # 单个矩阵
        R = T_matrix[:3, :3]  # (3, 3)
        T = T_matrix[:3, 3:4]  # (3, 1)
    else:
        # 批次矩阵
        R = T_matrix[:, :3, :3]  # (B, 3, 3)
        T = T_matrix[:, :3, 3:4]  # (B, 3, 1)

    return R, T


def compute_homography_from_transform(
    K: torch.Tensor,
    T_cam_to_sat: torch.Tensor,
    n: Optional[torch.Tensor] = None,
    d: float = 1.0,
) -> torch.Tensor:
    """
    从相机到卫星图的变换矩阵计算单应性矩阵

    这是 compute_homography_ground_plane 的便捷包装函数。

    Args:
        K: (3, 3) 或 (B, 3, 3) - 相机内参矩阵
        T_cam_to_sat: (4, 4) 或 (B, 4, 4) - 相机到卫星图的变换矩阵
        n: (3, 1) 或 (B, 3, 1) - 地面平面法向量 (默认: [0, -1, 0]^T)
        d: float - 地面平面距离 (默认: 1.0)

    Returns:
        H: (3, 3) 或 (B, 3, 3) - 单应性矩阵
    """
    # 提取 R 和 T
    R, T = extract_R_T_from_transform(T_cam_to_sat)

    # 计算单应性矩阵
    H = compute_homography_ground_plane(K, R, T, n, d)

    return H
