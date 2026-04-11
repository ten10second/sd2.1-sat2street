#!/usr/bin/env python3
"""
几何损失 (Pose Loss) 实现

通过单应性矩阵 H 计算 BEV 卫星图和相机图像之间的几何一致性损失。

流程：
1. 计算单应性矩阵 H = K · (R - T·n^T / d)
2. 生成 Warp 的图像 x̂_rgb：
   - Token 序列 T_gt → Token Embeddings (通过 token_embed)
   - Token Embeddings → 潜在特征 Z_latent (B × L × D)
   - Z_latent → 重塑为图像 Z_img (B × C_latent × H_latent × W_latent)
   - Z_img → 解码为卫星图 x̂_rgb (B × 3 × H_rgb × W_rgb)
3. 使用 H 将 x̂_rgb warp 到相机视角
4. 计算特征一致性损失

注意：
- 由于使用 Teacher Forcing，我们使用 GT Tokens 而不是 L_pose 辅助监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .bev_to_camera_warp import warp_bev_to_camera


class PoseLoss(nn.Module):
    """
    几何损失模块

    计算通过单应性变换后的 BEV 卫星图和相机图像之间的一致性损失。

    Args:
        use_teacher_forcing: bool - 是否使用 Teacher Forcing (使用 GT tokens)
        loss_type: str - 损失类型 ('l1', 'l2', 'smooth_l1')
        loss_weight: float - 损失权重
    """

    def __init__(
        self,
        use_teacher_forcing: bool = True,
        loss_type: str = 'l1',
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.use_teacher_forcing = use_teacher_forcing
        self.loss_type = loss_type
        self.loss_weight = loss_weight

    def compute_loss(
        self,
        warped_sat: torch.Tensor,
        cam_image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算 warp 后的卫星图和相机图像之间的损失

        Args:
            warped_sat: (B, 3, H, W) - Warp 后的卫星图
            cam_image: (B, 3, H, W) - 相机图像
            mask: (B, 1, H, W) - 可选的掩码（标记有效区域）

        Returns:
            loss: scalar - 损失值
        """
        if self.loss_type == 'l1':
            loss = F.l1_loss(warped_sat, cam_image, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(warped_sat, cam_image, reduction='none')
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(warped_sat, cam_image, reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # 应用掩码（如果提供）
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss * self.loss_weight

    def forward(
        self,
        gt_tokens: torch.Tensor,
        vqgan_tokenizer: nn.Module,
        cam_image: torch.Tensor,
        K: torch.Tensor,
        T_cam_to_world: torch.Tensor,
        resolution: float = 0.2,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算几何损失

        流程：
        1. GT Tokens → VQGAN Decode → 卫星图 x̂_sat
        2. 卫星图 x̂_sat → 通过射线投影到相机平面 → x̂_sat_warped (H_cam×W_cam)
        3. L_pose = ||x̂_sat_warped - x_cam|| (只计算有效区域)

        Args:
            gt_tokens: (B, L) - GT token 序列
            vqgan_tokenizer: nn.Module - 完整的 VQGAN tokenizer (包含 decode 方法)
            cam_image: (B, 3, H_cam, W_cam) - 相机图像
            K: (3, 3) 或 (B, 3, 3) - 相机内参
            T_cam_to_world: (4, 4) 或 (B, 4, 4) - 相机到世界的变换
            resolution: float - 卫星图分辨率 m/pixel (默认 0.2)
            mask: (B, 1, H_cam, W_cam) - 可选的掩码（相机图尺寸）

        Returns:
            dict: {
                'loss': 总损失,
                'warped_sat': Warp 后的卫星图（投影到相机平面）,
                'sat_image': 解码后的卫星图,
                'valid_mask': 有效像素掩码,
            }
        """
        # 步骤 1: 使用 VQGAN 解码 GT tokens → 卫星图
        with torch.no_grad():  # VQGAN 参数不更新
            sat_image = vqgan_tokenizer.decode(gt_tokens)  # (B, 3, H_sat, W_sat)

        # 步骤 2: 将卫星图 warp 到相机平面（通过射线与地面平面相交）
        H_cam, W_cam = cam_image.shape[2:]
        warped_sat, valid_mask = warp_bev_to_camera(
            sat_image,
            K,
            T_cam_to_world,
            cam_height=H_cam,
            cam_width=W_cam,
            resolution=resolution,
        )

        # 步骤 3: 计算损失（在相机平面上比较，只计算有效区域）
        if mask is not None:
            valid_mask = valid_mask * mask

        loss = self.compute_loss(warped_sat, cam_image, valid_mask)

        return {
            'loss': loss,
            'warped_sat': warped_sat.detach(),
            'sat_image': sat_image.detach(),
            'valid_mask': valid_mask.detach(),
        }
