"""Satellite image condition encoder with ground-plane perspective PE."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.conditioning import SatelliteMemoryState

from .perspective_position_encoder import (
    PerspectivePositionEncoder,
    compute_sat_patch_perspective_uv,
)


class SatelliteConditionEncoder(nn.Module):
    """Encode satellite patches with grid PE, perspective PE, and self-attention."""

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        sat_resolution: float = 0.2,
        sat_size: int = 512,
        perspective_pe_enabled: bool = True,
        num_heads: int = 12,
        num_layers: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.sat_resolution = float(sat_resolution)
        self.sat_size = int(sat_size)
        self.perspective_pe_enabled = bool(perspective_pe_enabled)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)

        self.grid_size = self.sat_size // self.patch_size
        self.grid_num_patches = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        # Learnable grid position embedding (row-major order)
        self.grid_pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_num_patches, self.embed_dim)
        )
        nn.init.trunc_normal_(self.grid_pos_embed, std=0.02)

        self.perspective_pos_encoder = PerspectivePositionEncoder(dim=self.embed_dim)
        self.token_norm = nn.LayerNorm(self.embed_dim)

        # Self-attention layers
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=4 * self.embed_dim,
                dropout=float(attn_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.num_layers,
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)

        # Zero-init output projections so self-attention starts as near-identity
        for layer in self.self_attn.layers:
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
        nn.init.zeros_(self.attn_norm.weight)
        nn.init.zeros_(self.attn_norm.bias)

    def _compute_patch_bev_coords(self, B: int, H: int, W: int) -> torch.Tensor:
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        patch_pixel_h = (
            torch.arange(patch_h, dtype=torch.float32) * self.patch_size + self.patch_size / 2
        )
        patch_pixel_w = (
            torch.arange(patch_w, dtype=torch.float32) * self.patch_size + self.patch_size / 2
        )
        w_grid, h_grid = torch.meshgrid(patch_pixel_w, patch_pixel_h, indexing="xy")

        half_w = float(W) / 2.0
        half_h = float(H) / 2.0
        x_meters = (w_grid - half_w) * self.sat_resolution
        y_meters = (half_h - h_grid) * self.sat_resolution
        coords = torch.stack([x_meters.reshape(-1), y_meters.reshape(-1)], dim=-1)
        return coords.unsqueeze(0).expand(B, -1, 2)

    def _compute_patch_normalized_coords(self, B: int, H: int, W: int) -> torch.Tensor:
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        u = torch.arange(patch_w, dtype=torch.float32) * self.patch_size + self.patch_size / 2
        v = torch.arange(patch_h, dtype=torch.float32) * self.patch_size + self.patch_size / 2
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        x_norm = (uu - (W / 2.0)) / (W / 2.0)
        y_norm = ((H / 2.0) - vv) / (H / 2.0)
        coords = torch.stack([x_norm.reshape(-1), y_norm.reshape(-1)], dim=-1)
        return coords.unsqueeze(0).expand(B, -1, 2)

    @staticmethod
    def _validate_geometry(
        K: Optional[torch.Tensor],
        T_cam_to_world: Optional[torch.Tensor],
        T_imu_to_world: Optional[torch.Tensor],
        image_size: Optional[Tuple[int, int]],
    ) -> None:
        missing = []
        if K is None:
            missing.append("K")
        if T_cam_to_world is None:
            missing.append("T_cam_to_world")
        if T_imu_to_world is None:
            missing.append("T_imu_to_world")
        if image_size is None:
            missing.append("image_size")
        if missing:
            raise ValueError(
                "perspective PE is enabled but geometry inputs are missing: " + ", ".join(missing)
            )

    def forward(
        self,
        sat_images: torch.Tensor,
        *,
        K: Optional[torch.Tensor] = None,
        T_cam_to_world: Optional[torch.Tensor] = None,
        T_imu_to_world: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> SatelliteMemoryState:
        B, _, H, W = sat_images.shape
        device = sat_images.device

        patches = self.patch_embed(sat_images)
        tokens = patches.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        tokens = self.token_norm(tokens)

        # Add learnable grid position embedding
        grid_pe = self.grid_pos_embed[:, : tokens.shape[1], :].to(device=device, dtype=tokens.dtype)
        tokens = tokens + grid_pe

        bev_coords = self._compute_patch_bev_coords(B, H, W).to(device=device, dtype=sat_images.dtype)
        sat_xy = self._compute_patch_normalized_coords(B, H, W).to(device=device, dtype=sat_images.dtype)

        perspective_uv = None
        perspective_valid = None
        if self.perspective_pe_enabled:
            self._validate_geometry(K, T_cam_to_world, T_imu_to_world, image_size)
            image_h, image_w = int(image_size[0]), int(image_size[1])
            perspective_uv, perspective_valid = compute_sat_patch_perspective_uv(
                bev_coords=bev_coords,
                K=K.to(device=device, dtype=sat_images.dtype),
                T_cam_to_world=T_cam_to_world.to(device=device, dtype=sat_images.dtype),
                T_imu_to_world=T_imu_to_world.to(device=device, dtype=sat_images.dtype),
                image_w=image_w,
                image_h=image_h,
            )
            tokens = tokens + self.perspective_pos_encoder(perspective_uv, perspective_valid)
        elif K is not None and T_cam_to_world is not None and T_imu_to_world is not None and image_size is not None:
            image_h, image_w = int(image_size[0]), int(image_size[1])
            perspective_uv, perspective_valid = compute_sat_patch_perspective_uv(
                bev_coords=bev_coords,
                K=K.to(device=device, dtype=sat_images.dtype),
                T_cam_to_world=T_cam_to_world.to(device=device, dtype=sat_images.dtype),
                T_imu_to_world=T_imu_to_world.to(device=device, dtype=sat_images.dtype),
                image_w=image_w,
                image_h=image_h,
            )

        # Self-attention among satellite patches
        tokens = tokens + self.self_attn(tokens)
        tokens = self.attn_norm(tokens)

        return SatelliteMemoryState(
            tokens=tokens,
            xy=sat_xy,
            bev_coords=bev_coords,
            perspective_uv=perspective_uv,
            perspective_valid=perspective_valid,
        )
