"""Satellite image condition encoder with 2D RoPE self-attention and projected geometry."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioning import SatelliteMemoryState

from .perspective_position_encoder import compute_sat_patch_perspective_uv


logger = logging.getLogger(__name__)


def _apply_1d_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to the last dim of ``x`` using per-token positions."""
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")

    half_dim = dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / max(half_dim, 1))
    )
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = angles.cos().to(dtype=x.dtype).view(1, 1, -1, half_dim)
    sin = angles.sin().to(dtype=x.dtype).view(1, 1, -1, half_dim)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack(
        (
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ),
        dim=-1,
    ).flatten(-2)


def apply_2d_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    grid_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply separable 2D RoPE to query/key tensors shaped [B, heads, N, head_dim]."""
    head_dim = query.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError(
            "2D RoPE requires attention head_dim divisible by 4, "
            f"got head_dim={head_dim}"
        )

    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    expected_tokens = grid_h * grid_w
    if query.shape[2] != expected_tokens:
        raise ValueError(
            f"2D RoPE token count mismatch: got {query.shape[2]}, expected {expected_tokens}"
        )

    rows = torch.arange(grid_h, device=query.device, dtype=torch.float32)
    cols = torch.arange(grid_w, device=query.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    row_pos = yy.reshape(-1)
    col_pos = xx.reshape(-1)

    split = head_dim // 2
    q_x, q_y = query[..., :split], query[..., split:]
    k_x, k_y = key[..., :split], key[..., split:]
    return (
        torch.cat([_apply_1d_rope(q_x, col_pos), _apply_1d_rope(q_y, row_pos)], dim=-1),
        torch.cat([_apply_1d_rope(k_x, col_pos), _apply_1d_rope(k_y, row_pos)], dim=-1),
    )


class RoPESelfAttention(nn.Module):
    """Multi-head satellite self-attention with 2D RoPE applied to q/k."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"embed_dim={self.dim} must be divisible by num_heads={self.num_heads}")
        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                "2D RoPE requires embed_dim / num_heads divisible by 4, "
                f"got {self.head_dim}"
            )

        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = float(dropout)

    def forward(self, hidden_states: torch.Tensor, *, grid_hw: Tuple[int, int]) -> torch.Tensor:
        batch_size, num_tokens, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        query, key = apply_2d_rope(query, key, grid_hw=grid_hw)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, num_tokens, self.dim)
        return self.out_proj(attended)


class RoPETransformerEncoderLayer(nn.Module):
    """Norm-first transformer encoder layer using 2D RoPE self-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = RoPESelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * dim, dim),
        )
        self.dropout2 = nn.Dropout(float(dropout))

    def forward(self, hidden_states: torch.Tensor, *, grid_hw: Tuple[int, int]) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout1(
            self.self_attn(self.norm1(hidden_states), grid_hw=grid_hw)
        )
        hidden_states = hidden_states + self.dropout2(self.mlp(self.norm2(hidden_states)))
        return hidden_states


class RoPETransformerEncoder(nn.Module):
    """Stack of satellite self-attention layers with 2D RoPE spatial structure."""

    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RoPETransformerEncoderLayer(dim=dim, num_heads=num_heads, dropout=dropout)
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, hidden_states: torch.Tensor, *, grid_hw: Tuple[int, int]) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, grid_hw=grid_hw)
        return hidden_states


class SatelliteConditionEncoder(nn.Module):
    """Encode satellite patches as pure content tokens and projected perspective UV."""

    _DEPRECATED_PE_KWARGS = {
        "perspective_pe_enabled",
        "perspective_num_freqs",
        "perspective_pe_gate_init",
        "perspective_invalid_mode",
        "perspective_use_validity_embedding",
        "perspective_ooi_init_std",
        "perspective_validity_embed_init_std",
        "perspective_pe_injection",
        "perspective_pe_scale_mode",
        "perspective_pe_target_ratio",
    }

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        sat_resolution: float = 0.2,
        sat_size: int = 512,
        num_heads: int = 12,
        num_layers: int = 4,
        attn_dropout: float = 0.1,
        **deprecated_pe_kwargs,
    ):
        super().__init__()
        unknown_kwargs = set(deprecated_pe_kwargs) - self._DEPRECATED_PE_KWARGS
        if unknown_kwargs:
            names = ", ".join(sorted(unknown_kwargs))
            raise TypeError(f"Unexpected SatelliteConditionEncoder kwargs: {names}")
        if deprecated_pe_kwargs:
            logger.warning(
                "Ignoring deprecated additive perspective PE satellite encoder kwargs: %s",
                ", ".join(sorted(deprecated_pe_kwargs)),
            )
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.sat_resolution = float(sat_resolution)
        self.sat_size = int(sat_size)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        self.token_norm = nn.LayerNorm(self.embed_dim)

        self.self_attn = RoPETransformerEncoder(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=float(attn_dropout),
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)

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
        camera_height_m: Optional[torch.Tensor],
        image_size: Optional[Tuple[int, int]],
    ) -> None:
        missing = []
        if K is None:
            missing.append("K")
        if T_cam_to_world is None:
            missing.append("T_cam_to_world")
        if T_imu_to_world is None:
            missing.append("T_imu_to_world")
        if camera_height_m is None:
            missing.append("camera_height_m")
        if image_size is None:
            missing.append("image_size")
        if missing:
            raise ValueError(
                "projected satellite geometry inputs are missing: " + ", ".join(missing)
            )

    def forward(
        self,
        sat_images: torch.Tensor,
        *,
        K: Optional[torch.Tensor] = None,
        T_cam_to_world: Optional[torch.Tensor] = None,
        T_imu_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> SatelliteMemoryState:
        B, _, H, W = sat_images.shape
        device = sat_images.device

        patches = self.patch_embed(sat_images)
        patch_h, patch_w = int(patches.shape[2]), int(patches.shape[3])
        tokens = patches.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        tokens = self.token_norm(tokens)

        bev_coords = self._compute_patch_bev_coords(B, H, W).to(device=device, dtype=sat_images.dtype)
        sat_xy = self._compute_patch_normalized_coords(B, H, W).to(device=device, dtype=sat_images.dtype)

        perspective_uv = None
        perspective_valid = None

        geometry_inputs = (
            K is not None
            or T_cam_to_world is not None
            or T_imu_to_world is not None
            or camera_height_m is not None
            or image_size is not None
        )

        if geometry_inputs:
            self._validate_geometry(K, T_cam_to_world, T_imu_to_world, camera_height_m, image_size)
            image_h, image_w = int(image_size[0]), int(image_size[1])
            # Keep T_cam_to_world / T_imu_to_world in their original dtype
            # (fp32) – compute_sat_patch_perspective_uv internally works in
            # fp32 to avoid bf16 truncation of UTM-scale (~1e6 m) translations.
            perspective_uv, perspective_valid = compute_sat_patch_perspective_uv(
                bev_coords=bev_coords,
                K=K,
                T_cam_to_world=T_cam_to_world,
                T_imu_to_world=T_imu_to_world,
                camera_height_m=camera_height_m,
                image_w=image_w,
                image_h=image_h,
            )

        tokens = self.self_attn(tokens, grid_hw=(patch_h, patch_w))
        tokens = self.attn_norm(tokens)

        return SatelliteMemoryState(
            tokens=tokens,
            xy=sat_xy,
            bev_coords=bev_coords,
            perspective_uv=perspective_uv,
            perspective_valid=perspective_valid,
        )
