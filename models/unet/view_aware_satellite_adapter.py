"""
View-aware satellite scene reader.

This version reads from a lightweight triplane scene representation:
- `xy`: overhead satellite-aligned memory
- `xz` / `yz`: learned vertical memories derived from the satellite image

The public output interface stays unchanged so training and inference can keep
using `readout_tokens` and `readout_map`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewAwareSatelliteAdapter(nn.Module):
    def __init__(
        self,
        sat_in_dim: int = 768,
        out_dim: int = 1024,
        grid_h: int = 8,
        grid_w: int = 20,
        query_dim: int = 768,
        num_heads: int = 8,
        scale: float = 1.0,
        geo_bias_weight: float = 1.0,
        geo_sigma: float = 0.35,
        local_topk: int = 25,
        geo_target_sigma: float = 0.20,
        gate_hidden_dim: int = 256,
        token_pool_num_tokens: int = 8,
        token_pool_num_heads: Optional[int] = None,
        token_scale: float = 1.0,
        save_attention_heatmap: bool = True,
        heatmap_max_tokens: int = 16,
        ray_num_samples: int = 32,
        ray_depth_min: float = 1.0,
        ray_depth_max: float = 80.0,
        ray_offset_scale: float = 0.50,
        ray_scene_extent_x_m: float = 51.2,
        ray_scene_extent_y_m: float = 51.2,
        ray_scene_z_min_m: float = 0.0,
        ray_scene_z_max_m: float = 20.0,
        triplane_enabled: bool = True,
    ):
        super().__init__()
        if sat_in_dim <= 0 or out_dim <= 0 or query_dim <= 0:
            raise ValueError(
                f"sat_in_dim/out_dim/query_dim must be positive, got {sat_in_dim}/{out_dim}/{query_dim}"
            )
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"grid_h/grid_w must be positive, got {grid_h}/{grid_w}")
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim must be divisible by num_heads, got {query_dim}/{num_heads}")
        if token_pool_num_tokens <= 0:
            raise ValueError(f"token_pool_num_tokens must be positive, got {token_pool_num_tokens}")
        if ray_num_samples <= 0:
            raise ValueError(f"ray_num_samples must be positive, got {ray_num_samples}")
        if ray_depth_max <= ray_depth_min:
            raise ValueError(
                f"ray_depth_max must be larger than ray_depth_min, got {ray_depth_min}/{ray_depth_max}"
            )
        if ray_scene_extent_x_m <= 0.0 or ray_scene_extent_y_m <= 0.0:
            raise ValueError(
                "ray_scene_extent_x_m/ray_scene_extent_y_m must be positive, "
                f"got {ray_scene_extent_x_m}/{ray_scene_extent_y_m}"
            )
        if ray_scene_z_max_m <= ray_scene_z_min_m:
            raise ValueError(
                f"ray_scene_z_max_m must be larger than ray_scene_z_min_m, got "
                f"{ray_scene_z_min_m}/{ray_scene_z_max_m}"
            )
        token_pool_num_heads = int(token_pool_num_heads or num_heads)
        if query_dim % token_pool_num_heads != 0:
            raise ValueError(
                f"query_dim must be divisible by token_pool_num_heads, got {query_dim}/{token_pool_num_heads}"
            )

        self.sat_in_dim = int(sat_in_dim)
        self.out_dim = int(out_dim)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.query_dim = int(query_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.query_dim // self.num_heads
        self.scale = float(scale)
        self.map_scale = float(scale)
        self.token_pool_num_tokens = int(token_pool_num_tokens)
        self.token_pool_num_heads = int(token_pool_num_heads)
        self.token_scale = float(token_scale)
        self.save_attention_heatmap = bool(save_attention_heatmap)
        self.heatmap_max_tokens = int(max(1, heatmap_max_tokens))
        self.ray_num_samples = int(ray_num_samples)
        self.ray_depth_min = float(ray_depth_min)
        self.ray_depth_max = float(ray_depth_max)
        self.ray_offset_scale = float(ray_offset_scale)
        self.ray_scene_extent_x_m = float(ray_scene_extent_x_m)
        self.ray_scene_extent_y_m = float(ray_scene_extent_y_m)
        self.ray_scene_z_min_m = float(ray_scene_z_min_m)
        self.ray_scene_z_max_m = float(ray_scene_z_max_m)
        self.triplane_enabled = bool(triplane_enabled)
        self.geo_bias_weight = float(geo_bias_weight)
        self.geo_sigma = float(geo_sigma)
        self.local_topk = int(local_topk)
        self.geo_target_sigma = float(geo_target_sigma)

        self.query_mlp = nn.Sequential(
            nn.Linear(9, query_dim),
            nn.LayerNorm(query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
        )
        self.sat_norm = nn.LayerNorm(sat_in_dim)
        self.world_proj = nn.Linear(sat_in_dim, query_dim)
        self.coord_proj = nn.Sequential(
            nn.Linear(2, query_dim),
            nn.LayerNorm(query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
        )
        self.perspective_offset_net = nn.Linear(query_dim, self.num_heads * self.ray_num_samples * 3)
        self.perspective_attn_net = nn.Linear(query_dim, self.num_heads * self.ray_num_samples)
        self.perspective_value_proj = nn.Linear(query_dim, query_dim)
        self.perspective_head_mix = nn.Linear(query_dim, query_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.token_pool_query = nn.Parameter(
            torch.randn(self.token_pool_num_tokens, query_dim) / math.sqrt(query_dim)
        )
        self.token_pool_query_norm = nn.LayerNorm(query_dim)
        self.token_pool_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=self.token_pool_num_heads,
            batch_first=True,
        )
        self.token_out = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(query_dim + 9, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.register_buffer(
            "ray_sample_scales",
            torch.linspace(self.ray_depth_min, self.ray_depth_max, steps=self.ray_num_samples),
            persistent=False,
        )
        self._reset_perspective_ray_attention()

        self.last_stats: Dict[str, torch.Tensor] = {}
        self.last_attention_heatmap: Optional[torch.Tensor] = None
        self.last_attention_index: Optional[torch.Tensor] = None
        self.last_sat_xy: Optional[torch.Tensor] = None
        self.last_view_xy: Optional[torch.Tensor] = None
        self.last_query_tokens: Optional[torch.Tensor] = None
        self.last_readout_tokens: Optional[torch.Tensor] = None
        self.last_readout_map: Optional[torch.Tensor] = None
        self.last_readout_tokens_raw: Optional[torch.Tensor] = None

    def _reset_perspective_ray_attention(self) -> None:
        """Start from uniform ray sampling: zero offsets and uniform depth weights."""
        nn.init.zeros_(self.perspective_offset_net.weight)
        nn.init.zeros_(self.perspective_offset_net.bias)
        nn.init.zeros_(self.perspective_attn_net.weight)
        nn.init.zeros_(self.perspective_attn_net.bias)
        nn.init.eye_(self.perspective_value_proj.weight)
        nn.init.zeros_(self.perspective_value_proj.bias)
        nn.init.eye_(self.perspective_head_mix.weight)
        nn.init.zeros_(self.perspective_head_mix.bias)

    @staticmethod
    def _as_batched_intrinsics(
        value: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if value is None or not torch.is_tensor(value):
            return None
        x = value.to(device=device, dtype=dtype)
        if x.ndim == 1 and x.shape[0] == 4:
            fx, fy, cx, cy = x.unbind(dim=0)
            K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            K[:, 0, 0] = fx
            K[:, 1, 1] = fy
            K[:, 0, 2] = cx
            K[:, 1, 2] = cy
            return K.expand(batch, -1, -1)
        if x.ndim == 2 and x.shape == (3, 3):
            return x.unsqueeze(0).expand(batch, -1, -1)
        if x.ndim == 2 and x.shape[-1] == 4:
            if x.shape[0] == 1:
                x = x.expand(batch, -1)
            if x.shape[0] != batch:
                return None
            K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1).clone()
            K[:, 0, 0] = x[:, 0]
            K[:, 1, 1] = x[:, 1]
            K[:, 0, 2] = x[:, 2]
            K[:, 1, 2] = x[:, 3]
            return K
        if x.ndim == 3 and x.shape[-2:] == (3, 3):
            if x.shape[0] == 1 and batch != 1:
                return x.expand(batch, -1, -1)
            if x.shape[0] == batch:
                return x
        return None

    @staticmethod
    def _as_batched_transform(
        value: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if value is None or not torch.is_tensor(value):
            return None
        x = value.to(device=device, dtype=dtype)
        if x.ndim == 2 and x.shape == (4, 4):
            return x.unsqueeze(0).expand(batch, -1, -1)
        if x.ndim == 3 and x.shape[-2:] == (4, 4):
            if x.shape[0] == 1 and batch != 1:
                return x.expand(batch, -1, -1)
            if x.shape[0] == batch:
                return x
        return None

    @staticmethod
    def _as_batched_scalar(
        value: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if torch.is_tensor(value):
            x = value.to(device=device, dtype=dtype)
        else:
            try:
                x = torch.as_tensor(value, device=device, dtype=dtype)
            except (TypeError, ValueError):
                return None
        if x.ndim == 0:
            return x.reshape(1).expand(batch)
        x = x.reshape(-1)
        if x.numel() == 1 and batch != 1:
            return x.expand(batch)
        if x.numel() == batch:
            return x
        return None

    @staticmethod
    def _coerce_source_image_size(source_image_size: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if source_image_size is None:
            return None
        if len(source_image_size) != 2:
            return None
        height, width = int(source_image_size[0]), int(source_image_size[1])
        if height <= 0 or width <= 0:
            return None
        return height, width

    def _build_perspective_query_inputs(
        self,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        ego_to_world: Optional[torch.Tensor],
        camera_height_m: Optional[torch.Tensor],
        source_image_size: Tuple[int, int],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_h, source_w = int(source_image_size[0]), int(source_image_size[1])
        geom_dtype = torch.float32
        K = intrinsics.to(device=device, dtype=geom_dtype)
        T_cam = cam_to_world.to(device=device, dtype=geom_dtype)
        T_ego = ego_to_world.to(device=device, dtype=geom_dtype) if ego_to_world is not None else None
        cam_height = (
            camera_height_m.to(device=device, dtype=geom_dtype).reshape(batch)
            if camera_height_m is not None
            else None
        )

        u = (torch.arange(self.grid_w, device=device, dtype=geom_dtype) + 0.5) * (
            float(source_w) / float(max(self.grid_w, 1))
        )
        v = (torch.arange(self.grid_h, device=device, dtype=geom_dtype) + 0.5) * (
            float(source_h) / float(max(self.grid_h, 1))
        )
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        pixels = torch.stack(
            [uu.reshape(-1), vv.reshape(-1), torch.ones_like(uu.reshape(-1))],
            dim=0,
        ).unsqueeze(0).expand(batch, -1, -1)

        dirs_cam = torch.bmm(torch.inverse(K), pixels).transpose(1, 2)
        dirs_world = torch.bmm(T_cam[:, :3, :3], dirs_cam.transpose(1, 2)).transpose(1, 2)
        dirs_world = F.normalize(dirs_world, dim=-1)

        cam_center = T_cam[:, :3, 3]
        if T_ego is not None:
            center_xy = T_ego[:, :2, 3]
        else:
            center_xy = cam_center[:, :2]
        if cam_height is not None:
            ground_z = cam_center[:, 2] - cam_height
        elif T_ego is not None:
            ground_z = T_ego[:, 2, 3]
        else:
            ground_z = torch.zeros(batch, device=device, dtype=geom_dtype)

        origin_x = (cam_center[:, 0] - center_xy[:, 0]) / self.ray_scene_extent_x_m
        origin_y = (cam_center[:, 1] - center_xy[:, 1]) / self.ray_scene_extent_y_m
        origin_z = (
            (cam_center[:, 2] - ground_z - self.ray_scene_z_min_m)
            / (self.ray_scene_z_max_m - self.ray_scene_z_min_m)
            * 2.0
            - 1.0
        )
        origin_norm = torch.stack([origin_x, origin_y, origin_z], dim=-1)
        origin_norm = origin_norm[:, None, :].expand(-1, self.grid_h * self.grid_w, -1)

        pixel_x = uu.reshape(-1) / float(max(source_w, 1)) * 2.0 - 1.0
        pixel_y = 1.0 - vv.reshape(-1) / float(max(source_h, 1)) * 2.0
        pixel_xy = torch.stack([pixel_x, pixel_y], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        valid_tokens = torch.ones(batch, self.grid_h * self.grid_w, 1, device=device, dtype=geom_dtype)

        query_input = torch.cat([pixel_xy, dirs_world, origin_norm, valid_tokens], dim=-1).to(dtype=dtype)
        query_tokens = self.query_mlp(query_input)
        return query_input, query_tokens, valid_tokens.to(dtype=dtype)

    def _prepare_plane_maps(
        self,
        scene_memory: Dict[str, Union[torch.Tensor, Tuple[int, int], int]],
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        batch = int(scene_memory["xy_tokens"].shape[0])
        device = scene_memory["xy_tokens"].device
        xy_h, xy_w = scene_memory["xy_hw"]
        z_tokens = int(scene_memory["z_tokens"])
        plane_maps: Dict[str, torch.Tensor] = {}
        plane_shapes = {
            "xy": (int(xy_h), int(xy_w)),
            "xz": (z_tokens, int(xy_w)),
            "yz": (z_tokens, int(xy_h)),
        }
        for plane_name in ("xy", "xz", "yz"):
            tokens = scene_memory[f"{plane_name}_tokens"]
            coords = scene_memory[f"{plane_name}_coords"].to(device=device, dtype=dtype)
            world = self.world_proj(self.sat_norm(tokens)) + self.coord_proj(coords)
            plane_h, plane_w = plane_shapes[plane_name]
            plane_maps[plane_name] = world.transpose(1, 2).reshape(
                batch, self.query_dim, plane_h, plane_w
            ).contiguous()
        return plane_maps

    def _coerce_scene_memory(
        self,
        scene_memory: Optional[Dict[str, Union[torch.Tensor, Tuple[int, int], int]]],
        sat_tokens: Optional[torch.Tensor],
        sat_xy: Optional[torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, Tuple[int, int], int]]:
        if isinstance(scene_memory, dict):
            required = ("xy_tokens", "xy_coords", "xz_tokens", "xz_coords", "yz_tokens", "yz_coords", "xy_hw", "z_tokens")
            for key in required:
                if key not in scene_memory:
                    raise ValueError(f"scene_memory missing required key: {key}")
            return scene_memory

        if sat_tokens is None or sat_xy is None:
            raise ValueError("Either scene_memory or both sat_tokens/sat_xy must be provided")
        if sat_tokens.ndim != 3 or sat_tokens.shape[-1] != self.sat_in_dim:
            raise ValueError(f"sat_tokens must be [B,N,{self.sat_in_dim}], got {list(sat_tokens.shape)}")
        if sat_xy.ndim != 3 or sat_xy.shape[:2] != sat_tokens.shape[:2] or sat_xy.shape[-1] != 2:
            raise ValueError(f"sat_xy must be [B,{sat_tokens.shape[1]},2], got {list(sat_xy.shape)}")

        batch = sat_tokens.shape[0]
        token_count = sat_tokens.shape[1]
        xy_h = int(round(math.sqrt(token_count)))
        while xy_h > 1 and token_count % xy_h != 0:
            xy_h -= 1
        xy_w = token_count // max(1, xy_h)
        z_tokens = max(8, xy_h // 2)

        xy_plane = sat_tokens.reshape(batch, xy_h, xy_w, self.sat_in_dim).permute(0, 3, 1, 2).contiguous()
        x_reduced = xy_plane.mean(dim=2, keepdim=True).expand(-1, -1, z_tokens, -1).contiguous()
        y_reduced = xy_plane.mean(dim=3, keepdim=True).expand(-1, -1, z_tokens, -1).contiguous()

        sat_xy_grid = sat_xy.reshape(batch, xy_h, xy_w, 2)
        x_axis = sat_xy_grid[0, 0, :, 0]
        y_axis = sat_xy_grid[0, :, 0, 1]
        z_axis = 1.0 - (
            (torch.arange(z_tokens, device=sat_tokens.device, dtype=sat_tokens.dtype) + 0.5) / float(max(z_tokens, 1))
        ) * 2.0
        xz_x, xz_z = torch.meshgrid(x_axis, z_axis, indexing="xy")
        yz_y, yz_z = torch.meshgrid(y_axis, z_axis, indexing="xy")

        return {
            "xy_tokens": sat_tokens,
            "xy_coords": sat_xy,
            "xz_tokens": x_reduced.permute(0, 2, 3, 1).reshape(batch, -1, self.sat_in_dim).contiguous(),
            "xz_coords": torch.stack([xz_x.reshape(-1), xz_z.reshape(-1)], dim=-1).unsqueeze(0).expand(batch, -1, -1),
            "yz_tokens": y_reduced.permute(0, 2, 3, 1).reshape(batch, -1, self.sat_in_dim).contiguous(),
            "yz_coords": torch.stack([yz_y.reshape(-1), yz_z.reshape(-1)], dim=-1).unsqueeze(0).expand(batch, -1, -1),
            "xy_hw": (xy_h, xy_w),
            "z_tokens": z_tokens,
        }

    def _sample_plane(
        self,
        plane_map: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        sampled = F.grid_sample(
            plane_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.permute(0, 2, 3, 1).contiguous()

    def _build_perspective_ray_samples(
        self,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        ego_to_world: Optional[torch.Tensor],
        camera_height_m: Optional[torch.Tensor],
        source_image_size: Tuple[int, int],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_h, source_w = int(source_image_size[0]), int(source_image_size[1])
        geom_dtype = torch.float32
        K = intrinsics.to(device=device, dtype=geom_dtype)
        T_cam = cam_to_world.to(device=device, dtype=geom_dtype)
        T_ego = ego_to_world.to(device=device, dtype=geom_dtype) if ego_to_world is not None else None
        cam_height = (
            camera_height_m.to(device=device, dtype=geom_dtype).reshape(batch)
            if camera_height_m is not None
            else None
        )

        u = (torch.arange(self.grid_w, device=device, dtype=geom_dtype) + 0.5) * (
            float(source_w) / float(max(self.grid_w, 1))
        )
        v = (torch.arange(self.grid_h, device=device, dtype=geom_dtype) + 0.5) * (
            float(source_h) / float(max(self.grid_h, 1))
        )
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        pixels = torch.stack(
            [uu.reshape(-1), vv.reshape(-1), torch.ones_like(uu.reshape(-1))],
            dim=0,
        ).unsqueeze(0).expand(batch, -1, -1)

        dirs_cam = torch.bmm(torch.inverse(K), pixels).transpose(1, 2)
        depths = self.ray_sample_scales.to(device=device, dtype=geom_dtype)
        pts_cam = dirs_cam.unsqueeze(2) * depths.view(1, 1, self.ray_num_samples, 1)

        R = T_cam[:, :3, :3]
        cam_center = T_cam[:, :3, 3]
        pts_world = torch.einsum("bij,bnkj->bnki", R, pts_cam) + cam_center[:, None, None, :]

        if T_ego is not None:
            center_xy = T_ego[:, :2, 3]
        else:
            center_xy = cam_center[:, :2]

        if cam_height is not None:
            ground_z = cam_center[:, 2] - cam_height
        elif T_ego is not None:
            ground_z = T_ego[:, 2, 3]
        else:
            ground_z = torch.zeros(batch, device=device, dtype=geom_dtype)

        x_norm = (pts_world[..., 0] - center_xy[:, None, None, 0]) / self.ray_scene_extent_x_m
        y_norm = (pts_world[..., 1] - center_xy[:, None, None, 1]) / self.ray_scene_extent_y_m
        z_m = pts_world[..., 2] - ground_z[:, None, None]
        z_norm = (
            (z_m - self.ray_scene_z_min_m)
            / (self.ray_scene_z_max_m - self.ray_scene_z_min_m)
        ) * 2.0 - 1.0
        xyz_norm = torch.stack([x_norm, y_norm, z_norm], dim=-1).to(dtype=dtype)

        valid = (
            (depths.view(1, 1, -1) > 0.0)
            & (x_norm >= -1.0)
            & (x_norm <= 1.0)
            & (y_norm >= -1.0)
            & (y_norm <= 1.0)
            & (z_norm >= -1.0)
            & (z_norm <= 1.0)
        )
        return xyz_norm, valid

    def _normalized_offset_from_metric(self, offsets_m: torch.Tensor) -> torch.Tensor:
        z_extent = self.ray_scene_z_max_m - self.ray_scene_z_min_m
        scale = offsets_m.new_tensor(
            [
                1.0 / self.ray_scene_extent_x_m,
                1.0 / self.ray_scene_extent_y_m,
                2.0 / z_extent,
            ]
        )
        return offsets_m * scale.view(1, 1, 1, 1, 3)

    @staticmethod
    def _normalized_points_valid(xyz_norm: torch.Tensor) -> torch.Tensor:
        return (
            (xyz_norm[..., 0] >= -1.0)
            & (xyz_norm[..., 0] <= 1.0)
            & (xyz_norm[..., 1] >= -1.0)
            & (xyz_norm[..., 1] <= 1.0)
            & (xyz_norm[..., 2] >= -1.0)
            & (xyz_norm[..., 2] <= 1.0)
        )

    def _triplane_lookup(
        self,
        plane_maps: Dict[str, torch.Tensor],
        xyz_norm: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_queries, num_heads, num_samples = xyz_norm.shape[:4]
        flat_xyz = xyz_norm.reshape(batch, num_queries * num_heads, num_samples, 3)

        xy_grid = torch.stack([flat_xyz[..., 0], -flat_xyz[..., 1]], dim=-1)
        xz_grid = torch.stack([flat_xyz[..., 0], -flat_xyz[..., 2]], dim=-1)
        yz_grid = torch.stack([flat_xyz[..., 1], -flat_xyz[..., 2]], dim=-1)

        sampled = (
            self._sample_plane(plane_maps["xy"], xy_grid)
            + self._sample_plane(plane_maps["xz"], xz_grid)
            + self._sample_plane(plane_maps["yz"], yz_grid)
        )
        return sampled.reshape(batch, num_queries, num_heads, num_samples, self.query_dim)

    def _sample_scene_with_perspective_rays(
        self,
        scene_memory: Dict[str, Union[torch.Tensor, Tuple[int, int], int]],
        pose_query_tokens: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        ego_to_world: Optional[torch.Tensor],
        camera_height_m: Optional[torch.Tensor],
        source_image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_queries = pose_query_tokens.shape[:2]
        device = pose_query_tokens.device
        dtype = pose_query_tokens.dtype
        plane_maps = self._prepare_plane_maps(scene_memory, dtype=dtype)
        xy_h, xy_w = scene_memory["xy_hw"]

        base_xyz, base_valid = self._build_perspective_ray_samples(
            intrinsics=intrinsics,
            cam_to_world=cam_to_world,
            ego_to_world=ego_to_world,
            camera_height_m=camera_height_m,
            source_image_size=source_image_size,
            batch=batch,
            device=device,
            dtype=dtype,
        )
        offsets_m = self.perspective_offset_net(pose_query_tokens)
        offsets_m = offsets_m.view(batch, num_queries, self.num_heads, self.ray_num_samples, 3)
        offsets_m = torch.tanh(offsets_m) * self.ray_offset_scale
        offsets_norm = self._normalized_offset_from_metric(offsets_m)

        sample_xyz = base_xyz.unsqueeze(2) + offsets_norm
        sample_valid = base_valid.unsqueeze(2) & self._normalized_points_valid(sample_xyz)

        feats = self._triplane_lookup(plane_maps, sample_xyz)
        feats = feats * sample_valid.unsqueeze(-1).to(dtype=feats.dtype)
        values_full = self.perspective_value_proj(feats)
        values_full = values_full.reshape(
            batch,
            num_queries,
            self.num_heads,
            self.ray_num_samples,
            self.num_heads,
            self.head_dim,
        )
        values = torch.stack(
            [values_full[:, :, head_idx, :, head_idx, :] for head_idx in range(self.num_heads)],
            dim=2,
        )

        attn_logits = self.perspective_attn_net(pose_query_tokens)
        attn_logits = attn_logits.view(batch, num_queries, self.num_heads, self.ray_num_samples)
        attn_logits = attn_logits.masked_fill(~sample_valid, -1.0e4)
        attn_weights = torch.softmax(attn_logits.float(), dim=-1).to(dtype=dtype)
        attn_weights = attn_weights * sample_valid.to(dtype=dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        readout_tokens_raw = (attn_weights.unsqueeze(-1) * values).sum(dim=3)
        readout_tokens_raw = readout_tokens_raw.reshape(batch, num_queries, self.query_dim)
        readout_tokens_raw = self.perspective_head_mix(readout_tokens_raw)

        local_attn_mean = attn_weights.float().mean(dim=2)
        sample_xy = base_xyz[..., :2]
        sample_x_idx = (
            ((sample_xy[..., 0] + 1.0) * 0.5 * float(max(int(xy_w) - 1, 1)))
            .round()
            .to(dtype=torch.long)
            .clamp(0, max(int(xy_w) - 1, 0))
        )
        sample_y_idx = (
            ((1.0 - sample_xy[..., 1]) * 0.5 * float(max(int(xy_h) - 1, 1)))
            .round()
            .to(dtype=torch.long)
            .clamp(0, max(int(xy_h) - 1, 0))
        )
        sample_token_index = sample_y_idx * int(xy_w) + sample_x_idx
        geo_loss = local_attn_mean.new_zeros(())
        return readout_tokens_raw, geo_loss, local_attn_mean, sample_token_index, sample_xy

    def _apply_query_gate(
        self,
        readout_tokens_raw: torch.Tensor,
        pose_query_tokens: torch.Tensor,
        query_input: torch.Tensor,
        valid_tokens: torch.Tensor,
        condition_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([pose_query_tokens, query_input], dim=-1)
        token_gate = torch.sigmoid(self.gate_mlp(gate_input))
        token_gate = token_gate * valid_tokens
        if condition_mask is not None:
            token_gate = token_gate * condition_mask.to(
                device=readout_tokens_raw.device,
                dtype=readout_tokens_raw.dtype,
            ).view(readout_tokens_raw.shape[0], 1, 1)
        return readout_tokens_raw * token_gate, token_gate

    def _pool_readout_tokens(self, readout_tokens_raw: torch.Tensor) -> torch.Tensor:
        batch = readout_tokens_raw.shape[0]
        token_queries = self.token_pool_query.unsqueeze(0).expand(batch, -1, -1)
        token_queries = self.token_pool_query_norm(token_queries)
        pooled_tokens, _ = self.token_pool_attn(
            token_queries,
            readout_tokens_raw,
            readout_tokens_raw,
            need_weights=False,
        )
        return self.token_out(pooled_tokens) * self.token_scale

    def _build_readout_map(self, readout_tokens_raw: torch.Tensor) -> torch.Tensor:
        readout_map_tokens = self.out(readout_tokens_raw) * self.map_scale
        return readout_map_tokens.reshape(
            readout_tokens_raw.shape[0],
            self.grid_h,
            self.grid_w,
            self.out_dim,
        ).permute(0, 3, 1, 2).contiguous()

    def read_scene(
        self,
        scene_memory: Optional[Dict[str, Union[torch.Tensor, Tuple[int, int], int]]] = None,
        sat_tokens: Optional[torch.Tensor] = None,
        sat_xy: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        cam_to_world: Optional[torch.Tensor] = None,
        ego_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
        source_image_size: Optional[Tuple[int, int]] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        scene_memory = self._coerce_scene_memory(scene_memory=scene_memory, sat_tokens=sat_tokens, sat_xy=sat_xy)

        batch = int(scene_memory["xy_tokens"].shape[0])
        device = scene_memory["xy_tokens"].device
        dtype = scene_memory["xy_tokens"].dtype

        source_size = self._coerce_source_image_size(source_image_size)
        intrinsics_b = self._as_batched_intrinsics(intrinsics, batch, device, torch.float32)
        cam_to_world_b = self._as_batched_transform(cam_to_world, batch, device, torch.float32)
        ego_to_world_b = self._as_batched_transform(ego_to_world, batch, device, torch.float32)
        camera_height_b = self._as_batched_scalar(camera_height_m, batch, device, torch.float32)
        missing = []
        if source_size is None:
            missing.append("source_image_size")
        if intrinsics_b is None:
            missing.append("intrinsics")
        if cam_to_world_b is None:
            missing.append("cam_to_world")
        if missing:
            raise ValueError(
                "Perspective ray reader requires "
                + ", ".join(missing)
                + "; no fallback path is available."
            )

        query_input, query_tokens, valid_tokens = self._build_perspective_query_inputs(
            intrinsics=intrinsics_b,
            cam_to_world=cam_to_world_b,
            ego_to_world=ego_to_world_b,
            camera_height_m=camera_height_b,
            source_image_size=source_size,
            device=device,
            dtype=dtype,
            batch=batch,
        )
        readout_tokens_raw, geo_loss, local_attn_mean, local_index, sample_xy = (
            self._sample_scene_with_perspective_rays(
                scene_memory=scene_memory,
                pose_query_tokens=query_tokens,
                intrinsics=intrinsics_b,
                cam_to_world=cam_to_world_b,
                ego_to_world=ego_to_world_b,
                camera_height_m=camera_height_b,
                source_image_size=source_size,
            )
        )
        readout_tokens_raw, token_gate = self._apply_query_gate(
            readout_tokens_raw=readout_tokens_raw,
            pose_query_tokens=query_tokens,
            query_input=query_input,
            valid_tokens=valid_tokens,
            condition_mask=condition_mask,
        )
        readout_tokens = self._pool_readout_tokens(readout_tokens_raw)
        readout_map = self._build_readout_map(readout_tokens_raw)

        entropy = -(local_attn_mean * local_attn_mean.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = (
            entropy * valid_tokens.squeeze(-1).float()
        ).sum() / valid_tokens.squeeze(-1).float().sum().clamp_min(1.0)
        query_anchor_xy = sample_xy[..., 0, :]
        sample_geo_dist = (sample_xy - query_anchor_xy.unsqueeze(2)).pow(2).sum(dim=-1).sqrt().float()
        attn_geo_dist = (local_attn_mean * sample_geo_dist).sum(dim=-1)
        attn_geo_dist = (
            attn_geo_dist * valid_tokens.squeeze(-1).float()
        ).sum() / valid_tokens.squeeze(-1).float().sum().clamp_min(1.0)
        nearest_geo_dist = sample_geo_dist.min(dim=-1).values
        nearest_geo_dist = (
            nearest_geo_dist * valid_tokens.squeeze(-1).float()
        ).sum() / valid_tokens.squeeze(-1).float().sum().clamp_min(1.0)

        if self.save_attention_heatmap:
            keep_tokens = min(self.heatmap_max_tokens, local_attn_mean.shape[1])
            self.last_attention_heatmap = local_attn_mean[:, :keep_tokens].detach().cpu()
            self.last_attention_index = local_index[:, :keep_tokens].detach().cpu()
            self.last_sat_xy = scene_memory["xy_coords"].detach().cpu()
            self.last_view_xy = query_anchor_xy[:, :keep_tokens].detach().cpu()
        else:
            self.last_attention_heatmap = None
            self.last_attention_index = None
            self.last_sat_xy = None
            self.last_view_xy = None

        self.last_query_tokens = query_tokens.detach().cpu()
        self.last_readout_tokens = readout_tokens.detach().cpu()
        self.last_readout_map = readout_map.detach().cpu()
        self.last_readout_tokens_raw = readout_tokens_raw.detach().cpu()
        self.last_stats = {
            "valid_ratio": valid_tokens.float().mean().detach(),
            "gate_mean": token_gate.float().mean().detach(),
            "attn_entropy": entropy.detach(),
            "attn_geo_dist": attn_geo_dist.detach(),
            "nearest_geo_dist": nearest_geo_dist.detach(),
            "geo_kl": geo_loss.detach(),
        }
        return {
            "readout_tokens": readout_tokens,
            "readout_map": readout_map,
            "query_tokens": query_tokens,
            "readout_tokens_raw": readout_tokens_raw,
            "geo_loss": geo_loss,
        }

    def forward(
        self,
        scene_memory: Optional[Dict[str, Union[torch.Tensor, Tuple[int, int], int]]] = None,
        sat_tokens: Optional[torch.Tensor] = None,
        sat_xy: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        cam_to_world: Optional[torch.Tensor] = None,
        ego_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
        source_image_size: Optional[Tuple[int, int]] = None,
        condition_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.read_scene(
            scene_memory=scene_memory,
            sat_tokens=sat_tokens,
            sat_xy=sat_xy,
            intrinsics=intrinsics,
            cam_to_world=cam_to_world,
            ego_to_world=ego_to_world,
            camera_height_m=camera_height_m,
            source_image_size=source_image_size,
            condition_mask=condition_mask,
        )
        if return_dict:
            return outputs
        return outputs["readout_map"]
