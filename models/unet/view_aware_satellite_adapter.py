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
        ray_num_samples: int = 8,
        ray_depth_min: float = 0.15,
        ray_depth_max: float = 1.25,
        ray_offset_scale: float = 0.10,
        ray_boundary_scale: float = 0.95,
        ray_height_scale: float = 0.75,
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
        self.ray_boundary_scale = float(ray_boundary_scale)
        self.ray_height_scale = float(ray_height_scale)
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
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ray_depth_proj = nn.Sequential(
            nn.Linear(1, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
        )
        self.ray_offset_mlp = nn.Sequential(
            nn.Linear(query_dim + 9, query_dim),
            nn.LayerNorm(query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, self.ray_num_samples * 3),
        )
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
        self.plane_embeddings = nn.Parameter(
            torch.randn(3, query_dim) / math.sqrt(query_dim)
        )
        self.register_buffer(
            "ray_sample_scales",
            torch.linspace(self.ray_depth_min, self.ray_depth_max, steps=self.ray_num_samples),
            persistent=False,
        )

        self.last_stats: Dict[str, torch.Tensor] = {}
        self.last_attention_heatmap: Optional[torch.Tensor] = None
        self.last_attention_index: Optional[torch.Tensor] = None
        self.last_sat_xy: Optional[torch.Tensor] = None
        self.last_view_xy: Optional[torch.Tensor] = None
        self.last_query_tokens: Optional[torch.Tensor] = None
        self.last_readout_tokens: Optional[torch.Tensor] = None
        self.last_readout_map: Optional[torch.Tensor] = None
        self.last_readout_tokens_raw: Optional[torch.Tensor] = None

    @staticmethod
    def _as_bchw_map(
        value: Optional[torch.Tensor],
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if value is None or not torch.is_tensor(value):
            return None
        x = value.to(device=device, dtype=dtype)
        if x.ndim == 4 and x.shape[1] == channels:
            return x
        if x.ndim == 4 and x.shape[-1] == channels:
            return x.permute(0, 3, 1, 2)
        if x.ndim == 3 and x.shape[-1] == channels:
            token_count = x.shape[1]
            h = int(math.sqrt(token_count))
            while h > 1 and token_count % h != 0:
                h -= 1
            w = token_count // h
            return x.reshape(x.shape[0], h, w, channels).permute(0, 3, 1, 2)
        return None

    @staticmethod
    def _pool_map(x: torch.Tensor, size: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
        if mode == "nearest":
            pooled = F.interpolate(x, size=size, mode=mode)
        else:
            pooled = F.interpolate(x, size=size, mode=mode, align_corners=False)
        return pooled.permute(0, 2, 3, 1).reshape(x.shape[0], size[0] * size[1], x.shape[1])

    @staticmethod
    def _normalize_xy(xy: torch.Tensor) -> torch.Tensor:
        return xy / xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _normalize_xyz(xyz: torch.Tensor) -> torch.Tensor:
        return xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _build_pose_query_inputs(
        self,
        front_bev_xy: Optional[torch.Tensor],
        plucker: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        batch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = (self.grid_h, self.grid_w)

        xy_map = self._as_bchw_map(front_bev_xy, 2, device, dtype)
        if xy_map is None:
            xy_tokens = torch.zeros(batch, self.grid_h * self.grid_w, 2, device=device, dtype=dtype)
        else:
            xy_tokens = self._pool_map(xy_map, size)

        plucker_map = self._as_bchw_map(plucker, 6, device, dtype)
        if plucker_map is None:
            plucker_tokens = torch.zeros(batch, self.grid_h * self.grid_w, 6, device=device, dtype=dtype)
        else:
            plucker_tokens = self._pool_map(plucker_map, size)

        mask_map = self._as_bchw_map(valid_mask, 1, device, dtype)
        if mask_map is None:
            valid_tokens = torch.ones(batch, self.grid_h * self.grid_w, 1, device=device, dtype=dtype)
        else:
            valid_tokens = self._pool_map(mask_map, size, mode="nearest").clamp(0.0, 1.0)

        return xy_tokens, plucker_tokens, valid_tokens

    def _encode_pose_queries(
        self,
        xy_tokens: torch.Tensor,
        plucker_tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_input = torch.cat([xy_tokens, plucker_tokens, valid_tokens], dim=-1)
        query_tokens = self.query_mlp(query_input)
        return query_input, query_tokens

    def _build_query_anchor(
        self,
        xy_tokens: torch.Tensor,
        plucker_tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_binary = (valid_tokens > 0.5).to(dtype=xy_tokens.dtype)
        ray_dir_xyz = self._normalize_xyz(plucker_tokens[..., :3])
        ray_dir_xy = ray_dir_xyz[..., :2]
        xy_dir = self._normalize_xy(xy_tokens)
        ray_dir_xy_norm = ray_dir_xy.norm(dim=-1, keepdim=True)
        ray_dir_xy = torch.where(ray_dir_xy_norm > 1e-6, ray_dir_xy / ray_dir_xy_norm.clamp_min(1e-6), xy_dir)
        boundary_denom = ray_dir_xy.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        boundary_xy = (ray_dir_xy / boundary_denom) * self.ray_boundary_scale

        has_anchor = (xy_tokens.abs().amax(dim=-1, keepdim=True) > 1e-6).to(dtype=xy_tokens.dtype)
        use_anchor = valid_binary * has_anchor
        anchor_xy = use_anchor * xy_tokens + (1.0 - use_anchor) * boundary_xy

        anchor_dir = self._normalize_xy(anchor_xy)
        ray_dir_xy = torch.where(
            (ray_dir_xy.abs().amax(dim=-1, keepdim=True) > 1e-6),
            ray_dir_xy,
            anchor_dir,
        )
        ray_dir_xy = self._normalize_xy(ray_dir_xy)

        ray_dir_xyz = torch.cat([ray_dir_xy, ray_dir_xyz[..., 2:3]], dim=-1)
        ray_dir_xyz = self._normalize_xyz(ray_dir_xyz)
        return anchor_xy, ray_dir_xy, ray_dir_xyz

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

    def _sample_scene_along_rays(
        self,
        scene_memory: Dict[str, Union[torch.Tensor, Tuple[int, int], int]],
        pose_query_tokens: torch.Tensor,
        query_input: torch.Tensor,
        query_anchor_xy: torch.Tensor,
        ray_dir_xy: torch.Tensor,
        ray_dir_xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_queries = pose_query_tokens.shape[:2]
        device = pose_query_tokens.device
        dtype = pose_query_tokens.dtype
        plane_maps = self._prepare_plane_maps(scene_memory, dtype=dtype)
        xy_h, xy_w = scene_memory["xy_hw"]

        base_scales = self.ray_sample_scales.to(device=device, dtype=dtype).view(1, 1, self.ray_num_samples, 1)
        base_xy = query_anchor_xy.unsqueeze(2) * base_scales
        base_z = (-ray_dir_xyz[..., 2:3]).unsqueeze(2) * base_scales * self.ray_height_scale

        ray_perp_xy = torch.stack([-ray_dir_xy[..., 1], ray_dir_xy[..., 0]], dim=-1)
        raw_offsets = self.ray_offset_mlp(torch.cat([pose_query_tokens, query_input], dim=-1))
        raw_offsets = raw_offsets.view(batch, num_queries, self.ray_num_samples, 3)
        raw_offsets = torch.tanh(raw_offsets) * self.ray_offset_scale

        sample_xy = (
            base_xy
            + raw_offsets[..., :1] * ray_dir_xy.unsqueeze(2)
            + raw_offsets[..., 1:2] * ray_perp_xy.unsqueeze(2)
        ).clamp(
            min=-self.ray_boundary_scale * 1.25,
            max=self.ray_boundary_scale * 1.25,
        )
        sample_z = (base_z + raw_offsets[..., 2:3]).clamp(-1.0, 1.0)

        xy_grid = torch.stack([sample_xy[..., 0], -sample_xy[..., 1]], dim=-1)
        xz_grid = torch.stack([sample_xy[..., 0], -sample_z[..., 0]], dim=-1)
        yz_grid = torch.stack([sample_xy[..., 1], -sample_z[..., 0]], dim=-1)

        xy_sampled = self._sample_plane(plane_maps["xy"], xy_grid)
        xz_sampled = self._sample_plane(plane_maps["xz"], xz_grid)
        yz_sampled = self._sample_plane(plane_maps["yz"], yz_grid)

        xy_coord_tokens = self.coord_proj(sample_xy.reshape(batch, -1, 2)).reshape(
            batch, num_queries, self.ray_num_samples, self.query_dim
        )
        xz_coord = torch.cat([sample_xy[..., 0:1], sample_z], dim=-1)
        xz_coord_tokens = self.coord_proj(xz_coord.reshape(batch, -1, 2)).reshape(
            batch, num_queries, self.ray_num_samples, self.query_dim
        )
        yz_coord = torch.cat([sample_xy[..., 1:2], sample_z], dim=-1)
        yz_coord_tokens = self.coord_proj(yz_coord.reshape(batch, -1, 2)).reshape(
            batch, num_queries, self.ray_num_samples, self.query_dim
        )
        depth_tokens = self.ray_depth_proj(base_scales.expand(batch, num_queries, -1, -1))

        xy_tokens = xy_sampled + xy_coord_tokens + depth_tokens + self.plane_embeddings[0].view(1, 1, 1, -1)
        xz_tokens = xz_sampled + xz_coord_tokens + depth_tokens + self.plane_embeddings[1].view(1, 1, 1, -1)
        yz_tokens = yz_sampled + yz_coord_tokens + depth_tokens + self.plane_embeddings[2].view(1, 1, 1, -1)
        sample_tokens = torch.cat([xy_tokens, xz_tokens, yz_tokens], dim=2)

        q_weight, k_weight, v_weight = self.attn.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.attn.in_proj_bias.chunk(3, dim=0)
        q = F.linear(pose_query_tokens, q_weight, q_bias)
        k = F.linear(sample_tokens, k_weight, k_bias)
        v = F.linear(sample_tokens, v_weight, v_bias)

        total_candidates = sample_tokens.shape[2]
        q = q.reshape(batch, num_queries, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, num_queries, total_candidates, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.reshape(batch, num_queries, total_candidates, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        logits = (q.unsqueeze(3) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(logits.float(), dim=-1).to(dtype=dtype)

        readout_tokens_raw = (attn_weights.unsqueeze(-1) * v).sum(dim=3)
        readout_tokens_raw = readout_tokens_raw.permute(0, 2, 1, 3).reshape(batch, num_queries, self.query_dim)
        readout_tokens_raw = self.attn.out_proj(readout_tokens_raw)

        xy_attn_weights = attn_weights[..., : self.ray_num_samples]
        local_attn_mean = xy_attn_weights.float().mean(dim=1)
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
        front_bev_xy: Optional[torch.Tensor] = None,
        plucker: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        scene_memory = self._coerce_scene_memory(scene_memory=scene_memory, sat_tokens=sat_tokens, sat_xy=sat_xy)

        batch = int(scene_memory["xy_tokens"].shape[0])
        device = scene_memory["xy_tokens"].device
        dtype = scene_memory["xy_tokens"].dtype

        xy_tokens, plucker_tokens, valid_tokens = self._build_pose_query_inputs(
            front_bev_xy=front_bev_xy,
            plucker=plucker,
            valid_mask=valid_mask,
            device=device,
            dtype=dtype,
            batch=batch,
        )
        query_input, query_tokens = self._encode_pose_queries(
            xy_tokens=xy_tokens,
            plucker_tokens=plucker_tokens,
            valid_tokens=valid_tokens,
        )
        query_anchor_xy, ray_dir_xy, ray_dir_xyz = self._build_query_anchor(
            xy_tokens=xy_tokens,
            plucker_tokens=plucker_tokens,
            valid_tokens=valid_tokens,
        )
        readout_tokens_raw, geo_loss, local_attn_mean, local_index, sample_xy = self._sample_scene_along_rays(
            scene_memory=scene_memory,
            pose_query_tokens=query_tokens,
            query_input=query_input,
            query_anchor_xy=query_anchor_xy,
            ray_dir_xy=ray_dir_xy,
            ray_dir_xyz=ray_dir_xyz,
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
        front_bev_xy: Optional[torch.Tensor] = None,
        plucker: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.read_scene(
            scene_memory=scene_memory,
            sat_tokens=sat_tokens,
            sat_xy=sat_xy,
            front_bev_xy=front_bev_xy,
            plucker=plucker,
            valid_mask=valid_mask,
            condition_mask=condition_mask,
        )
        if return_dict:
            return outputs
        return outputs["readout_map"]
