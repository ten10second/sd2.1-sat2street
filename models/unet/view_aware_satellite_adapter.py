"""
View-aware satellite scene reader.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewAwareSatelliteAdapter(nn.Module):
    """
    Query local satellite scene memory with pose-conditioned queries.

    The reader keeps the previous map output for backward compatibility, while
    also exposing unified readout tokens so cross-attention and feature-map
    conditioning can come from the same scene readout.
    """

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
        if geo_sigma <= 0.0:
            raise ValueError(f"geo_sigma must be positive, got {geo_sigma}")
        if geo_target_sigma <= 0.0:
            raise ValueError(f"geo_target_sigma must be positive, got {geo_target_sigma}")
        if local_topk <= 0:
            raise ValueError(f"local_topk must be positive, got {local_topk}")
        if token_pool_num_tokens <= 0:
            raise ValueError(f"token_pool_num_tokens must be positive, got {token_pool_num_tokens}")
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
        self.geo_bias_weight = float(geo_bias_weight)
        self.geo_sigma = float(geo_sigma)
        self.local_topk = int(local_topk)
        self.geo_target_sigma = float(geo_target_sigma)
        self.token_pool_num_tokens = int(token_pool_num_tokens)
        self.token_pool_num_heads = int(token_pool_num_heads)
        self.token_scale = float(token_scale)
        self.save_attention_heatmap = bool(save_attention_heatmap)
        self.heatmap_max_tokens = int(max(1, heatmap_max_tokens))
        self.scale = float(scale)
        self.map_scale = float(scale)

        self.query_mlp = nn.Sequential(
            nn.Linear(9, query_dim),
            nn.LayerNorm(query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
        )
        self.sat_norm = nn.LayerNorm(sat_in_dim)
        self.world_proj = nn.Linear(sat_in_dim, query_dim)
        self.world_xy_proj = nn.Sequential(
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

    def _build_query_features(
        self,
        front_bev_xy: Optional[torch.Tensor],
        plucker: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        batch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Backward-compatible wrapper used by older call sites/tests.
        return self._build_pose_query_inputs(
            front_bev_xy=front_bev_xy,
            plucker=plucker,
            valid_mask=valid_mask,
            device=device,
            dtype=dtype,
            batch=batch,
        )

    def _encode_pose_queries(
        self,
        xy_tokens: torch.Tensor,
        plucker_tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_input = torch.cat([xy_tokens, plucker_tokens, valid_tokens], dim=-1)
        query_tokens = self.query_mlp(query_input)
        return query_input, query_tokens

    def _read_scene_locally(
        self,
        scene_tokens: torch.Tensor,
        scene_xy: torch.Tensor,
        pose_query_tokens: torch.Tensor,
        query_xy_tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = scene_tokens.shape[0]
        device = scene_tokens.device
        dtype = scene_tokens.dtype

        world = self.world_proj(self.sat_norm(scene_tokens)) + self.world_xy_proj(
            scene_xy.to(device=device, dtype=dtype)
        )
        q, k, v = F._in_projection_packed(
            pose_query_tokens,
            world,
            world,
            self.attn.in_proj_weight,
            self.attn.in_proj_bias,
        )
        q = q.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        geo_dist2 = (
            query_xy_tokens[:, :, None, :] - scene_xy.to(device=device, dtype=dtype)[:, None, :, :]
        ).pow(2).sum(dim=-1)
        local_topk = min(self.local_topk, scene_tokens.shape[1])
        local_dist2, local_index = torch.topk(geo_dist2, k=local_topk, dim=-1, largest=False)

        head_index = local_index[:, None, :, :].expand(-1, self.num_heads, -1, -1)
        k_expanded = k.unsqueeze(2).expand(-1, -1, q.shape[2], -1, -1)
        v_expanded = v.unsqueeze(2).expand(-1, -1, q.shape[2], -1, -1)
        gather_index = head_index.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        local_k = torch.gather(k_expanded, dim=3, index=gather_index)
        local_v = torch.gather(v_expanded, dim=3, index=gather_index)

        logits = (q.unsqueeze(3) * local_k).sum(dim=-1) / math.sqrt(self.head_dim)
        if self.geo_bias_weight != 0.0:
            local_geo_bias = -self.geo_bias_weight * local_dist2 / (self.geo_sigma ** 2 + 1e-6)
            logits = logits + local_geo_bias.unsqueeze(1)
        attn_weights = torch.softmax(logits.float(), dim=-1).to(dtype=dtype)

        readout_tokens_raw = (attn_weights.unsqueeze(-1) * local_v).sum(dim=3)
        readout_tokens_raw = readout_tokens_raw.transpose(1, 2).reshape(batch, -1, self.query_dim)
        readout_tokens_raw = self.attn.out_proj(readout_tokens_raw)

        local_attn_mean = attn_weights.float().mean(dim=1)
        local_geo_target = torch.softmax(
            (-local_dist2.float() / (self.geo_target_sigma ** 2 + 1e-6)),
            dim=-1,
        )
        geo_kl = (
            local_geo_target
            * (local_geo_target.clamp_min(1e-8).log() - local_attn_mean.clamp_min(1e-8).log())
        ).sum(dim=-1)
        geo_kl = (
            geo_kl * valid_tokens.squeeze(-1).float()
        ).sum() / valid_tokens.squeeze(-1).float().sum().clamp_min(1.0)
        return readout_tokens_raw, geo_kl, local_attn_mean, local_index

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

    def _pool_readout_tokens(
        self,
        readout_tokens_raw: torch.Tensor,
    ) -> torch.Tensor:
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

    def _build_readout_map(
        self,
        readout_tokens_raw: torch.Tensor,
    ) -> torch.Tensor:
        readout_map_tokens = self.out(readout_tokens_raw) * self.map_scale
        return readout_map_tokens.reshape(
            readout_tokens_raw.shape[0],
            self.grid_h,
            self.grid_w,
            self.out_dim,
        ).permute(0, 3, 1, 2).contiguous()

    def read_scene(
        self,
        sat_tokens: torch.Tensor,
        sat_xy: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor],
        plucker: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if sat_tokens.ndim != 3 or sat_tokens.shape[-1] != self.sat_in_dim:
            raise ValueError(
                f"sat_tokens must be [B,N,{self.sat_in_dim}], got {list(sat_tokens.shape)}"
            )
        if sat_xy.ndim != 3 or sat_xy.shape[:2] != sat_tokens.shape[:2] or sat_xy.shape[-1] != 2:
            raise ValueError(
                f"sat_xy must be [B,{sat_tokens.shape[1]},2], got {list(sat_xy.shape)}"
            )

        batch = sat_tokens.shape[0]
        device = sat_tokens.device
        dtype = sat_tokens.dtype

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
        readout_tokens_raw, geo_kl, local_attn_mean, local_index = self._read_scene_locally(
            scene_tokens=sat_tokens,
            scene_xy=sat_xy,
            pose_query_tokens=query_tokens,
            query_xy_tokens=xy_tokens,
            valid_tokens=valid_tokens,
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
        local_geo_dist = torch.gather(
            (xy_tokens[:, :, None, :] - sat_xy.to(device=device, dtype=dtype)[:, None, :, :]).pow(2).sum(dim=-1).sqrt(),
            dim=-1,
            index=local_index,
        ).float()
        attn_geo_dist = (local_attn_mean * local_geo_dist).sum(dim=-1)
        attn_geo_dist = (
            attn_geo_dist * valid_tokens.squeeze(-1).float()
        ).sum() / valid_tokens.squeeze(-1).float().sum().clamp_min(1.0)
        nearest_geo_dist = local_geo_dist[..., 0]
        nearest_geo_dist = (
            nearest_geo_dist * valid_tokens.squeeze(-1).float()
        ).sum() / valid_tokens.squeeze(-1).float().sum().clamp_min(1.0)

        if self.save_attention_heatmap:
            keep_tokens = min(self.heatmap_max_tokens, local_attn_mean.shape[1])
            self.last_attention_heatmap = local_attn_mean[:, :keep_tokens].detach().cpu()
            self.last_attention_index = local_index[:, :keep_tokens].detach().cpu()
            self.last_sat_xy = sat_xy.detach().cpu()
            self.last_view_xy = xy_tokens[:, :keep_tokens].detach().cpu()
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
            "geo_kl": geo_kl.detach(),
        }
        return {
            "readout_tokens": readout_tokens,
            "readout_map": readout_map,
            "query_tokens": query_tokens,
            "readout_tokens_raw": readout_tokens_raw,
            "geo_loss": geo_kl,
        }

    def forward(
        self,
        sat_tokens: torch.Tensor,
        sat_xy: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor],
        plucker: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.read_scene(
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
