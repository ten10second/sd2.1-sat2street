"""
Satellite reading attention with GeoRoPE and geometry bias.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .continuous_xy_georope import ContinuousXYGeoRoPE


class SatelliteReadingAttention(nn.Module):
    """
    Build Q/K/V, apply GeoRoPE, read satellite tokens for front queries.

    Queries combine geometric position and the current U-Net front feature so the
    reading policy can change across denoising steps and feature hierarchies.
    """

    def __init__(
        self,
        sat_in_dim: int,
        front_in_dim: int,
        num_heads: int,
        head_dim: int,
        geo_ratio: float = 0.5,
        rope_base: float = 10000.0,
        lambda_geo: float = 1.0,
        use_geom_bias: bool = True,
        use_sat_layer_norm: bool = True,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError(f"num_heads/head_dim must be positive, got {num_heads}/{head_dim}")

        model_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.lambda_geo = lambda_geo
        self.use_geom_bias = use_geom_bias

        self.position_mlp = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.front_adapter = nn.Sequential(
            nn.Linear(front_in_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.query_norm = nn.LayerNorm(model_dim)

        adapter = [nn.Linear(sat_in_dim, model_dim)]
        if use_sat_layer_norm:
            adapter.append(nn.LayerNorm(model_dim))
        self.sat_adapter = nn.Sequential(*adapter)

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

        self.rope = ContinuousXYGeoRoPE(
            head_dim=head_dim,
            geo_ratio=geo_ratio,
            rope_base=rope_base,
        )
        self.attn_dropout = nn.Dropout(attn_dropout)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tensor.shape
        return tensor.reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def _geometry_bias(self, q_xy: torch.Tensor, k_xy: torch.Tensor) -> torch.Tensor:
        dist2 = (q_xy.unsqueeze(2) - k_xy.unsqueeze(1)).pow(2).sum(dim=-1)
        return -self.lambda_geo * dist2

    def forward(
        self,
        front_feat: torch.Tensor,
        front_bev_xy: torch.Tensor,
        sat_tokens: torch.Tensor,
        sat_xy: torch.Tensor,
        return_attn_map: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if front_feat.ndim != 4:
            raise ValueError("front_feat must be rank-4 [B,C,H,W]")
        if front_bev_xy.ndim != 3 or sat_tokens.ndim != 3 or sat_xy.ndim != 3:
            raise ValueError("front_bev_xy, sat_tokens, sat_xy must all be rank-3 tensors")
        if front_bev_xy.shape[-1] != 2 or sat_xy.shape[-1] != 2:
            raise ValueError("front_bev_xy/sat_xy last dim must be 2")
        if (
            front_feat.shape[0] != front_bev_xy.shape[0] or
            sat_tokens.shape[0] != front_bev_xy.shape[0] or
            sat_xy.shape[0] != front_bev_xy.shape[0]
        ):
            raise ValueError("Batch size mismatch among front_bev_xy/sat_tokens/sat_xy")
        if sat_tokens.shape[1] != sat_xy.shape[1]:
            raise ValueError("sat_tokens and sat_xy token count mismatch")

        batch_size, _, height, width = front_feat.shape
        front_feat_flat = front_feat.flatten(2).transpose(1, 2)
        expected_nf = height * width
        if front_bev_xy.shape[1] != expected_nf:
            raise ValueError(
                f"front_bev_xy token count mismatch: expected {expected_nf}, got {front_bev_xy.shape[1]}"
            )
        if front_feat_flat.shape[:2] != front_bev_xy.shape[:2]:
            raise ValueError(
                f"front_feat token count mismatch: expected {list(front_bev_xy.shape[:2])}, "
                f"got {list(front_feat_flat.shape[:2])}"
            )

        pos_embed = self.position_mlp(front_bev_xy)
        feat_embed = self.front_adapter(front_feat_flat)
        q_embed = self.query_norm(pos_embed + feat_embed)
        sat_feat = self.sat_adapter(sat_tokens)

        q = self._reshape_heads(self.q_proj(q_embed))
        k = self._reshape_heads(self.k_proj(sat_feat))
        v = self._reshape_heads(self.v_proj(sat_feat))

        q_tilde, k_tilde = self.rope(q, k, front_bev_xy, sat_xy)

        logits = torch.matmul(q_tilde, k_tilde.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.use_geom_bias:
            logits = logits + self._geometry_bias(front_bev_xy, sat_xy).unsqueeze(1)

        attn_map = torch.softmax(logits, dim=-1)
        attn_map = self.attn_dropout(attn_map)

        read_tokens = torch.matmul(attn_map, v)
        read_tokens = read_tokens.transpose(1, 2).reshape(batch_size, front_bev_xy.shape[1], self.model_dim)

        return read_tokens, (attn_map if return_attn_map else None)
