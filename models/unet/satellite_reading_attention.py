"""
Satellite reading attention with GeoRoPE and geometry bias.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import math

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
        lambda_geom: float = 1.0,
        geom_hidden_dim: int = 128,
        geom_head_dim: int = 16,
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
        self.lambda_geom = float(lambda_geom)
        self.use_geom_bias = use_geom_bias
        self.geom_head_dim = int(geom_head_dim)
        self.geom_model_dim = self.num_heads * self.geom_head_dim
        if geom_hidden_dim <= 0:
            raise ValueError(f"geom_hidden_dim must be positive, got {geom_hidden_dim}")
        if self.geom_head_dim <= 0:
            raise ValueError(f"geom_head_dim must be positive, got {self.geom_head_dim}")

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
        self.plk_encoder = nn.Sequential(
            nn.Linear(6, geom_hidden_dim),
            nn.SiLU(),
            nn.Linear(geom_hidden_dim, geom_hidden_dim),
        )
        self.sat_xy_encoder = nn.Sequential(
            nn.Linear(2, geom_hidden_dim),
            nn.SiLU(),
            nn.Linear(geom_hidden_dim, geom_hidden_dim),
        )
        self.geom_q_norm = nn.LayerNorm(geom_hidden_dim)
        self.geom_k_norm = nn.LayerNorm(geom_hidden_dim)
        self.q_geom_proj = nn.Linear(geom_hidden_dim, self.geom_model_dim)
        self.k_geom_proj = nn.Linear(geom_hidden_dim, self.geom_model_dim)

        self.rope = ContinuousXYGeoRoPE(
            head_dim=head_dim,
            geo_ratio=geo_ratio,
            rope_base=rope_base,
        )
        self.attn_dropout = nn.Dropout(attn_dropout)

    def _reshape_heads(self, tensor: torch.Tensor, head_dim: Optional[int] = None) -> torch.Tensor:
        head_dim = self.head_dim if head_dim is None else int(head_dim)
        batch_size, token_count, _ = tensor.shape
        expected_dim = self.num_heads * head_dim
        if tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"Cannot reshape tensor with last dim {tensor.shape[-1]} into "
                f"[num_heads={self.num_heads}, head_dim={head_dim}]"
            )
        return tensor.reshape(batch_size, token_count, self.num_heads, head_dim).transpose(1, 2)

    def _geometry_bias(self, q_xy: torch.Tensor, k_xy: torch.Tensor) -> torch.Tensor:
        dist2 = (q_xy.unsqueeze(2) - k_xy.unsqueeze(1)).pow(2).sum(dim=-1)
        return -self.lambda_geo * dist2

    def forward(
        self,
        front_feat: torch.Tensor,
        front_bev_xy: torch.Tensor,
        sat_tokens: torch.Tensor,
        sat_xy: torch.Tensor,
        front_plucker: Optional[torch.Tensor] = None,
        return_attn_map: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
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
        if front_plucker is not None:
            if front_plucker.ndim != 3 or front_plucker.shape[-1] != 6:
                raise ValueError("front_plucker must be [B,Nf,6]")
            if front_plucker.shape[:2] != front_bev_xy.shape[:2]:
                raise ValueError(
                    f"front_plucker token count mismatch: expected {list(front_bev_xy.shape[:2])}, "
                    f"got {list(front_plucker.shape[:2])}"
                )

        pos_embed = self.position_mlp(front_bev_xy)
        feat_embed = self.front_adapter(front_feat_flat)
        q_embed = self.query_norm(pos_embed + feat_embed)
        sat_feat = self.sat_adapter(sat_tokens)

        q = self._reshape_heads(self.q_proj(q_embed), self.head_dim)
        k = self._reshape_heads(self.k_proj(sat_feat), self.head_dim)
        v = self._reshape_heads(self.v_proj(sat_feat))

        q_tilde, k_tilde = self.rope(q, k, front_bev_xy, sat_xy)

        logits_sem = torch.matmul(q_tilde, k_tilde.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits_sem
        if front_plucker is not None:
            plk_feat = self.geom_q_norm(self.plk_encoder(front_plucker))
            sat_geom_feat = self.geom_k_norm(self.sat_xy_encoder(sat_xy))
            q_geom = self._reshape_heads(self.q_geom_proj(plk_feat), self.geom_head_dim)
            k_geom = self._reshape_heads(self.k_geom_proj(sat_geom_feat), self.geom_head_dim)
            logits_geom = torch.matmul(q_geom, k_geom.transpose(-2, -1)) / math.sqrt(self.geom_head_dim)
            logits = logits + self.lambda_geom * logits_geom
        else:
            logits_geom = None
        if self.use_geom_bias:
            logits = logits + self._geometry_bias(front_bev_xy, sat_xy).unsqueeze(1)

        attn_map = torch.softmax(logits, dim=-1)
        attn_map = self.attn_dropout(attn_map)

        read_tokens = torch.matmul(attn_map, v)
        read_tokens = read_tokens.transpose(1, 2).reshape(batch_size, front_bev_xy.shape[1], self.model_dim)

        logits_sem_std = logits_sem.float().std(unbiased=False)
        if logits_geom is None:
            logits_geom_std = logits_sem_std.new_zeros(())
        else:
            logits_geom_std = (self.lambda_geom * logits_geom).float().std(unbiased=False)
        stats = {
            "logits_sem_std": logits_sem_std.detach(),
            "logits_geom_std": logits_geom_std.detach(),
            "logits_geom_to_sem_ratio": (logits_geom_std / logits_sem_std.clamp_min(1e-8)).detach(),
        }

        return read_tokens, (attn_map if return_attn_map else None), stats
