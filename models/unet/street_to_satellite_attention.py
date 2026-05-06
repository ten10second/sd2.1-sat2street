"""
Street-to-satellite attention for bidirectional cross-view refinement.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import math

import torch
import torch.nn as nn

from .continuous_xy_georope import ContinuousXYGeoRoPE


class StreetToSatelliteAttention(nn.Module):
    """
    Use street-view features to update the satellite memory at aligned ground locations.
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
        use_front_layer_norm: bool = True,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError(f"num_heads/head_dim must be positive, got {num_heads}/{head_dim}")

        model_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.lambda_geo = float(lambda_geo)
        self.lambda_geom = float(lambda_geom)
        self.use_geom_bias = bool(use_geom_bias)
        self.geom_head_dim = int(geom_head_dim)
        self.geom_model_dim = self.num_heads * self.geom_head_dim
        if geom_hidden_dim <= 0:
            raise ValueError(f"geom_hidden_dim must be positive, got {geom_hidden_dim}")
        if self.geom_head_dim <= 0:
            raise ValueError(f"geom_head_dim must be positive, got {self.geom_head_dim}")

        self.sat_adapter = nn.Sequential(
            nn.Linear(sat_in_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.front_position_mlp = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.front_adapter = nn.Sequential(
            nn.Linear(front_in_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.front_norm = nn.LayerNorm(model_dim)

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, sat_in_dim)
        self.out_norm = nn.LayerNorm(sat_in_dim)

        self.sat_xy_encoder = nn.Sequential(
            nn.Linear(2, geom_hidden_dim),
            nn.SiLU(),
            nn.Linear(geom_hidden_dim, geom_hidden_dim),
        )
        self.front_plucker_encoder = nn.Sequential(
            nn.Linear(6, geom_hidden_dim),
            nn.SiLU(),
            nn.Linear(geom_hidden_dim, geom_hidden_dim),
        )
        self.front_xy_encoder = nn.Sequential(
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

        if use_front_layer_norm:
            self.front_token_norm = nn.LayerNorm(model_dim)
        else:
            self.front_token_norm = nn.Identity()

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

    def _geometry_bias(self, sat_xy: torch.Tensor, front_xy: torch.Tensor) -> torch.Tensor:
        dist2 = (sat_xy.unsqueeze(2) - front_xy.unsqueeze(1)).pow(2).sum(dim=-1)
        return -self.lambda_geo * dist2

    @staticmethod
    def _apply_key_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected key mask [B,N], got {list(mask.shape)}")
        return logits.masked_fill(~mask[:, None, None, :], -1e4)

    @staticmethod
    def _normalize_masked_attention(attn_map: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return attn_map
        masked = attn_map * mask[:, None, None, :].to(dtype=attn_map.dtype)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return masked / denom

    def forward(
        self,
        front_feat: torch.Tensor,
        front_bev_xy: torch.Tensor,
        sat_tokens: torch.Tensor,
        sat_xy: torch.Tensor,
        front_plucker: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        return_attn_map: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if front_feat.ndim != 4:
            raise ValueError("front_feat must be rank-4 [B,C,H,W]")
        if front_bev_xy.ndim != 3 or sat_tokens.ndim != 3 or sat_xy.ndim != 3:
            raise ValueError("front_bev_xy, sat_tokens, sat_xy must all be rank-3 tensors")
        if front_bev_xy.shape[-1] != 2 or sat_xy.shape[-1] != 2:
            raise ValueError("front_bev_xy/sat_xy last dim must be 2")
        if sat_tokens.shape[0] != front_bev_xy.shape[0] or sat_xy.shape[0] != sat_tokens.shape[0]:
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

        if front_ground_valid_mask is not None:
            if front_ground_valid_mask.ndim == 3 and front_ground_valid_mask.shape[-1] == 1:
                front_ground_valid_mask = front_ground_valid_mask.squeeze(-1)
            if front_ground_valid_mask.ndim != 2 or front_ground_valid_mask.shape != front_bev_xy.shape[:2]:
                raise ValueError(
                    f"front_ground_valid_mask must be [B,Nf], got {list(front_ground_valid_mask.shape)} "
                    f"for expected {list(front_bev_xy.shape[:2])}"
                )
            front_ground_valid_mask = front_ground_valid_mask.to(device=front_feat.device, dtype=torch.bool)

        front_pos_embed = self.front_position_mlp(front_bev_xy)
        front_feat_embed = self.front_adapter(front_feat_flat)
        front_tokens = self.front_token_norm(front_pos_embed + front_feat_embed)
        sat_queries = self.sat_adapter(sat_tokens)

        q = self._reshape_heads(self.q_proj(sat_queries))
        k = self._reshape_heads(self.k_proj(front_tokens))
        v = self._reshape_heads(self.v_proj(front_tokens))

        q_tilde, k_tilde = self.rope(q, k, sat_xy, front_bev_xy)

        logits_sem = torch.matmul(q_tilde, k_tilde.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits_sem
        if front_plucker is not None:
            if front_plucker.ndim != 3 or front_plucker.shape[-1] != 6:
                raise ValueError("front_plucker must be [B,Nf,6]")
            if front_plucker.shape[:2] != front_bev_xy.shape[:2]:
                raise ValueError(
                    f"front_plucker token count mismatch: expected {list(front_bev_xy.shape[:2])}, "
                    f"got {list(front_plucker.shape[:2])}"
                )
            sat_geom_feat = self.geom_q_norm(self.sat_xy_encoder(sat_xy))
            front_geom_source = self.front_plucker_encoder(front_plucker)
            q_geom = self._reshape_heads(self.q_geom_proj(sat_geom_feat), self.geom_head_dim)
            k_geom = self._reshape_heads(self.k_geom_proj(self.geom_k_norm(front_geom_source)), self.geom_head_dim)
            logits_geom = torch.matmul(q_geom, k_geom.transpose(-2, -1)) / math.sqrt(self.geom_head_dim)
            logits = logits + self.lambda_geom * logits_geom
        else:
            front_geom_source = self.front_xy_encoder(front_bev_xy)
            q_geom = self._reshape_heads(self.q_geom_proj(self.geom_q_norm(self.sat_xy_encoder(sat_xy))), self.geom_head_dim)
            k_geom = self._reshape_heads(self.k_geom_proj(self.geom_k_norm(front_geom_source)), self.geom_head_dim)
            logits_geom = torch.matmul(q_geom, k_geom.transpose(-2, -1)) / math.sqrt(self.geom_head_dim)
            logits = logits + self.lambda_geom * logits_geom

        if self.use_geom_bias:
            logits = logits + self._geometry_bias(sat_xy, front_bev_xy).unsqueeze(1)
        if front_ground_valid_mask is not None:
            logits = self._apply_key_mask(logits, front_ground_valid_mask)

        attn_map = torch.softmax(logits, dim=-1)
        attn_map = self.attn_dropout(attn_map)
        attn_map = self._normalize_masked_attention(attn_map, front_ground_valid_mask)

        sat_update = torch.matmul(attn_map, v)
        sat_update = sat_update.transpose(1, 2).reshape(batch_size, sat_xy.shape[1], self.model_dim)
        sat_tokens_out = self.out_norm(sat_tokens + self.out_proj(sat_update))

        logits_sem_std = logits_sem.float().std(unbiased=False)
        logits_geom_std = (self.lambda_geom * logits_geom).float().std(unbiased=False)
        update_norm = (sat_tokens_out - sat_tokens).float().norm(dim=-1).mean()
        stats = {
            "logits_sem_std": logits_sem_std.detach(),
            "logits_geom_std": logits_geom_std.detach(),
            "logits_geom_to_sem_ratio": (logits_geom_std / logits_sem_std.clamp_min(1e-8)).detach(),
            "sat_update_norm": update_norm.detach(),
        }

        return sat_tokens_out, (attn_map if return_attn_map else None), stats
