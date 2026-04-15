"""
Satellite reading block: attention reading + gated residual injection.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .gated_residual_inject import GatedResidualInject
from .satellite_reading_attention import SatelliteReadingAttention


class SatelliteReadingBlock(nn.Module):
    """
    End-to-end reading block for one U-Net feature map scale.
    """

    def __init__(
        self,
        front_dim: int,
        sat_in_dim: int,
        num_heads: int,
        head_dim: int,
        geo_ratio: float = 0.5,
        rope_base: float = 10000.0,
        lambda_geo: float = 1.0,
        gate_hidden_ratio: float = 0.25,
        use_geom_bias: bool = True,
        use_gated_residual: bool = True,
    ):
        super().__init__()
        self.model_dim = num_heads * head_dim

        self.read_attn = SatelliteReadingAttention(
            sat_in_dim=sat_in_dim,
            front_in_dim=front_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            geo_ratio=geo_ratio,
            rope_base=rope_base,
            lambda_geo=lambda_geo,
            use_geom_bias=use_geom_bias,
        )
        self.inject = GatedResidualInject(
            front_dim=front_dim,
            read_dim=self.model_dim,
            gate_hidden_ratio=gate_hidden_ratio,
            use_gated_residual=use_gated_residual,
        )

    def forward(
        self,
        front_feat: torch.Tensor,
        sat_tokens: torch.Tensor,
        sat_xy: torch.Tensor,
        front_bev_xy: torch.Tensor,
        front_plucker: Optional[torch.Tensor] = None,
        return_attn_map: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if front_feat.ndim != 4:
            raise ValueError("front_feat must be rank-4 [B,C,H,W]")
        if front_bev_xy.ndim != 3 or front_bev_xy.shape[-1] != 2:
            raise ValueError("front_bev_xy must be [B,Nf,2]")

        batch_size, _, height, width = front_feat.shape
        expected_nf = height * width
        if front_bev_xy.shape[0] != batch_size or front_bev_xy.shape[1] != expected_nf:
            raise ValueError(
                f"front_bev_xy shape mismatch, expected [B,{expected_nf},2], got {list(front_bev_xy.shape)}"
            )

        read_tokens, attn_map = self.read_attn(
            front_feat=front_feat,
            front_bev_xy=front_bev_xy,
            sat_tokens=sat_tokens,
            sat_xy=sat_xy,
            front_plucker=front_plucker,
            return_attn_map=return_attn_map,
        )
        front_feat_out, read_feat, gate = self.inject(front_feat, read_tokens)

        return {
            "front_feat_out": front_feat_out,
            "read_feat": read_feat,
            "attn_map": attn_map,
            "gate": gate,
        }
