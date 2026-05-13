"""
Cross-view refinement block:
street -> satellite update, then satellite self-refine.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.conditioning import SatelliteMemoryState

from .relative_position_attention import RelativePositionAttention
from .street_to_satellite_attention import StreetToSatelliteAttention


class CrossViewRefinementBlock(nn.Module):
    """
    One U-Net-scale refinement block with ground-PE-guided satellite memory updates.
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
        lambda_geom: float = 1.0,
        geom_hidden_dim: int = 128,
        geom_head_dim: int = 16,
        use_geom_bias: bool = True,
        sat_update_layers: int = 1,
        sat_self_attn_dropout: float = 0.1,
    ):
        super().__init__()
        if sat_update_layers <= 0:
            raise ValueError(f"sat_update_layers must be positive, got {sat_update_layers}")
        self.sat_update_layers = int(sat_update_layers)

        self.street_to_sat_layers = nn.ModuleList()
        self.sat_self_refine_layers = nn.ModuleList()
        for _ in range(self.sat_update_layers):
            self.street_to_sat_layers.append(
                StreetToSatelliteAttention(
                    sat_in_dim=sat_in_dim,
                    front_in_dim=front_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    geo_ratio=geo_ratio,
                    rope_base=rope_base,
                    lambda_geo=lambda_geo,
                    lambda_geom=lambda_geom,
                    geom_hidden_dim=geom_hidden_dim,
                    geom_head_dim=geom_head_dim,
                    use_geom_bias=use_geom_bias,
                )
            )
            self.sat_self_refine_layers.append(
                RelativePositionAttention(
                    embed_dim=sat_in_dim,
                    num_heads=num_heads,
                    dropout=sat_self_attn_dropout,
                    use_relative_pos=True,
                )
            )

    def forward(
        self,
        front_feat: torch.Tensor,
        satellite_state: SatelliteMemoryState,
        front_bev_xy: torch.Tensor,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        return_attn_map: bool = False,
    ) -> Dict[str, Any]:
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

        sat_tokens_refined = satellite_state.tokens
        street_to_sat_attn = None
        stats: Dict[str, torch.Tensor] = {}
        sat_positions = satellite_state.bev_coords if satellite_state.bev_coords is not None else satellite_state.xy
        for layer_index, (street_to_sat, sat_self_refine) in enumerate(
            zip(self.street_to_sat_layers, self.sat_self_refine_layers)
        ):
            sat_tokens_updated, street_to_sat_attn, update_stats = street_to_sat(
                front_feat=front_feat,
                front_bev_xy=front_bev_xy,
                sat_tokens=sat_tokens_refined,
                sat_xy=satellite_state.xy,
                front_ground_valid_mask=front_ground_valid_mask,
                return_attn_map=return_attn_map,
            )
            sat_tokens_refined = sat_self_refine(sat_tokens_updated, sat_positions)
            for key, value in update_stats.items():
                stats[f"sat_update_l{layer_index}_{key}"] = value
                if layer_index == self.sat_update_layers - 1:
                    stats[key] = value
        updated_satellite_state = satellite_state.replace(tokens=sat_tokens_refined)

        attn_maps = None
        if return_attn_map:
            attn_maps = {
                "street_to_sat": street_to_sat_attn,
            }

        return {
            "attn_map": attn_maps,
            "stats": stats,
            "satellite_state": updated_satellite_state,
        }
