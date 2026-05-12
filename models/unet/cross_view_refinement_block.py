"""
Bidirectional cross-view refinement block:
street -> satellite update, satellite self-refine, satellite -> street read.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.conditioning import SatelliteMemoryState

from .gated_residual_inject import GatedResidualInject
from .relative_position_attention import RelativePositionAttention
from .satellite_reading_attention import SatelliteReadingAttention
from .street_to_satellite_attention import StreetToSatelliteAttention


class CrossViewRefinementBlock(nn.Module):
    """
    One U-Net-scale refinement block with bidirectional cross-view interaction.
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
        gate_hidden_ratio: float = 0.25,
        use_geom_bias: bool = True,
        street_to_sat_use_plucker_geom: bool = False,
        use_gated_residual: bool = True,
        sat_self_attn_dropout: float = 0.1,
        enable_front_refinement: bool = True,
    ):
        super().__init__()
        self.model_dim = num_heads * head_dim
        self.enable_front_refinement = bool(enable_front_refinement)

        self.street_to_sat = StreetToSatelliteAttention(
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
            use_plucker_geom=street_to_sat_use_plucker_geom,
        )
        self.sat_self_refine = RelativePositionAttention(
            embed_dim=sat_in_dim,
            num_heads=num_heads,
            dropout=sat_self_attn_dropout,
            use_relative_pos=True,
        )
        if self.enable_front_refinement:
            self.sat_to_street = SatelliteReadingAttention(
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
            self.inject = GatedResidualInject(
                front_dim=front_dim,
                read_dim=self.model_dim,
                gate_hidden_ratio=gate_hidden_ratio,
                use_gated_residual=use_gated_residual,
            )
        else:
            self.sat_to_street = None
            self.inject = None

    def forward(
        self,
        front_feat: torch.Tensor,
        satellite_state: SatelliteMemoryState,
        front_bev_xy: torch.Tensor,
        front_plucker: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
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

        sat_tokens_updated, street_to_sat_attn, update_stats = self.street_to_sat(
            front_feat=front_feat,
            front_bev_xy=front_bev_xy,
            sat_tokens=satellite_state.tokens,
            sat_xy=satellite_state.xy,
            front_plucker=front_plucker,
            front_ground_valid_mask=front_ground_valid_mask,
            return_attn_map=return_attn_map,
        )
        sat_tokens_refined = self.sat_self_refine(
            sat_tokens_updated,
            satellite_state.bev_coords if satellite_state.bev_coords is not None else satellite_state.xy,
        )
        updated_satellite_state = satellite_state.replace(tokens=sat_tokens_refined)

        stats = dict(update_stats)
        if self.enable_front_refinement:
            read_tokens, sat_to_street_attn, read_stats = self.sat_to_street(
                front_feat=front_feat,
                front_bev_xy=front_bev_xy,
                sat_tokens=updated_satellite_state.tokens,
                sat_xy=updated_satellite_state.xy,
                front_plucker=front_plucker,
                front_ground_valid_mask=front_ground_valid_mask,
                return_attn_map=return_attn_map,
            )
            front_feat_out, read_feat, gate = self.inject(front_feat, read_tokens)
            for key, value in read_stats.items():
                if key not in stats:
                    stats[key] = value
                else:
                    stats[f"read_{key}"] = value
        else:
            sat_to_street_attn = None
            front_feat_out = front_feat
            read_feat = None
            gate = None

        attn_maps = None
        if return_attn_map:
            attn_maps = {
                "street_to_sat": street_to_sat_attn,
                "sat_to_street": sat_to_street_attn,
            }

        return {
            "front_feat_out": front_feat_out,
            "read_feat": read_feat,
            "attn_map": attn_maps,
            "gate": gate,
            "stats": stats,
            "satellite_state": updated_satellite_state,
        }
