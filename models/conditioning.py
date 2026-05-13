"""
Structured conditioning state for cross-view satellite/street refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class SatelliteMemoryState:
    """
    Mutable satellite memory carried across U-Net refinement sites.
    """

    tokens: torch.Tensor
    xy: torch.Tensor
    bev_coords: Optional[torch.Tensor] = None

    def replace(
        self,
        *,
        tokens: Optional[torch.Tensor] = None,
        xy: Optional[torch.Tensor] = None,
        bev_coords: Optional[torch.Tensor] = None,
    ) -> "SatelliteMemoryState":
        return SatelliteMemoryState(
            tokens=self.tokens if tokens is None else tokens,
            xy=self.xy if xy is None else xy,
            bev_coords=self.bev_coords if bev_coords is None else bev_coords,
        )


@dataclass
class CrossViewConditioningState:
    """
    Structured conditioning payload threaded through explicit UNet refinement.
    """

    satellite: SatelliteMemoryState
    front_bev_xy: Any
    front_ground_valid_mask: Any = None
    condition_mask: Optional[torch.Tensor] = None
    return_attn_map: bool = False
    attn_maps: Dict[str, torch.Tensor] = field(default_factory=dict)
    refinement_stats: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
