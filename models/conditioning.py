"""Structured conditioning state for satellite conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SatelliteMemoryState:
    """
    Satellite token state carried through Stable Diffusion conditioning.
    """

    tokens: torch.Tensor
    xy: torch.Tensor
    bev_coords: Optional[torch.Tensor] = None
    perspective_uv: Optional[torch.Tensor] = None
    perspective_valid: Optional[torch.Tensor] = None

    def replace(
        self,
        *,
        tokens: Optional[torch.Tensor] = None,
        xy: Optional[torch.Tensor] = None,
        bev_coords: Optional[torch.Tensor] = None,
        perspective_uv: Optional[torch.Tensor] = None,
        perspective_valid: Optional[torch.Tensor] = None,
    ) -> "SatelliteMemoryState":
        return SatelliteMemoryState(
            tokens=self.tokens if tokens is None else tokens,
            xy=self.xy if xy is None else xy,
            bev_coords=self.bev_coords if bev_coords is None else bev_coords,
            perspective_uv=self.perspective_uv if perspective_uv is None else perspective_uv,
            perspective_valid=self.perspective_valid if perspective_valid is None else perspective_valid,
        )
