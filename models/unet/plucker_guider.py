"""
Lightweight Plucker-ray guider for latent residual conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PluckerGuider(nn.Module):
    """Project a 6-channel Plucker map into a latent-space residual."""

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 32,
        out_channels: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        final_conv = self.net[-1]
        if isinstance(final_conv, nn.Conv2d):
            nn.init.zeros_(final_conv.weight)
            if final_conv.bias is not None:
                nn.init.zeros_(final_conv.bias)

    def forward(self, plucker_map: torch.Tensor) -> torch.Tensor:
        if plucker_map.ndim != 4 or plucker_map.shape[1] != 6:
            raise ValueError(
                f"plucker_map must be [B, 6, H, W], got {list(plucker_map.shape)}"
            )
        return self.net(plucker_map)
