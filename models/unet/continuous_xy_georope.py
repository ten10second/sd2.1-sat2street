"""
Continuous xy rotary position encoding for geometry-aware satellite attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousXYGeoRoPE(nn.Module):
    """Apply xy-separable continuous rotary encoding to multi-head Q/K tensors."""

    def __init__(
        self,
        head_dim: int,
        geo_ratio: float = 1.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if not 0.0 < geo_ratio <= 1.0:
            raise ValueError(f"geo_ratio must be in (0, 1], got {geo_ratio}")

        rotary_dim = int(head_dim * geo_ratio)
        rotary_dim = max(4, rotary_dim - (rotary_dim % 4))
        if rotary_dim > head_dim:
            rotary_dim = head_dim - (head_dim % 4)
        if rotary_dim <= 0 or rotary_dim % 4 != 0:
            raise ValueError(
                f"GeoRoPE requires a positive rotary dim divisible by 4, got head_dim={head_dim}, "
                f"geo_ratio={geo_ratio}, rotary_dim={rotary_dim}"
            )

        axis_dim = rotary_dim // 2
        if axis_dim % 2 != 0:
            raise ValueError(f"GeoRoPE x/y axis dims must be even, got {axis_dim}")

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
        self.head_dim = int(head_dim)
        self.rotary_dim = int(rotary_dim)
        self.axis_dim = int(axis_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _apply_axis(self, tensor: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        freqs = coord[..., None].to(dtype=torch.float32) * self.inv_freq[None, None, :]
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1).to(device=tensor.device, dtype=tensor.dtype)
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        return tensor * cos + self._rotate_half(tensor) * sin

    def _apply(self, tensor: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor [B,H,N,D], got {list(tensor.shape)}")
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError(f"Expected xy [B,N,2], got {list(xy.shape)}")
        if tensor.shape[0] != xy.shape[0] or tensor.shape[2] != xy.shape[1]:
            raise ValueError(f"Tensor/xy shape mismatch: {list(tensor.shape)} vs {list(xy.shape)}")

        rotary = tensor[..., : self.rotary_dim]
        tail = tensor[..., self.rotary_dim :]
        x_part = rotary[..., : self.axis_dim]
        y_part = rotary[..., self.axis_dim :]
        x_rot = self._apply_axis(x_part, xy[..., 0])
        y_rot = self._apply_axis(y_part, xy[..., 1])
        return torch.cat([x_rot, y_rot, tail], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_xy: torch.Tensor,
        k_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply(q, q_xy), self._apply(k, k_xy)
