"""
Continuous XY GeoRoPE for query/key geometric addressing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousXYGeoRoPE(nn.Module):
    """
    Apply xy-separable continuous rotary embedding to Q/K.

    Input tensors follow shape [B, heads, N, head_dim].
    Coordinates follow shape [B, N, 2] with normalized ego-centric xy.
    """

    def __init__(
        self,
        head_dim: int,
        geo_ratio: float = 0.5,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        d_geo = int(head_dim * geo_ratio)
        d_geo = max(0, min(d_geo, head_dim))
        d_base = head_dim - d_geo
        d_x = d_geo // 2
        d_y = d_geo - d_x

        if d_x % 2 != 0 or d_y % 2 != 0:
            raise ValueError(
                "GeoRoPE requires even d_x and d_y for rotary pairs, "
                f"got d_x={d_x}, d_y={d_y} (head_dim={head_dim}, geo_ratio={geo_ratio})"
            )

        self.head_dim = head_dim
        self.geo_ratio = geo_ratio
        self.rope_base = rope_base
        self.d_base = d_base
        self.d_x = d_x
        self.d_y = d_y

    def _inv_freq(self, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if dim == 0:
            return torch.empty(0, device=device, dtype=dtype)
        half_dim = dim // 2
        index = torch.arange(half_dim, device=device, dtype=dtype)
        return self.rope_base ** (-2.0 * index / dim)

    def _apply_1d_rotary(self, tensor: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] == 0:
            return tensor

        inv_freq = self._inv_freq(tensor.shape[-1], tensor.device, tensor.dtype)
        angles = coord.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)

        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos

        out = torch.empty_like(tensor)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out

    def _apply_xy(self, tensor: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        base = tensor[..., : self.d_base]
        x_part = tensor[..., self.d_base : self.d_base + self.d_x]
        y_part = tensor[..., self.d_base + self.d_x :]

        x_rot = self._apply_1d_rotary(x_part, xy[..., 0])
        y_rot = self._apply_1d_rotary(y_part, xy[..., 1])
        return torch.cat([base, x_rot, y_rot], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_xy: torch.Tensor,
        k_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(
                f"Q/K head_dim mismatch: expected {self.head_dim}, "
                f"got q={q.shape[-1]}, k={k.shape[-1]}"
            )
        if q_xy.shape[-1] != 2 or k_xy.shape[-1] != 2:
            raise ValueError(
                f"q_xy/k_xy last dim must be 2, got {q_xy.shape[-1]} and {k_xy.shape[-1]}"
            )

        q_tilde = self._apply_xy(q, q_xy)
        k_tilde = self._apply_xy(k, k_xy)
        return q_tilde, k_tilde

