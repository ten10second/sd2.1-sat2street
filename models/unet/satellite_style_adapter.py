"""
Low-capacity global satellite style tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SatelliteStyleAdapter(nn.Module):
    """Compress satellite tokens into a few global style tokens for native attn2."""

    def __init__(
        self,
        sat_in_dim: int = 768,
        out_dim: int = 1024,
        num_tokens: int = 4,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        scale: float = 0.5,
    ):
        super().__init__()
        if sat_in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"sat_in_dim/out_dim must be positive, got {sat_in_dim}/{out_dim}")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        hidden_dim = int(hidden_dim or sat_in_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads, got {hidden_dim}/{num_heads}")

        self.sat_in_dim = int(sat_in_dim)
        self.out_dim = int(out_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.scale = float(scale)

        self.query = nn.Parameter(torch.randn(num_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.sat_norm = nn.LayerNorm(sat_in_dim)
        self.kv_proj = nn.Linear(sat_in_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        sat_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sat_tokens.ndim != 3 or sat_tokens.shape[-1] != self.sat_in_dim:
            raise ValueError(
                f"sat_tokens must be [B,N,{self.sat_in_dim}], got {list(sat_tokens.shape)}"
            )
        batch = sat_tokens.shape[0]
        kv = self.kv_proj(self.sat_norm(sat_tokens))
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        query = self.query_norm(query)
        style, _ = self.attn(query, kv, kv, need_weights=False)
        style = self.out(style) * self.scale
        if condition_mask is not None:
            style = style * condition_mask.to(device=style.device, dtype=style.dtype).view(-1, 1, 1)
        return style
