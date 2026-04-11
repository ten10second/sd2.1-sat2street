"""
Relative Position Attention for satellite image features.

Self-attention with relative position encoding based on BEV coordinates.

This module implements the core idea:
"给卫星图加入另一个位置编码——透视平面上每个 token 在 BEV 图上的相对位置，然后做 self-attention"

Key idea:
- Each satellite patch has a position in BEV space (physical coordinates)
- When computing self-attention, we use these BEV coordinates to compute
  relative positions between patches
- This maintains spatial consistency: two patches that are close in BEV space
  will have stronger attention between them, regardless of their position
  in the satellite image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RelativePositionAttention(nn.Module):
    """
    Self-attention with relative position encoding based on BEV coordinates.

    This attention layer uses BEV space coordinates (not image grid coordinates)
    to compute relative positions. This ensures that the self-attention respects
    the actual physical spatial layout.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_relative_pos: Whether to use relative position encoding
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        use_relative_pos: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.head_dim = embed_dim // num_heads
        self.use_relative_pos = use_relative_pos

        # Linear layers for q, k, v
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Linear layer for output
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Relative position encoding - encodes relative BEV coordinates
        if self.use_relative_pos:
            self.relative_pos_encoding = nn.Linear(2, num_heads)

        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _compute_relative_positions(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute relative positions between all pairs of patches in BEV space.

        This is the key: for each pair of patches (i, j), we compute
        (x_i - x_j, y_i - y_j) where x, y are BEV coordinates in meters.

        Args:
            coords: (B, N, 2) - BEV coordinates for each patch

        Returns:
            relative_pos: (B, N, N, 2) - Relative BEV coordinates
        """
        # coords: (B, N, 2)
        B, N, _ = coords.shape
        # (B, N, 1, 2) - (B, 1, N, 2) -> (B, N, N, 2)
        relative_pos = coords.unsqueeze(2) - coords.unsqueeze(1)
        return relative_pos

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with relative position attention.

        Args:
            x: (B, N, embed_dim) - Input features (satellite patch embeddings)
            coords: (B, N, 2) - BEV coordinates for each patch (in meters)

        Returns:
            out: (B, N, embed_dim) - Output features
        """
        B, N, _ = x.shape

        # Layer norm
        x = self.norm1(x)

        # Compute q, k, v
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention weights
        # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add relative position encoding based on BEV coordinates
        if self.use_relative_pos and coords is not None:
            # Compute relative positions in BEV space
            relative_pos = self._compute_relative_positions(coords)
            # Encode relative positions directly as per-head attention bias.
            relative_bias = self.relative_pos_encoding(relative_pos).permute(0, 3, 1, 2)
            relative_bias = relative_bias / (self.head_dim ** 0.5)
            attn_weights = attn_weights + relative_bias

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection
        x = x + out

        # Feed forward
        out = self.norm2(x)
        out = self.ffn(out)
        out = self.dropout(out)

        # Residual connection
        x = x + out

        return x
