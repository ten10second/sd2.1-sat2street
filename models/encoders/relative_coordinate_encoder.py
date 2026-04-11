"""
Relative Coordinate Encoder for satellite image features.

Encodes relative coordinates of frontview tokens on BEV map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RelativeCoordinateEncoder(nn.Module):
    """
    Encode relative coordinates of tokens in BEV space.

    Supports multiple encoding types:
    - 'fourier': Fourier features encoding (sin/cos)
    - 'sincos': Sine-cosine positional encoding
    - 'linear': Linear projection

    Args:
        encode_type: Type of coordinate encoding
        embed_dim: Output embedding dimension
        fourier_scale: Scale factor for Fourier encoding
        num_frequencies: Number of frequency bands for Fourier encoding
    """

    def __init__(
        self,
        encode_type: str = 'fourier',
        embed_dim: int = 256,
        fourier_scale: float = 10.0,
        num_frequencies: int = 10,
    ):
        super().__init__()

        self.encode_type = encode_type.lower()
        self.embed_dim = embed_dim
        self.fourier_scale = fourier_scale
        self.num_frequencies = num_frequencies

        if self.encode_type == 'fourier':
            # Fourier features: 2 * num_frequencies * 2 (x, y)
            self.num_features = 2 * num_frequencies * 2
            self.freq_bands = torch.tensor(
                [fourier_scale * (2.0 ** i) for i in range(num_frequencies)]
            )
        elif self.encode_type == 'sincos':
            # Sine-cosine encoding
            self.num_features = 2 * 2  # x, y each with sin and cos
        elif self.encode_type == 'linear':
            # Linear projection from 2D coords
            self.num_features = 2
        else:
            raise ValueError(f"Unknown encode_type: {encode_type}")

        # Projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Fourier features encoding."""
        freq_bands = self.freq_bands.to(coords.device)
        # coords: (B, N, 2)
        B, N, _ = coords.shape

        # Apply frequencies
        # (B, N, 2) -> (B, N, 2, F)
        encoded = coords.unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
        # Compute sin and cos
        # (B, N, 2, F) -> (B, N, 2*2*F)
        encoded = torch.cat([
            torch.sin(encoded),
            torch.cos(encoded)
        ], dim=-1).flatten(-2)

        return encoded

    def _sincos_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Sine-cosine positional encoding."""
        # coords: (B, N, 2)
        x, y = coords[..., 0:1], coords[..., 1:2]
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        sin_y = torch.sin(y)
        cos_y = torch.cos(y)
        return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode relative coordinates.

        Args:
            coords: (B, N, 2) - Relative coordinates in BEV space

        Returns:
            emb: (B, N, embed_dim) - Encoded coordinates
        """
        if self.encode_type == 'fourier':
            encoded = self._fourier_encode(coords)
        elif self.encode_type == 'sincos':
            encoded = self._sincos_encode(coords)
        elif self.encode_type == 'linear':
            encoded = coords
        else:
            raise ValueError(f"Unknown encode_type: {self.encode_type}")

        return self.projection(encoded)
