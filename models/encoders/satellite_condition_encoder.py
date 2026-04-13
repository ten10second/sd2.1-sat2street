"""
Satellite Image Condition Encoder.

Encodes satellite images with coordinate positional encoding for Stable Diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..unet.relative_position_attention import RelativePositionAttention


class SatelliteConditionEncoder(nn.Module):
    """
    Encode satellite images with coordinate positional encoding.

    This encoder implements the core idea:
    "给卫星图加入另一个位置编码——透视平面上每个 token 在 BEV 图上的相对位置，然后做 self-attention"

    Key insight:
    - Satellite image patches are in BEV space with fixed physical coordinates
    - We use BEV space coordinates (not image grid coordinates) for relative position
    - This maintains spatial consistency through self-attention

    Args:
        embed_dim: Embedding dimension
        patch_size: Patch size for dividing satellite image
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        use_relative_pos: Whether to use relative position attention
        sat_resolution: Meters per pixel in satellite image
        sat_size: Size of satellite image
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 12,
        use_relative_pos: bool = True,
        sat_resolution: float = 0.2,
        sat_size: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_relative_pos = use_relative_pos
        self.sat_resolution = sat_resolution
        self.sat_size = sat_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Coordinate encoder - for BEV space coordinates (meters)
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Transformer layers with relative position attention
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                RelativePositionAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    use_relative_pos=use_relative_pos,
                )
            )

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def _compute_patch_bev_coords(
        self,
        B: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Compute BEV space coordinates (in meters) for each satellite patch.

        This is the key: each satellite patch has a fixed physical position
        in BEV space, not just a grid position in the image.

        Args:
            B: Batch size
            H: Height of satellite image
            W: Width of satellite image

        Returns:
            patch_coords: (B, N, 2) - BEV coordinates in meters for each patch
        """
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # Compute patch centers in pixel coordinates
        patch_pixel_h = torch.arange(patch_h, dtype=torch.float32) * self.patch_size + self.patch_size / 2
        patch_pixel_w = torch.arange(patch_w, dtype=torch.float32) * self.patch_size + self.patch_size / 2

        # Create meshgrid
        w_grid, h_grid = torch.meshgrid(patch_pixel_w, patch_pixel_h, indexing='xy')

        # Convert to BEV space coordinates in meters
        # Satellite image is centered at (0, 0) in BEV space
        # Positive x is east, positive y is north
        half_sat = self.sat_size / 2.0
        x_meters = (w_grid - half_sat) * self.sat_resolution  # (patch_h, patch_w)
        y_meters = (half_sat - h_grid) * self.sat_resolution  # (patch_h, patch_w)

        # Stack to (patch_h*patch_w, 2)
        coords = torch.stack([x_meters.reshape(-1), y_meters.reshape(-1)], dim=-1)

        # Expand for batch
        return coords.unsqueeze(0).expand(B, -1, 2)

    def _compute_patch_normalized_coords(
        self,
        B: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Compute normalized ego-centric xy for each patch center in [-1, 1].

        Coordinate origin is the satellite image center.
        x is positive to the right, y is positive upward.
        """
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        patch_pixel_h = torch.arange(patch_h, dtype=torch.float32) * self.patch_size + self.patch_size / 2
        patch_pixel_w = torch.arange(patch_w, dtype=torch.float32) * self.patch_size + self.patch_size / 2
        w_grid, h_grid = torch.meshgrid(patch_pixel_w, patch_pixel_h, indexing='xy')

        x_norm = (w_grid - (W / 2.0)) / (W / 2.0)
        y_norm = ((H / 2.0) - h_grid) / (H / 2.0)
        coords = torch.stack([x_norm.reshape(-1), y_norm.reshape(-1)], dim=-1)
        return coords.unsqueeze(0).expand(B, -1, 2)

    def forward(
        self,
        sat_images: torch.Tensor,
        coords_map: Optional[torch.Tensor] = None,
        return_sat_xy: bool = False,
    ):
        """
        Encode satellite images.

        Args:
            sat_images: (B, 3, H, W) - Satellite images
            coords_map: (B, 2, H_cam, W_cam) - Optional, not used in this implementation
                        Kept for backward compatibility

        Returns:
            sat_emb: (B, N, embed_dim) - Encoded satellite features
            sat_xy: (B, N, 2) - Optional normalized patch-center coordinates in [-1, 1]
        """
        B, C, H, W = sat_images.shape

        # Step 1: Patch embedding
        patches = self.patch_embed(sat_images)  # (B, D, H/P, W/P)
        patches_flat = patches.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)  # (B, N, D)

        # Step 2: Compute BEV space coordinates for each patch
        # These are fixed physical positions in meters
        bev_coords = self._compute_patch_bev_coords(B, H, W)
        bev_coords = bev_coords.to(sat_images.device)  # (B, N, 2)

        # Step 3: Encode BEV coordinates
        coord_emb = self.coord_encoder(bev_coords)  # (B, N, D)

        # Step 4: Combine patch features and coordinate encoding
        x = patches_flat + coord_emb  # (B, N, D)

        # Step 5: Pass through transformer layers
        # The self-attention uses BEV coordinates to compute relative positions
        for layer in self.layers:
            x = layer(x, bev_coords)  # (B, N, D)

        # Step 6: Apply final layer norm
        x = self.norm(x)

        if not return_sat_xy:
            return x

        sat_xy = self._compute_patch_normalized_coords(B, H, W).to(sat_images.device)
        return x, sat_xy
