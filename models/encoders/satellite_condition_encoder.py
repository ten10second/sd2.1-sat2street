"""
Satellite Image Condition Encoder.

Encodes satellite images into a shared scene representation for Stable Diffusion.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unet.relative_position_attention import RelativePositionAttention


class ResNetXYBackbone(nn.Module):
    """Return an intermediate ResNet feature map for the satellite XY plane."""

    _VALID_STAGES = ("layer1", "layer2", "layer3", "layer4")

    def __init__(
        self,
        name: str = "resnet34",
        output_stage: str = "layer3",
        pretrained: bool = False,
    ):
        super().__init__()
        output_stage = str(output_stage)
        if output_stage not in self._VALID_STAGES:
            raise ValueError(f"output_stage must be one of {self._VALID_STAGES}, got {output_stage}")

        try:
            from torchvision import models as tv_models
        except ImportError as exc:
            raise ImportError("ResNetXYBackbone requires torchvision to be installed") from exc

        builder = getattr(tv_models, str(name), None)
        if builder is None:
            raise ValueError(f"Unknown torchvision ResNet backbone: {name}")

        if pretrained:
            weight_key = f"{str(name).lower()}_weights"
            weights_enum = next(
                (
                    value
                    for enum_name, value in vars(tv_models).items()
                    if enum_name.lower() == weight_key
                ),
                None,
            )
            weights = weights_enum.DEFAULT if weights_enum is not None else None
            backbone = builder(weights=weights)
        else:
            backbone = builder(weights=None)

        self.name = str(name)
        self.output_stage = output_stage
        self.pretrained = bool(pretrained)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = self._stage_out_channels(getattr(backbone, output_stage))
        if self.pretrained:
            self.register_buffer(
                "image_mean",
                torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "image_std",
                torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )

    @staticmethod
    def _stage_out_channels(stage: nn.Module) -> int:
        block = list(stage.children())[-1]
        for attr_name in ("conv3", "conv2", "conv1"):
            conv = getattr(block, attr_name, None)
            out_channels = getattr(conv, "out_channels", None)
            if isinstance(out_channels, int):
                return out_channels
        raise ValueError(f"Unable to infer ResNet stage channels from {type(block).__name__}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = (x - self.image_mean.to(device=x.device, dtype=x.dtype)) / self.image_std.to(
                device=x.device,
                dtype=x.dtype,
            )
        x = self.stem(x)
        x = self.layer1(x)
        if self.output_stage == "layer1":
            return x
        x = self.layer2(x)
        if self.output_stage == "layer2":
            return x
        x = self.layer3(x)
        if self.output_stage == "layer3":
            return x
        return self.layer4(x)


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
        num_layers: int = 0,
        num_heads: int = 12,
        use_relative_pos: bool = True,
        sat_resolution: float = 0.2,
        sat_size: int = 512,
        xy_feature_source: str = "resnet",
        resnet_name: str = "resnet34",
        resnet_stage: str = "layer3",
        resnet_pretrained: bool = False,
        triplane_enabled: bool = True,
        triplane_height_tokens: int = 16,
        triplane_num_cvha_layers: int = 1,
        triplane_cvha_num_self_points: int = 4,
        triplane_cvha_num_cross_points: int = 8,
        triplane_cvha_local_radius: float = 1.0,
        triplane_cvha_offset_scale: float = 1.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_relative_pos = use_relative_pos
        self.sat_resolution = sat_resolution
        self.sat_size = sat_size
        self.xy_feature_source = str(xy_feature_source).lower()
        self.resnet_name = str(resnet_name)
        self.resnet_stage = str(resnet_stage)
        self.resnet_pretrained = bool(resnet_pretrained)
        self.triplane_enabled = bool(triplane_enabled)
        self.triplane_height_tokens = int(triplane_height_tokens)
        self.triplane_num_cvha_layers = int(triplane_num_cvha_layers)
        self.triplane_cvha_num_self_points = int(triplane_cvha_num_self_points)
        self.triplane_cvha_num_cross_points = int(triplane_cvha_num_cross_points)
        self.triplane_cvha_local_radius = float(triplane_cvha_local_radius)
        self.triplane_cvha_offset_scale = float(triplane_cvha_offset_scale)
        if self.triplane_height_tokens <= 0:
            raise ValueError(
                f"triplane_height_tokens must be positive, got {self.triplane_height_tokens}"
            )
        if self.triplane_num_cvha_layers < 0:
            raise ValueError(
                f"triplane_num_cvha_layers must be non-negative, got {self.triplane_num_cvha_layers}"
            )
        if self.triplane_cvha_num_self_points <= 0:
            raise ValueError(
                f"triplane_cvha_num_self_points must be positive, got {self.triplane_cvha_num_self_points}"
            )
        if self.triplane_cvha_num_cross_points <= 0:
            raise ValueError(
                f"triplane_cvha_num_cross_points must be positive, got {self.triplane_cvha_num_cross_points}"
            )
        if self.triplane_cvha_local_radius <= 0.0:
            raise ValueError(
                f"triplane_cvha_local_radius must be positive, got {self.triplane_cvha_local_radius}"
            )
        if self.triplane_cvha_offset_scale <= 0.0:
            raise ValueError(
                f"triplane_cvha_offset_scale must be positive, got {self.triplane_cvha_offset_scale}"
            )

        if self.xy_feature_source not in {"resnet", "patch"}:
            raise ValueError(f"xy_feature_source must be 'resnet' or 'patch', got {xy_feature_source}")

        if self.xy_feature_source == "resnet":
            self.xy_backbone = ResNetXYBackbone(
                name=self.resnet_name,
                output_stage=self.resnet_stage,
                pretrained=self.resnet_pretrained,
            )
            self.xy_feature_proj = nn.Sequential(
                nn.Conv2d(self.xy_backbone.out_channels, embed_dim, kernel_size=1),
                self._make_group_norm(embed_dim),
                nn.SiLU(),
            )
            self.patch_embed = None
        else:
            self.xy_backbone = None
            self.xy_feature_proj = None
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

        if self.triplane_enabled:
            self.height_embedding = nn.Parameter(
                torch.randn(self.triplane_height_tokens, embed_dim) / math.sqrt(embed_dim)
            )
            base_plane_w = max(1, self.sat_size // self.patch_size)
            # XZ/YZ start as content-free learnable positional tokens. CVHA is
            # responsible for pulling satellite content from XY into them.
            self.xz_plane_init = nn.Parameter(
                torch.randn(1, embed_dim, self.triplane_height_tokens, base_plane_w) / math.sqrt(embed_dim)
            )
            self.yz_plane_init = nn.Parameter(
                torch.randn(1, embed_dim, self.triplane_height_tokens, base_plane_w) / math.sqrt(embed_dim)
            )
            self.cvha_layers = nn.ModuleList(
                [
                    CrossViewHybridAttentionV1(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_self_points=self.triplane_cvha_num_self_points,
                        num_cross_points=self.triplane_cvha_num_cross_points,
                        local_radius=self.triplane_cvha_local_radius,
                        offset_scale=self.triplane_cvha_offset_scale,
                    )
                    for _ in range(self.triplane_num_cvha_layers)
                ]
            )

    @staticmethod
    def _make_group_norm(num_channels: int) -> nn.GroupNorm:
        for num_groups in (32, 24, 16, 12, 8, 4, 2, 1):
            if num_channels % num_groups == 0:
                return nn.GroupNorm(num_groups, num_channels)
        return nn.GroupNorm(1, num_channels)

    @staticmethod
    def _make_axis_centers(
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        descending: bool = False,
    ) -> torch.Tensor:
        axis = (torch.arange(length, device=device, dtype=dtype) + 0.5) / float(max(length, 1))
        axis = axis * 2.0 - 1.0
        if descending:
            axis = -axis
        return axis

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

    def _compute_feature_bev_coords(
        self,
        B: int,
        image_h: int,
        image_w: int,
        feat_h: int,
        feat_w: int,
    ) -> torch.Tensor:
        """Compute BEV meter coordinates for centers of an arbitrary XY feature map."""
        cell_h = float(image_h) / float(max(feat_h, 1))
        cell_w = float(image_w) / float(max(feat_w, 1))
        center_h = (torch.arange(feat_h, dtype=torch.float32) + 0.5) * cell_h
        center_w = (torch.arange(feat_w, dtype=torch.float32) + 0.5) * cell_w
        w_grid, h_grid = torch.meshgrid(center_w, center_h, indexing="xy")

        x_meters = (w_grid - (float(image_w) / 2.0)) * self.sat_resolution
        y_meters = ((float(image_h) / 2.0) - h_grid) * self.sat_resolution
        coords = torch.stack([x_meters.reshape(-1), y_meters.reshape(-1)], dim=-1)
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

    @staticmethod
    def _compute_plane_normalized_coords(
        B: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return normalized XY centers for a plane grid using x-right/y-up convention."""
        x_axis = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(max(width, 1))
        y_axis = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(max(height, 1))
        x_axis = x_axis * 2.0 - 1.0
        y_axis = 1.0 - y_axis * 2.0
        x_grid, y_grid = torch.meshgrid(x_axis, y_axis, indexing="xy")
        coords = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
        return coords.unsqueeze(0).expand(B, -1, 2)

    def _compute_triplane_coords(
        self,
        B: int,
        xy_h: int,
        xy_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        sat_xy = self._compute_plane_normalized_coords(
            B,
            height=xy_h,
            width=xy_w,
            device=device,
            dtype=dtype,
        )
        sat_xy_grid = sat_xy.reshape(B, xy_h, xy_w, 2)
        x_axis = sat_xy_grid[0, 0, :, 0]
        y_axis = sat_xy_grid[0, :, 0, 1]
        z_axis = self._make_axis_centers(
            self.triplane_height_tokens,
            device=device,
            dtype=dtype,
            descending=True,
        )

        xz_x, xz_z = torch.meshgrid(x_axis, z_axis, indexing="xy")
        yz_y, yz_z = torch.meshgrid(y_axis, z_axis, indexing="xy")

        xz = torch.stack([xz_x.reshape(-1), xz_z.reshape(-1)], dim=-1).unsqueeze(0).expand(B, -1, -1)
        yz = torch.stack([yz_y.reshape(-1), yz_z.reshape(-1)], dim=-1).unsqueeze(0).expand(B, -1, -1)

        return {
            "xy": sat_xy,
            "xz": xz,
            "yz": yz,
        }

    def _build_triplane_memory(
        self,
        xy_tokens: torch.Tensor,
        B: int,
        xy_h: int,
        xy_w: int,
    ) -> Dict[str, torch.Tensor]:
        xy_plane = xy_tokens.reshape(B, xy_h, xy_w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        xz_plane = self.xz_plane_init
        if xz_plane.shape[-2:] != (self.triplane_height_tokens, xy_w):
            xz_plane = F.interpolate(
                xz_plane,
                size=(self.triplane_height_tokens, xy_w),
                mode="bilinear",
                align_corners=False,
            )
        xz_plane = xz_plane.expand(B, -1, -1, -1).contiguous()

        yz_plane = self.yz_plane_init
        if yz_plane.shape[-2:] != (self.triplane_height_tokens, xy_h):
            yz_plane = F.interpolate(
                yz_plane,
                size=(self.triplane_height_tokens, xy_h),
                mode="bilinear",
                align_corners=False,
            )
        yz_plane = yz_plane.expand(B, -1, -1, -1).contiguous()

        height_bias = self.height_embedding.transpose(0, 1).reshape(1, self.embed_dim, self.triplane_height_tokens, 1)
        xz_plane = xz_plane + height_bias
        yz_plane = yz_plane + height_bias

        coords = self._compute_triplane_coords(
            B=B,
            xy_h=xy_h,
            xy_w=xy_w,
            device=xy_tokens.device,
            dtype=xy_tokens.dtype,
        )

        for layer in self.cvha_layers:
            xy_plane, xz_plane, yz_plane = layer(
                xy_plane,
                xz_plane,
                yz_plane,
                xy_coords=coords["xy"],
                xz_coords=coords["xz"],
                yz_coords=coords["yz"],
            )

        return {
            "xy_tokens": xy_plane.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim).contiguous(),
            "xz_tokens": xz_plane.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim).contiguous(),
            "yz_tokens": yz_plane.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim).contiguous(),
            "xy_coords": coords["xy"],
            "xz_coords": coords["xz"],
            "yz_coords": coords["yz"],
            "xy_hw": (xy_h, xy_w),
            "z_tokens": self.triplane_height_tokens,
        }

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

        # Step 1: extract the satellite XY plane feature map.
        if self.xy_feature_source == "resnet":
            xy_map = self.xy_backbone(sat_images)
            xy_map = self.xy_feature_proj(xy_map)
        else:
            xy_map = self.patch_embed(sat_images)
        xy_h, xy_w = int(xy_map.shape[-2]), int(xy_map.shape[-1])
        patches_flat = xy_map.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)  # (B, N, D)

        # Step 2: Compute BEV space coordinates for each patch
        # These are fixed physical positions in meters
        bev_coords = self._compute_feature_bev_coords(B, H, W, xy_h, xy_w)
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

        if self.triplane_enabled:
            scene_memory = self._build_triplane_memory(x, B=B, xy_h=xy_h, xy_w=xy_w)
            if not return_sat_xy:
                return scene_memory
            return scene_memory, scene_memory["xy_coords"]

        if not return_sat_xy:
            return x

        sat_xy = self._compute_plane_normalized_coords(
            B,
            height=xy_h,
            width=xy_w,
            device=sat_images.device,
            dtype=sat_images.dtype,
        )
        return x, sat_xy


class CrossViewHybridAttentionV1(nn.Module):
    """
    Minimal true CVHA-style block.

    For each target-plane token:
    - sample local points on the same plane
    - sample aligned points on the other two planes along the missing axis
    - predict deformable offsets and attention weights from the query token
    - aggregate sampled values with a residual Transformer-style update
    """

    PLANE_ORDER = ("xy", "xz", "yz")

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_self_points: int = 4,
        num_cross_points: int = 8,
        local_radius: float = 1.0,
        offset_scale: float = 1.0,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim}/{num_heads}")
        if num_self_points <= 0 or num_cross_points <= 0:
            raise ValueError(
                f"num_self_points and num_cross_points must be positive, got {num_self_points}/{num_cross_points}"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.num_self_points = int(num_self_points)
        self.num_cross_points = int(num_cross_points)
        self.total_points = self.num_self_points + 2 * self.num_cross_points
        self.local_radius = float(local_radius)
        self.offset_scale = float(offset_scale)

        self.attn_norms = nn.ModuleDict({name: nn.LayerNorm(self.embed_dim) for name in self.PLANE_ORDER})
        self.offset_projs = nn.ModuleDict(
            {
                name: nn.Linear(self.embed_dim, self.num_heads * self.total_points * 2)
                for name in self.PLANE_ORDER
            }
        )
        self.attn_projs = nn.ModuleDict(
            {
                name: nn.Linear(self.embed_dim, self.num_heads * self.total_points)
                for name in self.PLANE_ORDER
            }
        )
        self.out_projs = nn.ModuleDict({name: nn.Linear(self.embed_dim, self.embed_dim) for name in self.PLANE_ORDER})
        self.ffn_norms = nn.ModuleDict({name: nn.LayerNorm(self.embed_dim) for name in self.PLANE_ORDER})
        self.ffns = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.embed_dim, ffn_multiplier * self.embed_dim),
                    nn.GELU(),
                    nn.Linear(ffn_multiplier * self.embed_dim, self.embed_dim),
                )
                for name in self.PLANE_ORDER
            }
        )
        self.value_projs = nn.ModuleDict({name: nn.Linear(self.embed_dim, self.head_dim) for name in self.PLANE_ORDER})
        self.value_norms = nn.ModuleDict({name: nn.LayerNorm(self.embed_dim) for name in self.PLANE_ORDER})
        self.plane_embeddings = nn.Parameter(
            torch.randn(len(self.PLANE_ORDER), self.embed_dim) / math.sqrt(self.embed_dim)
        )
        self.register_buffer(
            "horizontal_axis_positions",
            ((torch.arange(self.num_cross_points, dtype=torch.float32) + 0.5) / float(self.num_cross_points)) * 2.0
            - 1.0,
            persistent=False,
        )
        self.register_buffer(
            "vertical_axis_positions",
            1.0
            - ((torch.arange(self.num_cross_points, dtype=torch.float32) + 0.5) / float(self.num_cross_points)) * 2.0,
            persistent=False,
        )
        self.register_buffer(
            "local_pattern",
            self._build_local_pattern(self.num_self_points),
            persistent=False,
        )

    @staticmethod
    def _build_local_pattern(num_points: int) -> torch.Tensor:
        points = [(0.0, 0.0)]
        radius = 1
        while len(points) < num_points:
            ring = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    ring.append((float(dx), float(dy)))
            ring.sort(key=lambda item: (item[0] ** 2 + item[1] ** 2, abs(item[1]), abs(item[0])))
            points.extend(ring)
            radius += 1
        return torch.tensor(points[:num_points], dtype=torch.float32)

    @staticmethod
    def _flatten_plane(plane: torch.Tensor) -> torch.Tensor:
        return plane.permute(0, 2, 3, 1).reshape(plane.shape[0], -1, plane.shape[1]).contiguous()

    @staticmethod
    def _unflatten_plane(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _grid_from_coords(coords: torch.Tensor) -> torch.Tensor:
        return torch.stack([coords[..., 0], -coords[..., 1]], dim=-1)

    @staticmethod
    def _sample_from_plane(plane_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch, num_queries, num_heads, num_points = coords.shape[:4]
        grid = CrossViewHybridAttentionV1._grid_from_coords(coords).reshape(batch, num_queries * num_heads * num_points, 1, 2)
        sampled = F.grid_sample(
            plane_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(batch, num_queries, num_heads, num_points, plane_map.shape[1])
        return sampled.contiguous()

    @staticmethod
    def _cell_scale(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [2.0 / max(width, 1), 2.0 / max(height, 1)],
            device=device,
            dtype=dtype,
        )

    def _build_reference_sets(
        self,
        target_plane: str,
        target_coords: torch.Tensor,
        target_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[str, torch.Tensor], tuple[str, torch.Tensor]]:
        batch, num_queries = target_coords.shape[:2]
        horizontal_axis = self.horizontal_axis_positions.to(device=device, dtype=dtype).view(1, 1, self.num_cross_points)
        vertical_axis = self.vertical_axis_positions.to(device=device, dtype=dtype).view(1, 1, self.num_cross_points)
        local_offsets = self.local_pattern.to(device=device, dtype=dtype).view(1, 1, self.num_self_points, 2)
        local_offsets = local_offsets * self._cell_scale(target_shape[0], target_shape[1], device, dtype).view(1, 1, 1, 2)
        self_refs = (target_coords.unsqueeze(2) + local_offsets * self.local_radius).clamp(-1.0, 1.0)

        if target_plane == "xy":
            x = target_coords[..., 0:1].expand(batch, num_queries, self.num_cross_points)
            y = target_coords[..., 1:2].expand(batch, num_queries, self.num_cross_points)
            xz_refs = torch.stack([x, vertical_axis.expand_as(x)], dim=-1)
            yz_refs = torch.stack([y, vertical_axis.expand_as(y)], dim=-1)
            return self_refs, ("xz", xz_refs), ("yz", yz_refs)

        if target_plane == "xz":
            x = target_coords[..., 0:1].expand(batch, num_queries, self.num_cross_points)
            z = target_coords[..., 1:2].expand(batch, num_queries, self.num_cross_points)
            axis_expand = vertical_axis.expand_as(x)
            xy_refs = torch.stack([x, axis_expand], dim=-1)
            yz_refs = torch.stack([horizontal_axis.expand_as(z), z], dim=-1)
            return self_refs, ("xy", xy_refs), ("yz", yz_refs)

        y = target_coords[..., 0:1].expand(batch, num_queries, self.num_cross_points)
        z = target_coords[..., 1:2].expand(batch, num_queries, self.num_cross_points)
        xy_refs = torch.stack([horizontal_axis.expand_as(y), y], dim=-1)
        xz_refs = torch.stack([horizontal_axis.expand_as(z), z], dim=-1)
        return self_refs, ("xy", xy_refs), ("xz", xz_refs)

    def _apply_offsets(
        self,
        base_refs: torch.Tensor,
        offset_slice: torch.Tensor,
        source_shape: tuple[int, int],
    ) -> torch.Tensor:
        scale = self._cell_scale(
            source_shape[0],
            source_shape[1],
            device=base_refs.device,
            dtype=base_refs.dtype,
        ).view(1, 1, 1, 1, 2)
        refs = base_refs.unsqueeze(2) + torch.tanh(offset_slice) * scale * self.offset_scale
        return refs.clamp(-1.0, 1.0)

    def _sample_source_values(
        self,
        source_name: str,
        source_plane: torch.Tensor,
        source_refs: torch.Tensor,
    ) -> torch.Tensor:
        sampled = self._sample_from_plane(source_plane, source_refs)
        sampled = sampled + self.plane_embeddings[self.PLANE_ORDER.index(source_name)].view(1, 1, 1, 1, -1)
        sampled = self.value_norms[source_name](sampled)
        sampled = self.value_projs[source_name](sampled)
        return sampled

    def _update_plane(
        self,
        target_name: str,
        target_plane: torch.Tensor,
        plane_maps: dict[str, torch.Tensor],
        plane_coords: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch, _, height, width = target_plane.shape
        target_tokens = self._flatten_plane(target_plane)
        query_tokens = self.attn_norms[target_name](target_tokens)

        offsets = self.offset_projs[target_name](query_tokens).reshape(
            batch, target_tokens.shape[1], self.num_heads, self.total_points, 2
        )
        attn_logits = self.attn_projs[target_name](query_tokens).reshape(
            batch, target_tokens.shape[1], self.num_heads, self.total_points
        )
        attn_weights = torch.softmax(attn_logits.float(), dim=-1).to(dtype=target_tokens.dtype)

        self_refs, cross_a, cross_b = self._build_reference_sets(
            target_plane=target_name,
            target_coords=plane_coords[target_name],
            target_shape=(height, width),
            device=target_plane.device,
            dtype=target_plane.dtype,
        )
        self_offsets = offsets[:, :, :, : self.num_self_points, :]
        cross_a_offsets = offsets[:, :, :, self.num_self_points : self.num_self_points + self.num_cross_points, :]
        cross_b_offsets = offsets[:, :, :, self.num_self_points + self.num_cross_points :, :]

        self_values = self._sample_source_values(
            target_name,
            target_plane,
            self._apply_offsets(self_refs, self_offsets, (height, width)),
        )

        cross_a_name, cross_a_base = cross_a
        cross_a_plane = plane_maps[cross_a_name]
        cross_a_shape = cross_a_plane.shape[-2:]
        cross_a_values = self._sample_source_values(
            cross_a_name,
            cross_a_plane,
            self._apply_offsets(cross_a_base, cross_a_offsets, (int(cross_a_shape[0]), int(cross_a_shape[1]))),
        )

        cross_b_name, cross_b_base = cross_b
        cross_b_plane = plane_maps[cross_b_name]
        cross_b_shape = cross_b_plane.shape[-2:]
        cross_b_values = self._sample_source_values(
            cross_b_name,
            cross_b_plane,
            self._apply_offsets(cross_b_base, cross_b_offsets, (int(cross_b_shape[0]), int(cross_b_shape[1]))),
        )

        sampled_values = torch.cat([self_values, cross_a_values, cross_b_values], dim=3)
        aggregated = (attn_weights.unsqueeze(-1) * sampled_values).sum(dim=3)
        aggregated = aggregated.reshape(batch, target_tokens.shape[1], self.embed_dim)
        target_tokens = target_tokens + self.out_projs[target_name](aggregated)
        target_tokens = target_tokens + self.ffns[target_name](self.ffn_norms[target_name](target_tokens))
        return self._unflatten_plane(target_tokens, height, width)

    def forward(
        self,
        xy_plane: torch.Tensor,
        xz_plane: torch.Tensor,
        yz_plane: torch.Tensor,
        xy_coords: torch.Tensor,
        xz_coords: torch.Tensor,
        yz_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plane_maps = {
            "xy": xy_plane,
            "xz": xz_plane,
            "yz": yz_plane,
        }
        plane_coords = {
            "xy": xy_coords,
            "xz": xz_coords,
            "yz": yz_coords,
        }
        updated_xy = self._update_plane("xy", xy_plane, plane_maps, plane_coords)
        updated_xz = self._update_plane("xz", xz_plane, plane_maps, plane_coords)
        updated_yz = self._update_plane("yz", yz_plane, plane_maps, plane_coords)
        return updated_xy, updated_xz, updated_yz
