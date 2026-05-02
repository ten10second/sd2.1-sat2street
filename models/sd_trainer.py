"""
Stable Diffusion Trainer for satellite-to-frontview generation.

This module provides a simplified training interface using diffusers library.
"""

import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm
import os
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Tuple
import logging
from PIL import Image

from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.sd_model import DEFAULT_SATELLITE_EMBED_DIM, SatelliteConditionedUNet
from models.unet.view_aware_satellite_adapter import ViewAwareSatelliteAdapter


logger = logging.getLogger(__name__)


def _attention_to_patch_grid(
    attn_weights: torch.Tensor,
    attn_index: torch.Tensor,
    num_sat_tokens: int,
) -> torch.Tensor:
    batch, num_queries, local_topk = attn_weights.shape
    heatmap = torch.zeros(batch, num_queries, num_sat_tokens, dtype=attn_weights.dtype)
    heatmap.scatter_add_(2, attn_index.long(), attn_weights)
    side = int(round(math.sqrt(num_sat_tokens)))
    if side * side == num_sat_tokens:
        return heatmap.reshape(batch, num_queries, side, side)
    return heatmap.reshape(batch, num_queries, 1, num_sat_tokens)


def _coords_to_satellite_pixels(
    coords: torch.Tensor,
    sat_width: int,
    sat_height: int,
) -> list[tuple[float, float]]:
    if coords.numel() == 0:
        return []
    valid = (
        torch.isfinite(coords).all(dim=-1)
        & (coords[:, 0] >= -1.0)
        & (coords[:, 0] <= 1.0)
        & (coords[:, 1] >= -1.0)
        & (coords[:, 1] <= 1.0)
    )
    coords = coords[valid]
    if coords.numel() == 0:
        return []

    x_px = (coords[:, 0] + 1.0) * 0.5 * float(max(1, sat_width - 1))
    y_px = (1.0 - (coords[:, 1] + 1.0) * 0.5) * float(max(1, sat_height - 1))
    return list(zip(x_px.tolist(), y_px.tolist()))


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique_points = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique_points) <= 2:
        return unique_points

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _valid_boundary_mask(valid_hw: torch.Tensor) -> torch.Tensor:
    valid = valid_hw > 0.5
    up = torch.zeros_like(valid)
    down = torch.zeros_like(valid)
    left = torch.zeros_like(valid)
    right = torch.zeros_like(valid)
    up[1:, :] = valid[:-1, :]
    down[:-1, :] = valid[1:, :]
    left[:, 1:] = valid[:, :-1]
    right[:, :-1] = valid[:, 1:]
    interior = valid & up & down & left & right
    return valid & ~interior


def _coords_map_to_fov_polygon(
    coords_map: Optional[torch.Tensor],
    coords_valid_mask: Optional[torch.Tensor],
    sat_width: int,
    sat_height: int,
) -> list[tuple[float, float]]:
    if coords_map is None or not torch.is_tensor(coords_map):
        return []

    coords = coords_map.detach().cpu().to(torch.float32)
    if coords.ndim == 3 and coords.shape[0] == 2:
        coords_hw = coords.permute(1, 2, 0)
    elif coords.ndim == 3 and coords.shape[-1] == 2:
        coords_hw = coords
    else:
        return []

    if coords_valid_mask is not None and torch.is_tensor(coords_valid_mask):
        valid = coords_valid_mask.detach().cpu().to(torch.float32)
        if valid.ndim == 3 and valid.shape[0] == 1:
            valid_hw = valid[0]
        elif valid.ndim == 2:
            valid_hw = valid
        else:
            valid_hw = None
    else:
        valid_hw = None

    if valid_hw is not None and valid_hw.shape == coords_hw.shape[:2]:
        boundary = _valid_boundary_mask(valid_hw)
        coords_boundary = coords_hw[boundary]
        if coords_boundary.shape[0] < 3:
            coords_boundary = coords_hw[valid_hw > 0.5]
        points = _coords_to_satellite_pixels(coords_boundary, sat_width, sat_height)
        hull = _convex_hull(points)
        if len(hull) >= 3:
            return hull

    height, width = int(coords_hw.shape[0]), int(coords_hw.shape[1])
    if height < 2 or width < 2:
        return []
    top = coords_hw[0, :, :]
    right = coords_hw[:, width - 1, :]
    bottom = torch.flip(coords_hw[height - 1, :, :], dims=[0])
    left = torch.flip(coords_hw[:, 0, :], dims=[0])
    boundary = torch.cat([top, right, bottom, left], dim=0)
    points = _coords_to_satellite_pixels(boundary, sat_width, sat_height)
    return points if len(points) >= 3 else []


def load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "unet._view_condition_projs.",
        "view_satellite_adapter.token_pool_query",
        "view_satellite_adapter.token_pool_query_norm.",
        "view_satellite_adapter.token_pool_attn.",
        "view_satellite_adapter.token_out.",
    )
    missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "State dict mismatch. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )
    if incompatible.missing_keys:
        logger.warning(
            "Checkpoint missing view-condition or unified-reader weights; using freshly initialized modules for: %s",
            ", ".join(incompatible.missing_keys),
        )


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    load_model_state_dict(model, state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def _resolve_hf_snapshot_path(model_id: str, revision: Optional[str] = None) -> Optional[Path]:
    candidate_path = Path(model_id).expanduser()
    if candidate_path.is_dir():
        return candidate_path

    if "/" not in model_id:
        return None

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None

    cache_root = Path(hf_home).expanduser() / "hub" / f"models--{model_id.replace('/', '--')}"
    if not cache_root.is_dir():
        return None

    ref_name = revision or "main"
    ref_file = cache_root / "refs" / ref_name
    if ref_file.is_file():
        snapshot_hash = ref_file.read_text().strip()
        snapshot_dir = cache_root / "snapshots" / snapshot_hash
        if snapshot_dir.is_dir():
            return snapshot_dir

    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshot_dirs:
        return None

    return snapshot_dirs[-1]


class SatelliteConditionedSDModel(nn.Module):
    """
    Stable Diffusion model conditioned on satellite images with coordinate encoding.

    This model combines:
    - Base Stable Diffusion UNet
    - Satellite image encoder with coordinate positional encoding
    - Self-attention for spatial consistency
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        noise_scheduler: DDPMScheduler,
        satellite_encoder: Optional[SatelliteConditionEncoder] = None,
        freeze_base: bool = True,
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()

        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.cond_drop_prob = float(cond_drop_prob)
        if not 0.0 <= self.cond_drop_prob <= 1.0:
            raise ValueError(f"cond_drop_prob must be in [0, 1], got {self.cond_drop_prob}")

        # Satellite encoder
        if satellite_encoder is None:
            sat_embed_dim = DEFAULT_SATELLITE_EMBED_DIM
            sat_num_heads = 12
            if sat_embed_dim % sat_num_heads != 0:
                if sat_embed_dim % 64 == 0:
                    sat_num_heads = sat_embed_dim // 64
                else:
                    for candidate in range(min(sat_num_heads, sat_embed_dim), 0, -1):
                        if sat_embed_dim % candidate == 0:
                            sat_num_heads = candidate
                            break
            logger.info(
                f"[SatelliteConditionedSDModel] Satellite encoder config: embed_dim={sat_embed_dim}, num_heads={sat_num_heads}"
            )
            self.satellite_encoder = SatelliteConditionEncoder(
                embed_dim=sat_embed_dim,
                num_heads=sat_num_heads,
            )
        else:
            self.satellite_encoder = satellite_encoder
        sat_embed_dim = int(getattr(self.satellite_encoder, "embed_dim", DEFAULT_SATELLITE_EMBED_DIM))

        adapter_cfg = dict(getattr(self.unet, "reading_block_config", {}) or {})
        cross_attention_dim = int(getattr(self.unet.config, "cross_attention_dim", 1024) or 1024)
        token_pool_num_tokens = int(adapter_cfg.get("token_pool_num_tokens", 8))
        token_pool_num_heads = int(adapter_cfg.get("token_pool_num_heads", 8))
        view_query_dim = int(adapter_cfg.get("view_query_dim", sat_embed_dim))
        view_num_heads = int(adapter_cfg.get("view_num_heads", 8))
        if view_query_dim % view_num_heads != 0:
            view_num_heads = 1
            for candidate in range(min(12, view_query_dim), 0, -1):
                if view_query_dim % candidate == 0:
                    view_num_heads = candidate
                    break
        self.view_satellite_adapter = ViewAwareSatelliteAdapter(
            sat_in_dim=sat_embed_dim,
            out_dim=cross_attention_dim,
            grid_h=int(adapter_cfg.get("view_grid_h", 8)),
            grid_w=int(adapter_cfg.get("view_grid_w", 20)),
            query_dim=view_query_dim,
            num_heads=view_num_heads,
            scale=float(adapter_cfg.get("view_scale", 1.0)),
            geo_bias_weight=float(adapter_cfg.get("view_geo_bias_weight", 1.0)),
            geo_sigma=float(adapter_cfg.get("view_geo_sigma", 0.35)),
            local_topk=int(adapter_cfg.get("view_local_topk", 25)),
            geo_target_sigma=float(adapter_cfg.get("view_geo_target_sigma", 0.20)),
            gate_hidden_dim=int(adapter_cfg.get("view_gate_hidden_dim", 256)),
            token_pool_num_tokens=token_pool_num_tokens,
            token_pool_num_heads=token_pool_num_heads,
            token_scale=float(adapter_cfg.get("token_scale", 1.0)),
            save_attention_heatmap=bool(adapter_cfg.get("save_attention_heatmap", True)),
            heatmap_max_tokens=int(adapter_cfg.get("heatmap_max_tokens", 16)),
        )
        self.view_geo_loss_weight = float(adapter_cfg.get("view_geo_loss_weight", 0.1))
        self.scene_consistency_weight = float(adapter_cfg.get("scene_consistency_weight", 0.0))

        # Freeze base layers
        if freeze_base:
            # Freeze VAE
            for param in self.vae.parameters():
                param.requires_grad = False

            for param in self.unet.parameters():
                param.requires_grad = False

            for param in self.unet._view_condition_projs.parameters():
                param.requires_grad = True

        # Satellite encoder is always trainable
        for param in self.satellite_encoder.parameters():
            param.requires_grad = True
        for param in self.view_satellite_adapter.parameters():
            param.requires_grad = True

        logger.info(f"[SatelliteConditionedSDModel] Initialized")
        logger.info(f"  UNet trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
        logger.info(f"  Satellite encoder params: {sum(p.numel() for p in self.satellite_encoder.parameters())}")
        logger.info(f"  View-aware satellite adapter params: {sum(p.numel() for p in self.view_satellite_adapter.parameters())}")
        logger.info(f"  Scene consistency weight: {self.scene_consistency_weight}")
        logger.info(f"  Condition dropout: {self.cond_drop_prob}")

    def encode_scene_memory(
        self,
        sat_images: torch.Tensor,
        coords_map: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode satellite images into a shared scene memory."""
        return self.satellite_encoder(sat_images, coords_map, return_sat_xy=True)

    def read_scene_with_pose(
        self,
        sat_tokens: torch.Tensor,
        sat_xy: torch.Tensor,
        coords_map: Optional[torch.Tensor],
        coords_valid_mask: Optional[torch.Tensor],
        plucker_map: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.view_satellite_adapter(
            sat_tokens=sat_tokens,
            sat_xy=sat_xy,
            front_bev_xy=coords_map,
            plucker=plucker_map,
            valid_mask=coords_valid_mask,
            condition_mask=condition_mask,
            return_dict=True,
        )

    @staticmethod
    def _normalize_images_for_vae(images: torch.Tensor) -> torch.Tensor:
        """Map dataset images from [0, 1] to the SD VAE's expected [-1, 1] range."""
        return images * 2.0 - 1.0

    @staticmethod
    def _expand_condition_mask(condition_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        mask = condition_mask.to(device=reference.device, dtype=reference.dtype)
        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)
        return mask

    def _build_unet_kwargs(
        self,
        encoder_hidden_states: Optional[torch.Tensor],
        view_condition_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        unet_kwargs: Dict[str, Any] = {}
        if encoder_hidden_states is not None:
            unet_kwargs['encoder_hidden_states'] = encoder_hidden_states
        if view_condition_map is not None:
            unet_kwargs['view_condition_map'] = view_condition_map
        return unet_kwargs

    @staticmethod
    def _flatten_grouped_views(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None or not torch.is_tensor(tensor) or tensor.ndim < 5:
            return tensor
        batch, views = tensor.shape[:2]
        return tensor.reshape(batch * views, *tensor.shape[2:])

    def _compute_scene_consistency_loss(
        self,
        readout_tokens: torch.Tensor,
        grouped_views: bool,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        if (not grouped_views) or self.scene_consistency_weight <= 0.0:
            return readout_tokens.new_zeros(())
        if readout_tokens.ndim != 3:
            return readout_tokens.new_zeros(())

        pooled = readout_tokens.mean(dim=1)
        pooled = pooled.reshape(batch_size, num_views, pooled.shape[-1])
        pooled = F.normalize(pooled, dim=-1)
        anchor = pooled[:, :1, :]
        similarities = (pooled[:, 1:, :] * anchor).sum(dim=-1)
        return (1.0 - similarities).mean()

    def forward(
        self,
        sat_images: torch.Tensor,
        target_images: torch.Tensor,
        coords_map: torch.Tensor = None,
        coords_valid_mask: Optional[torch.Tensor] = None,
        plucker_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            sat_images: (B, 3, H_sat, W_sat) - Satellite images
            target_images: (B, 3, H, W) - Target frontview images
            coords_map: (B, 2, H_cam, W_cam) - BEV coordinates for each camera pixel
            coords_valid_mask: (B, 1, H_cam, W_cam) - Valid ground projection mask
            plucker_map: (B, 6, H_cam, W_cam) - Per-pixel Plucker ray map

        Returns:
            dict with 'loss' and other info
        """
        grouped_views = target_images.ndim == 5
        B = sat_images.shape[0]
        num_views = int(target_images.shape[1]) if grouped_views else 1
        device = sat_images.device

        # Encode satellite images
        sat_encoded = self.encode_scene_memory(sat_images, coords_map)
        if isinstance(sat_encoded, tuple):
            sat_tokens, sat_xy = sat_encoded
        else:
            sat_tokens = sat_encoded
            sat_xy = None
        condition_mask = torch.ones(B, device=device, dtype=torch.bool)
        if self.training and self.cond_drop_prob > 0.0:
            condition_mask = torch.rand(B, device=device) >= self.cond_drop_prob
            if not bool(condition_mask.any().item()):
                keep_index = torch.randint(0, B, (1,), device=device)
                condition_mask[keep_index] = True
            sat_tokens = sat_tokens * self._expand_condition_mask(condition_mask, sat_tokens)
            if sat_xy is not None:
                sat_xy = sat_xy * self._expand_condition_mask(condition_mask, sat_xy)
        plucker_for_unet = plucker_map
        plucker_dropout_prob = float(getattr(self.unet, "plucker_dropout_prob", 0.0))
        if self.training and plucker_for_unet is not None and plucker_dropout_prob > 0.0:
            if torch.rand((), device=device) < plucker_dropout_prob:
                plucker_for_unet = None
        if sat_xy is None:
            raise ValueError("Satellite encoder must return sat_xy for view-aware conditioning")
        if grouped_views:
            sat_tokens = sat_tokens[:, None].expand(-1, num_views, -1, -1).reshape(B * num_views, *sat_tokens.shape[1:])
            sat_xy = sat_xy[:, None].expand(-1, num_views, -1, -1).reshape(B * num_views, *sat_xy.shape[1:])
            condition_mask = condition_mask[:, None].expand(-1, num_views).reshape(B * num_views)
            target_images = self._flatten_grouped_views(target_images)
            coords_map = self._flatten_grouped_views(coords_map)
            coords_valid_mask = self._flatten_grouped_views(coords_valid_mask)
            plucker_for_unet = self._flatten_grouped_views(plucker_for_unet)

        reader_outputs = self.read_scene_with_pose(
            sat_tokens=sat_tokens,
            sat_xy=sat_xy,
            coords_map=coords_map,
            coords_valid_mask=coords_valid_mask,
            plucker_map=plucker_for_unet,
            condition_mask=condition_mask,
        )
        encoder_hidden_states = reader_outputs["readout_tokens"]
        view_condition_map = reader_outputs["readout_map"]

        # Encode target images to latents
        with torch.no_grad():
            target_images_vae = self._normalize_images_for_vae(target_images)
            latents = self.vae.encode(target_images_vae).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # Sample noise
        effective_batch = int(latents.shape[0])
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (effective_batch,),
            device=device, dtype=torch.long
        )

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        unet_kwargs = self._build_unet_kwargs(
            encoder_hidden_states=encoder_hidden_states,
            view_condition_map=view_condition_map,
        )

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            **unet_kwargs,
        ).sample

        # Compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        diffusion_loss = F.mse_loss(model_pred, target, reduction="mean")
        geo_loss = reader_outputs.get("geo_loss")
        if not torch.is_tensor(geo_loss):
            geo_loss = diffusion_loss.new_zeros(())
        else:
            geo_loss = geo_loss.to(diffusion_loss.device)
        scene_consistency_loss = self._compute_scene_consistency_loss(
            readout_tokens=encoder_hidden_states,
            grouped_views=grouped_views,
            batch_size=B,
            num_views=num_views,
        ).to(diffusion_loss.device)
        view_stats = {
            f"view_{key}": value.to(diffusion_loss.device)
            for key, value in self.view_satellite_adapter.last_stats.items()
            if torch.is_tensor(value)
        }
        loss = (
            diffusion_loss
            + self.view_geo_loss_weight * geo_loss
            + self.scene_consistency_weight * scene_consistency_loss
        )

        return {
            'loss': loss,
            'diffusion_loss': diffusion_loss,
            'view_geo_loss': geo_loss,
            'scene_consistency_loss': scene_consistency_loss,
            'model_pred': model_pred,
            'target': target,
            **view_stats,
        }

    @torch.no_grad()
    def _get_vae_scale_factor(self) -> int:
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels is None:
            return 8
        return max(1, 2 ** (len(block_out_channels) - 1))

    @staticmethod
    def _infer_image_size_from_coords_map(coords_map: Optional[torch.Tensor]) -> Optional[Tuple[int, int]]:
        if coords_map is None or not torch.is_tensor(coords_map):
            return None

        if coords_map.ndim == 4 and coords_map.shape[1] == 2:
            return int(coords_map.shape[2]), int(coords_map.shape[3])
        if coords_map.ndim == 4 and coords_map.shape[-1] == 2:
            return int(coords_map.shape[1]), int(coords_map.shape[2])
        if coords_map.ndim == 3 and coords_map.shape[0] == 2:
            return int(coords_map.shape[1]), int(coords_map.shape[2])
        if coords_map.ndim == 3 and coords_map.shape[-1] == 2:
            return int(coords_map.shape[0]), int(coords_map.shape[1])

        return None

    def _infer_generation_size(
        self,
        coords_map: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if target_size is not None:
            return int(target_size[0]), int(target_size[1])

        coords_size = self._infer_image_size_from_coords_map(coords_map)
        if coords_size is not None:
            return coords_size

        sample_size = self.unet.config.sample_size
        if isinstance(sample_size, (tuple, list)):
            return int(sample_size[0]), int(sample_size[1])
        return int(sample_size), int(sample_size)

    @torch.no_grad()
    def generate(
        self,
        sat_images: torch.Tensor,
        coords_map: torch.Tensor = None,
        coords_valid_mask: Optional[torch.Tensor] = None,
        plucker_map: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sat_condition_mode: str = "normal",
    ) -> torch.Tensor:
        """
        Generate frontview images from satellite images.

        Args:
            sat_images: (B, 3, H_sat, W_sat) - Satellite images
            coords_map: (B, 2, H_cam, W_cam) - BEV coordinates for each camera pixel
            coords_valid_mask: (B, 1, H_cam, W_cam) - Valid ground projection mask
            plucker_map: (B, 6, H_cam, W_cam) - Per-pixel Plucker ray map
            target_size: Optional target image size as (H, W)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            sat_condition_mode: "normal" uses encoded satellite conditioning,
                "zero" disables satellite conditioning by zeroing tokens and masks.

        Returns:
            generated_images: (B, 3, H, W) - Generated images
        """
        B = sat_images.shape[0]
        device = sat_images.device

        # Encode satellite images
        sat_encoded = self.encode_scene_memory(sat_images, coords_map)
        if isinstance(sat_encoded, tuple):
            sat_tokens, sat_xy = sat_encoded
        else:
            sat_tokens = sat_encoded
            sat_xy = None

        if sat_condition_mode == "normal":
            condition_mask = torch.ones(B, device=device, dtype=torch.bool)
        elif sat_condition_mode == "zero":
            sat_tokens = torch.zeros_like(sat_tokens)
            sat_xy = torch.zeros_like(sat_xy) if sat_xy is not None else None
            condition_mask = torch.zeros(B, device=device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown sat_condition_mode: {sat_condition_mode}")
        if sat_xy is None:
            raise ValueError("Satellite encoder must return sat_xy for view-aware conditioning")
        reader_outputs = self.read_scene_with_pose(
            sat_tokens=sat_tokens,
            sat_xy=sat_xy,
            coords_map=coords_map,
            coords_valid_mask=coords_valid_mask,
            plucker_map=plucker_map,
            condition_mask=condition_mask,
        )
        encoder_hidden_states = reader_outputs["readout_tokens"]
        view_condition_map = reader_outputs["readout_map"]

        image_h, image_w = self._infer_generation_size(coords_map=coords_map, target_size=target_size)
        vae_scale_factor = self._get_vae_scale_factor()
        latent_h = max(1, (image_h + vae_scale_factor - 1) // vae_scale_factor)
        latent_w = max(1, (image_w + vae_scale_factor - 1) // vae_scale_factor)

        # Initialize latents
        latents = torch.randn(
            (B, self.unet.config.in_channels, latent_h, latent_w),
            device=device,
            dtype=sat_tokens.dtype,
            generator=generator,
        )

        # Prepare CFG branches once and reuse them for every timestep.
        use_cfg = guidance_scale > 1.0
        if use_cfg:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            uncond_view_condition_map = torch.zeros_like(view_condition_map)
            encoder_hidden_states_double = torch.cat([encoder_hidden_states, uncond_encoder_hidden_states], dim=0)
            view_condition_map_double = torch.cat([view_condition_map, uncond_view_condition_map], dim=0)

        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            if use_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                unet_kwargs = self._build_unet_kwargs(
                    encoder_hidden_states=encoder_hidden_states_double,
                    view_condition_map=view_condition_map_double,
                )
                noise_pred_both = self.unet(
                    latent_model_input,
                    t,
                    **unet_kwargs,
                ).sample
                noise_pred_cond, noise_pred_uncond = noise_pred_both.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                unet_kwargs = self._build_unet_kwargs(
                    encoder_hidden_states=encoder_hidden_states,
                    view_condition_map=view_condition_map,
                )
                noise_pred = self.unet(
                    latents,
                    t,
                    **unet_kwargs,
                ).sample

            # Compute previous noisy sample
            if generator is not None:
                try:
                    latents = self.noise_scheduler.step(
                        noise_pred, t, latents, generator=generator
                    ).prev_sample
                except TypeError:
                    latents = self.noise_scheduler.step(
                        noise_pred, t, latents
                    ).prev_sample
            else:
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents
                ).prev_sample

        # Decode latents to images
        vae_param = next(self.vae.parameters(), None)
        vae_dtype = vae_param.dtype if vae_param is not None else latents.dtype
        latents = (latents / self.vae.config.scaling_factor).to(dtype=vae_dtype)
        generated_images = self.vae.decode(latents).sample

        # Normalize to [0, 1]
        generated_images = (generated_images / 2 + 0.5).clamp(0, 1)

        if generated_images.shape[-2:] != (image_h, image_w):
            generated_images = F.interpolate(
                generated_images,
                size=(image_h, image_w),
                mode="bilinear",
                align_corners=False,
            )

        return generated_images


def create_sd_model(
    base_model: str = 'stabilityai/stable-diffusion-2-1-base',
    freeze_base: bool = True,
    reading_block_config: Optional[Dict] = None,
    revision: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cond_drop_prob: float = 0.1,
) -> SatelliteConditionedSDModel:
    """
    Create a satellite-conditioned Stable Diffusion model.

    Args:
        base_model: Name of base Stable Diffusion model
        freeze_base: Whether to freeze base layers

    Returns:
        SatelliteConditionedSDModel
    """
    resolved_base_model = _resolve_hf_snapshot_path(base_model, revision=revision)
    load_source = str(resolved_base_model) if resolved_base_model is not None else base_model
    if resolved_base_model is not None:
        logger.info(f"Using cached base model snapshot: {resolved_base_model}")

    # Load base components
    component_load_kwargs: Dict[str, Any] = {}
    if revision is not None and resolved_base_model is None:
        component_load_kwargs["revision"] = revision
    if torch_dtype is not None:
        component_load_kwargs["torch_dtype"] = torch_dtype
    if resolved_base_model is not None:
        component_load_kwargs["local_files_only"] = True

    vae = AutoencoderKL.from_pretrained(
        load_source,
        subfolder="vae",
        **component_load_kwargs,
    )
    base_unet = UNet2DConditionModel.from_pretrained(
        load_source,
        subfolder="unet",
        **component_load_kwargs,
    )

    reading_cfg = dict(reading_block_config or {})
    requested_sat_embed_dim = int(
        reading_cfg.get('sat_in_dim', reading_cfg.get('satellite_embed_dim', DEFAULT_SATELLITE_EMBED_DIM))
    )
    sat_num_heads = 12
    if requested_sat_embed_dim % sat_num_heads != 0:
        if requested_sat_embed_dim % 64 == 0:
            sat_num_heads = requested_sat_embed_dim // 64
        else:
            for candidate in range(min(sat_num_heads, requested_sat_embed_dim), 0, -1):
                if requested_sat_embed_dim % candidate == 0:
                    sat_num_heads = candidate
                    break
    satellite_encoder = SatelliteConditionEncoder(embed_dim=requested_sat_embed_dim, num_heads=sat_num_heads)
    sat_embed_dim = int(getattr(satellite_encoder, "embed_dim", DEFAULT_SATELLITE_EMBED_DIM))
    reading_cfg["sat_in_dim"] = sat_embed_dim

    unet = SatelliteConditionedUNet(
        reading_block_config={
            'sat_in_dim': sat_embed_dim,
            'plucker_dropout_prob': reading_cfg.get('plucker_dropout_prob', 0.3),
            'view_grid_h': reading_cfg.get('view_grid_h', 8),
            'view_grid_w': reading_cfg.get('view_grid_w', 20),
            'view_query_dim': reading_cfg.get('view_query_dim', sat_embed_dim),
            'view_num_heads': reading_cfg.get('view_num_heads', 8),
            'view_scale': reading_cfg.get('view_scale', 1.0),
            'view_geo_bias_weight': reading_cfg.get('view_geo_bias_weight', 1.0),
            'view_geo_sigma': reading_cfg.get('view_geo_sigma', 0.35),
            'view_local_topk': reading_cfg.get('view_local_topk', 25),
            'view_geo_target_sigma': reading_cfg.get('view_geo_target_sigma', 0.20),
            'view_geo_loss_weight': reading_cfg.get('view_geo_loss_weight', 0.1),
            'view_gate_hidden_dim': reading_cfg.get('view_gate_hidden_dim', 256),
            'token_pool_num_tokens': reading_cfg.get('token_pool_num_tokens', 8),
            'token_pool_num_heads': reading_cfg.get('token_pool_num_heads', 8),
            'token_scale': reading_cfg.get('token_scale', 1.0),
            'scene_consistency_weight': reading_cfg.get('scene_consistency_weight', 0.0),
            'save_attention_heatmap': reading_cfg.get('save_attention_heatmap', False),
            'heatmap_max_tokens': reading_cfg.get('heatmap_max_tokens', 16),
        },
        **base_unet.config,
    )
    unet.load_state_dict(base_unet.state_dict(), strict=False)
    scheduler_load_kwargs: Dict[str, Any] = {}
    if revision is not None and resolved_base_model is None:
        scheduler_load_kwargs["revision"] = revision
    if resolved_base_model is not None:
        scheduler_load_kwargs["local_files_only"] = True

    noise_scheduler = DDPMScheduler.from_pretrained(
        load_source,
        subfolder="scheduler",
        **scheduler_load_kwargs,
    )

    model = SatelliteConditionedSDModel(
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        satellite_encoder=satellite_encoder,
        freeze_base=freeze_base,
        cond_drop_prob=cond_drop_prob,
    )

    return model


class SDTrainer:
    """
    Trainer for satellite-to-frontview generation with Stable Diffusion.

    Handles training loop, optimization, checkpointing, and evaluation.
    """

    def __init__(
        self,
        model: SatelliteConditionedSDModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_train_epochs: int = 100,
        lr_scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
        gradient_accumulation_steps: int = 1,
        output_dir: str = './output',
        save_every: int = 5,
        log_every: int = 100,
        device: str = 'cuda',
        use_wandb: bool = False,
        project_name: str = 'kitti360_sd',
        wandb_run_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_mode: str = 'online',
        use_tensorboard: bool = False,
        tensorboard_log_dir: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        mixed_precision: Optional[str] = None,
        max_grad_norm: float = 1.0,
        visualize_every: int = 1,
        num_visualizations: int = 4,
        visualization_inference_steps: int = 20,
        visualization_guidance_scale: float = 1.0,
        visualization_seed: int = 42,
        distributed: bool = False,
        local_rank: int = 0,
    ):
        self.model = model.to(device)
        self.distributed = bool(distributed)
        self.local_rank = int(local_rank)
        self.rank = dist.get_rank() if self.distributed and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if self.distributed and dist.is_initialized() else 1
        self.is_main_process = self.rank == 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_epochs = warmup_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = Path(output_dir)
        self.save_every = save_every
        self.log_every = log_every
        self.device = device
        self.use_wandb = bool(use_wandb) and self.is_main_process
        self.project_name = project_name
        self.wandb_run_name = wandb_run_name
        self.wandb_entity = wandb_entity
        self.wandb_mode = wandb_mode
        self.use_tensorboard = use_tensorboard
        self.max_grad_norm = float(max_grad_norm)
        self.visualize_every = max(0, int(visualize_every))
        self.num_visualizations = max(0, int(num_visualizations))
        self.visualization_inference_steps = int(visualization_inference_steps)
        self.visualization_guidance_scale = float(visualization_guidance_scale)
        self.visualization_seed = int(visualization_seed)
        self.visualization_dir = self.output_dir / "visualizations"
        self.monitor_dir = self.output_dir / "monitor"
        self.view_metrics_path = self.monitor_dir / "view_metrics.csv"
        self._view_metrics_header_written = False
        if self.is_main_process:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            self.monitor_dir.mkdir(parents=True, exist_ok=True)
            self._view_metrics_header_written = self.view_metrics_path.exists() and self.view_metrics_path.stat().st_size > 0
        self.tensorboard_log_dir = Path(tensorboard_log_dir) if tensorboard_log_dir is not None else self.output_dir / "tensorboard"
        self.tb_writer = None
        self.mixed_precision = None if mixed_precision is None else mixed_precision.lower()
        self.use_amp = device.startswith("cuda") and self.mixed_precision in {"fp16", "bf16"}
        self.amp_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16
            if self.mixed_precision == "bf16"
            else None
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        # Setup output dir
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        self._ensure_trainable_params_fp32()
        self._assert_no_trainable_fp16_params()
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.startswith("cuda") else None,
                output_device=self.local_rank if self.device.startswith("cuda") else None,
                find_unused_parameters=True,
            )

        # Setup optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup scheduler
        num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / gradient_accumulation_steps))
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        warmup_steps = warmup_epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Setup wandb
        if use_wandb and self.is_main_process:
            if not is_wandb_available():
                raise ImportError("Please install wandb to use logging: pip install wandb")
            import wandb
            wandb.init(
                project=project_name,
                entity=wandb_entity,
                name=wandb_run_name,
                mode=wandb_mode,
                config=run_config,
            )
        if use_tensorboard and self.is_main_process:
            try:
                from tensorboardX import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "Please install tensorboardX to use TensorBoard logging: pip install tensorboardX"
                ) from exc
            self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(logdir=str(self.tensorboard_log_dir))

        if self.is_main_process:
            logger.info(f"[SDTrainer] Initialized")
            logger.info(f"  Distributed: {self.distributed} (world_size={self.world_size})")
            logger.info(f"  Learning rate: {learning_rate}")
            logger.info(f"  Num epochs: {num_train_epochs}")
            logger.info(f"  Batch size per process: {train_dataloader.batch_size}")
            logger.info(f"  Mixed precision: {self.mixed_precision or 'disabled'}")
            logger.info(f"  Max grad norm: {self.max_grad_norm}")
            if self.use_wandb:
                logger.info(
                    f"  W&B logging: project={self.project_name}, "
                    f"run_name={self.wandb_run_name or 'auto'}, mode={self.wandb_mode}"
                )
            if self.tb_writer is not None:
                logger.info(f"  TensorBoard log dir: {self.tensorboard_log_dir}")
            if self.visualize_every > 0 and self.num_visualizations > 0:
                logger.info(
                    f"  Visualization: every {self.visualize_every} epoch(s), "
                    f"{self.num_visualizations} sample(s), {self.visualization_inference_steps} denoise steps"
                )

    @property
    def unwrapped_model(self) -> SatelliteConditionedSDModel:
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _reduce_mean(self, value: float, device: str) -> float:
        if not self.distributed or not dist.is_initialized():
            return float(value)
        tensor = torch.tensor(float(value), device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(self.world_size)
        return float(tensor.item())

    def _ensure_trainable_params_fp32(self) -> None:
        fp16_params = []
        converted_param_count = 0
        converted_numel = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.dtype != torch.float16:
                continue
            fp16_params.append(name)
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()
            converted_param_count += 1
            converted_numel += param.numel()

        if fp16_params:
            logger.info(
                f"Converted {converted_param_count} trainable parameter tensors "
                f"({converted_numel} values) from fp16 to fp32 for AMP stability"
            )

    def _assert_no_trainable_fp16_params(self) -> None:
        remaining_fp16 = [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.dtype == torch.float16
        ]
        if remaining_fp16:
            preview = ", ".join(remaining_fp16[:8])
            if len(remaining_fp16) > 8:
                preview += ", ..."
            raise RuntimeError(
                "Found trainable fp16 parameters after AMP preparation. "
                f"These must stay fp32 for GradScaler: {preview}"
            )

    def _global_step(self, epoch: int, step: int) -> int:
        return epoch * max(1, len(self.train_dataloader)) + step + 1

    def _log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        scalar_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if value is not None
        }
        if not scalar_metrics:
            return

        if self.use_wandb:
            import wandb
            wandb.log(scalar_metrics, step=step)

        if self.tb_writer is not None:
            for key, value in scalar_metrics.items():
                self.tb_writer.add_scalar(key, value, global_step=step)

    def _append_view_metrics(self, row: Dict[str, float]) -> None:
        if not self.is_main_process:
            return
        fieldnames = [
            "global_step",
            "epoch",
            "epoch_step",
            "num_batches",
            "raw_loss",
            "diffusion_loss",
            "view_geo_loss",
            "lr",
            "view_valid_ratio",
            "view_gate_mean",
            "view_attn_entropy",
            "view_attn_geo_dist",
            "view_nearest_geo_dist",
            "view_geo_over_nearest",
            "view_geo_kl",
        ]
        with self.view_metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._view_metrics_header_written:
                writer.writeheader()
                self._view_metrics_header_written = True
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1)

    def _log_visualizations(
        self,
        images: Sequence[Image.Image],
        captions: Sequence[str],
        step: int,
    ) -> None:
        if not images:
            return

        if self.use_wandb:
            import wandb
            wandb.log(
                {
                    "visualizations": [
                        wandb.Image(image, caption=caption)
                        for image, caption in zip(images, captions)
                    ]
                },
                step=step,
            )

        if self.tb_writer is not None:
            for idx, (image, caption) in enumerate(zip(images, captions)):
                tag = f"visualizations/sample_{idx:02d}"
                self.tb_writer.add_image(
                    tag,
                    self._pil_to_tensor(image),
                    global_step=step,
                    dataformats="CHW",
                )
                self.tb_writer.add_text(
                    f"{tag}_caption",
                    caption,
                    global_step=step,
                )

    def _close_loggers(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.use_wandb:
            import wandb
            wandb.finish()

    def train(self, resume_from: Optional[str] = None):
        """Run training."""
        start_epoch = 0

        if resume_from is not None:
            self._load_checkpoint(resume_from)

        try:
            for epoch in range(start_epoch, self.num_train_epochs):
                if self.is_main_process:
                    logger.info(f"Epoch {epoch + 1}/{self.num_train_epochs}")
                train_raw_loss = self._train_epoch(epoch)
                epoch_step = self._global_step(epoch, max(0, len(self.train_dataloader) - 1))
                if self.is_main_process:
                    logger.info(f"  Train raw loss: {train_raw_loss:.4f}")
                    self._log_scalars(
                        {
                            "train/epoch_raw_loss": train_raw_loss,
                            "train/epoch": epoch + 1,
                        },
                        step=epoch_step,
                    )

                # Validate
                if self.val_dataloader is not None and self.is_main_process:
                    val_loss = self._validate(epoch)
                    logger.info(f"  Val loss: {val_loss:.4f}")
                    self._log_scalars(
                        {
                            "val/loss": val_loss,
                            "train/epoch": epoch + 1,
                        },
                        step=epoch_step,
                    )
                self._barrier()

                if self.is_main_process and self.visualize_every > 0 and self.num_visualizations > 0 and (epoch + 1) % self.visualize_every == 0:
                    self._save_visualizations(epoch)
                self._barrier()

                # Save checkpoint
                if self.is_main_process and ((epoch + 1) % self.save_every == 0 or (epoch + 1) == self.num_train_epochs):
                    self._save_checkpoint(epoch)
                self._barrier()
        finally:
            self._close_loggers()

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_raw_loss = 0.0
        num_batches = len(self.train_dataloader)
        self.optimizer.zero_grad(set_to_none=True)

        sampler = getattr(self.train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Train Epoch {epoch+1}",
            disable=not self.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            # Move data to device
            sat_images = batch['sat'].to(self.device)
            target_images = batch['image'].to(self.device)

            # Get coords_map - BEV coordinates for each camera pixel (透视 token 在 BEV 图上的坐标)
            coords_map = batch.get('coords_map')
            if coords_map is not None:
                coords_map = coords_map.to(self.device)
            coords_valid_mask = batch.get('coords_valid_mask')
            if coords_valid_mask is not None:
                coords_valid_mask = coords_valid_mask.to(self.device)
            plucker_map = batch.get('plucker_map')
            if plucker_map is not None:
                plucker_map = plucker_map.to(self.device)

            # Forward pass
            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    sat_images,
                    target_images,
                    coords_map=coords_map,
                    coords_valid_mask=coords_valid_mask,
                    plucker_map=plucker_map,
                )
                raw_loss = outputs['loss']

            # Backward pass
            accumulation_start = (step // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
            accumulation_end = min(accumulation_start + self.gradient_accumulation_steps, num_batches)
            accumulation_window_size = max(1, accumulation_end - accumulation_start)
            loss = raw_loss / accumulation_window_size
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) == accumulation_end:
                if self.use_amp and self.amp_dtype == torch.float16:
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )
                    self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_raw_loss += raw_loss.item()
            postfix = {'raw_loss': f"{raw_loss.item():.3f}"}
            view_gate = outputs.get('view_gate_mean')
            scene_consistency_loss = outputs.get('scene_consistency_loss')
            if torch.is_tensor(view_gate):
                postfix['view_gate'] = f"{view_gate.item():.2f}"
            if torch.is_tensor(scene_consistency_loss) and self.scene_consistency_weight > 0.0:
                postfix['scene_cons'] = f"{scene_consistency_loss.item():.3f}"
            if self.is_main_process:
                progress_bar.set_postfix(postfix)

            # Log
            if self.is_main_process and (step + 1) % self.log_every == 0:
                view_valid_ratio = outputs.get('view_valid_ratio')
                view_gate = outputs.get('view_gate_mean')
                view_entropy = outputs.get('view_attn_entropy')
                view_attn_geo_dist = outputs.get('view_attn_geo_dist')
                view_nearest_geo_dist = outputs.get('view_nearest_geo_dist')
                view_geo_loss = outputs.get('view_geo_loss')
                view_geo_kl = outputs.get('view_geo_kl')
                scene_consistency_loss = outputs.get('scene_consistency_loss')
                if all(torch.is_tensor(v) for v in (view_valid_ratio, view_gate, view_entropy)):
                    logger.info(
                        "Train step %d/%d: raw_loss=%.6f geo_loss=%.6f scene_cons=%.6f view_gate=%.4f view_valid=%.4f view_entropy=%.4f view_geo=%.4f nearest_geo=%.4f",
                        step + 1,
                        num_batches,
                        raw_loss.item(),
                        view_geo_loss.item() if torch.is_tensor(view_geo_loss) else float("nan"),
                        scene_consistency_loss.item() if torch.is_tensor(scene_consistency_loss) else float("nan"),
                        view_gate.item(),
                        view_valid_ratio.item(),
                        view_entropy.item(),
                        view_attn_geo_dist.item() if torch.is_tensor(view_attn_geo_dist) else float("nan"),
                        view_nearest_geo_dist.item() if torch.is_tensor(view_nearest_geo_dist) else float("nan"),
                    )
                else:
                    logger.info(
                        "Train step %d/%d: raw_loss=%.6f",
                        step + 1,
                        num_batches,
                        raw_loss.item(),
                    )
                log_payload = {
                    'train/raw_loss': raw_loss.item(),
                    'train/diffusion_loss': outputs.get('diffusion_loss', raw_loss).item(),
                    'train/view_geo_loss': view_geo_loss.item() if torch.is_tensor(view_geo_loss) else 0.0,
                    'train/scene_consistency_loss': scene_consistency_loss.item() if torch.is_tensor(scene_consistency_loss) else 0.0,
                    'train/lr': self.lr_scheduler.get_last_lr()[0],
                    'train/epoch': epoch + 1,
                }
                if torch.is_tensor(view_valid_ratio):
                    log_payload['view/valid_ratio'] = view_valid_ratio.item()
                if torch.is_tensor(view_gate):
                    log_payload['view/gate_mean'] = view_gate.item()
                if torch.is_tensor(view_entropy):
                    log_payload['view/attn_entropy'] = view_entropy.item()
                if torch.is_tensor(view_attn_geo_dist):
                    log_payload['view/attn_geo_dist'] = view_attn_geo_dist.item()
                if torch.is_tensor(view_nearest_geo_dist):
                    log_payload['view/nearest_geo_dist'] = view_nearest_geo_dist.item()
                if torch.is_tensor(view_geo_kl):
                    log_payload['view/geo_kl'] = view_geo_kl.item()
                self._log_scalars(log_payload, step=self._global_step(epoch, step))

                geo_over_nearest = None
                if torch.is_tensor(view_attn_geo_dist) and torch.is_tensor(view_nearest_geo_dist):
                    geo_over_nearest = view_attn_geo_dist.item() / max(view_nearest_geo_dist.item(), 1e-8)
                self._append_view_metrics({
                    "global_step": self._global_step(epoch, step),
                    "epoch": epoch + 1,
                    "epoch_step": step + 1,
                    "num_batches": num_batches,
                    "raw_loss": raw_loss.item(),
                    "diffusion_loss": outputs.get('diffusion_loss', raw_loss).item(),
                    "view_geo_loss": view_geo_loss.item() if torch.is_tensor(view_geo_loss) else None,
                    "scene_consistency_loss": scene_consistency_loss.item() if torch.is_tensor(scene_consistency_loss) else None,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "view_valid_ratio": view_valid_ratio.item() if torch.is_tensor(view_valid_ratio) else None,
                    "view_gate_mean": view_gate.item() if torch.is_tensor(view_gate) else None,
                    "view_attn_entropy": view_entropy.item() if torch.is_tensor(view_entropy) else None,
                    "view_attn_geo_dist": view_attn_geo_dist.item() if torch.is_tensor(view_attn_geo_dist) else None,
                    "view_nearest_geo_dist": view_nearest_geo_dist.item() if torch.is_tensor(view_nearest_geo_dist) else None,
                    "view_geo_over_nearest": geo_over_nearest,
                    "view_geo_kl": view_geo_kl.item() if torch.is_tensor(view_geo_kl) else None,
                })
        local_mean = total_raw_loss / max(1, num_batches)
        return self._reduce_mean(local_mean, self.device)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Validate the model."""
        eval_model = self.unwrapped_model
        was_training = eval_model.training
        eval_model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_dataloader, desc=f"Val Epoch {epoch+1}"):
            sat_images = batch['sat'].to(self.device)
            target_images = batch['image'].to(self.device)

            # Get coords_map
            coords_map = batch.get('coords_map')
            if coords_map is not None:
                coords_map = coords_map.to(self.device)
            coords_valid_mask = batch.get('coords_valid_mask')
            if coords_valid_mask is not None:
                coords_valid_mask = coords_valid_mask.to(self.device)
            plucker_map = batch.get('plucker_map')
            if plucker_map is not None:
                plucker_map = plucker_map.to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = eval_model(
                    sat_images,
                    target_images,
                    coords_map=coords_map,
                    coords_valid_mask=coords_valid_mask,
                    plucker_map=plucker_map,
                )
                loss = outputs['loss']
            total_loss += loss.item()

        if was_training:
            eval_model.train()
        return total_loss / len(self.val_dataloader)

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }

        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
        logger.info(f"Checkpoint saved: {checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'}")

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        image = image.detach().cpu().clamp(0, 1)
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        image = (image * 255).to(torch.uint8).numpy()
        return Image.fromarray(image)

    @staticmethod
    def _make_heatmap_overlay(heatmap: torch.Tensor, size: tuple[int, int]) -> Image.Image:
        heatmap = heatmap.detach().cpu().float()
        heatmap = heatmap - heatmap.min()
        denom = float(heatmap.max().item())
        if denom > 1e-8:
            heatmap = heatmap / denom
        heatmap_np = heatmap.numpy()
        red = np.full_like(heatmap_np, 255.0)
        green = 255.0 * heatmap_np
        blue = np.zeros_like(heatmap_np)
        rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
        overlay = Image.fromarray(rgb).resize(size, resample=Image.BILINEAR)
        return overlay

    def _draw_satellite_attention_overlay(
        self,
        sat_image: torch.Tensor,
        coords_map: Optional[torch.Tensor],
        coords_valid_mask: Optional[torch.Tensor],
        attention_grid: Optional[torch.Tensor],
        frame_id: Optional[Any] = None,
        yaw_deg: Optional[float] = None,
        view_name: Optional[str] = None,
    ) -> Image.Image:
        sat_pil = self._tensor_to_pil(sat_image).convert("RGB")
        base = sat_pil.copy()
        draw = ImageDraw.Draw(base, "RGBA")

        polygon = _coords_map_to_fov_polygon(coords_map, coords_valid_mask, sat_pil.width, sat_pil.height)
        if polygon:
            draw.polygon(polygon, fill=(0, 180, 255, 45), outline=(255, 230, 0, 235))
            draw.line(polygon + [polygon[0]], fill=(255, 230, 0, 235), width=3)

        if attention_grid is not None and torch.is_tensor(attention_grid):
            overlay = self._make_heatmap_overlay(attention_grid, sat_pil.size)
            base = Image.blend(base, overlay, alpha=0.50)
            draw = ImageDraw.Draw(base, "RGBA")
            if polygon:
                draw.line(polygon + [polygon[0]], fill=(255, 230, 0, 255), width=3)

        center = (sat_pil.width / 2.0, sat_pil.height / 2.0)
        cross = 7
        draw.line((center[0] - cross, center[1], center[0] + cross, center[1]), fill=(255, 255, 255, 230), width=3)
        draw.line((center[0], center[1] - cross, center[0], center[1] + cross), fill=(255, 255, 255, 230), width=3)
        draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=(255, 64, 64, 240))

        label_parts = []
        if view_name:
            label_parts.append(str(view_name))
        if yaw_deg is not None:
            label_parts.append(f"yaw={float(yaw_deg):g}")
        if frame_id is not None:
            label_parts.append(f"frame={int(frame_id):010d}")
        label = " | ".join(label_parts) if label_parts else "satellite"
        draw.rectangle((4, 4, min(base.width - 4, 420), 28), fill=(0, 0, 0, 160))
        draw.text((8, 8), label, fill=(255, 255, 255, 255))
        return base

    def _compose_visualization(
        self,
        sat_image: torch.Tensor,
        generated_image: torch.Tensor,
        real_image: torch.Tensor,
        coords_map: Optional[torch.Tensor] = None,
        coords_valid_mask: Optional[torch.Tensor] = None,
        attention_grid: Optional[torch.Tensor] = None,
        frame_id: Optional[Any] = None,
        yaw_deg: Optional[float] = None,
        view_name: Optional[str] = None,
    ) -> Image.Image:
        target_h, target_w = int(real_image.shape[-2]), int(real_image.shape[-1])
        sat_pil = self._draw_satellite_attention_overlay(
            sat_image=sat_image,
            coords_map=coords_map,
            coords_valid_mask=coords_valid_mask,
            attention_grid=attention_grid,
            frame_id=frame_id,
            yaw_deg=yaw_deg,
            view_name=view_name,
        ).resize((target_h, target_h), resample=Image.BILINEAR)
        gen_pil = self._tensor_to_pil(generated_image)
        real_pil = self._tensor_to_pil(real_image)

        canvas = Image.new("RGB", (sat_pil.width + gen_pil.width + real_pil.width, target_h))
        x_offset = 0
        for img in (sat_pil, gen_pil, real_pil):
            canvas.paste(img, (x_offset, 0))
            x_offset += img.width
        return canvas

    @torch.no_grad()
    def _save_visualizations(self, epoch: int):
        data_loader = self.val_dataloader if self.val_dataloader is not None else self.train_dataloader
        if data_loader is None:
            return

        sat_chunks = []
        target_chunks = []
        coords_chunks = []
        coords_valid_chunks = []
        plucker_chunks = []
        frame_ids = []
        metas = []

        for batch in data_loader:
            batch_count = batch['sat'].shape[0]
            remaining = self.num_visualizations - len(frame_ids)
            if remaining <= 0:
                break

            take = min(remaining, batch_count)
            sat_chunks.append(batch['sat'][:take])
            target_chunks.append(batch['image'][:take])

            coords_map = batch.get('coords_map')
            if coords_map is not None:
                coords_chunks.append(coords_map[:take])
            coords_valid_mask = batch.get('coords_valid_mask')
            if coords_valid_mask is not None:
                coords_valid_chunks.append(coords_valid_mask[:take])
            plucker_map = batch.get('plucker_map')
            if plucker_map is not None:
                plucker_chunks.append(plucker_map[:take])

            batch_frame_ids = batch.get('frame_id')
            if batch_frame_ids is None:
                frame_ids.extend([None] * take)
            else:
                frame_ids.extend(list(batch_frame_ids[:take]))
            batch_meta = batch.get('meta')
            if batch_meta is None:
                metas.extend([None] * take)
            else:
                metas.extend(list(batch_meta[:take]))

            if len(frame_ids) >= self.num_visualizations:
                break

        if not sat_chunks:
            return

        sat_images = torch.cat(sat_chunks, dim=0).to(self.device)
        target_images = torch.cat(target_chunks, dim=0).to(self.device)
        coords_map = torch.cat(coords_chunks, dim=0).to(self.device) if coords_chunks else None
        coords_valid_mask = torch.cat(coords_valid_chunks, dim=0).to(self.device) if coords_valid_chunks else None
        plucker_map = torch.cat(plucker_chunks, dim=0).to(self.device) if plucker_chunks else None
        if target_images.ndim == 5:
            batch_count, view_count = target_images.shape[:2]
            sat_images = sat_images[:, None].expand(-1, view_count, -1, -1, -1).reshape(
                batch_count * view_count,
                *sat_images.shape[1:],
            )
            target_images = SatelliteConditionedSDModel._flatten_grouped_views(target_images)
            coords_map = SatelliteConditionedSDModel._flatten_grouped_views(coords_map)
            coords_valid_mask = SatelliteConditionedSDModel._flatten_grouped_views(coords_valid_mask)
            plucker_map = SatelliteConditionedSDModel._flatten_grouped_views(plucker_map)
            expanded_frame_ids = []
            for frame_id in frame_ids:
                expanded_frame_ids.extend([frame_id] * view_count)
            frame_ids = expanded_frame_ids
            expanded_metas = []
            for meta in metas:
                if isinstance(meta, dict) and isinstance(meta.get("views"), list):
                    expanded_metas.extend(meta["views"])
                else:
                    expanded_metas.extend([meta] * view_count)
            metas = expanded_metas

        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.visualization_seed)

        eval_model = self.unwrapped_model
        was_training = eval_model.training
        eval_model.eval()
        generated_images = eval_model.generate(
            sat_images,
            coords_map=coords_map,
            coords_valid_mask=coords_valid_mask,
            plucker_map=plucker_map,
            target_size=tuple(target_images.shape[-2:]),
            num_inference_steps=self.visualization_inference_steps,
            guidance_scale=self.visualization_guidance_scale,
            generator=generator,
        )
        if was_training:
            eval_model.train()

        attn_grid_all = None
        adapter = eval_model.view_satellite_adapter
        if adapter.last_attention_heatmap is not None and adapter.last_attention_index is not None and adapter.last_sat_xy is not None:
            attn_grid_all = _attention_to_patch_grid(
                attn_weights=adapter.last_attention_heatmap,
                attn_index=adapter.last_attention_index,
                num_sat_tokens=int(adapter.last_sat_xy.shape[1]),
            )
            if torch.is_tensor(attn_grid_all):
                attn_grid_all = attn_grid_all.mean(dim=1)

        epoch_dir = self.visualization_dir / f"epoch_{epoch + 1:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        comparison_images = []
        captions = []

        for idx in range(generated_images.shape[0]):
            frame_id = frame_ids[idx]
            frame_suffix = f"_frame_{int(frame_id):010d}" if frame_id is not None else ""
            meta = metas[idx] if idx < len(metas) else None
            yaw_deg = None
            view_name = None
            if isinstance(meta, dict):
                yaw_deg = meta.get("vehicle_relative_yaw_deg")
                view_name = meta.get("view_name")
            attention_grid = None
            if torch.is_tensor(attn_grid_all) and idx < attn_grid_all.shape[0]:
                attention_grid = attn_grid_all[idx]
            comparison = self._compose_visualization(
                sat_images[idx],
                generated_images[idx],
                target_images[idx],
                coords_map=coords_map[idx] if coords_map is not None else None,
                coords_valid_mask=coords_valid_mask[idx] if coords_valid_mask is not None else None,
                attention_grid=attention_grid,
                frame_id=frame_id,
                yaw_deg=yaw_deg,
                view_name=view_name,
            )
            comparison.save(epoch_dir / f"sample_{idx:02d}{frame_suffix}.png")
            comparison_images.append(comparison)
            caption = f"epoch={epoch + 1} sample={idx:02d}"
            if frame_id is not None:
                caption += f" frame={int(frame_id):010d}"
            if view_name is not None:
                caption += f" view={view_name}"
            if yaw_deg is not None:
                caption += f" yaw={float(yaw_deg):g}"
            captions.append(caption)

        logger.info(f"Saved visualizations: {epoch_dir}")
        self._log_visualizations(
            comparison_images,
            captions,
            step=self._global_step(epoch, max(0, len(self.train_dataloader) - 1)),
        )

    def _load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        load_model_state_dict(self.unwrapped_model, checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
