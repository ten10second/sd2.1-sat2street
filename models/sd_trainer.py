"""
Stable Diffusion Trainer for satellite-to-frontview generation.

This module provides a simplified training interface using diffusers library.
"""

import math
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
from typing import Optional, Dict, Any, List, Sequence, Tuple
import logging
from PIL import Image

from models.conditioning import SatelliteMemoryState
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.sd_model import SatelliteConditionedUNet


logger = logging.getLogger(__name__)


def _aggregate_refinement_stats(
    refinement_stats: Optional[Dict[str, Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    if not refinement_stats:
        return {}

    aggregated: Dict[str, torch.Tensor] = {}
    sem_values = []
    geom_values = []
    ratio_values = []
    for site, site_stats in refinement_stats.items():
        sem = site_stats.get("logits_sem_std")
        geom = site_stats.get("logits_geom_std")
        ratio = site_stats.get("logits_geom_to_sem_ratio")
        if sem is not None:
            aggregated[f"{site}_logits_sem_std"] = sem
            sem_values.append(sem)
        if geom is not None:
            aggregated[f"{site}_logits_geom_std"] = geom
            geom_values.append(geom)
        if ratio is not None:
            aggregated[f"{site}_logits_geom_to_sem_ratio"] = ratio
            ratio_values.append(ratio)
    if sem_values:
        aggregated["refinement_logits_sem_std_mean"] = torch.stack(sem_values).mean()
    if geom_values:
        aggregated["refinement_logits_geom_std_mean"] = torch.stack(geom_values).mean()
    if ratio_values:
        aggregated["refinement_logits_geom_to_sem_ratio_mean"] = torch.stack(ratio_values).mean()
    return aggregated


def load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Sequence[str], Sequence[str]]:
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    return missing_keys, unexpected_keys


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


@torch.no_grad()
def _materialize_lazy_modules(
    model: "SatelliteConditionedSDModel",
    sat_images: torch.Tensor,
    front_bev_xy: Optional[torch.Tensor],
    front_ground_valid_mask: Optional[torch.Tensor],
    target_size: Tuple[int, int],
) -> None:
    sat_state = model.encode_satellite(sat_images)

    vae_scale_factor = model._get_vae_scale_factor()
    latent_h = max(1, (target_size[0] + vae_scale_factor - 1) // vae_scale_factor)
    latent_w = max(1, (target_size[1] + vae_scale_factor - 1) // vae_scale_factor)
    latents = torch.randn(
        (sat_images.shape[0], model.unet.config.in_channels, latent_h, latent_w),
        device=sat_images.device,
        dtype=sat_state.tokens.dtype,
    )
    timestep = torch.zeros((sat_images.shape[0],), device=sat_images.device, dtype=torch.long)

    model.unet(
        latents,
        timestep,
        encoder_hidden_states=None,
        sat_tokens=sat_state.tokens,
        sat_xy=sat_state.xy,
        sat_bev_coords=sat_state.bev_coords,
        front_bev_xy=front_bev_xy,
        front_ground_valid_mask=front_ground_valid_mask,
        return_attn_map=False,
    )


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
            sat_embed_dim = int(unet.config.cross_attention_dim or 768)
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

        # Freeze base layers
        if freeze_base:
            # Freeze VAE
            for param in self.vae.parameters():
                param.requires_grad = False

            # Freeze the pretrained UNet backbone. Satellite-specific modules are
            # trained separately to preserve the SD prior and avoid scene drift.
            for name, param in self.unet.named_parameters():
                if (
                    ".attn2.to_k." in name
                    or ".attn2.to_v." in name
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Satellite encoder is always trainable
        for param in self.satellite_encoder.parameters():
            param.requires_grad = True

        logger.info(f"[SatelliteConditionedSDModel] Initialized")
        logger.info(f"  UNet trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
        logger.info(f"  Satellite encoder params: {sum(p.numel() for p in self.satellite_encoder.parameters())}")
        logger.info(
            f"  Trainable attn2 k/v params: "
            f"{sum(p.numel() for n, p in self.unet.named_parameters() if p.requires_grad and ('.attn2.to_k.' in n or '.attn2.to_v.' in n))}"
        )
        logger.info("  Main attn2 route: satellite tokens")
        logger.info(f"  Condition dropout: {self.cond_drop_prob}")

    def encode_satellite(self, sat_images: torch.Tensor) -> SatelliteMemoryState:
        """Encode satellite images into a structured satellite memory state."""
        return self.satellite_encoder(sat_images)

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
        sat_state: SatelliteMemoryState,
        front_bev_xy: Optional[torch.Tensor],
        front_ground_valid_mask: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        unet_kwargs: Dict[str, Any] = {}
        if encoder_hidden_states is not None:
            unet_kwargs['encoder_hidden_states'] = encoder_hidden_states
        if getattr(self.unet, 'supports_cross_view_refinement', False):
            unet_kwargs.update({
                'sat_tokens': sat_state.tokens,
                'sat_xy': sat_state.xy,
                'sat_bev_coords': sat_state.bev_coords,
                'front_bev_xy': front_bev_xy,
                'front_ground_valid_mask': front_ground_valid_mask,
                'condition_mask': condition_mask,
                'return_attn_map': False,
            })
        return unet_kwargs

    def _sample_condition_mask(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        condition_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        if self.training and self.cond_drop_prob > 0.0:
            condition_mask = torch.rand(batch_size, device=device) >= self.cond_drop_prob
            if not bool(condition_mask.any().item()):
                keep_index = torch.randint(0, batch_size, (1,), device=device)
                condition_mask[keep_index] = True
        return condition_mask

    def _apply_condition_dropout(
        self,
        sat_state: SatelliteMemoryState,
        condition_mask: torch.Tensor,
    ) -> SatelliteMemoryState:
        return sat_state.replace(
            tokens=sat_state.tokens * self._expand_condition_mask(condition_mask, sat_state.tokens),
            xy=sat_state.xy * self._expand_condition_mask(condition_mask, sat_state.xy),
            bev_coords=(
                sat_state.bev_coords * self._expand_condition_mask(condition_mask, sat_state.bev_coords)
                if sat_state.bev_coords is not None
                else None
            ),
        )

    @staticmethod
    def _merge_satellite_states(
        base_state: SatelliteMemoryState,
        updated_state: SatelliteMemoryState,
        condition_mask: torch.Tensor,
    ) -> SatelliteMemoryState:
        keep_mask = condition_mask.to(device=base_state.tokens.device, dtype=base_state.tokens.dtype).view(-1, 1, 1)
        merged_tokens = keep_mask * updated_state.tokens + (1.0 - keep_mask) * base_state.tokens
        merged_xy = keep_mask * updated_state.xy + (1.0 - keep_mask) * base_state.xy

        base_bev = base_state.bev_coords
        updated_bev = updated_state.bev_coords
        if base_bev is None or updated_bev is None:
            merged_bev = updated_bev if base_bev is None else base_bev
        else:
            merged_bev = keep_mask * updated_bev + (1.0 - keep_mask) * base_bev

        return SatelliteMemoryState(tokens=merged_tokens, xy=merged_xy, bev_coords=merged_bev)

    def forward_view_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        target_images: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run one street-view update/read step against a provided satellite memory.

        Returns the per-view diffusion loss together with the updated satellite state.
        """
        batch_size = target_images.shape[0]
        device = target_images.device

        if condition_mask is None:
            condition_mask = self._sample_condition_mask(batch_size=batch_size, device=device)
        conditioned_sat_state = self._apply_condition_dropout(sat_state, condition_mask)
        encoder_hidden_states = None

        with torch.no_grad():
            target_images_vae = self._normalize_images_for_vae(target_images)
            latents = self.vae.encode(target_images_vae).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        unet_kwargs = self._build_unet_kwargs(
            encoder_hidden_states=encoder_hidden_states,
            sat_state=conditioned_sat_state,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
            condition_mask=condition_mask,
        )
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            **unet_kwargs,
        ).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred, target, reduction="mean")
        updated_sat_state = getattr(self.unet, "last_satellite_state", None)
        if updated_sat_state is None:
            updated_sat_state = conditioned_sat_state
        updated_sat_state = self._merge_satellite_states(
            base_state=sat_state,
            updated_state=updated_sat_state,
            condition_mask=condition_mask,
        )

        refinement_stats_by_site = getattr(self.unet, "last_refinement_stats", {})
        refinement_stats = _aggregate_refinement_stats(refinement_stats_by_site)

        return {
            "loss": loss,
            "model_pred": model_pred,
            "target": target,
            "sat_state": updated_sat_state,
            "condition_mask": condition_mask,
            "refinement_stats": refinement_stats,
            "refinement_stats_by_site": refinement_stats_by_site,
            **refinement_stats,
        }

    def forward(
        self,
        sat_images: torch.Tensor,
        target_images: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            sat_images: (B, 3, H_sat, W_sat) - Satellite images
            target_images: (B, 3, H, W) - Target frontview images
            front_bev_xy: (B, 2, H_cam, W_cam) - BEV coordinates for each camera pixel

        Returns:
            dict with 'loss' and other info
        """
        if target_images.ndim != 4:
            raise ValueError(
                "This experiment trains one random-yaw street view per sample; "
                f"target_images must be [B,C,H,W], got {list(target_images.shape)}"
            )

        sat_state = self.encode_satellite(sat_images)
        return self.forward_view_with_satellite_state(
            sat_state=sat_state,
            target_images=target_images,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
        )

    @torch.no_grad()
    def _get_vae_scale_factor(self) -> int:
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels is None:
            return 8
        return max(1, 2 ** (len(block_out_channels) - 1))

    @staticmethod
    def _infer_image_size_from_front_bev_xy(front_bev_xy: Optional[torch.Tensor]) -> Optional[Tuple[int, int]]:
        if front_bev_xy is None or not torch.is_tensor(front_bev_xy):
            return None

        if front_bev_xy.ndim == 4 and front_bev_xy.shape[1] == 2:
            return int(front_bev_xy.shape[2]), int(front_bev_xy.shape[3])
        if front_bev_xy.ndim == 4 and front_bev_xy.shape[-1] == 2:
            return int(front_bev_xy.shape[1]), int(front_bev_xy.shape[2])
        if front_bev_xy.ndim == 3 and front_bev_xy.shape[0] == 2:
            return int(front_bev_xy.shape[1]), int(front_bev_xy.shape[2])
        if front_bev_xy.ndim == 3 and front_bev_xy.shape[-1] == 2:
            return int(front_bev_xy.shape[0]), int(front_bev_xy.shape[1])

        return None

    def _infer_generation_size(
        self,
        front_bev_xy: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if target_size is not None:
            return int(target_size[0]), int(target_size[1])

        coords_size = self._infer_image_size_from_front_bev_xy(front_bev_xy)
        if coords_size is not None:
            return coords_size

        sample_size = self.unet.config.sample_size
        if isinstance(sample_size, (tuple, list)):
            return int(sample_size[0]), int(sample_size[1])
        return int(sample_size), int(sample_size)

    @staticmethod
    def _first_batch_satellite_state(
        sat_state: SatelliteMemoryState,
        batch_size: int,
    ) -> SatelliteMemoryState:
        return SatelliteMemoryState(
            tokens=sat_state.tokens[:batch_size],
            xy=sat_state.xy[:batch_size],
            bev_coords=(
                sat_state.bev_coords[:batch_size]
                if sat_state.bev_coords is not None
                else None
            ),
        )

    @torch.no_grad()
    def generate_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sat_condition_mode: str = "normal",
    ) -> Tuple[torch.Tensor, SatelliteMemoryState]:
        """
        Generate images from a precomputed satellite memory state.

        The memory is initialized by the caller for one target yaw and updated
        only inside that denoising trajectory.

        Args:
            sat_state: Satellite memory state shared by the current view.
            front_bev_xy: (B, 2, H_cam, W_cam) - BEV coordinates for each camera pixel
            target_size: Optional target image size as (H, W)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            sat_condition_mode: "normal" uses encoded satellite conditioning,
                "zero" disables satellite conditioning by zeroing tokens and masks.

        Returns:
            generated_images: (B, 3, H, W) and the updated satellite state.
        """
        B = sat_state.tokens.shape[0]
        device = sat_state.tokens.device

        if sat_condition_mode == "normal":
            condition_mask = torch.ones(B, device=device, dtype=torch.bool)
        elif sat_condition_mode == "zero":
            sat_state = sat_state.replace(
                tokens=torch.zeros_like(sat_state.tokens),
                xy=torch.zeros_like(sat_state.xy),
                bev_coords=(torch.zeros_like(sat_state.bev_coords) if sat_state.bev_coords is not None else None),
            )
            condition_mask = torch.zeros(B, device=device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown sat_condition_mode: {sat_condition_mode}")

        encoder_hidden_states = None

        image_h, image_w = self._infer_generation_size(front_bev_xy=front_bev_xy, target_size=target_size)
        vae_scale_factor = self._get_vae_scale_factor()
        latent_h = max(1, (image_h + vae_scale_factor - 1) // vae_scale_factor)
        latent_w = max(1, (image_w + vae_scale_factor - 1) // vae_scale_factor)

        # Initialize latents
        latents = torch.randn(
            (B, self.unet.config.in_channels, latent_h, latent_w),
            device=device,
            dtype=sat_state.tokens.dtype,
            generator=generator,
        )

        # Prepare CFG branches once and reuse them for every timestep.
        use_cfg = guidance_scale > 1.0
        if use_cfg:
            uncond_state = sat_state.replace(
                tokens=torch.zeros_like(sat_state.tokens),
                xy=torch.zeros_like(sat_state.xy),
                bev_coords=(torch.zeros_like(sat_state.bev_coords) if sat_state.bev_coords is not None else None),
            )
            encoder_hidden_states_double = None
            sat_state_double = sat_state.replace(
                tokens=torch.cat([sat_state.tokens, uncond_state.tokens], dim=0),
                xy=torch.cat([sat_state.xy, uncond_state.xy], dim=0),
                bev_coords=(
                    torch.cat([sat_state.bev_coords, uncond_state.bev_coords], dim=0)
                    if sat_state.bev_coords is not None and uncond_state.bev_coords is not None
                    else None
                ),
            )
            front_bev_xy_double = torch.cat([front_bev_xy, front_bev_xy], dim=0) if front_bev_xy is not None else None
            front_ground_valid_mask_double = (
                torch.cat([front_ground_valid_mask, front_ground_valid_mask], dim=0)
                if front_ground_valid_mask is not None
                else None
            )
            condition_mask_double = torch.cat([
                condition_mask,
                torch.zeros_like(condition_mask),
            ], dim=0)

        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        updated_sat_state = sat_state

        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            if use_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                unet_kwargs = self._build_unet_kwargs(
                    encoder_hidden_states=encoder_hidden_states_double,
                    sat_state=sat_state_double,
                    front_bev_xy=front_bev_xy_double,
                    front_ground_valid_mask=front_ground_valid_mask_double,
                    condition_mask=condition_mask_double,
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
                last_sat_state = getattr(self.unet, "last_satellite_state", None)
                if last_sat_state is not None:
                    updated_sat_state = self._first_batch_satellite_state(last_sat_state, B)
            else:
                unet_kwargs = self._build_unet_kwargs(
                    encoder_hidden_states=encoder_hidden_states,
                    sat_state=sat_state,
                    front_bev_xy=front_bev_xy,
                    front_ground_valid_mask=front_ground_valid_mask,
                    condition_mask=condition_mask,
                )
                noise_pred = self.unet(
                    latents,
                    t,
                    **unet_kwargs,
                ).sample
                last_sat_state = getattr(self.unet, "last_satellite_state", None)
                if last_sat_state is not None:
                    updated_sat_state = last_sat_state

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

        return generated_images, updated_sat_state

    @torch.no_grad()
    def generate(
        self,
        sat_images: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sat_condition_mode: str = "normal",
    ) -> torch.Tensor:
        """
        Generate images from satellite images.

        This preserves the original independent single-view inference behavior.
        """
        sat_state = self.encode_satellite(sat_images)
        generated_images, _ = self.generate_with_satellite_state(
            sat_state,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
            target_size=target_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            sat_condition_mode=sat_condition_mode,
        )
        return generated_images


def create_sd_model(
    base_model: str = 'stabilityai/stable-diffusion-2-1-base',
    freeze_base: bool = True,
    refinement_block_config: Optional[Dict] = None,
    refinement_injection_sites: Optional[Tuple[str, ...]] = None,
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

    refinement_cfg = refinement_block_config or {}
    unet = SatelliteConditionedUNet(
        enable_cross_view_refinement=refinement_cfg.get('enable', True),
        refinement_injection_sites=list(refinement_injection_sites) if refinement_injection_sites is not None else None,
        refinement_block_config={
            'num_heads': refinement_cfg.get('num_heads', 8),
            'head_dim': refinement_cfg.get('head_dim', 64),
            'geo_ratio': refinement_cfg.get('geo_ratio', 0.5),
            'rope_base': refinement_cfg.get('rope_base', 10000.0),
            'lambda_geo': refinement_cfg.get('lambda_geo', 1.0),
            'lambda_geom': refinement_cfg.get('lambda_geom', 1.0),
            'geom_hidden_dim': refinement_cfg.get('geom_hidden_dim', 128),
            'geom_head_dim': refinement_cfg.get('geom_head_dim', 16),
            'sat_update_layers': refinement_cfg.get('sat_update_layers', 1),
            'use_geom_bias': refinement_cfg.get('use_geom_bias', True),
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
        if self.is_main_process:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
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
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        # Setup output dir
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        # Refinement blocks are created lazily on first forward. Materialize them before
        # building the optimizer so their parameters are actually trainable.
        self._materialize_lazy_condition_modules()
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

    def _any_rank_true(self, value: bool) -> bool:
        if not self.distributed or not dist.is_initialized():
            return bool(value)
        device = torch.device(self.device)
        flag = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    @torch.no_grad()
    def _materialize_lazy_condition_modules(self) -> None:
        unet = getattr(self.model, "unet", None)
        if unet is None or not getattr(unet, "enable_cross_view_refinement", False):
            return
        if len(getattr(unet, "refinement_blocks", {})) > 0:
            return

        try:
            batch = next(iter(self.train_dataloader))
        except StopIteration:
            logger.warning("Skipped lazy module materialization because the training dataloader is empty")
            return

        sat_images = batch.get("sat")
        target_images = batch.get("image")
        if sat_images is None or target_images is None:
            logger.warning("Skipped lazy module materialization because batch is missing 'sat' or 'image'")
            return

        sat_images = sat_images[:1].to(self.device)
        front_bev_xy = batch.get("front_bev_xy")
        if front_bev_xy is not None:
            front_bev_xy = front_bev_xy[:1].to(self.device)
            if front_bev_xy.ndim == 5:
                front_bev_xy = front_bev_xy[:, 0]
        front_ground_valid_mask = batch.get("front_ground_valid_mask")
        if front_ground_valid_mask is not None:
            front_ground_valid_mask = front_ground_valid_mask[:1].to(self.device)
            if front_ground_valid_mask.ndim == 5:
                front_ground_valid_mask = front_ground_valid_mask[:, 0]
        target_size = tuple(int(x) for x in target_images.shape[-2:])

        was_training = self.model.training
        self.model.eval()
        _materialize_lazy_modules(self.model, sat_images, front_bev_xy, front_ground_valid_mask, target_size)
        if was_training:
            self.model.train()

        refinement_param_count = sum(p.numel() for p in unet.refinement_blocks.parameters())
        logger.info(
            f"Materialized {len(unet.refinement_blocks)} refinement block(s) before optimizer init "
            f"({refinement_param_count} parameters)"
        )

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

    @staticmethod
    def _tensor_debug_summary(name: str, tensor: torch.Tensor) -> str:
        with torch.no_grad():
            detached = tensor.detach()
            finite_mask = torch.isfinite(detached)
            finite_count = int(finite_mask.sum().item())
            total_count = detached.numel()
            summary = (
                f"{name}: shape={list(detached.shape)} dtype={detached.dtype} "
                f"finite={finite_count}/{total_count}"
            )
            if finite_count > 0:
                finite_values = detached[finite_mask].float()
                summary += (
                    f" min={finite_values.min().item():.4g}"
                    f" max={finite_values.max().item():.4g}"
                    f" mean={finite_values.mean().item():.4g}"
                )
            return summary

    @staticmethod
    def _first_nonfinite_named_tensor(
        named_tensors: Sequence[Tuple[str, Optional[torch.Tensor]]],
    ) -> Optional[str]:
        for name, tensor in named_tensors:
            if tensor is None or not torch.is_tensor(tensor):
                continue
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                return SDTrainer._tensor_debug_summary(name, tensor)
        return None

    @staticmethod
    def _grad_global_norm(parameters: Sequence[torch.nn.Parameter], device: str) -> torch.Tensor:
        norms = []
        for param in parameters:
            if param.grad is None:
                continue
            norms.append(torch.linalg.vector_norm(param.grad.detach().float()))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.linalg.vector_norm(torch.stack([norm.to(device=device) for norm in norms]))

    def _log_nonfinite_training_state(
        self,
        reason: str,
        epoch: int,
        step: int,
        outputs: Optional[Dict[str, Any]] = None,
        batch_tensors: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> None:
        if not self.is_main_process:
            return

        logger.warning(
            "Skipping optimizer update because non-finite values were detected "
            "(reason=%s, epoch=%d, step=%d)",
            reason,
            epoch + 1,
            step + 1,
        )

        if outputs:
            output_tensors: List[Tuple[str, Optional[torch.Tensor]]] = []
            for key in ("loss", "per_view_loss", "model_pred", "target"):
                value = outputs.get(key)
                if torch.is_tensor(value):
                    output_tensors.append((f"outputs.{key}", value))
            sat_state = outputs.get("sat_state")
            if isinstance(sat_state, SatelliteMemoryState):
                output_tensors.extend(
                    [
                        ("outputs.sat_state.tokens", sat_state.tokens),
                        ("outputs.sat_state.xy", sat_state.xy),
                        ("outputs.sat_state.bev_coords", sat_state.bev_coords),
                    ]
                )
            refinement_stats = outputs.get("refinement_stats", {})
            if isinstance(refinement_stats, dict):
                for key, value in refinement_stats.items():
                    if torch.is_tensor(value):
                        output_tensors.append((f"outputs.refinement_stats.{key}", value))
            issue = self._first_nonfinite_named_tensor(output_tensors)
            if issue is not None:
                logger.warning("  First non-finite output: %s", issue)

        if batch_tensors:
            issue = self._first_nonfinite_named_tensor(
                [(f"batch.{key}", value) for key, value in batch_tensors.items()]
            )
            if issue is not None:
                logger.warning("  First non-finite batch tensor: %s", issue)

        param_issue = self._first_nonfinite_named_tensor(
            [(name, param) for name, param in self.unwrapped_model.named_parameters()]
        )
        if param_issue is not None:
            logger.warning("  First non-finite parameter: %s", param_issue)

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

        try:
            if resume_from is not None:
                start_epoch = self._load_checkpoint(resume_from)
            if start_epoch >= self.num_train_epochs:
                logger.info(
                    "Checkpoint already reached configured training horizon: "
                    "start_epoch=%d num_train_epochs=%d",
                    start_epoch,
                    self.num_train_epochs,
                )
                return
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
        finite_loss_batches = 0
        num_batches = len(self.train_dataloader)
        self.optimizer.zero_grad(set_to_none=True)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        skip_accumulation_until = -1
        last_grad_norm = None

        sampler = getattr(self.train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Train Epoch {epoch+1}",
            disable=not self.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            accumulation_start = (step // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
            accumulation_end = min(accumulation_start + self.gradient_accumulation_steps, num_batches)
            accumulation_window_size = max(1, accumulation_end - accumulation_start)
            if step < skip_accumulation_until:
                if self.is_main_process:
                    progress_bar.set_postfix({"raw_loss": "skipped"})
                if (step + 1) == skip_accumulation_until:
                    skip_accumulation_until = -1
                continue

            # Move data to device
            sat_images = batch['sat'].to(self.device)
            target_images = batch['image'].to(self.device)

            front_bev_xy = batch.get('front_bev_xy')
            if front_bev_xy is not None:
                front_bev_xy = front_bev_xy.to(self.device)
            front_ground_valid_mask = batch.get('front_ground_valid_mask')
            if front_ground_valid_mask is not None:
                front_ground_valid_mask = front_ground_valid_mask.to(self.device)

            # Forward pass
            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    sat_images,
                    target_images,
                    front_bev_xy=front_bev_xy,
                    front_ground_valid_mask=front_ground_valid_mask,
                )
                raw_loss = outputs['loss']

            batch_tensors = {
                "sat": sat_images,
                "image": target_images,
                "front_bev_xy": front_bev_xy,
                "front_ground_valid_mask": front_ground_valid_mask,
            }
            loss_is_finite = bool(torch.isfinite(raw_loss.detach()).all().item())
            if self._any_rank_true(not loss_is_finite):
                self._log_nonfinite_training_state(
                    reason="loss",
                    epoch=epoch,
                    step=step,
                    outputs=outputs,
                    batch_tensors=batch_tensors,
                )
                if self.is_main_process:
                    self._log_scalars(
                        {
                            "train/skipped_nonfinite_update": 1.0,
                            "train/epoch": epoch + 1,
                        },
                        step=self._global_step(epoch, step),
                    )
                self.optimizer.zero_grad(set_to_none=True)
                skip_accumulation_until = accumulation_end
                if (step + 1) == accumulation_end:
                    skip_accumulation_until = -1
                continue

            # Backward pass
            loss = raw_loss / accumulation_window_size
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) == accumulation_end:
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            self.max_grad_norm,
                        )
                    else:
                        grad_norm = self._grad_global_norm(trainable_params, self.device)
                else:
                    if self.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            self.max_grad_norm,
                        )
                    else:
                        grad_norm = self._grad_global_norm(trainable_params, self.device)

                last_grad_norm = float(grad_norm.detach().float().item())
                grad_is_finite = bool(torch.isfinite(grad_norm.detach()).all().item())
                if self._any_rank_true(not grad_is_finite):
                    self._log_nonfinite_training_state(
                        reason="grad",
                        epoch=epoch,
                        step=step,
                        outputs=outputs,
                        batch_tensors=batch_tensors,
                    )
                    if self.is_main_process:
                        self._log_scalars(
                            {
                                "train/skipped_nonfinite_update": 1.0,
                                "train/grad_norm": last_grad_norm,
                                "train/epoch": epoch + 1,
                            },
                            step=self._global_step(epoch, step),
                        )
                    if self.use_amp and self.amp_dtype == torch.float16:
                        self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_raw_loss += raw_loss.item()
            finite_loss_batches += 1
            postfix = {'raw_loss': f"{raw_loss.item():.3f}"}
            geom_ratio = outputs.get('refinement_logits_geom_to_sem_ratio_mean')
            if torch.is_tensor(geom_ratio):
                postfix['geom/sem'] = f"{geom_ratio.item():.2f}"
            if self.is_main_process:
                progress_bar.set_postfix(postfix)

            # Log
            if self.is_main_process and (step + 1) % self.log_every == 0:
                geom_std = outputs.get('refinement_logits_geom_std_mean')
                sem_std = outputs.get('refinement_logits_sem_std_mean')
                geom_ratio = outputs.get('refinement_logits_geom_to_sem_ratio_mean')
                if all(torch.is_tensor(v) for v in (geom_std, sem_std, geom_ratio)):
                    site_ratio_parts = []
                    for site, site_stats in outputs.get('refinement_stats_by_site', {}).items():
                        ratio = site_stats.get('logits_geom_to_sem_ratio')
                        if torch.is_tensor(ratio):
                            site_ratio_parts.append(f"{site}={ratio.item():.3f}")
                    site_ratio_text = f" ({', '.join(site_ratio_parts)})" if site_ratio_parts else ""
                    logger.info(
                        "Train step %d/%d: raw_loss=%.6f sem_std=%.6f geom_std=%.6f geom/sem=%.3f%s",
                        step + 1,
                        num_batches,
                        raw_loss.item(),
                        sem_std.item(),
                        geom_std.item(),
                        geom_ratio.item(),
                        site_ratio_text,
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
                    'train/lr': self.lr_scheduler.get_last_lr()[0],
                    'train/epoch': epoch + 1,
                }
                if last_grad_norm is not None:
                    log_payload['train/grad_norm'] = last_grad_norm
                if torch.is_tensor(sem_std):
                    log_payload['refinement/logits_sem_std_mean'] = sem_std.item()
                if torch.is_tensor(geom_std):
                    log_payload['refinement/logits_geom_std_mean'] = geom_std.item()
                if torch.is_tensor(geom_ratio):
                    log_payload['refinement/logits_geom_to_sem_ratio_mean'] = geom_ratio.item()
                self._log_scalars(log_payload, step=self._global_step(epoch, step))

        local_mean = total_raw_loss / max(1, finite_loss_batches)
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

            front_bev_xy = batch.get('front_bev_xy')
            if front_bev_xy is not None:
                front_bev_xy = front_bev_xy.to(self.device)
            front_ground_valid_mask = batch.get('front_ground_valid_mask')
            if front_ground_valid_mask is not None:
                front_ground_valid_mask = front_ground_valid_mask.to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = eval_model(
                    sat_images,
                    target_images,
                    front_bev_xy=front_bev_xy,
                    front_ground_valid_mask=front_ground_valid_mask,
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

    def _compose_visualization(
        self,
        sat_image: torch.Tensor,
        generated_image: torch.Tensor,
        real_image: torch.Tensor,
    ) -> Image.Image:
        target_h, target_w = int(real_image.shape[-2]), int(real_image.shape[-1])
        sat_resized = F.interpolate(
            sat_image.unsqueeze(0),
            size=(target_h, target_h),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        sat_pil = self._tensor_to_pil(sat_resized)
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
        front_bev_xy_chunks = []
        front_ground_valid_mask_chunks = []
        frame_ids = []

        for batch in data_loader:
            batch_count = batch['sat'].shape[0]
            remaining = self.num_visualizations - len(frame_ids)
            if remaining <= 0:
                break

            take = min(remaining, batch_count)
            sat_chunks.append(batch['sat'][:take])
            target_chunks.append(batch['image'][:take])

            front_bev_xy = batch.get('front_bev_xy')
            if front_bev_xy is not None:
                front_bev_xy_chunks.append(front_bev_xy[:take])
            front_ground_valid_mask = batch.get('front_ground_valid_mask')
            if front_ground_valid_mask is not None:
                front_ground_valid_mask_chunks.append(front_ground_valid_mask[:take])

            batch_frame_ids = batch.get('frame_id')
            if batch_frame_ids is None:
                frame_ids.extend([None] * take)
            else:
                frame_ids.extend(list(batch_frame_ids[:take]))

            if len(frame_ids) >= self.num_visualizations:
                break

        if not sat_chunks:
            return

        sat_images = torch.cat(sat_chunks, dim=0).to(self.device)
        target_images = torch.cat(target_chunks, dim=0).to(self.device)
        front_bev_xy = torch.cat(front_bev_xy_chunks, dim=0).to(self.device) if front_bev_xy_chunks else None
        front_ground_valid_mask = (
            torch.cat(front_ground_valid_mask_chunks, dim=0).to(self.device)
            if front_ground_valid_mask_chunks else None
        )

        generator_device = self.device if self.device.startswith("cuda") else "cpu"

        eval_model = self.unwrapped_model
        was_training = eval_model.training
        eval_model.eval()
        sat_state = eval_model.encode_satellite(sat_images)
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.visualization_seed)
        generated_images, _ = eval_model.generate_with_satellite_state(
            sat_state,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
            target_size=tuple(target_images.shape[-2:]),
            num_inference_steps=self.visualization_inference_steps,
            guidance_scale=self.visualization_guidance_scale,
            generator=generator,
        )
        if was_training:
            eval_model.train()

        epoch_dir = self.visualization_dir / f"epoch_{epoch + 1:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        comparison_images = []
        captions = []

        for idx in range(sat_images.shape[0]):
            frame_id = frame_ids[idx]
            frame_suffix = f"_frame_{int(frame_id):010d}" if frame_id is not None else ""
            comparison = self._compose_visualization(
                sat_images[idx],
                generated_images[idx],
                target_images[idx],
            )
            comparison.save(epoch_dir / f"sample_{idx:02d}{frame_suffix}.png")
            comparison_images.append(comparison)
            caption = f"epoch={epoch + 1} sample={idx:02d}"
            if frame_id is not None:
                caption += f" frame={int(frame_id):010d}"
            captions.append(caption)

        logger.info(f"Saved visualizations: {epoch_dir}")
        self._log_visualizations(
            comparison_images,
            captions,
            step=self._global_step(epoch, max(0, len(self.train_dataloader) - 1)),
        )

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load from checkpoint and return the next zero-based epoch index."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        load_model_state_dict(self.unwrapped_model, checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        checkpoint_epoch = checkpoint.get('epoch')
        start_epoch = int(checkpoint_epoch) + 1 if checkpoint_epoch is not None else 0
        if checkpoint_epoch is None:
            logger.info(
                "Checkpoint loaded: %s (no epoch metadata found; restarting epoch count from 1)",
                checkpoint_path,
            )
        else:
            logger.info(
                "Checkpoint loaded: %s (resuming at epoch %d/%d)",
                checkpoint_path,
                start_epoch + 1,
                self.num_train_epochs,
            )
        return start_epoch
