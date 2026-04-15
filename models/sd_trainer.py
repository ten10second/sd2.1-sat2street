"""
Stable Diffusion Trainer for satellite-to-frontview generation.

This module provides a simplified training interface using diffusers library.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models.sd_model import SatelliteConditionedUNet


logger = logging.getLogger(__name__)


def _filter_compatible_keys(keys: Sequence[str], allowed_prefixes: Sequence[str]) -> Sequence[str]:
    if not allowed_prefixes:
        return list(keys)
    return [
        key
        for key in keys
        if not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]


def load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    allow_missing_prefixes: Sequence[str] = (),
    allow_unexpected_prefixes: Sequence[str] = (),
) -> Tuple[Sequence[str], Sequence[str]]:
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    filtered_missing = _filter_compatible_keys(missing_keys, allow_missing_prefixes)
    filtered_unexpected = _filter_compatible_keys(unexpected_keys, allow_unexpected_prefixes)
    if filtered_missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {filtered_missing}")
    if filtered_unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {filtered_unexpected}")
    return missing_keys, unexpected_keys


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str,
    *,
    allow_missing_prefixes: Sequence[str] = (),
    allow_unexpected_prefixes: Sequence[str] = (),
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    load_model_state_dict(
        model,
        state_dict,
        allow_missing_prefixes=allow_missing_prefixes,
        allow_unexpected_prefixes=allow_unexpected_prefixes,
    )
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
    coords_map: Optional[torch.Tensor],
    target_size: Tuple[int, int],
) -> None:
    sat_encoded = model.encode_satellite(sat_images, coords_map)
    if isinstance(sat_encoded, tuple):
        sat_tokens, sat_xy = sat_encoded
    else:
        sat_tokens = sat_encoded
        sat_xy = None

    vae_scale_factor = model._get_vae_scale_factor()
    latent_h = max(1, (target_size[0] + vae_scale_factor - 1) // vae_scale_factor)
    latent_w = max(1, (target_size[1] + vae_scale_factor - 1) // vae_scale_factor)
    latents = torch.randn(
        (sat_images.shape[0], model.unet.config.in_channels, latent_h, latent_w),
        device=sat_images.device,
        dtype=sat_tokens.dtype,
    )
    timestep = torch.zeros((sat_images.shape[0],), device=sat_images.device, dtype=torch.long)

    model.unet(
        latents,
        timestep,
        encoder_hidden_states=None,
        sat_tokens=sat_tokens,
        sat_xy=sat_xy,
        front_bev_xy=coords_map,
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
                if ".attn2.to_k." in name or ".attn2.to_v." in name:
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

    def encode_satellite(self, sat_images: torch.Tensor, coords_map: torch.Tensor = None) -> torch.Tensor:
        """Encode satellite images to embeddings."""
        return self.satellite_encoder(sat_images, coords_map, return_sat_xy=True)

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
        sat_tokens: torch.Tensor,
        sat_xy: Optional[torch.Tensor],
        coords_map: Optional[torch.Tensor],
        plucker_map: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        unet_kwargs: Dict[str, Any] = {}
        if encoder_hidden_states is not None:
            unet_kwargs['encoder_hidden_states'] = encoder_hidden_states
        if getattr(self.unet, 'supports_satellite_reading', False):
            unet_kwargs.update({
                'sat_tokens': sat_tokens,
                'sat_xy': sat_xy,
                'front_bev_xy': coords_map,
                'front_plucker': plucker_map,
                'condition_mask': condition_mask,
                'return_attn_map': False,
            })
        return unet_kwargs

    def forward(
        self,
        sat_images: torch.Tensor,
        target_images: torch.Tensor,
        coords_map: torch.Tensor = None,
        plucker_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            sat_images: (B, 3, H_sat, W_sat) - Satellite images
            target_images: (B, 3, H, W) - Target frontview images
            coords_map: (B, 2, H_cam, W_cam) - BEV coordinates for each camera pixel
            plucker_map: (B, 6, H_cam, W_cam) - Per-pixel Plucker ray map

        Returns:
            dict with 'loss' and other info
        """
        B = sat_images.shape[0]
        device = sat_images.device

        # Encode satellite images
        sat_encoded = self.encode_satellite(sat_images, coords_map)
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
        encoder_hidden_states = None

        # Encode target images to latents
        with torch.no_grad():
            target_images_vae = self._normalize_images_for_vae(target_images)
            latents = self.vae.encode(target_images_vae).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,),
            device=device, dtype=torch.long
        )

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        unet_kwargs = self._build_unet_kwargs(
            encoder_hidden_states=encoder_hidden_states,
            sat_tokens=sat_tokens,
            sat_xy=sat_xy,
            coords_map=coords_map,
            plucker_map=plucker_map,
            condition_mask=condition_mask,
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

        loss = F.mse_loss(model_pred, target, reduction="mean")

        return {
            'loss': loss,
            'model_pred': model_pred,
            'target': target,
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
        sat_encoded = self.encode_satellite(sat_images, coords_map)
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

        encoder_hidden_states = None

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
            uncond_tokens = torch.zeros_like(sat_tokens)
            uncond_sat_xy = torch.zeros_like(sat_xy) if sat_xy is not None else None
            encoder_hidden_states_double = None
            sat_tokens_double = torch.cat([sat_tokens, uncond_tokens], dim=0)
            sat_xy_double = torch.cat([sat_xy, uncond_sat_xy], dim=0) if sat_xy is not None else None
            coords_map_double = torch.cat([coords_map, coords_map], dim=0) if coords_map is not None else None
            plucker_map_double = torch.cat([plucker_map, plucker_map], dim=0) if plucker_map is not None else None
            condition_mask_double = torch.cat([
                condition_mask,
                torch.zeros_like(condition_mask),
            ], dim=0)

        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            if use_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                unet_kwargs = self._build_unet_kwargs(
                    encoder_hidden_states=encoder_hidden_states_double,
                    sat_tokens=sat_tokens_double,
                    sat_xy=sat_xy_double,
                    coords_map=coords_map_double,
                    plucker_map=plucker_map_double,
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
            else:
                unet_kwargs = self._build_unet_kwargs(
                    encoder_hidden_states=encoder_hidden_states,
                    sat_tokens=sat_tokens,
                    sat_xy=sat_xy,
                    coords_map=coords_map,
                    plucker_map=plucker_map,
                    condition_mask=condition_mask,
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
    reading_injection_sites: Optional[Tuple[str, ...]] = None,
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

    reading_cfg = reading_block_config or {}
    unet = SatelliteConditionedUNet(
        use_satellite_reading=reading_cfg.get('enable', True),
        reading_injection_sites=list(reading_injection_sites) if reading_injection_sites is not None else None,
        reading_block_config={
            'num_heads': reading_cfg.get('num_heads', 8),
            'head_dim': reading_cfg.get('head_dim', 64),
            'geo_ratio': reading_cfg.get('geo_ratio', 0.5),
            'rope_base': reading_cfg.get('rope_base', 10000.0),
            'lambda_geo': reading_cfg.get('lambda_geo', 1.0),
            'gate_hidden_ratio': reading_cfg.get('gate_hidden_ratio', 0.25),
            'use_geom_bias': reading_cfg.get('use_geom_bias', True),
            'use_gated_residual': reading_cfg.get('use_gated_residual', True),
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
        mixed_precision: Optional[str] = None,
        max_grad_norm: float = 1.0,
        visualize_every: int = 1,
        num_visualizations: int = 4,
        visualization_inference_steps: int = 20,
        visualization_guidance_scale: float = 1.0,
        visualization_seed: int = 42,
    ):
        self.model = model.to(device)
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
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.max_grad_norm = float(max_grad_norm)
        self.visualize_every = max(0, int(visualize_every))
        self.num_visualizations = max(0, int(num_visualizations))
        self.visualization_inference_steps = int(visualization_inference_steps)
        self.visualization_guidance_scale = float(visualization_guidance_scale)
        self.visualization_seed = int(visualization_seed)
        self.visualization_dir = self.output_dir / "visualizations"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
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
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reading blocks are created lazily on first forward. Materialize them before
        # building the optimizer so their parameters are actually trainable.
        self._materialize_lazy_condition_modules()
        self._ensure_trainable_params_fp32()
        self._assert_no_trainable_fp16_params()

        # Setup optimizer
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
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
        if use_wandb:
            if not is_wandb_available():
                raise ImportError("Please install wandb to use logging: pip install wandb")
            import wandb
            wandb.init(project=project_name)

        logger.info(f"[SDTrainer] Initialized")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Num epochs: {num_train_epochs}")
        logger.info(f"  Batch size: {train_dataloader.batch_size}")
        logger.info(f"  Mixed precision: {self.mixed_precision or 'disabled'}")
        logger.info(f"  Max grad norm: {self.max_grad_norm}")
        if self.visualize_every > 0 and self.num_visualizations > 0:
            logger.info(
                f"  Visualization: every {self.visualize_every} epoch(s), "
                f"{self.num_visualizations} sample(s), {self.visualization_inference_steps} denoise steps"
            )

    @torch.no_grad()
    def _materialize_lazy_condition_modules(self) -> None:
        unet = getattr(self.model, "unet", None)
        if unet is None or not getattr(unet, "use_satellite_reading", False):
            return
        if len(getattr(unet, "reading_blocks", {})) > 0:
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
        coords_map = batch.get("coords_map")
        if coords_map is not None:
            coords_map = coords_map[:1].to(self.device)
        target_size = tuple(int(x) for x in target_images.shape[-2:])

        was_training = self.model.training
        self.model.eval()
        _materialize_lazy_modules(self.model, sat_images, coords_map, target_size)
        if was_training:
            self.model.train()

        reading_param_count = sum(p.numel() for p in unet.reading_blocks.parameters())
        logger.info(
            f"Materialized {len(unet.reading_blocks)} reading block(s) before optimizer init "
            f"({reading_param_count} parameters)"
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

    def train(self, resume_from: Optional[str] = None):
        """Run training."""
        start_epoch = 0

        if resume_from is not None:
            self._load_checkpoint(resume_from)

        for epoch in range(start_epoch, self.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_train_epochs}")
            train_raw_loss = self._train_epoch(epoch)
            logger.info(f"  Train raw loss: {train_raw_loss:.4f}")

            # Validate
            if self.val_dataloader is not None:
                val_loss = self._validate(epoch)
                logger.info(f"  Val loss: {val_loss:.4f}")

            if self.visualize_every > 0 and self.num_visualizations > 0 and (epoch + 1) % self.visualize_every == 0:
                self._save_visualizations(epoch)

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or (epoch + 1) == self.num_train_epochs:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_raw_loss = 0.0
        num_batches = len(self.train_dataloader)
        self.optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            # Move data to device
            sat_images = batch['sat'].to(self.device)
            target_images = batch['image'].to(self.device)

            # Get coords_map - BEV coordinates for each camera pixel (透视 token 在 BEV 图上的坐标)
            coords_map = batch.get('coords_map')
            if coords_map is not None:
                coords_map = coords_map.to(self.device)
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
            progress_bar.set_postfix({'raw_loss': f"{raw_loss.item():.3f}"})

            # Log
            if (step + 1) % self.log_every == 0:
                logger.info(
                    "Train step %d/%d: raw_loss=%.6f",
                    step + 1,
                    num_batches,
                    raw_loss.item(),
                )
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'train_raw_loss': raw_loss.item(),
                        'epoch': epoch,
                        'step': step,
                    })

        return total_raw_loss / num_batches

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_dataloader, desc=f"Val Epoch {epoch+1}"):
            sat_images = batch['sat'].to(self.device)
            target_images = batch['image'].to(self.device)

            # Get coords_map
            coords_map = batch.get('coords_map')
            if coords_map is not None:
                coords_map = coords_map.to(self.device)
            plucker_map = batch.get('plucker_map')
            if plucker_map is not None:
                plucker_map = plucker_map.to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    sat_images,
                    target_images,
                    coords_map=coords_map,
                    plucker_map=plucker_map,
                )
                loss = outputs['loss']
            total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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
        coords_chunks = []
        plucker_chunks = []
        frame_ids = []

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
            plucker_map = batch.get('plucker_map')
            if plucker_map is not None:
                plucker_chunks.append(plucker_map[:take])

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
        coords_map = torch.cat(coords_chunks, dim=0).to(self.device) if coords_chunks else None
        plucker_map = torch.cat(plucker_chunks, dim=0).to(self.device) if plucker_chunks else None

        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.visualization_seed)

        was_training = self.model.training
        self.model.eval()
        generated_images = self.model.generate(
            sat_images,
            coords_map=coords_map,
            plucker_map=plucker_map,
            target_size=tuple(target_images.shape[-2:]),
            num_inference_steps=self.visualization_inference_steps,
            guidance_scale=self.visualization_guidance_scale,
            generator=generator,
        )
        if was_training:
            self.model.train()

        epoch_dir = self.visualization_dir / f"epoch_{epoch + 1:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(generated_images.shape[0]):
            frame_id = frame_ids[idx]
            frame_suffix = f"_frame_{int(frame_id):010d}" if frame_id is not None else ""
            comparison = self._compose_visualization(
                sat_images[idx],
                generated_images[idx],
                target_images[idx],
            )
            comparison.save(epoch_dir / f"sample_{idx:02d}{frame_suffix}.png")

        logger.info(f"Saved visualizations: {epoch_dir}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        load_model_state_dict(self.model, checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
