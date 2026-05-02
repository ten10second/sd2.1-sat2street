"""
Stable Diffusion Model for Satellite-to-Frontview Generation.

Custom Stable Diffusion implementation with satellite condition encoder.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from diffusers import StableDiffusionPipeline
try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
except ImportError:
    from diffusers.models.unet_2d_condition import UNet2DConditionModel

try:
    from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
except ImportError:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.unet.satellite_style_adapter import SatelliteStyleAdapter

DEFAULT_SATELLITE_EMBED_DIM = 768


class SatelliteConditionedUNet(UNet2DConditionModel):
    """UNet2DConditionModel wrapper that only accepts native encoder_hidden_states."""

    def __init__(
        self,
        reading_block_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_satellite_reading = False
        self.use_satellite_reading = False
        self.reading_block_config = {
            "sat_in_dim": DEFAULT_SATELLITE_EMBED_DIM,
            "plucker_dropout_prob": 0.3,
            "style_num_tokens": 4,
            "style_num_heads": 8,
            "style_scale": 0.5,
            "view_grid_h": 8,
            "view_grid_w": 20,
            "view_query_dim": DEFAULT_SATELLITE_EMBED_DIM,
            "view_num_heads": 8,
            "view_scale": 1.0,
            "view_geo_bias_weight": 1.0,
            "view_geo_sigma": 0.35,
            "view_local_topk": 25,
            "view_geo_target_sigma": 0.20,
            "view_geo_loss_weight": 0.1,
            "view_gate_hidden_dim": 256,
            "save_attention_heatmap": True,
            "heatmap_max_tokens": 16,
        }
        if reading_block_config is not None:
            self.reading_block_config.update(reading_block_config)
        self.plucker_dropout_prob = float(self.reading_block_config.get("plucker_dropout_prob", 0.0))
        self.last_attn_maps: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        cross_attention_dim = int(self.config.cross_attention_dim or 1024)
        if encoder_hidden_states is None:
            encoder_hidden_states = sample.new_zeros((sample.shape[0], 1, cross_attention_dim))
        else:
            encoder_hidden_states = encoder_hidden_states.to(device=sample.device, dtype=sample.dtype)
            if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != sample.shape[0]:
                raise ValueError(
                    "encoder_hidden_states must be [B, N, C], got "
                    f"{list(encoder_hidden_states.shape)} for batch {sample.shape[0]}"
                )
            if encoder_hidden_states.shape[-1] != cross_attention_dim:
                raise ValueError(
                    f"encoder_hidden_states last dim must match cross_attention_dim={cross_attention_dim}, "
                    f"got {encoder_hidden_states.shape[-1]}"
                )

        self.last_attn_maps = {}
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )


class SatelliteConditionedSDPipeline(StableDiffusionPipeline):
    """
    Stable Diffusion pipeline with satellite condition support.

    Args:
        satellite_encoder: Satellite condition encoder
        **kwargs: Additional arguments for StableDiffusionPipeline
    """

    def __init__(
        self,
        satellite_encoder: nn.Module,
        satellite_style_adapter: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.satellite_encoder = satellite_encoder
        self.satellite_encoder.eval()
        self.satellite_style_adapter = satellite_style_adapter
        if self.satellite_style_adapter is not None:
            self.satellite_style_adapter.eval()

    def _encode_satellite_images(self, sat_images: torch.Tensor):
        """Encode satellite images to embeddings."""
        with torch.no_grad():
            sat_emb = self.satellite_encoder(sat_images)
        return sat_emb

    def _encode_satellite_style(self, sat_emb: torch.Tensor) -> torch.Tensor:
        if self.satellite_style_adapter is None:
            cross_attention_dim = int(self.unet.config.cross_attention_dim or sat_emb.shape[-1])
            return sat_emb.new_zeros((sat_emb.shape[0], 4, cross_attention_dim))
        with torch.no_grad():
            return self.satellite_style_adapter(sat_emb)

    def __call__(
        self,
        sat_images: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate frontview images from satellite images.

        Args:
            sat_images: (B, 3, H_sat, W_sat) or (3, H_sat, W_sat) - Satellite images
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for classifier-free guidance
            negative_prompt: Negative prompt for guidance
            **kwargs: Additional arguments

        Returns:
            output: StableDiffusionPipelineOutput
        """
        # Handle single image input
        if sat_images.ndim == 3:
            sat_images = sat_images.unsqueeze(0)

        # Move to device
        sat_images = sat_images.to(self.device)

        # Encode satellite images
        sat_emb = self._encode_satellite_images(sat_images)
        style_tokens = self._encode_satellite_style(sat_emb)

        # Encode negative prompt
        negative_emb = None
        if negative_prompt is not None:
            negative_emb = self._encode_prompt(negative_prompt, sat_images.shape[0])

        # Encode positive prompt (we use empty prompt for now)
        positive_emb = self._encode_prompt("", sat_images.shape[0])

        # Concatenate embeddings
        encoder_hidden_states = torch.cat([positive_emb, style_tokens], dim=1)
        if negative_emb is not None:
            negative_emb = torch.cat([negative_emb, torch.zeros_like(style_tokens)], dim=1)

        # Call super().__call__
        return super().__call__(
            prompt=None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            encoder_hidden_states=encoder_hidden_states,
            negative_prompt_embeds=negative_emb,
            **kwargs,
        )


def load_sd_model(
    base_model: str = 'stabilityai/stable-diffusion-2-1',
    freeze_base: bool = True,
    reading_block_config: Optional[Dict[str, Any]] = None,
) -> SatelliteConditionedSDPipeline:
    """
    Load Stable Diffusion model with satellite conditioning support.

    Args:
        base_model: Base model name from Hugging Face hub
        freeze_base: Whether to freeze base Stable Diffusion layers

    Returns:
        pipeline: Satellite-conditioned Stable Diffusion pipeline
    """
    # Load base model
    pipeline = StableDiffusionPipeline.from_pretrained(base_model)

    base_unet = pipeline.unet

    reading_cfg = dict(reading_block_config or {})
    sat_embed_dim = int(reading_cfg.get("sat_in_dim", DEFAULT_SATELLITE_EMBED_DIM))
    reading_cfg["sat_in_dim"] = sat_embed_dim
    sat_num_heads = 12
    if sat_embed_dim % sat_num_heads != 0:
        sat_num_heads = sat_embed_dim // 64 if sat_embed_dim % 64 == 0 else 1

    # Create satellite encoder (separate from UNet)
    satellite_encoder = SatelliteConditionEncoder(embed_dim=sat_embed_dim, num_heads=sat_num_heads)
    cross_attention_dim = int(base_unet.config.cross_attention_dim or 1024)
    style_num_tokens = int(reading_cfg.get("style_num_tokens", 4))
    style_num_heads = int(reading_cfg.get("style_num_heads", 8))
    if sat_embed_dim % style_num_heads != 0:
        if sat_embed_dim % 64 == 0:
            style_num_heads = sat_embed_dim // 64
        else:
            style_num_heads = 1
            for candidate in range(min(style_num_heads, sat_embed_dim), 0, -1):
                if sat_embed_dim % candidate == 0:
                    style_num_heads = candidate
                    break
    satellite_style_adapter = SatelliteStyleAdapter(
        sat_in_dim=sat_embed_dim,
        out_dim=cross_attention_dim,
        num_tokens=style_num_tokens,
        num_heads=style_num_heads,
        scale=float(reading_cfg.get("style_scale", 0.5)),
    )

    # Replace UNet with satellite-conditioned version
    pipeline.unet = SatelliteConditionedUNet(
        reading_block_config={
            'sat_in_dim': sat_embed_dim,
            'plucker_dropout_prob': reading_cfg.get('plucker_dropout_prob', 0.3),
            'style_num_tokens': reading_cfg.get('style_num_tokens', 4),
            'style_num_heads': reading_cfg.get('style_num_heads', 8),
            'style_scale': reading_cfg.get('style_scale', 0.5),
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
            'save_attention_heatmap': reading_cfg.get('save_attention_heatmap', True),
            'heatmap_max_tokens': reading_cfg.get('heatmap_max_tokens', 16),
        },
        **base_unet.config,
    )

    # Load weights from base model
    pipeline.unet.load_state_dict(base_unet.state_dict(), strict=False)

    if freeze_base:
        # Freeze most layers of the base model
        for param in pipeline.vae.parameters():
            param.requires_grad = False
        for param in pipeline.text_encoder.parameters():
            param.requires_grad = False
        for param in pipeline.unet.parameters():
            param.requires_grad = False

        # Unfreeze top layers
        for param in pipeline.unet.up_blocks[-1].parameters():
            param.requires_grad = True
        for param in pipeline.unet.mid_block.parameters():
            param.requires_grad = True

    return SatelliteConditionedSDPipeline(
        satellite_encoder=satellite_encoder,
        satellite_style_adapter=satellite_style_adapter,
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
        safety_checker=pipeline.safety_checker,
        feature_extractor=pipeline.feature_extractor,
        image_encoder=None,
        requires_safety_checker=pipeline.requires_safety_checker,
    )
