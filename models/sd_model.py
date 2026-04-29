"""
Stable Diffusion Model for Satellite-to-Frontview Generation.

Custom Stable Diffusion implementation with satellite condition encoder.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from diffusers import StableDiffusionPipeline
try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
except ImportError:
    from diffusers.models.unet_2d_condition import UNet2DConditionModel

try:
    from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
except ImportError:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from models.unet.georope_satellite_attn_processor import GeoRoPESatelliteAttnProcessor

DEFAULT_SATELLITE_EMBED_DIM = 768


class SatelliteConditionedUNet(UNet2DConditionModel):
    """
    UNet2DConditionModel with satellite condition support.

    Args:
        use_satellite_reading: Whether to use GeoRoPE satellite attn2 processors
        reading_injection_sites: Which U-Net blocks receive GeoRoPE satellite attn2 processors
        reading_block_config: Configuration for GeoRoPE satellite attn2 processors
        **kwargs: Additional arguments for UNet2DConditionModel
    """

    def __init__(
        self,
        use_satellite_reading: bool = True,
        reading_injection_sites: Optional[List[str]] = None,
        reading_block_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.supports_satellite_reading = True
        self.use_satellite_reading = use_satellite_reading
        self.reading_injection_sites = reading_injection_sites or ["down2", "down3", "mid", "up0", "up1"]
        self.reading_block_config = {
            "sat_in_dim": DEFAULT_SATELLITE_EMBED_DIM,
            "geo_ratio": 1.0,
            "rope_base": 10000.0,
            "invalid_conf_loss_weight": 0.05,
            "plucker_dropout_prob": 0.3,
        }
        if reading_block_config is not None:
            self.reading_block_config.update(reading_block_config)

        sat_in_dim = int(self.reading_block_config["sat_in_dim"])
        self.plucker_dropout_prob = float(self.reading_block_config.get("plucker_dropout_prob", 0.0))
        self._conditioning_context: Dict[str, Any] = {}
        self.georope_attn_processor_names: List[str] = []
        self.last_attn_maps: Dict[str, torch.Tensor] = {}
        self.last_reading_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.last_reading_losses: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.use_satellite_reading:
            self._register_georope_satellite_processors(sat_in_dim=sat_in_dim)

    def _get_conditioning_context(self) -> Dict[str, Any]:
        return self._conditioning_context

    @staticmethod
    def _processor_matches_site(processor_name: str, site: str) -> bool:
        if site == "mid":
            return processor_name.startswith("mid_block.") and ".attn2.processor" in processor_name
        if site.startswith("down"):
            return processor_name.startswith(f"down_blocks.{site[4:]}.") and ".attn2.processor" in processor_name
        if site.startswith("up"):
            return processor_name.startswith(f"up_blocks.{site[2:]}.") and ".attn2.processor" in processor_name
        return False

    def _register_georope_satellite_processors(self, sat_in_dim: int) -> None:
        processors = dict(self.attn_processors)
        module_by_processor_name = {
            f"{name}.processor": module
            for name, module in self.named_modules()
            if hasattr(module, "set_processor") and name
        }
        selected_sites = tuple(self.reading_injection_sites)
        for processor_name in list(processors.keys()):
            if not any(self._processor_matches_site(processor_name, site) for site in selected_sites):
                continue
            if ".attn2.processor" not in processor_name:
                continue
            attn_module = module_by_processor_name.get(processor_name)
            if attn_module is None:
                continue
            hidden_size = int(getattr(attn_module, "inner_dim", 0))
            num_heads = int(getattr(attn_module, "heads", 0))
            if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
                raise ValueError(f"Cannot infer attention dimensions for {processor_name}")
            processors[processor_name] = GeoRoPESatelliteAttnProcessor(
                name=processor_name.replace(".processor", ""),
                hidden_size=hidden_size,
                sat_in_dim=sat_in_dim,
                num_heads=num_heads,
                head_dim=hidden_size // num_heads,
                context_provider=self._get_conditioning_context,
                geo_ratio=float(self.reading_block_config.get("geo_ratio", 1.0)),
                rope_base=float(self.reading_block_config.get("rope_base", 10000.0)),
                invalid_conf_loss_weight=float(self.reading_block_config.get("invalid_conf_loss_weight", 0.05)),
            )
            self.georope_attn_processor_names.append(processor_name)
        if not self.georope_attn_processor_names:
            raise ValueError(
                "No attn2 processors matched reading_injection_sites="
                f"{list(self.reading_injection_sites)}"
            )
        self.set_attn_processor(processors)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        sat_tokens: Optional[torch.Tensor] = None,
        sat_xy: Optional[torch.Tensor] = None,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_bev_valid_mask: Optional[torch.Tensor] = None,
        front_plucker: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        return_attn_map: bool = False,
        **kwargs,
    ):
        """
        Forward pass with satellite conditioning.

        Args:
            sample: (B, 4, H, W) - Noisy latent representation
            timestep: (B,) or (1,) - Current timestep
            encoder_hidden_states: Optional text/context states for the base U-Net cross-attention.
            sat_tokens: (B, Ns, Cs) - Pre-encoded satellite tokens
            sat_xy: (B, Ns, 2) - Satellite token BEV coordinates
            front_bev_xy: (B, Nf, 2) - Frontview pixel BEV coordinates
            front_bev_valid_mask: (B, 1, H, W) or (B, Nf) - Valid ground-projection mask
            front_plucker: (B, 6, H, W) or (B, Nf, 6) - Per-pixel Plucker ray map
            condition_mask: (B,) - Per-sample mask controlling whether reading injection is enabled
            return_attn_map: Whether to return attention maps

        Returns:
            noise_pred: (B, 4, H, W) - Predicted noise
        """
        inferred_sat_tokens = sat_tokens
        inferred_sat_xy = sat_xy
        inferred_condition_mask = condition_mask

        if inferred_sat_tokens is not None and inferred_sat_tokens.ndim != 3:
            raise ValueError("sat_tokens must be [B, Ns, Cs]")
        if inferred_sat_xy is not None and (inferred_sat_xy.ndim != 3 or inferred_sat_xy.shape[-1] != 2):
            raise ValueError("sat_xy must be [B, Ns, 2]")
        if inferred_condition_mask is not None:
            if inferred_condition_mask.ndim != 1 or inferred_condition_mask.shape[0] != sample.shape[0]:
                raise ValueError(
                    f"condition_mask must be [B], got {list(inferred_condition_mask.shape)} for batch {sample.shape[0]}"
                )
            inferred_condition_mask = inferred_condition_mask.to(device=sample.device, dtype=torch.bool)

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

        enable_sat_condition = inferred_sat_tokens is not None
        enable_reading = (
            self.use_satellite_reading
            and enable_sat_condition
            and inferred_sat_xy is not None
            and front_bev_xy is not None
        )

        if enable_sat_condition:
            self._conditioning_context = {
                "sat_tokens": inferred_sat_tokens,
                "sat_xy": inferred_sat_xy,
                "front_bev_xy": front_bev_xy,
                "front_bev_valid_mask": front_bev_valid_mask,
                "front_plucker": front_plucker,
                "condition_mask": inferred_condition_mask,
                "latent_hw": (int(sample.shape[-2]), int(sample.shape[-1])),
                "return_attn_map": return_attn_map,
                "attn_maps": {},
                "reading_stats": {},
                "reading_losses": {},
            }
        else:
            self._conditioning_context = {}

        self.last_attn_maps = {}
        self.last_reading_stats = {}
        self.last_reading_losses = {}

        try:
            output = super().forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
        except Exception:
            self._conditioning_context = {}
            raise

        if enable_reading:
            self.last_attn_maps = self._conditioning_context.get("attn_maps", {})
            self.last_reading_stats = self._conditioning_context.get("reading_stats", {})
            self.last_reading_losses = self._conditioning_context.get("reading_losses", {})

        # Keep the conditioning context alive after a successful forward so
        # gradient checkpointing can re-run hooked submodules during backward
        # with the same satellite inputs. The next forward overwrites it.
        return output


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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.satellite_encoder = satellite_encoder
        self.satellite_encoder.eval()

    def _encode_satellite_images(self, sat_images: torch.Tensor):
        """Encode satellite images to embeddings."""
        with torch.no_grad():
            sat_emb = self.satellite_encoder(sat_images)
        return sat_emb

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

        # Encode negative prompt
        negative_emb = None
        if negative_prompt is not None:
            negative_emb = self._encode_prompt(negative_prompt, sat_images.shape[0])

        # Encode positive prompt (we use empty prompt for now)
        positive_emb = self._encode_prompt("", sat_images.shape[0])

        # Concatenate embeddings
        encoder_hidden_states = torch.cat([positive_emb, sat_emb], dim=1)
        if negative_emb is not None:
            negative_emb = torch.cat([negative_emb, torch.zeros_like(sat_emb)], dim=1)

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
    use_satellite_reading: bool = True,
    reading_injection_sites: Optional[List[str]] = None,
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

    # Replace UNet with satellite-conditioned version
    pipeline.unet = SatelliteConditionedUNet(
        use_satellite_reading=use_satellite_reading,
        reading_injection_sites=reading_injection_sites,
        reading_block_config=reading_cfg,
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
