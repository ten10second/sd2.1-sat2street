"""
Stable Diffusion Model for Satellite-to-Frontview Generation.

Custom Stable Diffusion implementation with satellite condition encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from models.unet.satellite_reading_block import SatelliteReadingBlock


class SatelliteConditionedUNet(UNet2DConditionModel):
    """
    UNet2DConditionModel with satellite condition support.

    Args:
        use_satellite_reading: Whether to use satellite reading blocks
        reading_injection_sites: Which U-Net layers to inject reading blocks into
        reading_block_config: Configuration for reading blocks
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
        self.reading_injection_sites = reading_injection_sites or ["down2", "mid"]
        sat_in_dim = int(self.config.cross_attention_dim or 768)
        self.reading_block_config = {
            "num_heads": 8,
            "head_dim": 64,
            "geo_ratio": 0.5,
            "rope_base": 10000.0,
            "lambda_geo": 1.0,
            "gate_hidden_ratio": 0.25,
            "use_geom_bias": True,
            "use_gated_residual": True,
        }
        if reading_block_config is not None:
            self.reading_block_config.update(reading_block_config)

        self.reading_blocks = nn.ModuleDict()
        self._conditioning_context: Dict[str, Any] = {}
        self._reading_hook_handles = []
        self._attn2_sat_hook_handles = []
        self.last_attn_maps: Dict[str, torch.Tensor] = {}

        self._register_attn2_sat_hooks(sat_in_dim=sat_in_dim)
        if self.use_satellite_reading:
            self._register_reading_hooks()

    def _register_reading_hooks(self):
        for site in self.reading_injection_sites:
            module = self._resolve_injection_module(site)
            if module is None:
                continue
            handle = module.register_forward_hook(self._make_reading_hook(site))
            self._reading_hook_handles.append(handle)

    def _register_attn2_sat_hooks(self, sat_in_dim: int):
        for name, module in list(self.named_modules()):
            if not name.endswith("attn2"):
                continue
            cross_attention_dim = getattr(module, "cross_attention_dim", None)
            if cross_attention_dim is None or int(cross_attention_dim) != sat_in_dim:
                raise ValueError(
                    f"attn2 module {name} has cross_attention_dim={cross_attention_dim}, "
                    f"expected {sat_in_dim} to match satellite token dim"
                )
            handle = module.register_forward_pre_hook(
                self._make_attn2_sat_hook(name),
                with_kwargs=True,
            )
            self._attn2_sat_hook_handles.append(handle)

    def _resolve_injection_module(self, site: str):
        if site == "mid":
            return self.mid_block
        if site.startswith("down"):
            try:
                index = int(site[4:])
            except ValueError:
                return None
            if 0 <= index < len(self.down_blocks):
                return self.down_blocks[index]
        if site.startswith("up"):
            try:
                index = int(site[2:])
            except ValueError:
                return None
            if 0 <= index < len(self.up_blocks):
                return self.up_blocks[index]
        return None

    def _get_or_create_reading_block(
        self,
        site: str,
        front_feat: torch.Tensor,
        sat_tokens: torch.Tensor,
    ) -> SatelliteReadingBlock:
        if site in self.reading_blocks:
            block = self.reading_blocks[site]
            block_param = next(block.parameters(), None)
            if block_param is not None and block_param.device != front_feat.device:
                block = block.to(device=front_feat.device)
                self.reading_blocks[site] = block
            return block

        block = SatelliteReadingBlock(
            front_dim=front_feat.shape[1],
            sat_in_dim=sat_tokens.shape[-1],
            num_heads=self.reading_block_config["num_heads"],
            head_dim=self.reading_block_config["head_dim"],
            geo_ratio=self.reading_block_config["geo_ratio"],
            rope_base=self.reading_block_config["rope_base"],
            lambda_geo=self.reading_block_config["lambda_geo"],
            gate_hidden_ratio=self.reading_block_config["gate_hidden_ratio"],
            use_geom_bias=self.reading_block_config["use_geom_bias"],
            use_gated_residual=self.reading_block_config["use_gated_residual"],
        )
        block = block.to(device=front_feat.device)
        self.reading_blocks[site] = block
        return block

    def _prepare_front_bev_xy(
        self,
        front_bev_xy: Any,
        site: str,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if front_bev_xy is None:
            return None

        if isinstance(front_bev_xy, dict):
            site_xy = front_bev_xy.get(site)
            if site_xy is None:
                site_xy = front_bev_xy.get("default")
            return self._prepare_front_bev_xy(site_xy, site, height, width, device, dtype)

        if not torch.is_tensor(front_bev_xy):
            return None

        front_bev_xy = front_bev_xy.to(device=device, dtype=dtype)

        if front_bev_xy.ndim == 3 and front_bev_xy.shape[-1] == 2:
            if front_bev_xy.shape[1] == height * width:
                return front_bev_xy
            return None

        if front_bev_xy.ndim == 4 and front_bev_xy.shape[1] == 2:
            resized = F.interpolate(front_bev_xy, size=(height, width), mode="bilinear", align_corners=False)
            return resized.permute(0, 2, 3, 1).reshape(front_bev_xy.shape[0], height * width, 2)

        if front_bev_xy.ndim == 4 and front_bev_xy.shape[-1] == 2:
            xy = front_bev_xy.permute(0, 3, 1, 2)
            resized = F.interpolate(xy, size=(height, width), mode="bilinear", align_corners=False)
            return resized.permute(0, 2, 3, 1).reshape(front_bev_xy.shape[0], height * width, 2)

        return None

    def _prepare_front_plucker(
        self,
        front_plucker: Any,
        site: str,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if front_plucker is None:
            return None

        if isinstance(front_plucker, dict):
            site_plucker = front_plucker.get(site)
            if site_plucker is None:
                site_plucker = front_plucker.get("default")
            return self._prepare_front_plucker(site_plucker, site, height, width, device, dtype)

        if not torch.is_tensor(front_plucker):
            return None

        front_plucker = front_plucker.to(device=device, dtype=dtype)

        if front_plucker.ndim == 3 and front_plucker.shape[-1] == 6:
            if front_plucker.shape[1] == height * width:
                return front_plucker
            return None

        if front_plucker.ndim == 4 and front_plucker.shape[1] == 6:
            resized = F.interpolate(front_plucker, size=(height, width), mode="bilinear", align_corners=False)
            return resized.permute(0, 2, 3, 1).reshape(front_plucker.shape[0], height * width, 6)

        if front_plucker.ndim == 4 and front_plucker.shape[-1] == 6:
            plucker = front_plucker.permute(0, 3, 1, 2)
            resized = F.interpolate(plucker, size=(height, width), mode="bilinear", align_corners=False)
            return resized.permute(0, 2, 3, 1).reshape(front_plucker.shape[0], height * width, 6)

        return None

    @staticmethod
    def _feature_condition_mask(condition_mask: torch.Tensor, front_feat: torch.Tensor) -> torch.Tensor:
        if condition_mask.ndim != 1 or condition_mask.shape[0] != front_feat.shape[0]:
            raise ValueError(
                f"condition_mask must be [B], got {list(condition_mask.shape)} for batch {front_feat.shape[0]}"
            )
        return condition_mask.to(device=front_feat.device, dtype=front_feat.dtype).view(-1, 1, 1, 1)

    @staticmethod
    def _token_condition_mask(condition_mask: torch.Tensor, token_states: torch.Tensor) -> torch.Tensor:
        if condition_mask.ndim != 1 or condition_mask.shape[0] != token_states.shape[0]:
            raise ValueError(
                f"condition_mask must be [B], got {list(condition_mask.shape)} for batch {token_states.shape[0]}"
            )
        return condition_mask.to(device=token_states.device, dtype=token_states.dtype).view(-1, 1, 1)

    def _make_attn2_sat_hook(self, module_name: str):
        def hook(module, args, kwargs):
            if not self._conditioning_context:
                return args, kwargs

            sat_tokens = self._conditioning_context.get("sat_tokens")
            condition_mask = self._conditioning_context.get("condition_mask")
            if sat_tokens is None:
                return args, kwargs

            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if not torch.is_tensor(hidden_states):
                return args, kwargs

            sat_encoder_hidden_states = sat_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if condition_mask is not None:
                sat_encoder_hidden_states = (
                    sat_encoder_hidden_states
                    * self._token_condition_mask(condition_mask, sat_encoder_hidden_states)
                )

            kwargs = dict(kwargs)
            kwargs["encoder_hidden_states"] = sat_encoder_hidden_states
            return args, kwargs

        return hook

    def _make_reading_hook(self, site: str):
        def hook(_module, _inputs, output):
            if not self._conditioning_context:
                return output

            if isinstance(output, tuple):
                if not output or not torch.is_tensor(output[0]):
                    return output
                front_feat = output[0]
                output_tail = output[1:]
                is_tuple = True
            elif torch.is_tensor(output):
                front_feat = output
                output_tail = ()
                is_tuple = False
            else:
                return output

            sat_tokens = self._conditioning_context.get("sat_tokens")
            sat_xy = self._conditioning_context.get("sat_xy")
            raw_front_bev_xy = self._conditioning_context.get("front_bev_xy")
            raw_front_plucker = self._conditioning_context.get("front_plucker")
            condition_mask = self._conditioning_context.get("condition_mask")
            if sat_tokens is None or sat_xy is None or raw_front_bev_xy is None:
                return output

            front_bev_xy = self._prepare_front_bev_xy(
                raw_front_bev_xy,
                site,
                front_feat.shape[2],
                front_feat.shape[3],
                front_feat.device,
                front_feat.dtype,
            )
            if front_bev_xy is None:
                return output
            front_plucker = self._prepare_front_plucker(
                raw_front_plucker,
                site,
                front_feat.shape[2],
                front_feat.shape[3],
                front_feat.device,
                front_feat.dtype,
            )

            block = self._get_or_create_reading_block(site, front_feat, sat_tokens)
            block_output = block(
                front_feat=front_feat,
                sat_tokens=sat_tokens.to(device=front_feat.device, dtype=front_feat.dtype),
                sat_xy=sat_xy.to(device=front_feat.device, dtype=front_feat.dtype),
                front_bev_xy=front_bev_xy,
                front_plucker=front_plucker,
                return_attn_map=self._conditioning_context.get("return_attn_map", False),
            )

            updated = block_output["front_feat_out"]
            if condition_mask is not None:
                feature_mask = self._feature_condition_mask(condition_mask, front_feat)
                updated = feature_mask * updated + (1.0 - feature_mask) * front_feat
            if self._conditioning_context.get("return_attn_map", False):
                attn_map = block_output.get("attn_map")
                if attn_map is not None:
                    self._conditioning_context.setdefault("attn_maps", {})[site] = attn_map.detach()

            if is_tuple:
                return (updated, *output_tail)
            return updated

        return hook

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        sat_tokens: Optional[torch.Tensor] = None,
        sat_xy: Optional[torch.Tensor] = None,
        front_bev_xy: Optional[torch.Tensor] = None,
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
            encoder_hidden_states: Optional legacy argument, ignored by attn2 hooks
            sat_tokens: (B, Ns, Cs) - Pre-encoded satellite tokens
            sat_xy: (B, Ns, 2) - Satellite token BEV coordinates
            front_bev_xy: (B, Nf, 2) - Frontview pixel BEV coordinates
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
                "front_plucker": front_plucker,
                "condition_mask": inferred_condition_mask,
                "return_attn_map": return_attn_map,
                "attn_maps": {},
            }
        else:
            self._conditioning_context = {}

        self.last_attn_maps = {}

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

    # Create satellite encoder (separate from UNet)
    satellite_encoder = SatelliteConditionEncoder()

    # Replace UNet with satellite-conditioned version
    pipeline.unet = SatelliteConditionedUNet(
        use_satellite_reading=use_satellite_reading,
        reading_injection_sites=reading_injection_sites,
        reading_block_config=reading_block_config,
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
