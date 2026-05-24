"""Satellite-conditioned Stable Diffusion model wrappers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler

try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
except ImportError:  # pragma: no cover - compatibility with older diffusers layouts
    from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

from models.conditioning import SatelliteMemoryState
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0, QueryUVSlicedAttnProcessor


logger = logging.getLogger(__name__)


def _resolve_module_path(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


class SatelliteConditionedUNet(UNet2DConditionModel):
    """Thin UNet wrapper that routes satellite tokens into cross-attention."""

    _logged_diag: bool = False

    def __init__(
        self,
        query_uv_pe_enabled: bool = False,
        query_geometry_bias_enabled: bool = False,
        query_geometry_bias_scale: float = 2.0,
        query_geometry_invalid_penalty: float = -1e4,
        query_uv_gate_init: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_uv_pe_enabled = bool(query_uv_pe_enabled)
        self.query_geometry_bias_enabled = bool(query_geometry_bias_enabled)
        self.query_geometry_bias_scale = float(query_geometry_bias_scale)
        self.query_geometry_invalid_penalty = float(query_geometry_invalid_penalty)
        self.query_uv_gate_init = float(query_uv_gate_init)
        self._install_query_uv_attention_processors()

    def _build_attention_processors(self):
        if not self.query_uv_pe_enabled and not self.query_geometry_bias_enabled:
            return AttnProcessor2_0()
        return self._build_query_uv_attention_processors()

    def _build_query_uv_attention_processors(self):
        processors = {}
        for name in self.attn_processors.keys():
            attn_module = _resolve_module_path(self, name.removesuffix(".processor"))
            query_dim = int(attn_module.to_q.out_features)
            processors[name] = QueryUVAttnProcessor2_0(
                query_dim=query_dim,
                query_uv_enabled=bool(self.query_uv_pe_enabled and name.endswith(".attn2.processor")),
                geometry_bias_enabled=bool(self.query_geometry_bias_enabled and name.endswith(".attn2.processor")),
                geometry_bias_scale=self.query_geometry_bias_scale,
                geometry_invalid_penalty=self.query_geometry_invalid_penalty,
                gate_init=self.query_uv_gate_init,
            )
        return processors

    def _install_query_uv_attention_processors(self) -> None:
        self.set_attn_processor(self._build_attention_processors())

    def set_attention_slice(self, slice_size="auto"):
        if not self.query_uv_pe_enabled and not self.query_geometry_bias_enabled:
            return super().set_attention_slice(slice_size)

        if slice_size is None:
            self._install_query_uv_attention_processors()
            return

        super().set_attention_slice(slice_size)
        sliced_processors = {}
        for name, processor in self.attn_processors.items():
            slice_value = getattr(processor, "slice_size", None)
            if slice_value is None:
                raise ValueError(
                    f"Unable to preserve query UV attention slicing for {name}: "
                    f"{type(processor).__name__} has no slice_size"
                )
            attn_module = _resolve_module_path(self, name.removesuffix(".processor"))
            query_dim = int(attn_module.to_q.out_features)
            sliced_processors[name] = QueryUVSlicedAttnProcessor(
                query_dim=query_dim,
                slice_size=int(slice_value),
                query_uv_enabled=bool(self.query_uv_pe_enabled and name.endswith(".attn2.processor")),
                geometry_bias_enabled=bool(self.query_geometry_bias_enabled and name.endswith(".attn2.processor")),
                geometry_bias_scale=self.query_geometry_bias_scale,
                geometry_invalid_penalty=self.query_geometry_invalid_penalty,
                gate_init=self.query_uv_gate_init,
            )
        self.set_attn_processor(sliced_processors)

    @staticmethod
    def _normalize_query_base_hw(cross_attention_kwargs):
        if cross_attention_kwargs is None:
            return None
        if not isinstance(cross_attention_kwargs, dict):
            raise ValueError(
                f"cross_attention_kwargs must be a dict when query-based geometry features are enabled, "
                f"got {type(cross_attention_kwargs)!r}"
            )
        query_base_hw = cross_attention_kwargs.get("query_base_hw")
        if query_base_hw is None:
            return None
        if isinstance(query_base_hw, torch.Tensor):
            if query_base_hw.numel() != 2:
                raise ValueError(f"query_base_hw tensor must contain 2 values, got {tuple(query_base_hw.shape)}")
            query_base_hw = tuple(int(x) for x in query_base_hw.reshape(-1).tolist())
        elif hasattr(query_base_hw, "__len__") and len(query_base_hw) != 2:
            raise ValueError(f"query_base_hw must contain 2 values, got {query_base_hw}")
        return tuple(int(x) for x in query_base_hw)

    def forward(self, *args, sat_tokens: Optional[torch.Tensor] = None, **kwargs):
        encoder_hidden_states = kwargs.pop("encoder_hidden_states", None)
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
        if sat_tokens is not None:
            encoder_hidden_states = sat_tokens
        if encoder_hidden_states is None:
            raise ValueError("SatelliteConditionedUNet requires sat_tokens or encoder_hidden_states")
        if self.query_uv_pe_enabled or self.query_geometry_bias_enabled:
            if self._normalize_query_base_hw(cross_attention_kwargs) is None:
                raise ValueError(
                    "query-based geometry features require cross_attention_kwargs['query_base_hw'] "
                    "to thread latent spatial coordinates into cross-attention"
                )
        if self.query_geometry_bias_enabled:
            if not isinstance(cross_attention_kwargs, dict) or cross_attention_kwargs.get("sat_perspective_uv") is None:
                raise ValueError(
                    "query_geometry_bias_enabled requires cross_attention_kwargs['sat_perspective_uv']"
                )

        # Diagnostic log (once per process)
        if not SatelliteConditionedUNet._logged_diag:
            _logger = logging.getLogger(__name__)
            _logger.info(
                "[SatelliteConditionedUNet] encoder_hidden_states: "
                "shape=%s mean=%.4f std=%.4f nonzero_frac=%.3f",
                tuple(encoder_hidden_states.shape),
                encoder_hidden_states.float().mean().item(),
                encoder_hidden_states.float().std().item(),
                (encoder_hidden_states.abs() > 1e-6).float().mean().item(),
            )
            SatelliteConditionedUNet._logged_diag = True

        return super().forward(
            *args,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )


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


class SatelliteConditionedSDModel(nn.Module):
    """Stable Diffusion model conditioned by satellite tokens with perspective PE."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        noise_scheduler: DDPMScheduler,
        satellite_encoder: Optional[SatelliteConditionEncoder] = None,
        freeze_base: bool = True,
        cond_drop_prob: float = 0.1,
        perspective_pe_enabled: bool = True,
    ):
        super().__init__()

        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.cond_drop_prob = float(cond_drop_prob)
        self.perspective_pe_enabled = bool(perspective_pe_enabled)

        if satellite_encoder is None:
            sat_embed_dim = int(getattr(unet.config, "cross_attention_dim", 768) or 768)
            satellite_encoder = SatelliteConditionEncoder(
                embed_dim=sat_embed_dim,
                perspective_pe_enabled=self.perspective_pe_enabled,
            )
        self.satellite_encoder = satellite_encoder

        if freeze_base:
            for param in self.vae.parameters():
                param.requires_grad = False

            for name, param in self.unet.named_parameters():
                if (
                    ".attn2.to_k." in name
                    or ".attn2.to_v." in name
                    or ".attn2.processor." in name
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for param in self.satellite_encoder.parameters():
            param.requires_grad = True

        logger.info("[SatelliteConditionedSDModel] Initialized")
        logger.info(f"  UNet trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
        logger.info(f"  Satellite encoder params: {sum(p.numel() for p in self.satellite_encoder.parameters())}")
        logger.info(
            f"  Trainable attn2 k/v params: "
            f"{sum(p.numel() for n, p in self.unet.named_parameters() if p.requires_grad and ('.attn2.to_k.' in n or '.attn2.to_v.' in n))}"
        )
        logger.info(
            f"  Trainable attn2 processor params: "
            f"{sum(p.numel() for n, p in self.unet.named_parameters() if p.requires_grad and '.attn2.processor.' in n)}"
        )
        logger.info(f"  Perspective PE enabled: {self.perspective_pe_enabled}")
        logger.info(f"  Query UV PE enabled: {bool(getattr(self.unet, 'query_uv_pe_enabled', False))}")
        logger.info(f"  Query geometry bias enabled: {bool(getattr(self.unet, 'query_geometry_bias_enabled', False))}")
        logger.info(f"  Condition dropout: {self.cond_drop_prob}")

    def encode_satellite(
        self,
        sat_images: torch.Tensor,
        *,
        K: Optional[torch.Tensor] = None,
        T_cam_to_world: Optional[torch.Tensor] = None,
        T_imu_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> SatelliteMemoryState:
        """Encode satellite images into a structured memory state."""
        encoder_param = next(self.satellite_encoder.parameters(), None)
        if encoder_param is not None:
            sat_images = sat_images.to(dtype=encoder_param.dtype)
        return self.satellite_encoder(
            sat_images,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            image_size=image_size,
        )

    @staticmethod
    def _normalize_images_for_vae(images: torch.Tensor) -> torch.Tensor:
        return images * 2.0 - 1.0

    @staticmethod
    def _expand_condition_mask(condition_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        mask = condition_mask.to(device=reference.device, dtype=reference.dtype)
        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)
        return mask

    def _sample_condition_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
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
            perspective_uv=(
                sat_state.perspective_uv * self._expand_condition_mask(condition_mask, sat_state.perspective_uv)
                if sat_state.perspective_uv is not None
                else None
            ),
            perspective_valid=(
                sat_state.perspective_valid * condition_mask.view(-1, 1)
                if sat_state.perspective_valid is not None
                else None
            ),
        )

    def _build_cross_attention_kwargs(
        self,
        reference: torch.Tensor,
        sat_state: SatelliteMemoryState,
    ):
        if not bool(getattr(self.unet, "query_uv_pe_enabled", False)) and not bool(
            getattr(self.unet, "query_geometry_bias_enabled", False)
        ):
            return None
        if reference.ndim != 4:
            raise ValueError(f"reference tensor must be [B,C,H,W], got {list(reference.shape)}")
        kwargs: Dict[str, Any] = {"query_base_hw": tuple(int(x) for x in reference.shape[-2:])}
        if bool(getattr(self.unet, "query_geometry_bias_enabled", False)):
            if sat_state.perspective_uv is None:
                raise ValueError("query_geometry_bias_enabled requires sat_state.perspective_uv")
            kwargs["sat_perspective_uv"] = sat_state.perspective_uv
            if sat_state.perspective_valid is not None:
                kwargs["sat_perspective_valid"] = sat_state.perspective_valid
        return kwargs

    @staticmethod
    def _zero_sat_geometry(sat_state: SatelliteMemoryState) -> SatelliteMemoryState:
        return sat_state.replace(
            perspective_uv=(
                torch.zeros_like(sat_state.perspective_uv)
                if sat_state.perspective_uv is not None
                else None
            ),
            perspective_valid=(
                torch.zeros_like(sat_state.perspective_valid)
                if sat_state.perspective_valid is not None
                else None
            ),
        )

    @staticmethod
    def _concat_optional_tensors(left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if left is None or right is None:
            return None
        return torch.cat([left, right], dim=0)

    def _build_cfg_sat_state(self, sat_state: SatelliteMemoryState) -> SatelliteMemoryState:
        zero_tokens = torch.zeros_like(sat_state.tokens)
        zero_geometry = self._zero_sat_geometry(sat_state)
        return SatelliteMemoryState(
            tokens=torch.cat([sat_state.tokens, zero_tokens], dim=0),
            xy=torch.cat([sat_state.xy, sat_state.xy], dim=0),
            bev_coords=self._concat_optional_tensors(sat_state.bev_coords, sat_state.bev_coords),
            perspective_uv=self._concat_optional_tensors(sat_state.perspective_uv, zero_geometry.perspective_uv),
            perspective_valid=self._concat_optional_tensors(sat_state.perspective_valid, zero_geometry.perspective_valid),
        )

    @torch.no_grad()
    def _get_vae_scale_factor(self) -> int:
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels is None:
            return 8
        return max(1, 2 ** (len(block_out_channels) - 1))

    @staticmethod
    def _infer_generation_size(
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if target_size is None:
            raise ValueError("target_size is required when satellite conditioning is used")
        return int(target_size[0]), int(target_size[1])

    @torch.no_grad()
    def generate_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        target_size: Optional[Tuple[int, int]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sat_condition_mode: str = "normal",
    ) -> Tuple[torch.Tensor, SatelliteMemoryState]:
        """Generate images from a precomputed satellite memory state."""
        B = sat_state.tokens.shape[0]
        device = sat_state.tokens.device

        if sat_condition_mode == "normal":
            condition_mask = torch.ones(B, device=device, dtype=torch.bool)
        elif sat_condition_mode == "zero":
            sat_state = sat_state.replace(
                tokens=torch.zeros_like(sat_state.tokens),
                perspective_uv=(
                    torch.zeros_like(sat_state.perspective_uv)
                    if sat_state.perspective_uv is not None
                    else None
                ),
                perspective_valid=(
                    torch.zeros_like(sat_state.perspective_valid)
                    if sat_state.perspective_valid is not None
                    else None
                ),
            )
            condition_mask = torch.zeros(B, device=device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown sat_condition_mode: {sat_condition_mode}")

        image_h, image_w = self._infer_generation_size(target_size=target_size)
        vae_scale_factor = self._get_vae_scale_factor()
        latent_h = max(1, (image_h + vae_scale_factor - 1) // vae_scale_factor)
        latent_w = max(1, (image_w + vae_scale_factor - 1) // vae_scale_factor)
        unet_param = next(self.unet.parameters(), None)
        latent_dtype = unet_param.dtype if unet_param is not None else sat_state.tokens.dtype

        latents = torch.randn(
            (B, self.unet.config.in_channels, latent_h, latent_w),
            device=device,
            dtype=latent_dtype,
            generator=generator,
        )

        use_cfg = guidance_scale > 1.0
        if use_cfg:
            sat_state_double = self._build_cfg_sat_state(sat_state)

        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            cross_attention_kwargs = self._build_cross_attention_kwargs(
                latents,
                sat_state_double if use_cfg else sat_state,
            )
            if use_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                noise_pred_both = self.unet(
                    latent_model_input,
                    t,
                    sat_tokens=sat_state_double.tokens,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                noise_pred_cond, noise_pred_uncond = noise_pred_both.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(
                    latents,
                    t,
                    sat_tokens=sat_state.tokens,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            if generator is not None:
                try:
                    latents = self.noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
                except TypeError:
                    latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            else:
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        vae_param = next(self.vae.parameters(), None)
        vae_dtype = vae_param.dtype if vae_param is not None else latents.dtype
        latents = (latents / self.vae.config.scaling_factor).to(dtype=vae_dtype)
        generated_images = self.vae.decode(latents).sample
        generated_images = (generated_images / 2 + 0.5).clamp(0, 1)

        if generated_images.shape[-2:] != (image_h, image_w):
            generated_images = F.interpolate(
                generated_images,
                size=(image_h, image_w),
                mode="bilinear",
                align_corners=False,
            )

        return generated_images, sat_state

    @torch.no_grad()
    def generate(
        self,
        sat_images: torch.Tensor,
        *,
        K: Optional[torch.Tensor] = None,
        T_cam_to_world: Optional[torch.Tensor] = None,
        T_imu_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        sat_condition_mode: str = "normal",
    ) -> torch.Tensor:
        sat_state = self.encode_satellite(
            sat_images,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            image_size=target_size,
        )
        generated_images, _ = self.generate_with_satellite_state(
            sat_state,
            target_size=target_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            sat_condition_mode=sat_condition_mode,
        )
        return generated_images

    def forward(
        self,
        sat_images: torch.Tensor,
        target_images: torch.Tensor,
        *,
        K: Optional[torch.Tensor] = None,
        T_cam_to_world: Optional[torch.Tensor] = None,
        T_imu_to_world: Optional[torch.Tensor] = None,
        camera_height_m: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if target_images.ndim != 4:
            raise ValueError(
                "This experiment trains one random-yaw street view per sample; "
                f"target_images must be [B,C,H,W], got {list(target_images.shape)}"
            )

        sat_state = self.encode_satellite(
            sat_images,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            image_size=tuple(int(x) for x in target_images.shape[-2:]),
        )

        batch_size = target_images.shape[0]
        condition_mask = self._sample_condition_mask(batch_size=batch_size, device=target_images.device)
        conditioned_sat_state = self._apply_condition_dropout(sat_state, condition_mask)

        with torch.no_grad():
            target_images_vae = self._normalize_images_for_vae(target_images)
            latents = self.vae.encode(target_images_vae).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=target_images.device,
            dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        cross_attention_kwargs = self._build_cross_attention_kwargs(noisy_latents, conditioned_sat_state)
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            sat_tokens=conditioned_sat_state.tokens,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred, target, reduction="mean")
        return {
            "loss": loss,
            "model_pred": model_pred,
            "target": target,
            "sat_state": conditioned_sat_state,
            "condition_mask": condition_mask,
        }


def create_sd_model(
    base_model: str = "stabilityai/stable-diffusion-2-1-base",
    freeze_base: bool = True,
    revision: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cond_drop_prob: float = 0.1,
    perspective_pe_enabled: bool = True,
    query_uv_pe_enabled: bool = False,
    query_geometry_bias_enabled: bool = False,
    query_geometry_bias_scale: float = 2.0,
    query_geometry_invalid_penalty: float = -1e4,
    query_uv_gate_init: float = 0.0,
    satellite_encoder_config: Optional[Dict[str, Any]] = None,
) -> SatelliteConditionedSDModel:
    """Create a satellite-conditioned Stable Diffusion model."""
    resolved_base_model = _resolve_hf_snapshot_path(base_model, revision=revision)
    load_source = str(resolved_base_model) if resolved_base_model is not None else base_model
    if resolved_base_model is not None:
        logger.info(f"Using cached base model snapshot: {resolved_base_model}")

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

    sat_encoder_cfg = dict(satellite_encoder_config or {})
    sat_encoder_cfg.pop("name", None)
    sat_encoder_cfg.setdefault("perspective_pe_enabled", perspective_pe_enabled)
    sat_embed_dim = int(getattr(base_unet.config, "cross_attention_dim", 768) or 768)
    requested_embed_dim = sat_encoder_cfg.get("embed_dim")
    if requested_embed_dim is not None and int(requested_embed_dim) != sat_embed_dim:
        logger.warning(
            "Overriding satellite_encoder.embed_dim=%s with base UNet cross_attention_dim=%s",
            requested_embed_dim,
            sat_embed_dim,
        )
    sat_encoder_cfg["embed_dim"] = sat_embed_dim
    satellite_encoder = SatelliteConditionEncoder(**sat_encoder_cfg)

    unet = SatelliteConditionedUNet(
        query_uv_pe_enabled=query_uv_pe_enabled,
        query_geometry_bias_enabled=query_geometry_bias_enabled,
        query_geometry_bias_scale=query_geometry_bias_scale,
        query_geometry_invalid_penalty=query_geometry_invalid_penalty,
        query_uv_gate_init=query_uv_gate_init,
        **base_unet.config,
    )
    unet.load_state_dict(base_unet.state_dict(), strict=False)
    if torch_dtype is not None:
        unet = unet.to(dtype=torch_dtype)

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
        perspective_pe_enabled=perspective_pe_enabled,
    )
    return model
