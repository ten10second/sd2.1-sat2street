"""
Satellite-conditioned UNet for satellite-to-frontview generation.
"""

import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from models.conditioning import CrossViewConditioningState, SatelliteMemoryState
try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
except ImportError:
    from diffusers.models.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers

from models.unet.geometry_masked_attention_processor import apply_geometry_masked_attn_processors


class SatelliteConditionedUNet(UNet2DConditionModel):
    """
    UNet2DConditionModel that routes satellite tokens through native cross-attention.
    """

    def __init__(
        self,
        native_cross_attention_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.supports_satellite_conditioning = True
        self.last_attn_maps: Dict[str, torch.Tensor] = {}
        self.last_refinement_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.last_satellite_state: Optional[SatelliteMemoryState] = None
        self._geometry_attention_context: Optional[Dict[str, Any]] = None
        self._attention_debug_config: Optional[Dict[str, Any]] = None

        self.native_cross_attention_config = {
            "enable_geometry_mask": False,
            "mask_sites": ["down1", "down2", "mid"],
            "mask_mode": "topk",
            "topk": 32,
            "mask_invalid_queries": True,
            "fallback_to_unmasked": True,
            "use_metric_coords": False,
            "enable_geometry_bias": False,
            "lambda_dist": 2.0,
            "lambda_dir": 0.5,
            "learnable_geometry_bias": False,
        }
        if native_cross_attention_config is not None:
            self.native_cross_attention_config.update(native_cross_attention_config)
        self.geometry_masked_attn_sites: List[str] = []
        if self.native_cross_attention_config.get("enable_geometry_mask", False):
            self._apply_native_geometry_mask_processors()

    def _get_geometry_attention_context(self) -> Optional[Dict[str, Any]]:
        return self._geometry_attention_context

    def enable_attention_debug(
        self,
        *,
        layers: Optional[List[str]] = None,
        storage: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if storage is None:
            storage = {}
        self._attention_debug_config = {
            "enabled": True,
            "layers": set(layers or []),
            "storage": storage,
        }
        return storage

    def disable_attention_debug(self) -> None:
        self._attention_debug_config = None

    def _apply_native_geometry_mask_processors(self) -> List[str]:
        self.geometry_masked_attn_sites = []
        if not self.native_cross_attention_config.get("enable_geometry_mask", False):
            return self.geometry_masked_attn_sites
        mask_mode = str(self.native_cross_attention_config.get("mask_mode", "topk")).lower()
        if mask_mode != "topk":
            raise ValueError(f"Only native_cross_attention.mask_mode='topk' is supported, got {mask_mode!r}")
        self.geometry_masked_attn_sites = apply_geometry_masked_attn_processors(
            self,
            sites=self.native_cross_attention_config.get("mask_sites", ["down1", "down2", "mid"]),
            context_provider=self._get_geometry_attention_context,
            topk=int(self.native_cross_attention_config.get("topk", 32)),
            mask_invalid_queries=bool(self.native_cross_attention_config.get("mask_invalid_queries", True)),
            fallback_to_unmasked=bool(self.native_cross_attention_config.get("fallback_to_unmasked", True)),
            use_metric_coords=bool(self.native_cross_attention_config.get("use_metric_coords", False)),
            enable_geometry_bias=bool(self.native_cross_attention_config.get("enable_geometry_bias", False)),
            geometry_bias_type=str(self.native_cross_attention_config.get("geometry_bias_type", "dist_dir")),
            lambda_dist=float(self.native_cross_attention_config.get("lambda_dist", 2.0)),
            lambda_dir=float(self.native_cross_attention_config.get("lambda_dir", 0.5)),
            learnable_geometry_bias=bool(self.native_cross_attention_config.get("learnable_geometry_bias", False)),
        )
        return self.geometry_masked_attn_sites

    def set_attention_slice(self, slice_size: Union[str, int, List[int], None]) -> None:
        super().set_attention_slice(slice_size)
        self._apply_native_geometry_mask_processors()

    @staticmethod
    def _token_condition_mask(condition_mask: torch.Tensor, token_states: torch.Tensor) -> torch.Tensor:
        if condition_mask.ndim != 1 or condition_mask.shape[0] != token_states.shape[0]:
            raise ValueError(
                f"condition_mask must be [B], got {list(condition_mask.shape)} for batch {token_states.shape[0]}"
            )
        return condition_mask.to(device=token_states.device, dtype=token_states.dtype).view(-1, 1, 1)

    @staticmethod
    def _move_satellite_state(
        satellite_state: SatelliteMemoryState,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SatelliteMemoryState:
        return SatelliteMemoryState(
            tokens=satellite_state.tokens.to(device=device, dtype=dtype),
            xy=satellite_state.xy.to(device=device, dtype=dtype),
            bev_coords=(
                satellite_state.bev_coords.to(device=device, dtype=dtype)
                if satellite_state.bev_coords is not None
                else None
            ),
        )

    def _apply_satellite_condition_mask(
        self,
        satellite_state: SatelliteMemoryState,
        condition_mask: Optional[torch.Tensor],
    ) -> SatelliteMemoryState:
        if condition_mask is None:
            return satellite_state

        token_mask = self._token_condition_mask(condition_mask, satellite_state.tokens)
        # Only tokens are cleared for CFG; xy and bev_coords stay intact
        # so geometric bias remains valid on the unconditioned branch.
        return satellite_state.replace(tokens=satellite_state.tokens * token_mask)

    def _build_conditioning_state(
        self,
        sat_tokens: Optional[torch.Tensor],
        sat_xy: Optional[torch.Tensor],
        sat_bev_coords: Optional[torch.Tensor],
        front_bev_xy: Optional[torch.Tensor],
        front_ground_valid_mask: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        return_attn_map: bool,
        batch_size: int,
        device: torch.device,
    ) -> Optional[CrossViewConditioningState]:
        if sat_tokens is None:
            return None

        if sat_tokens.ndim != 3:
            raise ValueError("sat_tokens must be [B, Ns, Cs]")
        if sat_xy is None or sat_xy.ndim != 3 or sat_xy.shape[-1] != 2:
            raise ValueError("sat_xy must be provided as [B, Ns, 2] when sat_tokens are used")
        if sat_tokens.shape[:2] != sat_xy.shape[:2]:
            raise ValueError(
                f"sat_tokens and sat_xy token shapes must match, got {list(sat_tokens.shape[:2])} "
                f"vs {list(sat_xy.shape[:2])}"
            )
        if sat_bev_coords is not None and sat_bev_coords.shape[:2] != sat_tokens.shape[:2]:
            raise ValueError(
                "sat_bev_coords must match sat_tokens token layout when provided"
            )
        if condition_mask is not None:
            if condition_mask.ndim != 1 or condition_mask.shape[0] != batch_size:
                raise ValueError(
                    f"condition_mask must be [B], got {list(condition_mask.shape)} for batch {batch_size}"
                )
            condition_mask = condition_mask.to(device=device, dtype=torch.bool)

        satellite_state = SatelliteMemoryState(
            tokens=sat_tokens,
            xy=sat_xy,
            bev_coords=sat_bev_coords,
        )
        satellite_state = self._apply_satellite_condition_mask(satellite_state, condition_mask)

        return CrossViewConditioningState(
            satellite=satellite_state,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
            condition_mask=condition_mask,
            return_attn_map=return_attn_map,
        )

    def _current_encoder_hidden_states(
        self,
        conditioning_state: Optional[CrossViewConditioningState],
        encoder_hidden_states: Any,
        hidden_states: torch.Tensor,
    ) -> Any:
        if conditioning_state is not None:
            satellite_state = self._move_satellite_state(
                conditioning_state.satellite,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            satellite_state = self._apply_satellite_condition_mask(
                satellite_state,
                conditioning_state.condition_mask,
            )
            return satellite_state.tokens

        if torch.is_tensor(encoder_hidden_states):
            return encoder_hidden_states.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if isinstance(encoder_hidden_states, tuple):
            return tuple(
                value.to(device=hidden_states.device, dtype=hidden_states.dtype)
                if torch.is_tensor(value)
                else value
                for value in encoder_hidden_states
            )
        return encoder_hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        sat_tokens: Optional[torch.Tensor] = None,
        sat_xy: Optional[torch.Tensor] = None,
        sat_bev_coords: Optional[torch.Tensor] = None,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        return_attn_map: bool = False,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        conditioning_state = self._build_conditioning_state(
            sat_tokens=sat_tokens,
            sat_xy=sat_xy,
            sat_bev_coords=sat_bev_coords,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
            condition_mask=condition_mask,
            return_attn_map=return_attn_map,
            batch_size=sample.shape[0],
            device=sample.device,
        )
        if conditioning_state is None and encoder_hidden_states is None:
            raise ValueError("SatelliteConditionedUNet requires sat_tokens or encoder_hidden_states")

        self.last_attn_maps = {}
        self.last_refinement_stats = {}
        self.last_satellite_state = None
        self._geometry_attention_context = None
        if conditioning_state is not None:
            self._geometry_attention_context = {
                "sat_xy": conditioning_state.satellite.xy,
                "sat_bev_coords": conditioning_state.satellite.bev_coords,
                "front_bev_xy": conditioning_state.front_bev_xy,
                "front_ground_valid_mask": conditioning_state.front_ground_valid_mask,
                "condition_mask": conditioning_state.condition_mask,
                "attention_debug": self._attention_debug_config,
                "timestep": timestep,
            }
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_encoder_hidden_states = (
            conditioning_state.satellite.tokens if conditioning_state is not None else encoder_hidden_states
        )
        aug_emb = self.get_aug_embed(
            emb=emb,
            encoder_hidden_states=aug_encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if conditioning_state is None:
            encoder_hidden_states = self.process_encoder_hidden_states(
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            )

        sample = self.conv_in(sample)

        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        is_controlnet = (
            mid_block_additional_residual is not None
            and down_block_additional_residuals is not None
        )
        is_adapter = down_intrablock_additional_residuals is not None
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated "
                "and will be removed in diffusers 1.3.0. `down_block_additional_residuals` should only be used "
                "for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead.",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for index, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                current_encoder_hidden_states = self._current_encoder_hidden_states(
                    conditioning_state,
                    encoder_hidden_states,
                    sample,
                )
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=current_encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample = sample + down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                new_down_block_res_samples = new_down_block_res_samples + (
                    down_block_res_sample + down_block_additional_residual,
                )
            down_block_res_samples = new_down_block_res_samples

        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                current_encoder_hidden_states = self._current_encoder_hidden_states(
                    conditioning_state,
                    encoder_hidden_states,
                    sample,
                )
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=current_encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample = sample + down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        for index, upsample_block in enumerate(self.up_blocks):
            is_final_block = index == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                current_encoder_hidden_states = self._current_encoder_hidden_states(
                    conditioning_state,
                    encoder_hidden_states,
                    sample,
                )
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=current_encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if conditioning_state is not None:
            self.last_attn_maps = conditioning_state.attn_maps
            self.last_refinement_stats = conditioning_state.refinement_stats
            self.last_satellite_state = conditioning_state.satellite

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
