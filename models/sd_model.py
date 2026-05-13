"""
Cross-view-refined UNet for satellite-to-frontview generation.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from models.conditioning import CrossViewConditioningState, SatelliteMemoryState
try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
except ImportError:
    from diffusers.models.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers

from models.unet.cross_view_refinement_block import CrossViewRefinementBlock


class SatelliteConditionedUNet(UNet2DConditionModel):
    """
    UNet2DConditionModel with explicit cross-view refinement support.

    Args:
        enable_cross_view_refinement: Whether to run cross-view refinement blocks
        refinement_injection_sites: Which U-Net layers run refinement
        refinement_block_config: Configuration for refinement blocks
        **kwargs: Additional arguments for UNet2DConditionModel
    """

    def __init__(
        self,
        enable_cross_view_refinement: bool = True,
        refinement_injection_sites: Optional[List[str]] = None,
        refinement_block_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.supports_cross_view_refinement = True
        self.enable_cross_view_refinement = enable_cross_view_refinement
        self.refinement_injection_sites = list(refinement_injection_sites or ["mid", "up0", "up1"])
        self._refinement_site_set = set(self.refinement_injection_sites)
        self.refinement_block_config = {
            "num_heads": 8,
            "head_dim": 64,
            "geo_ratio": 0.5,
            "rope_base": 10000.0,
            "lambda_geo": 1.0,
            "lambda_geom": 1.0,
            "geom_hidden_dim": 128,
            "geom_head_dim": 16,
            "sat_update_layers": 1,
            "use_geom_bias": True,
            "adapter_residual": True,
            "adapter_residual_scale": 1.0,
        }
        if refinement_block_config is not None:
            self.refinement_block_config.update(refinement_block_config)

        self.refinement_blocks = torch.nn.ModuleDict()
        self.last_attn_maps: Dict[str, torch.Tensor] = {}
        self.last_refinement_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.last_satellite_state: Optional[SatelliteMemoryState] = None

    def _get_or_create_refinement_block(
        self,
        site: str,
        front_feat: torch.Tensor,
        sat_tokens: torch.Tensor,
    ) -> CrossViewRefinementBlock:
        if site in self.refinement_blocks:
            block = self.refinement_blocks[site]
            block_param = next(block.parameters(), None)
            if block_param is not None and block_param.device != front_feat.device:
                block = block.to(device=front_feat.device)
                self.refinement_blocks[site] = block
            return block

        block = CrossViewRefinementBlock(
            front_dim=front_feat.shape[1],
            sat_in_dim=sat_tokens.shape[-1],
            num_heads=self.refinement_block_config["num_heads"],
            head_dim=self.refinement_block_config["head_dim"],
            geo_ratio=self.refinement_block_config["geo_ratio"],
            rope_base=self.refinement_block_config["rope_base"],
            lambda_geo=self.refinement_block_config["lambda_geo"],
            lambda_geom=self.refinement_block_config["lambda_geom"],
            geom_hidden_dim=self.refinement_block_config["geom_hidden_dim"],
            geom_head_dim=self.refinement_block_config["geom_head_dim"],
            sat_update_layers=self.refinement_block_config["sat_update_layers"],
            use_geom_bias=self.refinement_block_config["use_geom_bias"],
            adapter_residual=self.refinement_block_config["adapter_residual"],
            adapter_residual_scale=self.refinement_block_config["adapter_residual_scale"],
        )
        block = block.to(device=front_feat.device)
        self.refinement_blocks[site] = block
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

    def _prepare_front_ground_valid_mask(
        self,
        front_ground_valid_mask: Any,
        site: str,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if front_ground_valid_mask is None:
            return None

        if isinstance(front_ground_valid_mask, dict):
            site_mask = front_ground_valid_mask.get(site)
            if site_mask is None:
                site_mask = front_ground_valid_mask.get("default")
            return self._prepare_front_ground_valid_mask(site_mask, site, height, width, device, dtype)

        if not torch.is_tensor(front_ground_valid_mask):
            return None

        front_ground_valid_mask = front_ground_valid_mask.to(device=device, dtype=dtype)

        if front_ground_valid_mask.ndim == 2:
            if front_ground_valid_mask.shape == (front_ground_valid_mask.shape[0], height * width):
                return front_ground_valid_mask
            return None

        if front_ground_valid_mask.ndim == 3 and front_ground_valid_mask.shape[-1] == 1:
            if front_ground_valid_mask.shape[1] == height * width:
                return front_ground_valid_mask.squeeze(-1)
            return None

        if front_ground_valid_mask.ndim == 4 and front_ground_valid_mask.shape[1] == 1:
            resized = F.interpolate(front_ground_valid_mask, size=(height, width), mode="nearest")
            return resized.reshape(front_ground_valid_mask.shape[0], height * width)

        if front_ground_valid_mask.ndim == 4 and front_ground_valid_mask.shape[-1] == 1:
            mask = front_ground_valid_mask.permute(0, 3, 1, 2)
            resized = F.interpolate(mask, size=(height, width), mode="nearest")
            return resized.reshape(front_ground_valid_mask.shape[0], height * width)

        return None

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
        xy_mask = self._token_condition_mask(condition_mask, satellite_state.xy)
        bev_coords = satellite_state.bev_coords
        if bev_coords is not None:
            bev_mask = self._token_condition_mask(condition_mask, bev_coords)
            bev_coords = bev_coords * bev_mask

        return satellite_state.replace(
            tokens=satellite_state.tokens * token_mask,
            xy=satellite_state.xy * xy_mask,
            bev_coords=bev_coords,
        )

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

    def _run_refinement(
        self,
        site: str,
        front_feat: torch.Tensor,
        conditioning_state: Optional[CrossViewConditioningState],
    ) -> torch.Tensor:
        if conditioning_state is None:
            return front_feat
        if not self.enable_cross_view_refinement or site not in self._refinement_site_set:
            return front_feat
        if conditioning_state.front_bev_xy is None:
            return front_feat
        if conditioning_state.condition_mask is not None and not bool(conditioning_state.condition_mask.any().item()):
            return front_feat

        front_bev_xy = self._prepare_front_bev_xy(
            conditioning_state.front_bev_xy,
            site,
            front_feat.shape[2],
            front_feat.shape[3],
            front_feat.device,
            front_feat.dtype,
        )
        if front_bev_xy is None:
            return front_feat

        front_ground_valid_mask = self._prepare_front_ground_valid_mask(
            conditioning_state.front_ground_valid_mask,
            site,
            front_feat.shape[2],
            front_feat.shape[3],
            front_feat.device,
            front_feat.dtype,
        )
        satellite_state = self._move_satellite_state(
            conditioning_state.satellite,
            device=front_feat.device,
            dtype=front_feat.dtype,
        )
        block = self._get_or_create_refinement_block(site, front_feat, satellite_state.tokens)
        block_output = block(
            front_feat=front_feat,
            satellite_state=satellite_state,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
            return_attn_map=conditioning_state.return_attn_map,
        )

        updated_satellite = block_output["satellite_state"]
        if conditioning_state.condition_mask is not None:
            updated_satellite = self._apply_satellite_condition_mask(
                updated_satellite,
                conditioning_state.condition_mask,
            )

        conditioning_state.satellite = updated_satellite
        if conditioning_state.return_attn_map:
            attn_map = block_output.get("attn_map")
            if attn_map is not None:
                conditioning_state.attn_maps[site] = attn_map

        stats = block_output.get("stats")
        if stats:
            detached_stats = {}
            for key, value in stats.items():
                if torch.is_tensor(value):
                    detached_stats[key] = value.detach()
            if detached_stats:
                conditioning_state.refinement_stats[site] = detached_stats

        adapter_residual = block_output.get("adapter_residual")
        if torch.is_tensor(adapter_residual):
            if conditioning_state.condition_mask is not None:
                residual_mask = conditioning_state.condition_mask.to(
                    device=adapter_residual.device,
                    dtype=adapter_residual.dtype,
                ).view(-1, 1, 1, 1)
                adapter_residual = adapter_residual * residual_mask
            return front_feat + adapter_residual
        return front_feat

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

            sample = self._run_refinement(f"down{index}", sample, conditioning_state)
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

        sample = self._run_refinement("mid", sample, conditioning_state)

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

            sample = self._run_refinement(f"up{index}", sample, conditioning_state)

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
