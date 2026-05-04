"""
Stable Diffusion Model for Satellite-to-Frontview Generation.

Custom Stable Diffusion implementation with satellite condition encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional

try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
except ImportError:
    from diffusers.models.unet_2d_condition import UNet2DConditionModel

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
            "token_pool_num_tokens": 8,
            "token_pool_num_heads": 8,
            "token_scale": 1.0,
            "scene_consistency_weight": 0.0,
            "save_attention_heatmap": False,
            "heatmap_max_tokens": 16,
            "ray_num_samples": 32,
            "ray_depth_min": 1.0,
            "ray_depth_max": 80.0,
            "ray_offset_scale": 0.50,
            "ray_scene_extent_x_m": 51.2,
            "ray_scene_extent_y_m": 51.2,
            "ray_scene_z_min_m": 0.0,
            "ray_scene_z_max_m": 20.0,
            "triplane_enabled": True,
            "triplane_height_tokens": 16,
            "triplane_num_cvha_layers": 1,
            "triplane_cvha_num_self_points": 4,
            "triplane_cvha_num_cross_points": 8,
            "triplane_cvha_local_radius": 1.0,
            "triplane_cvha_offset_scale": 1.0,
            "view_injection_sites": ("down2", "down3", "mid", "up0", "up1"),
            "view_modulation_scale": 0.4,
        }
        if reading_block_config is not None:
            self.reading_block_config.update(reading_block_config)
        self.last_attn_maps: Dict[str, torch.Tensor] = {}
        self.view_injection_sites = tuple(self.reading_block_config.get("view_injection_sites", ("down2", "down3", "mid", "up0", "up1")))
        self.view_modulation_scale = float(self.reading_block_config.get("view_modulation_scale", 0.4))
        self.view_condition_in_dim = int(
            self.config.cross_attention_dim or self.reading_block_config.get("sat_in_dim", DEFAULT_SATELLITE_EMBED_DIM)
        )
        self._view_condition_projs = nn.ModuleDict(self._build_view_condition_projs())
        self._view_condition_handles = []
        self._view_condition_map: Optional[torch.Tensor] = None
        self._register_view_condition_hooks()

    @staticmethod
    def _build_view_proj(view_in_dim: int, out_channels: int) -> nn.Module:
        hidden_dim = max(out_channels, view_in_dim // 2)
        return nn.Sequential(
            nn.Conv2d(view_in_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def _get_view_injection_modules(self) -> Dict[str, Optional[nn.Module]]:
        down_blocks = list(getattr(self, "down_blocks", []))
        up_blocks = list(getattr(self, "up_blocks", []))
        return {
            "down2": down_blocks[2] if len(down_blocks) > 2 else None,
            "down3": down_blocks[3] if len(down_blocks) > 3 else None,
            "mid": getattr(self, "mid_block", None),
            "up0": up_blocks[0] if len(up_blocks) > 0 else None,
            "up1": up_blocks[1] if len(up_blocks) > 1 else None,
        }

    @staticmethod
    def _infer_module_out_channels(module: nn.Module) -> int:
        resnets = getattr(module, "resnets", None)
        if resnets:
            resnet = resnets[-1]
            out_channels = getattr(resnet, "out_channels", None)
            if isinstance(out_channels, int):
                return out_channels
            conv2 = getattr(resnet, "conv2", None)
            out_channels = getattr(conv2, "out_channels", None)
            if isinstance(out_channels, int):
                return out_channels

        for attr_name in ("downsamplers", "upsamplers"):
            samplers = getattr(module, attr_name, None)
            if samplers:
                sampler = samplers[-1]
                conv = getattr(sampler, "conv", None)
                out_channels = getattr(conv, "out_channels", None)
                if isinstance(out_channels, int):
                    return out_channels

        raise ValueError(f"Unable to infer output channels for module {type(module).__name__}")

    def _build_view_condition_projs(self) -> Dict[str, nn.Module]:
        projs: Dict[str, nn.Module] = {}
        for name in self.view_injection_sites:
            module = self._get_view_injection_modules().get(str(name))
            if module is None:
                continue
            out_channels = self._infer_module_out_channels(module)
            projs[str(name)] = self._build_view_proj(self.view_condition_in_dim, out_channels)
        return projs

    def _apply_view_condition(self, hidden_states: torch.Tensor, name: str) -> torch.Tensor:
        if self._view_condition_map is None:
            return hidden_states
        if name not in self._view_condition_projs:
            return hidden_states
        view_map = self._view_condition_map
        if view_map.shape[0] != hidden_states.shape[0]:
            return hidden_states
        view_map = F.interpolate(
            view_map.to(device=hidden_states.device, dtype=hidden_states.dtype),
            size=hidden_states.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        proj = self._view_condition_projs[name]
        proj_param = next(proj.parameters(), None)
        proj_dtype = proj_param.dtype if proj_param is not None else hidden_states.dtype
        delta = proj(view_map.to(dtype=proj_dtype)).to(dtype=hidden_states.dtype)
        delta = delta * self.view_modulation_scale
        return hidden_states + delta

    def _make_view_condition_hook(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                if not output:
                    return output
                first = output[0]
                if torch.is_tensor(first):
                    modulated = self._apply_view_condition(first, name)
                    return (modulated, *output[1:])
                return output
            if torch.is_tensor(output):
                return self._apply_view_condition(output, name)
            return output
        return hook

    def _register_view_condition_hooks(self) -> None:
        for handle in self._view_condition_handles:
            handle.remove()
        self._view_condition_handles = []

        block_map = self._get_view_injection_modules()
        for name in self.view_injection_sites:
            module = block_map.get(str(name))
            if module is None:
                continue
            handle = module.register_forward_hook(self._make_view_condition_hook(str(name)))
            self._view_condition_handles.append(handle)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        view_condition_map: Optional[torch.Tensor] = None,
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
        self._view_condition_map = view_condition_map
        try:
            return super().forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
        finally:
            self._view_condition_map = None
