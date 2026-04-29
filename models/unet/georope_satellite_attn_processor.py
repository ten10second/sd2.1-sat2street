"""
Geometry-aware satellite cross-attention processor for diffusers Attention modules.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_xy_georope import ContinuousXYGeoRoPE


class GeoRoPESatelliteAttnProcessor(nn.Module):
    """Use native U-Net attn2 Q with GeoRoPE-encoded satellite K/V conditioning."""

    def __init__(
        self,
        name: str,
        hidden_size: int,
        sat_in_dim: int,
        num_heads: int,
        head_dim: int,
        context_provider: Callable[[], Dict[str, Any]],
        geo_ratio: float = 1.0,
        rope_base: float = 10000.0,
        invalid_conf_loss_weight: float = 0.05,
    ):
        super().__init__()
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                f"hidden_size must equal num_heads * head_dim, got {hidden_size} vs {num_heads}*{head_dim}"
            )
        self.name = name
        self.hidden_size = int(hidden_size)
        self.sat_in_dim = int(sat_in_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.context_provider = context_provider
        self.invalid_conf_loss_weight = float(invalid_conf_loss_weight)

        self.sat_norm = nn.LayerNorm(sat_in_dim)
        self.sat_to_k = nn.Linear(sat_in_dim, hidden_size)
        self.sat_to_v = nn.Linear(sat_in_dim, hidden_size)
        self.plucker_adapter = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate_norm = nn.LayerNorm(hidden_size)
        self.gate_head = nn.Linear(hidden_size, 1)
        self.rope = ContinuousXYGeoRoPE(head_dim=head_dim, geo_ratio=geo_ratio, rope_base=rope_base)

        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -1.0)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    @staticmethod
    def _factor_hw(token_count: int, aspect_hw: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if aspect_hw is not None:
            aspect_h, aspect_w = max(1, int(aspect_hw[0])), max(1, int(aspect_hw[1]))
            aspect = aspect_w / aspect_h
            h = max(1, int(round(math.sqrt(token_count / max(aspect, 1e-6)))))
            candidates = []
            for delta in range(-4, 5):
                cand_h = max(1, h + delta)
                if token_count % cand_h == 0:
                    cand_w = token_count // cand_h
                    candidates.append((abs((cand_w / cand_h) - aspect), cand_h, cand_w))
            if candidates:
                _, best_h, best_w = min(candidates, key=lambda item: item[0])
                return best_h, best_w

        h = int(math.sqrt(token_count))
        while h > 1 and token_count % h != 0:
            h -= 1
        return h, token_count // h

    @staticmethod
    def _prepare_xy(raw_xy: Any, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if raw_xy is None or not torch.is_tensor(raw_xy):
            return None
        xy = raw_xy.to(device=device, dtype=dtype)
        if xy.ndim == 3 and xy.shape[-1] == 2:
            if xy.shape[1] == height * width:
                return xy
            src_h, src_w = GeoRoPESatelliteAttnProcessor._factor_hw(xy.shape[1], None)
            xy = xy.reshape(xy.shape[0], src_h, src_w, 2).permute(0, 3, 1, 2)
            return F.interpolate(xy, size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(xy.shape[0], height * width, 2)
        if xy.ndim == 4 and xy.shape[1] == 2:
            return F.interpolate(xy, size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(xy.shape[0], height * width, 2)
        if xy.ndim == 4 and xy.shape[-1] == 2:
            xy = xy.permute(0, 3, 1, 2)
            return F.interpolate(xy, size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(xy.shape[0], height * width, 2)
        return None

    @staticmethod
    def _prepare_mask(raw_mask: Any, batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        if raw_mask is None or not torch.is_tensor(raw_mask):
            return torch.ones(batch, height * width, device=device, dtype=torch.bool)
        mask = raw_mask.to(device=device, dtype=torch.float32)
        if mask.ndim == 2:
            if mask.shape[1] == height * width:
                return mask > 0.5
            src_h, src_w = GeoRoPESatelliteAttnProcessor._factor_hw(mask.shape[1], None)
            mask = mask.reshape(mask.shape[0], 1, src_h, src_w)
            return F.interpolate(mask, size=(height, width), mode="nearest").reshape(mask.shape[0], height * width) > 0.5
        if mask.ndim == 3 and mask.shape[-1] == 1:
            return GeoRoPESatelliteAttnProcessor._prepare_mask(mask.squeeze(-1), batch, height, width, device)
        if mask.ndim == 4 and mask.shape[1] == 1:
            return F.interpolate(mask, size=(height, width), mode="nearest").reshape(mask.shape[0], height * width) > 0.5
        if mask.ndim == 4 and mask.shape[-1] == 1:
            return GeoRoPESatelliteAttnProcessor._prepare_mask(mask.permute(0, 3, 1, 2), batch, height, width, device)
        return torch.ones(batch, height * width, device=device, dtype=torch.bool)

    @staticmethod
    def _prepare_plucker(raw_plucker: Any, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if raw_plucker is None or not torch.is_tensor(raw_plucker):
            return None
        plucker = raw_plucker.to(device=device, dtype=dtype)
        if plucker.ndim == 3 and plucker.shape[-1] == 6:
            if plucker.shape[1] == height * width:
                return plucker
            src_h, src_w = GeoRoPESatelliteAttnProcessor._factor_hw(plucker.shape[1], None)
            plucker = plucker.reshape(plucker.shape[0], src_h, src_w, 6).permute(0, 3, 1, 2)
            return F.interpolate(plucker, size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(plucker.shape[0], height * width, 6)
        if plucker.ndim == 4 and plucker.shape[1] == 6:
            return F.interpolate(plucker, size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(plucker.shape[0], height * width, 6)
        if plucker.ndim == 4 and plucker.shape[-1] == 6:
            plucker = plucker.permute(0, 3, 1, 2)
            return F.interpolate(plucker, size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(plucker.shape[0], height * width, 6)
        return None

    def _record(self, context: Dict[str, Any], stats: Dict[str, torch.Tensor], loss: torch.Tensor, attn_map: Optional[torch.Tensor]) -> None:
        detached_stats = {key: value.detach() for key, value in stats.items() if torch.is_tensor(value)}
        context.setdefault("reading_stats", {})[self.name] = detached_stats
        context.setdefault("reading_losses", {})[self.name] = {"invalid_conf_loss": loss}
        if attn_map is not None:
            context.setdefault("attn_maps", {})[self.name] = attn_map.detach()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        context = self.context_provider()
        sat_tokens = context.get("sat_tokens") if context else None
        sat_xy = context.get("sat_xy") if context else None
        raw_front_bev_xy = context.get("front_bev_xy") if context else None
        if sat_tokens is None or sat_xy is None or raw_front_bev_xy is None:
            return torch.zeros_like(hidden_states)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, token_count, channel = hidden_states.shape
            height, width = self._factor_hw(token_count, context.get("latent_hw") if context else None)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        front_bev_xy = self._prepare_xy(raw_front_bev_xy, height, width, hidden_states.device, hidden_states.dtype)
        if front_bev_xy is None:
            return torch.zeros_like(residual)
        valid_mask = self._prepare_mask(
            context.get("front_bev_valid_mask"),
            batch_size,
            height,
            width,
            hidden_states.device,
        )
        front_plucker = self._prepare_plucker(
            context.get("front_plucker"),
            height,
            width,
            hidden_states.device,
            hidden_states.dtype,
        )

        sat_tokens = sat_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype)
        sat_xy = sat_xy.to(device=hidden_states.device, dtype=hidden_states.dtype)
        query = attn.to_q(hidden_states)
        key = self.sat_to_k(self.sat_norm(sat_tokens))
        value = self.sat_to_v(self.sat_norm(sat_tokens))

        q = query.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, front_bev_xy, sat_xy)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_logits.float(), dim=-1).to(dtype=q.dtype)
        hidden_states_out = torch.matmul(attn_probs, v)
        hidden_states_out = hidden_states_out.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)

        gate_context = hidden_states
        if front_plucker is not None:
            gate_context = gate_context + self.plucker_adapter(front_plucker)
        gate = torch.sigmoid(self.gate_head(self.gate_norm(gate_context))).squeeze(-1)
        gate = gate * valid_mask.to(device=gate.device, dtype=gate.dtype)
        condition_mask = context.get("condition_mask")
        if condition_mask is not None:
            gate = gate * condition_mask.to(device=gate.device, dtype=gate.dtype).view(-1, 1)
        hidden_states_out = hidden_states_out * gate.unsqueeze(-1)

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        if input_ndim == 4:
            hidden_states_out = hidden_states_out.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states_out = hidden_states_out + residual
        hidden_states_out = hidden_states_out / attn.rescale_output_factor

        valid_float = valid_mask.to(dtype=gate.dtype)
        invalid_float = 1.0 - valid_float
        valid_gate = self._masked_mean(gate, valid_float)
        invalid_gate = self._masked_mean(gate, invalid_float)
        entropy = -(attn_probs.float() * attn_probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        stats = {
            "confidence": gate.float().mean(),
            "valid_confidence": valid_gate,
            "invalid_confidence": invalid_gate,
            "valid_ratio": valid_float.float().mean(),
            "georope_attn_entropy": entropy,
            "gate_mean": gate.float().mean(),
            "valid_gate_mean": valid_gate,
            "invalid_gate_mean": invalid_gate,
        }
        invalid_conf_loss = invalid_gate * self.invalid_conf_loss_weight
        return_attn_map = bool(context.get("return_attn_map", False))
        self._record(context, stats, invalid_conf_loss, attn_probs if return_attn_map else None)
        return hidden_states_out
