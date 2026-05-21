"""Cross-attention processor that can add query UV PE and geometry bias."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.perspective_position_encoder import PerspectivePositionEncoder


def build_normalized_image_uv_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return row-major pixel-center UV coords normalized as ``2 * pixel / size - 1``."""
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be positive, got {height}/{width}")

    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    uv = torch.stack((2.0 * xx - 1.0, 2.0 * yy - 1.0), dim=-1)
    return uv.reshape(1, height * width, 2)


def infer_spatial_hw(sequence_length: int, base_hw: Tuple[int, int]) -> Tuple[int, int]:
    """Infer an HxW grid for a flattened UNet feature map from the latent aspect ratio."""
    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    base_h, base_w = (int(base_hw[0]), int(base_hw[1]))
    if base_h <= 0 or base_w <= 0:
        raise ValueError(f"query_base_hw must be positive, got {base_hw}")

    aspect = float(base_w) / float(base_h)
    target_h = math.sqrt(float(sequence_length) / max(aspect, 1e-8))
    best = None
    limit = int(math.sqrt(sequence_length))
    for h in range(1, limit + 1):
        if sequence_length % h != 0:
            continue
        w = sequence_length // h
        for cand_h, cand_w in ((h, w), (w, h)):
            ratio = float(cand_w) / float(cand_h)
            score = (abs(ratio - aspect), abs(float(cand_h) - target_h))
            if best is None or score < best[0]:
                best = (score, cand_h, cand_w)

    if best is None:
        raise ValueError(f"Unable to infer spatial grid for sequence length {sequence_length}")
    return int(best[1]), int(best[2])


class _QueryUVAttnProcessorBase(nn.Module):
    def __init__(
        self,
        query_dim: int,
        *,
        query_uv_enabled: bool,
        geometry_bias_enabled: bool,
        geometry_bias_scale: float = 2.0,
        geometry_invalid_penalty: float = -1e4,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.query_uv_enabled = bool(query_uv_enabled)
        self.geometry_bias_enabled = bool(geometry_bias_enabled)
        self.geometry_bias_scale = float(geometry_bias_scale)
        self.geometry_invalid_penalty = float(geometry_invalid_penalty)

        if self.query_uv_enabled:
            self.query_uv_encoder = PerspectivePositionEncoder(dim=self.query_dim)
            self.query_uv_gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.query_uv_encoder = None
            self.register_parameter("query_uv_gate", None)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        query_base_hw: Optional[Tuple[int, int]] = None,
        query_uv: Optional[torch.Tensor] = None,
        sat_perspective_uv: Optional[torch.Tensor] = None,
        sat_perspective_valid: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            query_base_hw=query_base_hw,
            query_uv=query_uv,
            sat_perspective_uv=sat_perspective_uv,
            sat_perspective_valid=sat_perspective_valid,
            **kwargs,
        )

    def _resolve_query_uv(
        self,
        hidden_states: torch.Tensor,
        *,
        batch_size: int,
        query_base_hw: Optional[Tuple[int, int]],
        query_uv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if query_uv is not None:
            uv = query_uv.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if uv.ndim != 3 or uv.shape[-1] != 2:
                raise ValueError(f"query_uv must be [B,N,2], got {tuple(uv.shape)}")
            if uv.shape[1] != hidden_states.shape[1]:
                raise ValueError(
                    f"query_uv token count must match query tokens: {uv.shape[1]} vs {hidden_states.shape[1]}"
                )
            if uv.shape[0] == 1 and batch_size != 1:
                uv = uv.expand(batch_size, -1, -1)
            if uv.shape[0] != batch_size:
                raise ValueError(f"query_uv batch must be 1 or {batch_size}, got {uv.shape[0]}")
            return uv

        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be flattened before query UV resolution, got {hidden_states.ndim}D")
        if query_base_hw is None:
            raise ValueError("query_base_hw is required for flattened UNet attention states")

        height, width = infer_spatial_hw(hidden_states.shape[1], query_base_hw)
        uv = build_normalized_image_uv_grid(
            height,
            width,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return uv.expand(batch_size, -1, -1)

    def _resolve_sat_perspective_uv(
        self,
        hidden_states: torch.Tensor,
        *,
        batch_size: int,
        key_length: int,
        sat_perspective_uv: Optional[torch.Tensor],
        sat_perspective_valid: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if sat_perspective_uv is None:
            return None, None

        uv = sat_perspective_uv.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if uv.ndim != 3 or uv.shape[-1] != 2:
            raise ValueError(f"sat_perspective_uv must be [B,N,2], got {tuple(uv.shape)}")
        if uv.shape[1] != key_length:
            raise ValueError(
                f"sat_perspective_uv token count must match key tokens: {uv.shape[1]} vs {key_length}"
            )
        if uv.shape[0] == 1 and batch_size != 1:
            uv = uv.expand(batch_size, -1, -1)
        if uv.shape[0] != batch_size:
            raise ValueError(f"sat_perspective_uv batch must be 1 or {batch_size}, got {uv.shape[0]}")

        valid = None
        if sat_perspective_valid is not None:
            valid = sat_perspective_valid.to(device=hidden_states.device)
            if valid.ndim != 2:
                raise ValueError(f"sat_perspective_valid must be [B,N], got {tuple(valid.shape)}")
            if valid.shape[1] != key_length:
                raise ValueError(
                    f"sat_perspective_valid token count must match key tokens: {valid.shape[1]} vs {key_length}"
                )
            if valid.shape[0] == 1 and batch_size != 1:
                valid = valid.expand(batch_size, -1)
            if valid.shape[0] != batch_size:
                raise ValueError(
                    f"sat_perspective_valid batch must be 1 or {batch_size}, got {valid.shape[0]}"
                )
            valid = valid.to(dtype=torch.bool)

        return uv, valid

    def _build_geometry_bias(
        self,
        query_uv: torch.Tensor,
        sat_perspective_uv: Optional[torch.Tensor],
        sat_perspective_valid: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.geometry_bias_enabled or sat_perspective_uv is None:
            return None

        if query_uv.ndim != 3 or sat_perspective_uv.ndim != 3:
            raise ValueError(
                f"query_uv and sat_perspective_uv must be [B,N,2], got {tuple(query_uv.shape)} and {tuple(sat_perspective_uv.shape)}"
            )
        if query_uv.shape[0] != sat_perspective_uv.shape[0]:
            raise ValueError(
                f"query_uv batch must match sat_perspective_uv batch: {query_uv.shape[0]} vs {sat_perspective_uv.shape[0]}"
            )

        dist2 = (query_uv[:, :, None, :] - sat_perspective_uv[:, None, :, :]).pow(2).sum(dim=-1)
        bias = -self.geometry_bias_scale * dist2

        if sat_perspective_valid is not None:
            if sat_perspective_valid.ndim != 2:
                raise ValueError(
                    f"sat_perspective_valid must be [B,N], got {tuple(sat_perspective_valid.shape)}"
                )
            if sat_perspective_valid.shape[0] != bias.shape[0]:
                raise ValueError(
                    f"sat_perspective_valid batch must match query batch: {sat_perspective_valid.shape[0]} vs {bias.shape[0]}"
                )
            invalid_penalty = torch.tensor(
                self.geometry_invalid_penalty,
                device=bias.device,
                dtype=bias.dtype,
            )
            bias = bias + (~sat_perspective_valid).to(dtype=bias.dtype).unsqueeze(1) * invalid_penalty

        return bias

class QueryUVAttnProcessor2_0(_QueryUVAttnProcessorBase):
    """
    Diffusers AttnProcessor2_0-compatible processor with optional query UV PE
    and optional geometry bias on the cross-attention scores.
    """

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        query_base_hw: Optional[Tuple[int, int]] = None,
        query_uv: Optional[torch.Tensor] = None,
        sat_perspective_uv: Optional[torch.Tensor] = None,
        sat_perspective_valid: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del args, kwargs

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        elif input_ndim == 3:
            batch_size = hidden_states.shape[0]
            channel = None
        else:
            raise ValueError(f"hidden_states must be rank 3 or 4, got {input_ndim}")

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query_uv_tensor = None
        if encoder_hidden_states is not None and (self.query_uv_enabled or self.geometry_bias_enabled):
            query_uv_tensor = self._resolve_query_uv(
                hidden_states,
                batch_size=hidden_states.shape[0],
                query_base_hw=query_base_hw,
                query_uv=query_uv,
            ).to(dtype=query.dtype)
        if self.query_uv_enabled and query_uv_tensor is not None:
            uv = query_uv_tensor
            query_pe = self.query_uv_encoder(uv)
            query = query + self.query_uv_gate.to(dtype=query.dtype) * query_pe

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(hidden_states.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(hidden_states.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(hidden_states.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.geometry_bias_enabled and encoder_hidden_states is not None:
            sat_uv, sat_valid = self._resolve_sat_perspective_uv(
                hidden_states,
                batch_size=hidden_states.shape[0],
                key_length=key.shape[2],
                sat_perspective_uv=sat_perspective_uv,
                sat_perspective_valid=sat_perspective_valid,
            )
            if query_uv_tensor is None:
                query_uv_tensor = self._resolve_query_uv(
                    hidden_states,
                    batch_size=hidden_states.shape[0],
                    query_base_hw=query_base_hw,
                    query_uv=query_uv,
                ).to(dtype=query.dtype)
            geometry_bias = self._build_geometry_bias(
                query_uv_tensor,
                sat_uv,
                sat_valid,
            )
            if geometry_bias is not None:
                geometry_bias = geometry_bias.unsqueeze(1)
                attention_mask = geometry_bias if attention_mask is None else attention_mask + geometry_bias

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(hidden_states.shape[0], channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class QueryUVSlicedAttnProcessor(_QueryUVAttnProcessorBase):
    """Sliced attention processor that preserves query UV PE and geometry bias."""

    def __init__(
        self,
        query_dim: int,
        *,
        slice_size: int,
        query_uv_enabled: bool,
        geometry_bias_enabled: bool,
        geometry_bias_scale: float = 2.0,
        geometry_invalid_penalty: float = -1e4,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__(
            query_dim=query_dim,
            query_uv_enabled=query_uv_enabled,
            geometry_bias_enabled=geometry_bias_enabled,
            geometry_bias_scale=geometry_bias_scale,
            geometry_invalid_penalty=geometry_invalid_penalty,
            gate_init=gate_init,
        )
        self.slice_size = int(slice_size)

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        query_base_hw: Optional[Tuple[int, int]] = None,
        query_uv: Optional[torch.Tensor] = None,
        sat_perspective_uv: Optional[torch.Tensor] = None,
        sat_perspective_valid: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del args, kwargs

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        elif input_ndim == 3:
            batch_size = hidden_states.shape[0]
            channel = None
        else:
            raise ValueError(f"hidden_states must be rank 3 or 4, got {input_ndim}")

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query_uv_tensor = None
        if encoder_hidden_states is not None and (self.query_uv_enabled or self.geometry_bias_enabled):
            query_uv_tensor = self._resolve_query_uv(
                hidden_states,
                batch_size=hidden_states.shape[0],
                query_base_hw=query_base_hw,
                query_uv=query_uv,
            ).to(dtype=query.dtype)
        if self.query_uv_enabled and query_uv_tensor is not None:
            uv = query_uv_tensor
            query_pe = self.query_uv_encoder(uv)
            query = query + self.query_uv_gate.to(dtype=query.dtype) * query_pe

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.geometry_bias_enabled and encoder_hidden_states is not None:
            sat_uv, sat_valid = self._resolve_sat_perspective_uv(
                hidden_states,
                batch_size=batch_size,
                key_length=key.shape[1],
                sat_perspective_uv=sat_perspective_uv,
                sat_perspective_valid=sat_perspective_valid,
            )
            if query_uv_tensor is None:
                query_uv_tensor = self._resolve_query_uv(
                    hidden_states,
                    batch_size=batch_size,
                    query_base_hw=query_base_hw,
                    query_uv=query_uv,
                ).to(dtype=query.dtype)
            geometry_bias = self._build_geometry_bias(
                query_uv_tensor,
                sat_uv,
                sat_valid,
            )
            if geometry_bias is not None:
                geometry_bias = geometry_bias.repeat_interleave(attn.heads, dim=0)
                attention_mask = geometry_bias if attention_mask is None else attention_mask + geometry_bias

        batch_size_attention, query_tokens, dim = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim),
            device=query.device,
            dtype=query.dtype,
        )

        for i in range((batch_size_attention - 1) // self.slice_size + 1):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(hidden_states.shape[0], channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
