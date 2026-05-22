"""
Geometry-masked native cross-attention processors.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel as torch_sdpa_kernel
except ImportError:  # pragma: no cover - compatibility with older torch layouts
    SDPBackend = Any
    torch_sdpa_kernel = None

try:
    from diffusers.models.attention_processor import Attention
except ImportError:  # pragma: no cover - compatibility with older diffusers layouts
    Attention = Any


GeometryContextProvider = Callable[[], Optional[dict[str, Any]]]


def build_topk_mask(
    front_xy: torch.Tensor,
    sat_xy: torch.Tensor,
    *,
    topk: int,
    query_mask: Optional[torch.Tensor] = None,
    key_mask: Optional[torch.Tensor] = None,
    mask_invalid_queries: bool = True,
    dtype: Optional[torch.dtype] = None,
    min_value: float = -10000.0,
) -> torch.Tensor:
    """Build an additive SDPA mask that keeps only the top-k nearest satellite tokens."""
    if front_xy.ndim != 3 or sat_xy.ndim != 3 or front_xy.shape[-1] != 2 or sat_xy.shape[-1] != 2:
        raise ValueError("front_xy and sat_xy must be [B,N,2]")
    if front_xy.shape[0] != sat_xy.shape[0]:
        raise ValueError("front_xy and sat_xy batch size must match")

    dtype = front_xy.dtype if dtype is None else dtype
    key_length = int(sat_xy.shape[1])
    topk = min(int(topk), key_length)

    dist2 = (front_xy.unsqueeze(2) - sat_xy.unsqueeze(1)).pow(2).sum(dim=-1)
    if key_mask is not None:
        key_mask = key_mask.to(device=front_xy.device, dtype=torch.bool)
        if key_mask.shape == sat_xy.shape[:2]:
            dist2 = dist2.masked_fill(~key_mask.unsqueeze(1), torch.inf)

    nearest = torch.topk(dist2, k=topk, dim=-1, largest=False).indices
    keep = torch.zeros(dist2.shape, device=front_xy.device, dtype=torch.bool)
    keep.scatter_(dim=-1, index=nearest, value=True)

    if mask_invalid_queries and query_mask is not None:
        keep = keep | (~query_mask.to(device=front_xy.device, dtype=torch.bool)).unsqueeze(-1)

    if key_mask is not None:
        key_mask = key_mask.to(device=front_xy.device, dtype=torch.bool)
        if key_mask.shape == sat_xy.shape[:2]:
            keep = keep & key_mask.unsqueeze(1)

    mask = torch.zeros(keep.shape, device=front_xy.device, dtype=dtype)
    return mask.masked_fill(~keep, min_value)


class GeometryMaskedAttnProcessor2_0(nn.Module):
    """
    PyTorch-2 SDPA attention processor with BEV top-k masking for satellite cross-attn.

    The processor is intended for selected UNet attn2 modules. Query tokens come from
    the current street/latent feature map; key/value tokens come from satellite memory.
    """

    def __init__(
        self,
        *,
        site: str,
        module_name: Optional[str] = None,
        context_provider: GeometryContextProvider,
        topk: int = 32,
        mask_invalid_queries: bool = True,
        fallback_to_unmasked: bool = True,
        use_metric_coords: bool = False,
        enable_geometry_bias: bool = False,
        geometry_bias_type: str = "dist_dir",
        lambda_dist: float = 2.0,
        lambda_dir: float = 0.5,
        learnable_geometry_bias: bool = False,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("GeometryMaskedAttnProcessor2_0 requires PyTorch 2.0 SDPA support")
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        self.site = str(site)
        self.module_name = str(module_name or site)
        self.context_provider = context_provider
        self.topk = int(topk)
        self.mask_invalid_queries = bool(mask_invalid_queries)
        self.fallback_to_unmasked = bool(fallback_to_unmasked)
        self.use_metric_coords = bool(use_metric_coords)
        self.enable_geometry_bias = bool(enable_geometry_bias)
        self.geometry_bias_type = str(geometry_bias_type).lower()
        if self.geometry_bias_type != "dist_dir":
            raise ValueError(f"Only geometry_bias_type='dist_dir' is supported, got {geometry_bias_type!r}")
        self.learnable_geometry_bias = bool(learnable_geometry_bias)
        if self.learnable_geometry_bias:
            self.lambda_dist = nn.Parameter(torch.tensor(float(lambda_dist)))
            self.lambda_dir = nn.Parameter(torch.tensor(float(lambda_dir)))
        else:
            self.register_buffer("lambda_dist", torch.tensor(float(lambda_dist)), persistent=True)
            self.register_buffer("lambda_dir", torch.tensor(float(lambda_dir)), persistent=True)

    @staticmethod
    def _front_xy_to_bchw(front_bev_xy: torch.Tensor) -> Optional[torch.Tensor]:
        if front_bev_xy.ndim == 4 and front_bev_xy.shape[1] == 2:
            return front_bev_xy
        if front_bev_xy.ndim == 4 and front_bev_xy.shape[-1] == 2:
            return front_bev_xy.permute(0, 3, 1, 2)
        return None

    @staticmethod
    def _front_mask_to_b1hw(front_ground_valid_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if front_ground_valid_mask is None:
            return None
        if front_ground_valid_mask.ndim == 4 and front_ground_valid_mask.shape[1] == 1:
            return front_ground_valid_mask
        if front_ground_valid_mask.ndim == 4 and front_ground_valid_mask.shape[-1] == 1:
            return front_ground_valid_mask.permute(0, 3, 1, 2)
        return None

    @staticmethod
    def _front_mask_to_bn(front_ground_valid_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if front_ground_valid_mask is None:
            return None
        if front_ground_valid_mask.ndim == 2:
            return front_ground_valid_mask
        if front_ground_valid_mask.ndim == 3 and front_ground_valid_mask.shape[-1] == 1:
            return front_ground_valid_mask.squeeze(-1)
        return None

    @staticmethod
    def _infer_hw_from_tokens(
        token_count: int,
        source_height: int,
        source_width: int,
    ) -> Optional[tuple[int, int]]:
        if token_count <= 0:
            return None
        if source_height <= 0 or source_width <= 0:
            side = int(math.isqrt(token_count))
            return (side, side) if side * side == token_count else None

        ratio = float(source_width) / float(source_height)
        height = max(1, int(round(math.sqrt(float(token_count) / max(ratio, 1e-8)))))
        width = max(1, int(round(float(token_count) / float(height))))
        if height * width == token_count:
            return height, width

        candidates: list[tuple[float, int, int]] = []
        root = int(math.isqrt(token_count))
        for h in range(1, root + 1):
            if token_count % h != 0:
                continue
            w = token_count // h
            candidates.append((abs((float(w) / float(h)) - ratio), h, w))
            candidates.append((abs((float(h) / float(w)) - ratio), w, h))
        if not candidates:
            return None
        _, best_h, best_w = min(candidates, key=lambda item: item[0])
        return best_h, best_w

    def _prepare_front_geometry(
        self,
        *,
        context: dict[str, Any],
        query_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[tuple[int, int]]]:
        front_bev_xy = context.get("front_bev_xy")
        if not torch.is_tensor(front_bev_xy):
            return None, None, None
        if front_bev_xy.ndim == 3 and front_bev_xy.shape[-1] == 2:
            if front_bev_xy.shape[1] != query_length:
                return None, None, None
            front_xy = front_bev_xy.to(device=device, dtype=dtype)
            query_mask = self._front_mask_to_bn(context.get("front_ground_valid_mask"))
            if query_mask is not None:
                query_mask = query_mask.to(device=device, dtype=dtype)
                if query_mask.shape[:1] != front_xy.shape[:1] or query_mask.shape[1] != query_length:
                    query_mask = None
                else:
                    query_mask = query_mask > 0.5
            return front_xy, query_mask, self._infer_grid_hw_from_xy(front_xy[0])

        front_xy_bchw = self._front_xy_to_bchw(front_bev_xy)
        if front_xy_bchw is None:
            return None, None, None

        source_height = int(front_xy_bchw.shape[-2])
        source_width = int(front_xy_bchw.shape[-1])
        hw = self._infer_hw_from_tokens(query_length, source_height, source_width)
        if hw is None:
            return None, None, None
        height, width = hw

        front_xy_bchw = front_xy_bchw.to(device=device, dtype=dtype)
        resized_xy = F.interpolate(front_xy_bchw, size=(height, width), mode="bilinear", align_corners=False)
        front_xy = resized_xy.permute(0, 2, 3, 1).reshape(front_xy_bchw.shape[0], height * width, 2)

        front_mask = self._front_mask_to_b1hw(context.get("front_ground_valid_mask"))
        if front_mask is None:
            return front_xy, None, hw
        front_mask = front_mask.to(device=device, dtype=dtype)
        resized_mask = F.interpolate(front_mask, size=(height, width), mode="nearest")
        query_mask = resized_mask.reshape(front_mask.shape[0], height * width) > 0.5
        return front_xy, query_mask, hw

    def _prepare_satellite_xy(
        self,
        *,
        context: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        sat_xy = context.get("sat_bev_coords") if self.use_metric_coords else context.get("sat_xy")
        if not torch.is_tensor(sat_xy):
            sat_xy = context.get("sat_xy")
        if not torch.is_tensor(sat_xy) or sat_xy.ndim != 3 or sat_xy.shape[-1] != 2:
            return None
        return sat_xy.to(device=device, dtype=dtype)

    @staticmethod
    def _normalize_token_mask(
        mask: torch.Tensor,
        *,
        key_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.flatten(2).squeeze(1)
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.reshape(mask.shape[0], -1)
        if mask.ndim != 2 or mask.shape[1] != key_length:
            return None
        return mask.to(device=device, dtype=torch.bool)

    def _prepare_satellite_key_mask(
        self,
        *,
        context: dict[str, Any],
        sat_xy: torch.Tensor,
        key_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        for key in (
            "sat_valid_mask",
            "sat_key_mask",
            "sat_bev_valid_mask",
            "sat_perspective_valid_mask",
            "sat_perspective_uv_valid_mask",
        ):
            mask = context.get(key)
            if torch.is_tensor(mask):
                normalized = self._normalize_token_mask(mask, key_length=key_length, device=device)
                if normalized is not None:
                    return normalized

        sat_perspective_uv = context.get("sat_perspective_uv")
        if torch.is_tensor(sat_perspective_uv) and sat_perspective_uv.shape[:2] == sat_xy.shape[:2]:
            uv = sat_perspective_uv.to(device=device)
            finite = torch.isfinite(uv).all(dim=-1)
            in_bounds = (uv >= -1.0).all(dim=-1) & (uv <= 1.0).all(dim=-1)
            not_sentinel_zero = uv.abs().sum(dim=-1) > 1e-6
            return finite & in_bounds & not_sentinel_zero

        return None

    def _prepare_geometry_pair(
        self,
        *,
        context: dict[str, Any],
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], tuple[int, int]]]:
        front_xy, query_mask, query_hw = self._prepare_front_geometry(
            context=context,
            query_length=query_length,
            device=device,
            dtype=dtype,
        )
        sat_xy = self._prepare_satellite_xy(context=context, device=device, dtype=dtype)
        if front_xy is None or sat_xy is None or query_hw is None:
            return None
        if front_xy.shape[0] != sat_xy.shape[0]:
            return None
        if front_xy.shape[1] != query_length or sat_xy.shape[1] != key_length:
            return None
        key_mask = self._prepare_satellite_key_mask(
            context=context,
            sat_xy=sat_xy,
            key_length=key_length,
            device=device,
        )
        return front_xy, sat_xy, query_mask, key_mask, query_hw

    def _build_geometry_attention_bias(
        self,
        *,
        front_xy: torch.Tensor,
        sat_xy: torch.Tensor,
        dist2: torch.Tensor,
        keep: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        min_value = -10000.0
        if not self.enable_geometry_bias:
            mask = torch.zeros(keep.shape, device=keep.device, dtype=dtype)
            return mask.masked_fill(~keep, min_value)

        front_dir = F.normalize(front_xy.float(), dim=-1, eps=1e-6).to(dtype=front_xy.dtype)
        sat_dir = F.normalize(sat_xy.float(), dim=-1, eps=1e-6).to(dtype=sat_xy.dtype)
        dir_cos = (front_dir.unsqueeze(2) * sat_dir.unsqueeze(1)).sum(dim=-1)

        lambda_dist = self.lambda_dist.to(device=front_xy.device, dtype=front_xy.dtype).clamp_min(0.0)
        lambda_dir = self.lambda_dir.to(device=front_xy.device, dtype=front_xy.dtype)
        geo_bias = -lambda_dist * dist2 + lambda_dir * dir_cos
        masked_bias = torch.full_like(geo_bias, min_value)
        return torch.where(keep, geo_bias, masked_bias).to(dtype=dtype)

    def _build_geometry_attention_mask_from_xy(
        self,
        *,
        context: dict[str, Any],
        front_xy: torch.Tensor,
        sat_xy: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        key_mask: Optional[torch.Tensor],
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        topk = min(self.topk, int(key_length))

        if not self.enable_geometry_bias:
            mask = build_topk_mask(
                front_xy,
                sat_xy,
                topk=topk,
                query_mask=query_mask,
                key_mask=key_mask,
                mask_invalid_queries=self.mask_invalid_queries,
                dtype=dtype,
            )
            return mask[:, None, :, :]

        dist2 = (front_xy.unsqueeze(2) - sat_xy.unsqueeze(1)).pow(2).sum(dim=-1)
        dist2_for_topk = dist2
        if key_mask is not None and key_mask.shape == sat_xy.shape[:2]:
            key_mask = key_mask.to(device=device, dtype=torch.bool)
            dist2_for_topk = dist2.masked_fill(~key_mask.unsqueeze(1), torch.inf)

        nearest = torch.topk(dist2_for_topk, k=topk, dim=-1, largest=False).indices
        keep = torch.zeros(dist2.shape, device=device, dtype=torch.bool)
        keep.scatter_(dim=-1, index=nearest, value=True)

        if self.mask_invalid_queries and query_mask is not None:
            keep = keep | (~query_mask).unsqueeze(-1)

        if key_mask is not None and key_mask.shape == sat_xy.shape[:2]:
            keep = keep & key_mask.to(device=device, dtype=torch.bool).unsqueeze(1)

        bias = self._build_geometry_attention_bias(
            front_xy=front_xy,
            sat_xy=sat_xy,
            dist2=dist2,
            keep=keep,
            dtype=dtype,
        )
        return bias[:, None, :, :]

    def _build_geometry_attention_mask(
        self,
        *,
        context: dict[str, Any],
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        geometry = self._prepare_geometry_pair(
            context=context,
            query_length=query_length,
            key_length=key_length,
            device=device,
            dtype=dtype,
        )
        if geometry is None:
            return None
        front_xy, sat_xy, query_mask, key_mask, _ = geometry
        return self._build_geometry_attention_mask_from_xy(
            context=context,
            front_xy=front_xy,
            sat_xy=sat_xy,
            query_mask=query_mask,
            key_mask=key_mask,
            key_length=key_length,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _infer_grid_hw_from_xy(xy: torch.Tensor) -> Optional[tuple[int, int]]:
        if xy.ndim != 2 or xy.shape[-1] != 2:
            return None
        token_count = int(xy.shape[0])
        if token_count <= 0:
            return None
        rounded = torch.round(xy.detach().float() * 10000.0) / 10000.0
        unique_x = torch.unique(rounded[:, 0])
        unique_y = torch.unique(rounded[:, 1])
        if int(unique_x.numel()) * int(unique_y.numel()) == token_count:
            return int(unique_y.numel()), int(unique_x.numel())
        side = int(math.isqrt(token_count))
        if side * side == token_count:
            return side, side
        return None

    @staticmethod
    def _merge_attention_masks(
        attention_mask: Optional[torch.Tensor],
        geometry_mask: torch.Tensor,
    ) -> torch.Tensor:
        if attention_mask is None:
            return geometry_mask
        return attention_mask + geometry_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)

    def _maybe_capture_attention_debug(
        self,
        *,
        context: dict[str, Any],
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        key_length: int,
        head_dim: int,
    ) -> None:
        debug = context.get("attention_debug")
        if not isinstance(debug, dict) or not debug.get("enabled", False):
            return

        layers = debug.get("layers")
        if layers and self.module_name not in layers and self.site not in layers:
            return

        storage = debug.get("storage")
        if not isinstance(storage, dict):
            return

        front_xy, query_mask, query_hw = self._prepare_front_geometry(
            context=context,
            query_length=query_length,
            device=query.device,
            dtype=query.dtype,
        )
        sat_xy = self._prepare_satellite_xy(context=context, device=query.device, dtype=query.dtype)
        if front_xy is None or sat_xy is None or query_hw is None:
            return
        if front_xy.shape[0] != sat_xy.shape[0] or sat_xy.shape[1] != key_length:
            return

        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores / math.sqrt(float(head_dim))
        if attention_mask is not None:
            scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)
        probs = torch.softmax(scores, dim=-1).mean(dim=1)

        timestep = context.get("timestep")
        if torch.is_tensor(timestep):
            timestep_value: Any = timestep.detach().cpu()
        else:
            timestep_value = timestep

        storage[self.module_name] = {
            "site": self.site,
            "module_name": self.module_name,
            "attention": probs.detach().float().cpu(),
            "front_xy": front_xy.detach().float().cpu(),
            "query_mask": query_mask.detach().cpu() if torch.is_tensor(query_mask) else None,
            "sat_xy": sat_xy.detach().float().cpu(),
            "query_hw": query_hw,
            "sat_hw": self._infer_grid_hw_from_xy(sat_xy[0].detach().cpu()),
            "topk": self.topk,
            "timestep": timestep_value,
        }

    @staticmethod
    def _manual_scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        head_dim = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(head_dim))
        if attention_mask is not None:
            scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, value)

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        def _call_without_cudnn() -> torch.Tensor:
            if torch_sdpa_kernel is not None:
                with torch_sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
                ):
                    return F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )

            if hasattr(torch.backends.cuda, "sdp_kernel"):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True,
                    enable_cudnn=False,
                ):
                    return F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )

            return self._manual_scaled_dot_product_attention(query, key, value, attention_mask)

        try:
            return _call_without_cudnn()
        except RuntimeError as exc:
            message = str(exc).lower()
            cudnn_plan_error = "cudnn frontend" in message or "no execution plans support the graph" in message
            if not cudnn_plan_error:
                raise

            return self._manual_scaled_dot_product_attention(query, key, value, attention_mask)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = None

        key_length = hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if is_cross_attention:
            context = self.context_provider()
            if context is not None:
                geometry = self._prepare_geometry_pair(
                    context=context,
                    query_length=query.shape[2],
                    key_length=key.shape[2],
                    device=query.device,
                    dtype=query.dtype,
                )
                if geometry is not None:
                    front_xy, sat_xy, query_mask, key_mask, _ = geometry
                    geometry_mask = self._build_geometry_attention_mask_from_xy(
                        context=context,
                        front_xy=front_xy,
                        sat_xy=sat_xy,
                        query_mask=query_mask,
                        key_mask=key_mask,
                        key_length=key.shape[2],
                        device=query.device,
                        dtype=query.dtype,
                    )
                    attention_mask = self._merge_attention_masks(attention_mask, geometry_mask)
                elif not self.fallback_to_unmasked:
                    raise ValueError(f"Could not build geometry mask for attention site '{self.site}'")
                self._maybe_capture_attention_debug(
                    context=context,
                    query=query,
                    key=key,
                    attention_mask=attention_mask,
                    query_length=query.shape[2],
                    key_length=key.shape[2],
                    head_dim=head_dim,
                )

        hidden_states = self._scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def apply_geometry_masked_attn_processors(
    unet: torch.nn.Module,
    *,
    sites: Sequence[str],
    context_provider: GeometryContextProvider,
    topk: int = 32,
    mask_invalid_queries: bool = True,
    fallback_to_unmasked: bool = True,
    use_metric_coords: bool = False,
    enable_geometry_bias: bool = False,
    geometry_bias_type: str = "dist_dir",
    lambda_dist: float = 2.0,
    lambda_dir: float = 0.5,
    learnable_geometry_bias: bool = False,
) -> list[str]:
    """
    Replace selected native attn2 processors and return the module names changed.
    """
    site_set = {str(site) for site in sites}
    changed: list[str] = []
    for name, module in unet.named_modules():
        if not name.endswith(".attn2"):
            continue
        parts = name.split(".")
        if len(parts) < 2:
            continue
        site = None
        if parts[0] == "down_blocks" and len(parts) > 1:
            site = f"down{parts[1]}"
        elif parts[0] == "up_blocks" and len(parts) > 1:
            site = f"up{parts[1]}"
        elif parts[0] == "mid_block":
            site = "mid"
        if site not in site_set:
            continue
        if not hasattr(module, "set_processor"):
            continue
        module.set_processor(
            GeometryMaskedAttnProcessor2_0(
                site=site,
                module_name=name,
                context_provider=context_provider,
                topk=topk,
                mask_invalid_queries=mask_invalid_queries,
                fallback_to_unmasked=fallback_to_unmasked,
                use_metric_coords=use_metric_coords,
                enable_geometry_bias=enable_geometry_bias,
                geometry_bias_type=geometry_bias_type,
                lambda_dist=lambda_dist,
                lambda_dir=lambda_dir,
                learnable_geometry_bias=learnable_geometry_bias,
            )
        )
        changed.append(name)
    return changed
