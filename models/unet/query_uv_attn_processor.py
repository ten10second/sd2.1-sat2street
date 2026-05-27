"""Cross-attention processor with logit-level query/satellite geometry addressing."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        query_uv_enabled: bool = False,
        geometry_bias_enabled: bool = False,
        geometry_score_enabled: bool = False,
        geometry_score_dim: int = 64,
        geometry_score_num_freqs: int = 6,
        geometry_score_gate_init: float = 1.0,
        geometry_score_layers: Optional[Tuple[str, ...]] = None,
        geometry_score_max_query_tokens: Optional[int] = None,
        geometry_score_mode: str = "geometry_first_semantic_refine",
        candidate_radius: float = 0.35,
        candidate_min_k: int = 16,
        candidate_invalid_penalty: float = -1e4,
        semantic_score_dim: int = 64,
        semantic_score_alpha: float = 0.25,
        geometry_bias_scale: float = 2.0,
        geometry_invalid_penalty: float = -1e4,
        gate_init: float = 0.0,
        layer_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        del query_uv_enabled, geometry_bias_enabled, geometry_bias_scale
        del geometry_invalid_penalty, gate_init
        self.query_dim = int(query_dim)
        self.query_uv_enabled = False
        self.geometry_bias_enabled = False
        self.geometry_score_enabled = bool(geometry_score_enabled)
        self.geometry_score_dim = int(geometry_score_dim)
        self.geometry_score_num_freqs = int(geometry_score_num_freqs)
        self.geometry_score_mode = str(geometry_score_mode)
        self.candidate_radius = float(candidate_radius)
        self.candidate_min_k = max(1, int(candidate_min_k))
        self.candidate_invalid_penalty = float(candidate_invalid_penalty)
        self.semantic_score_dim = int(semantic_score_dim)
        self.semantic_score_alpha = float(semantic_score_alpha)
        self.semantic_score_runtime_alpha = float(semantic_score_alpha)
        self.geometry_score_layers = (
            None
            if geometry_score_layers is None
            else tuple(str(layer) for layer in geometry_score_layers)
        )
        self.geometry_score_max_query_tokens = (
            None
            if geometry_score_max_query_tokens is None
            else int(geometry_score_max_query_tokens)
        )
        self.layer_name = layer_name
        self.query_uv_encoder = None
        self.query_uv_gate = None
        self.geometry_score_runtime_scale = 1.0

        if self.geometry_score_enabled:
            fourier_dim = 4 * self.geometry_score_num_freqs
            self.geometry_score_proj = nn.Linear(fourier_dim, self.geometry_score_dim, bias=False)
            nn.init.orthogonal_(self.geometry_score_proj.weight)
            self.geometry_score_gate = nn.Parameter(torch.tensor(float(geometry_score_gate_init)))
            self.semantic_query_proj = nn.Linear(self.query_dim, self.semantic_score_dim, bias=False)
            self.semantic_key_proj = nn.Linear(self.query_dim, self.semantic_score_dim, bias=False)
            nn.init.orthogonal_(self.semantic_query_proj.weight)
            nn.init.orthogonal_(self.semantic_key_proj.weight)
        else:
            self.geometry_score_proj = None
            self.semantic_query_proj = None
            self.semantic_key_proj = None
            self.register_parameter("geometry_score_gate", None)

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
        attention_alignment: Optional[Dict[str, Any]] = None,
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
            attention_alignment=attention_alignment,
            **kwargs,
        )

    def set_geometry_score_runtime_scale(self, scale: float) -> None:
        self.geometry_score_runtime_scale = float(scale)

    def set_semantic_score_runtime_alpha(self, alpha: float) -> None:
        self.semantic_score_runtime_alpha = float(alpha)

    def _uses_geometry_first_mode(self) -> bool:
        return self.geometry_score_mode == "geometry_first_semantic_refine"

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

    def _should_collect_attention_alignment(
        self,
        attention_alignment: Optional[Dict[str, Any]],
        *,
        query_tokens: int,
        is_cross_attention: bool,
    ) -> bool:
        if not is_cross_attention or not isinstance(attention_alignment, dict):
            return False
        if not bool(attention_alignment.get("enabled", False)):
            return False
        if self.layer_name is None:
            return False

        layers = attention_alignment.get("layers")
        if layers is not None and self.layer_name not in set(str(layer) for layer in layers):
            return False

        max_query_tokens = attention_alignment.get("max_query_tokens")
        if max_query_tokens is not None and int(query_tokens) > int(max_query_tokens):
            return False
        return True

    def _should_apply_geometry_score(
        self,
        *,
        query_tokens: int,
        is_cross_attention: bool,
    ) -> bool:
        if not is_cross_attention or not self.geometry_score_enabled:
            return False
        if self.layer_name is None:
            return False
        if self.geometry_score_layers is not None and self.layer_name not in self.geometry_score_layers:
            return False
        if (
            self.geometry_score_max_query_tokens is not None
            and int(query_tokens) > int(self.geometry_score_max_query_tokens)
        ):
            return False
        return True

    @staticmethod
    def _fourier_encode_geometry(uv: torch.Tensor, num_freqs: int) -> torch.Tensor:
        freqs = (2.0 ** torch.arange(num_freqs, dtype=uv.dtype, device=uv.device)) * torch.pi
        uv_exp = uv.unsqueeze(-1) * freqs
        enc = torch.cat([torch.sin(uv_exp), torch.cos(uv_exp)], dim=-1)
        return enc.flatten(-2)

    def _build_geometry_score_bias(
        self,
        query_uv: torch.Tensor,
        sat_perspective_uv: Optional[torch.Tensor],
        sat_perspective_valid: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.geometry_score_proj is None or self.geometry_score_gate is None or sat_perspective_uv is None:
            return None, {}

        query_features = self._fourier_encode_geometry(
            query_uv.float(),
            self.geometry_score_num_freqs,
        )
        sat_features = self._fourier_encode_geometry(
            sat_perspective_uv.float(),
            self.geometry_score_num_freqs,
        )
        query_geo = self.geometry_score_proj(query_features.to(dtype=self.geometry_score_proj.weight.dtype))
        sat_geo = self.geometry_score_proj(sat_features.to(dtype=self.geometry_score_proj.weight.dtype))
        query_geo = F.normalize(query_geo.float(), dim=-1)
        sat_geo = F.normalize(sat_geo.float(), dim=-1)

        if sat_perspective_valid is not None:
            sat_valid = sat_perspective_valid.to(device=sat_geo.device, dtype=torch.bool)
            sat_geo = sat_geo * sat_valid.unsqueeze(-1).to(dtype=sat_geo.dtype)

        raw_score = torch.matmul(query_geo, sat_geo.transpose(-1, -2))
        runtime_scale = torch.tensor(
            float(getattr(self, "geometry_score_runtime_scale", 1.0)),
            device=raw_score.device,
            dtype=torch.float32,
        )
        gate = self.geometry_score_gate.float() * runtime_scale
        score = (gate * raw_score).to(dtype=dtype)
        metrics = {
            "geometry_score_gate": gate.detach(),
            "geometry_score_gate_raw": self.geometry_score_gate.detach().float(),
            "geometry_score_runtime_scale": runtime_scale.detach(),
            "geometry_score_raw_std": raw_score.detach().float().std(),
            "geometry_score_bias_std": score.detach().float().std(),
        }
        return score, metrics

    def _build_semantic_score_bias(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.semantic_query_proj is None or self.semantic_key_proj is None:
            return None, {}

        query_sem = self.semantic_query_proj(query_states.to(dtype=self.semantic_query_proj.weight.dtype))
        key_sem = self.semantic_key_proj(key_states.to(dtype=self.semantic_key_proj.weight.dtype))
        query_sem = F.normalize(query_sem.float(), dim=-1)
        key_sem = F.normalize(key_sem.float(), dim=-1)
        raw_score = torch.matmul(query_sem, key_sem.transpose(-1, -2))
        alpha = torch.tensor(
            float(getattr(self, "semantic_score_runtime_alpha", self.semantic_score_alpha)),
            device=raw_score.device,
            dtype=torch.float32,
        )
        score = (alpha * raw_score).to(dtype=dtype)
        metrics = {
            "semantic_score_alpha": alpha.detach(),
            "semantic_score_raw_std": raw_score.detach().float().std(),
            "semantic_score_bias_std": score.detach().float().std(),
        }
        return score, metrics

    def _build_candidate_mask(
        self,
        query_uv: torch.Tensor,
        sat_perspective_uv: Optional[torch.Tensor],
        sat_perspective_valid: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if sat_perspective_uv is None:
            return None, None, {}

        sat_uv = sat_perspective_uv.float()
        query_uv_f = query_uv.float()
        sat_valid = (
            sat_perspective_valid.to(device=sat_uv.device, dtype=torch.bool)
            if sat_perspective_valid is not None
            else torch.ones(sat_uv.shape[:2], device=sat_uv.device, dtype=torch.bool)
        )
        sat_valid = sat_valid & torch.isfinite(sat_uv).all(dim=-1)

        dist2 = (query_uv_f[:, :, None, :] - sat_uv[:, None, :, :]).pow(2).sum(dim=-1)
        valid_dist2 = dist2.masked_fill(~sat_valid[:, None, :], torch.finfo(dist2.dtype).max)
        window_candidate = sat_valid[:, None, :] & (valid_dist2 <= self.candidate_radius * self.candidate_radius)
        window_count = window_candidate.sum(dim=-1)
        valid_count = sat_valid.sum(dim=-1, keepdim=True)

        k = min(self.candidate_min_k, int(sat_uv.shape[1]))
        nearest_indices = torch.topk(valid_dist2, k=k, dim=-1, largest=False).indices
        nearest_candidate = torch.zeros_like(window_candidate)
        nearest_candidate.scatter_(-1, nearest_indices, True)
        nearest_candidate = nearest_candidate & sat_valid[:, None, :]

        fallback = (window_count < self.candidate_min_k) & (valid_count > 0)
        candidate = torch.where(fallback.unsqueeze(-1), nearest_candidate, window_candidate)
        additive_mask = torch.full(
            candidate.shape,
            self.candidate_invalid_penalty,
            device=query_uv.device,
            dtype=dtype,
        )
        additive_mask = additive_mask.masked_fill(candidate, 0.0)
        metrics = {
            "candidate_count_mean": candidate.sum(dim=-1).float().mean().detach(),
            "window_candidate_count_mean": window_count.float().mean().detach(),
            "window_fallback_ratio": fallback.float().mean().detach(),
            "candidate_valid_query_ratio": candidate.any(dim=-1).float().mean().detach(),
        }
        return additive_mask, candidate, metrics

    @staticmethod
    def _infer_square_hw(token_count: int) -> Optional[Tuple[int, int]]:
        side = int(math.isqrt(int(token_count)))
        if side * side != int(token_count):
            return None
        return side, side

    @staticmethod
    def _manual_scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        content_scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        content_scores = content_scores * (1.0 / math.sqrt(float(query.shape[-1])))
        scores = content_scores
        if attention_mask is not None:
            scores = scores + attention_mask.float()
        attention_probs = torch.softmax(scores, dim=-1)
        hidden_states = torch.matmul(attention_probs.to(dtype=value.dtype), value)
        return hidden_states, attention_probs, scores, content_scores

    @staticmethod
    def _manual_attention_from_scores(
        value: torch.Tensor,
        attention_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_probs = torch.softmax(attention_scores.float(), dim=-1)
        hidden_states = torch.matmul(attention_probs.to(dtype=value.dtype), value)
        return hidden_states, attention_probs

    @staticmethod
    def _raw_content_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        content_scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        return content_scores * (1.0 / math.sqrt(float(query.shape[-1])))

    @staticmethod
    def _reshape_sliced_attention_mask(
        attention_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        heads: int,
        query_tokens: int,
        key_tokens: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.shape[0] != batch_size * heads or attention_mask.shape[-1] != key_tokens:
            raise ValueError(
                "sliced attention mask must be [B*heads,Q,K] before alignment collection, "
                f"got {tuple(attention_mask.shape)} for B={batch_size}, heads={heads}, K={key_tokens}"
            )
        mask_query_tokens = attention_mask.shape[1]
        if mask_query_tokens not in (1, query_tokens):
            raise ValueError(
                "sliced attention mask query dimension must be 1 or match query tokens, "
                f"got {mask_query_tokens} vs {query_tokens}"
            )
        return attention_mask.view(batch_size, heads, mask_query_tokens, key_tokens)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _finite_std(values: torch.Tensor) -> torch.Tensor:
        finite = values[torch.isfinite(values)]
        if finite.numel() <= 1:
            return values.detach().float().new_tensor(0.0)
        return finite.float().std(unbiased=False)

    @staticmethod
    def _finite_abs_mean(values: torch.Tensor) -> torch.Tensor:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return values.detach().float().new_tensor(0.0)
        return finite.float().abs().mean()

    @staticmethod
    def _mean_top_logit_gap(scores: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        if scores.shape[-1] < 2:
            return scores.detach().new_tensor(0.0)
        finite_scores = scores.masked_fill(~torch.isfinite(scores), -torch.finfo(scores.dtype).max)
        top2 = torch.topk(finite_scores, k=2, dim=-1).values
        gap = top2[..., 0] - top2[..., 1]
        return gap[torch.isfinite(gap)].mean() if torch.isfinite(gap).any() else scores.detach().new_tensor(0.0)

    def _record_attention_alignment(
        self,
        *,
        attention_probs: torch.Tensor,
        attention_scores: Optional[torch.Tensor],
        content_scores: Optional[torch.Tensor],
        geometry_bias: Optional[torch.Tensor],
        semantic_scores: Optional[torch.Tensor],
        candidate_mask: Optional[torch.Tensor],
        geometry_only_scores: Optional[torch.Tensor],
        semantic_only_scores: Optional[torch.Tensor],
        query_uv: torch.Tensor,
        sat_perspective_uv: Optional[torch.Tensor],
        sat_perspective_valid: Optional[torch.Tensor],
        query_base_hw: Optional[Tuple[int, int]],
        attention_alignment: Optional[Dict[str, Any]],
        query_pe_metrics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        if not isinstance(attention_alignment, dict) or sat_perspective_uv is None:
            return

        losses = attention_alignment.setdefault("losses", [])
        metrics = attention_alignment.setdefault("metrics", [])
        debug_storage = attention_alignment.get("debug_storage")

        sat_uv = sat_perspective_uv.float()
        query_uv_f = query_uv.float()
        sat_valid = (
            sat_perspective_valid.bool()
            if sat_perspective_valid is not None
            else torch.ones(sat_uv.shape[:2], device=sat_uv.device, dtype=torch.bool)
        )
        sat_valid = sat_valid & torch.isfinite(sat_uv).all(dim=-1)

        valid_radius = float(attention_alignment.get("valid_radius", 0.25))
        invalid_attention_weight = float(attention_alignment.get("invalid_attention_weight", 0.1))

        dist2 = (query_uv_f[:, :, None, :] - sat_uv[:, None, :, :]).pow(2).sum(dim=-1)
        dist2 = dist2.masked_fill(~sat_valid[:, None, :], torch.finfo(dist2.dtype).max)
        nearest_dist2 = dist2.min(dim=-1).values
        has_valid_sat = sat_valid.any(dim=1, keepdim=True)
        query_mask = has_valid_sat & (nearest_dist2 <= valid_radius * valid_radius)
        target_mask = sat_valid[:, None, :] & (dist2 <= valid_radius * valid_radius)

        attention_mean = attention_probs.float().mean(dim=1)
        valid_weights = attention_mean * sat_valid.to(dtype=attention_mean.dtype).unsqueeze(1)
        valid_mass = valid_weights.sum(dim=-1)
        target_mass = (attention_mean * target_mask.to(dtype=attention_mean.dtype)).sum(dim=-1)
        target_token_fraction = target_mask.to(dtype=attention_mean.dtype).sum(dim=-1) / float(attention_mean.shape[-1])
        target_lift = target_mass / target_token_fraction.clamp_min(1e-6)
        nearest_indices = dist2.argmin(dim=-1)
        nearest_mass = torch.gather(attention_mean, dim=-1, index=nearest_indices.unsqueeze(-1)).squeeze(-1)
        predicted_uv = torch.matmul(valid_weights, sat_uv) / valid_mass.clamp_min(1e-6).unsqueeze(-1)
        error2 = (predicted_uv - query_uv_f).pow(2).sum(dim=-1)
        loss_map = error2 + invalid_attention_weight * (1.0 - valid_mass).clamp_min(0.0).pow(2)

        loss = self._masked_mean(loss_map, query_mask)
        mean_error = self._masked_mean(error2.sqrt(), query_mask)
        mean_valid_mass = self._masked_mean(valid_mass, query_mask)
        mean_target_mass = self._masked_mean(target_mass, query_mask)
        mean_target_fraction = self._masked_mean(target_token_fraction, query_mask)
        mean_target_lift = self._masked_mean(target_lift, query_mask)
        mean_nearest_mass = self._masked_mean(nearest_mass, query_mask)

        metric_payload: Dict[str, Any] = {
            "layer": self.layer_name or "unknown",
            "loss": loss.detach(),
            "mean_error": mean_error.detach(),
            "valid_query_ratio": query_mask.float().mean().detach(),
            "valid_attention_mass": mean_valid_mass.detach(),
            "target_attention_mass": mean_target_mass.detach(),
            "target_token_fraction": mean_target_fraction.detach(),
            "target_attention_lift": mean_target_lift.detach(),
            "target_attention_lift_mixed": mean_target_lift.detach(),
            "nearest_attention_mass": mean_nearest_mass.detach(),
        }

        def score_ablation_metrics(prefix: str, scores: torch.Tensor) -> None:
            probs = torch.softmax(scores.float(), dim=-1).mean(dim=1)
            target_mass_for_scores = (probs * target_mask.to(dtype=probs.dtype)).sum(dim=-1)
            lift_for_scores = target_mass_for_scores / target_token_fraction.clamp_min(1e-6)
            scores_mean = scores.float().mean(dim=1)
            near_weights = target_mask.to(dtype=scores_mean.dtype)
            far_mask = sat_valid[:, None, :] & ~target_mask
            far_weights = far_mask.to(dtype=scores_mean.dtype)
            near_score = (scores_mean * near_weights).sum(dim=-1) / near_weights.sum(dim=-1).clamp_min(1.0)
            far_score = (scores_mean * far_weights).sum(dim=-1) / far_weights.sum(dim=-1).clamp_min(1.0)
            gap_mask = query_mask & far_mask.any(dim=-1)
            metric_payload[f"target_attention_lift_{prefix}"] = self._masked_mean(
                lift_for_scores,
                query_mask,
            ).detach()
            metric_payload[f"target_logit_gap_{prefix}"] = self._masked_mean(
                near_score - far_score,
                gap_mask,
            ).detach()

        if candidate_mask is not None:
            candidate_bool = candidate_mask.to(device=target_mask.device, dtype=torch.bool)
            target_counts = target_mask.to(dtype=torch.float32).sum(dim=-1)
            covered_counts = (target_mask & candidate_bool).to(dtype=torch.float32).sum(dim=-1)
            candidate_recall = covered_counts / target_counts.clamp_min(1.0)
            metric_payload["candidate_recall"] = self._masked_mean(candidate_recall, query_mask).detach()
            metric_payload["candidate_count_mean"] = candidate_bool.sum(dim=-1).float().mean().detach()

        if content_scores is not None:
            content_scores_f = content_scores.float()
            content_probs = torch.softmax(content_scores_f, dim=-1)
            content_attention_mean = content_probs.mean(dim=1)
            content_valid_weights = content_attention_mean * sat_valid.to(dtype=content_attention_mean.dtype).unsqueeze(1)
            content_valid_mass = content_valid_weights.sum(dim=-1)
            content_target_mass = (
                content_attention_mean * target_mask.to(dtype=content_attention_mean.dtype)
            ).sum(dim=-1)
            content_target_lift = content_target_mass / target_token_fraction.clamp_min(1e-6)
            content_nearest_mass = torch.gather(
                content_attention_mean,
                dim=-1,
                index=nearest_indices.unsqueeze(-1),
            ).squeeze(-1)
            content_std = self._finite_std(content_scores_f)
            content_abs_mean = self._finite_abs_mean(content_scores_f)
            content_top_gap = self._mean_top_logit_gap(content_scores_f)

            eps = torch.finfo(content_probs.dtype).eps
            geom_probs = attention_probs.float().clamp_min(eps)
            kl_per_query = (geom_probs * (geom_probs.log() - content_probs.clamp_min(eps).log())).sum(dim=-1).mean(dim=1)

            metric_payload.update(
                {
                    "content_logits_std": content_std.detach(),
                    "content_logits_abs_mean": content_abs_mean.detach(),
                    "content_logits_top_gap": content_top_gap.detach(),
                    "raw_content_qk_std": content_std.detach(),
                    "raw_content_qk_abs_mean": content_abs_mean.detach(),
                    "raw_content_qk_top_gap": content_top_gap.detach(),
                    "valid_attention_mass_without_geometry": self._masked_mean(
                        content_valid_mass,
                        query_mask,
                    ).detach(),
                    "target_attention_mass_without_geometry": self._masked_mean(
                        content_target_mass,
                        query_mask,
                    ).detach(),
                    "target_attention_lift_without_geometry": self._masked_mean(
                        content_target_lift,
                        query_mask,
                    ).detach(),
                    "target_attention_lift_geometry_delta": self._masked_mean(
                        target_lift - content_target_lift,
                        query_mask,
                    ).detach(),
                    "nearest_attention_mass_without_geometry": self._masked_mean(
                        content_nearest_mass,
                        query_mask,
                    ).detach(),
                    "attention_geometry_kl": self._masked_mean(
                        kl_per_query,
                        query_mask,
                    ).detach(),
                }
            )

        if geometry_bias is not None:
            geometry_bias_f = geometry_bias.float()
            if geometry_bias_f.shape[1] == 1 and attention_probs.shape[1] != 1:
                geometry_bias_f = geometry_bias_f.expand(-1, attention_probs.shape[1], -1, -1)
            geometry_std = self._finite_std(geometry_bias_f)
            geometry_abs_mean = self._finite_abs_mean(geometry_bias_f)
            geometry_top_gap = self._mean_top_logit_gap(geometry_bias_f)
            metric_payload.update(
                {
                    "geometry_bias_std": geometry_std.detach(),
                    "geometry_bias_abs_mean": geometry_abs_mean.detach(),
                    "geometry_bias_top_gap": geometry_top_gap.detach(),
                    "geometry_logits_std": geometry_std.detach(),
                }
            )
            if content_scores is not None:
                content_std = metric_payload["content_logits_std"].float()
                content_abs_mean = metric_payload["content_logits_abs_mean"].float()
                content_top_gap = metric_payload["content_logits_top_gap"].float()
                metric_payload.update(
                    {
                        "geometry_to_content_std_ratio": (
                            geometry_std / content_std.clamp_min(1e-6)
                        ).detach(),
                        "geometry_to_content_abs_ratio": (
                            geometry_abs_mean / content_abs_mean.clamp_min(1e-6)
                        ).detach(),
                        "geometry_to_content_top_gap_ratio": (
                            geometry_top_gap / content_top_gap.clamp_min(1e-6)
                        ).detach(),
                        "raw_content_qk_to_geometry_ratio": (
                            content_std / geometry_std.clamp_min(1e-6)
                        ).detach(),
                    }
                )

        if semantic_scores is not None:
            semantic_scores_f = semantic_scores.float()
            if semantic_scores_f.shape[1] == 1 and attention_probs.shape[1] != 1:
                semantic_scores_f = semantic_scores_f.expand(-1, attention_probs.shape[1], -1, -1)
            semantic_std = self._finite_std(semantic_scores_f)
            semantic_abs_mean = self._finite_abs_mean(semantic_scores_f)
            semantic_top_gap = self._mean_top_logit_gap(semantic_scores_f)
            metric_payload.update(
                {
                    "semantic_logits_std": semantic_std.detach(),
                    "semantic_logits_abs_mean": semantic_abs_mean.detach(),
                    "semantic_logits_top_gap": semantic_top_gap.detach(),
                }
            )
            if geometry_bias is not None:
                geometry_bias_f = geometry_bias.float()
                if geometry_bias_f.shape[1] == 1 and semantic_scores_f.shape[1] != 1:
                    geometry_bias_f = geometry_bias_f.expand(-1, semantic_scores_f.shape[1], -1, -1)
                geometry_std = self._finite_std(geometry_bias_f)
                metric_payload["semantic_to_geometry_ratio"] = (
                    semantic_std / geometry_std.clamp_min(1e-6)
                ).detach()

        if attention_scores is not None:
            scores_mean = attention_scores.float().mean(dim=1)
            near_weights = target_mask.to(dtype=scores_mean.dtype)
            far_mask = sat_valid[:, None, :] & ~target_mask
            far_weights = far_mask.to(dtype=scores_mean.dtype)
            near_score = (scores_mean * near_weights).sum(dim=-1) / near_weights.sum(dim=-1).clamp_min(1.0)
            far_score = (scores_mean * far_weights).sum(dim=-1) / far_weights.sum(dim=-1).clamp_min(1.0)
            gap_mask = query_mask & far_mask.any(dim=-1)
            target_logit_gap = self._masked_mean(near_score - far_score, gap_mask)
            metric_payload["target_logit_gap"] = target_logit_gap.detach()
            metric_payload["target_logit_gap_mixed"] = target_logit_gap.detach()
            if content_scores is not None:
                content_scores_mean = content_scores.float().mean(dim=1)
                content_near_score = (
                    content_scores_mean * near_weights
                ).sum(dim=-1) / near_weights.sum(dim=-1).clamp_min(1.0)
                content_far_score = (
                    content_scores_mean * far_weights
                ).sum(dim=-1) / far_weights.sum(dim=-1).clamp_min(1.0)
                content_logit_gap = self._masked_mean(
                    content_near_score - content_far_score,
                    gap_mask,
                )
                metric_payload["target_logit_gap_without_geometry"] = content_logit_gap.detach()
                metric_payload["target_logit_gap_geometry_delta"] = (
                    target_logit_gap - content_logit_gap
                ).detach()

        if geometry_only_scores is not None:
            score_ablation_metrics("geometry_only", geometry_only_scores)
        if semantic_only_scores is not None:
            score_ablation_metrics("semantic_only", semantic_only_scores)

        if query_pe_metrics:
            for name, value in query_pe_metrics.items():
                if torch.is_tensor(value):
                    metric_payload[name] = value.detach()

        losses.append(loss)
        metrics.append(metric_payload)

        if isinstance(debug_storage, dict) and self.layer_name is not None:
            query_hw = (
                infer_spatial_hw(query_uv.shape[1], query_base_hw)
                if query_base_hw is not None
                else (1, query_uv.shape[1])
            )
            debug_storage[self.layer_name] = {
                "attention": attention_mean.detach().cpu(),
                "front_xy": query_uv_f.detach().cpu(),
                "sat_xy": sat_uv.detach().cpu(),
                "query_mask": query_mask.detach().cpu(),
                "target_mask": target_mask.detach().cpu(),
                "valid_radius": valid_radius,
                "query_hw": query_hw,
                "sat_hw": self._infer_square_hw(sat_uv.shape[1]),
            }

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
        attention_alignment: Optional[Dict[str, Any]] = None,
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

        is_cross_attention = encoder_hidden_states is not None
        collect_alignment = self._should_collect_attention_alignment(
            attention_alignment,
            query_tokens=hidden_states.shape[1],
            is_cross_attention=is_cross_attention,
        )
        apply_geometry_score = self._should_apply_geometry_score(
            query_tokens=hidden_states.shape[1],
            is_cross_attention=is_cross_attention,
        )
        query = attn.to_q(hidden_states)
        query_pe_metrics: Dict[str, torch.Tensor] = {}
        if collect_alignment:
            query_norm = query.float().norm(dim=-1).mean()
            query_pe_metrics["query_content_norm"] = query_norm
        query_uv_tensor = None
        if is_cross_attention and (
            apply_geometry_score
            or collect_alignment
        ):
            query_uv_tensor = self._resolve_query_uv(
                hidden_states,
                batch_size=hidden_states.shape[0],
                query_base_hw=query_base_hw,
                query_uv=query_uv,
            ).to(dtype=query.dtype)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query_states_for_semantic = query
        key = attn.to_k(encoder_hidden_states)
        key_states_for_semantic = key
        value = attn.to_v(encoder_hidden_states)
        if collect_alignment:
            query_pe_metrics["key_content_norm"] = key.float().norm(dim=-1).mean()

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(hidden_states.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(hidden_states.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(hidden_states.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        sat_uv = None
        sat_valid = None
        geometry_bias_for_metrics = None
        semantic_scores_for_metrics = None
        candidate_bool_for_metrics = None
        geometry_only_scores_for_metrics = None
        semantic_only_scores_for_metrics = None
        if (apply_geometry_score or collect_alignment) and is_cross_attention:
            sat_uv, sat_valid = self._resolve_sat_perspective_uv(
                hidden_states,
                batch_size=hidden_states.shape[0],
                key_length=key.shape[2],
                sat_perspective_uv=sat_perspective_uv,
                sat_perspective_valid=sat_perspective_valid,
            )
        use_geometry_first = apply_geometry_score and self._uses_geometry_first_mode()
        if apply_geometry_score and query_uv_tensor is not None and not use_geometry_first:
            geometry_score_bias, geometry_score_metrics = self._build_geometry_score_bias(
                query_uv_tensor,
                sat_uv,
                sat_valid,
                dtype=query.dtype,
            )
            if collect_alignment:
                query_pe_metrics.update(geometry_score_metrics)
            if geometry_score_bias is not None:
                geometry_score_bias = geometry_score_bias.unsqueeze(1)
                geometry_bias_for_metrics = geometry_score_bias
                attention_mask = (
                    geometry_score_bias
                    if attention_mask is None
                    else attention_mask + geometry_score_bias
                )
        if use_geometry_first and query_uv_tensor is not None:
            geometry_score_bias, geometry_score_metrics = self._build_geometry_score_bias(
                query_uv_tensor,
                sat_uv,
                sat_valid,
                dtype=query.dtype,
            )
            semantic_score_bias, semantic_score_metrics = self._build_semantic_score_bias(
                query_states_for_semantic,
                key_states_for_semantic,
                dtype=query.dtype,
            )
            candidate_mask, candidate_bool, candidate_metrics = self._build_candidate_mask(
                query_uv_tensor,
                sat_uv,
                sat_valid,
                dtype=query.dtype,
            )
            query_pe_metrics.update(geometry_score_metrics)
            query_pe_metrics.update(semantic_score_metrics)
            query_pe_metrics.update(candidate_metrics)

            content_scores = self._raw_content_scores(query, key)
            if candidate_mask is None:
                active_scores = torch.zeros_like(content_scores)
            else:
                active_scores = candidate_mask.unsqueeze(1).float()
            if geometry_score_bias is not None:
                geometry_bias_for_metrics = geometry_score_bias.unsqueeze(1)
                active_scores = active_scores + geometry_bias_for_metrics.float()
            if semantic_score_bias is not None:
                semantic_scores_for_metrics = semantic_score_bias.unsqueeze(1)
                active_scores = active_scores + semantic_scores_for_metrics.float()
            if attention_mask is not None:
                active_scores = active_scores + attention_mask.float()

            geometry_only_scores_for_metrics = (
                (candidate_mask.unsqueeze(1).float() if candidate_mask is not None else torch.zeros_like(content_scores))
                + (geometry_bias_for_metrics.float() if geometry_bias_for_metrics is not None else 0.0)
                + (attention_mask.float() if attention_mask is not None else 0.0)
            )
            semantic_only_scores_for_metrics = (
                (candidate_mask.unsqueeze(1).float() if candidate_mask is not None else torch.zeros_like(content_scores))
                + (semantic_scores_for_metrics.float() if semantic_scores_for_metrics is not None else 0.0)
                + (attention_mask.float() if attention_mask is not None else 0.0)
            )
            candidate_bool_for_metrics = candidate_bool
            hidden_states, attention_probs = self._manual_attention_from_scores(value, active_scores)
            attention_scores = active_scores
            if collect_alignment and query_uv_tensor is not None:
                self._record_attention_alignment(
                    attention_probs=attention_probs,
                    attention_scores=attention_scores,
                    content_scores=content_scores,
                    geometry_bias=geometry_bias_for_metrics,
                    semantic_scores=semantic_scores_for_metrics,
                    candidate_mask=candidate_bool_for_metrics,
                    geometry_only_scores=geometry_only_scores_for_metrics,
                    semantic_only_scores=semantic_only_scores_for_metrics,
                    query_uv=query_uv_tensor,
                    sat_perspective_uv=sat_uv,
                    sat_perspective_valid=sat_valid,
                    query_base_hw=query_base_hw,
                    attention_alignment=attention_alignment,
                    query_pe_metrics=query_pe_metrics,
                )
        elif collect_alignment:
            hidden_states, attention_probs, attention_scores, content_scores = self._manual_scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask,
            )
            if query_uv_tensor is not None:
                self._record_attention_alignment(
                    attention_probs=attention_probs,
                    attention_scores=attention_scores,
                    content_scores=content_scores,
                    geometry_bias=geometry_bias_for_metrics,
                    semantic_scores=semantic_scores_for_metrics,
                    candidate_mask=candidate_bool_for_metrics,
                    geometry_only_scores=geometry_only_scores_for_metrics,
                    semantic_only_scores=semantic_only_scores_for_metrics,
                    query_uv=query_uv_tensor,
                    sat_perspective_uv=sat_uv,
                    sat_perspective_valid=sat_valid,
                    query_base_hw=query_base_hw,
                    attention_alignment=attention_alignment,
                    query_pe_metrics=query_pe_metrics,
                )
        else:
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
    """Sliced attention processor that preserves logit-level geometry score."""

    def __init__(
        self,
        query_dim: int,
        *,
        slice_size: int,
        query_uv_enabled: bool = False,
        geometry_bias_enabled: bool = False,
        geometry_score_enabled: bool = False,
        geometry_score_dim: int = 64,
        geometry_score_num_freqs: int = 6,
        geometry_score_gate_init: float = 1.0,
        geometry_score_layers: Optional[Tuple[str, ...]] = None,
        geometry_score_max_query_tokens: Optional[int] = None,
        geometry_score_mode: str = "geometry_first_semantic_refine",
        candidate_radius: float = 0.35,
        candidate_min_k: int = 16,
        candidate_invalid_penalty: float = -1e4,
        semantic_score_dim: int = 64,
        semantic_score_alpha: float = 0.25,
        geometry_bias_scale: float = 2.0,
        geometry_invalid_penalty: float = -1e4,
        gate_init: float = 0.0,
        layer_name: Optional[str] = None,
    ) -> None:
        del query_uv_enabled, geometry_bias_enabled, geometry_bias_scale
        del geometry_invalid_penalty, gate_init
        super().__init__(
            query_dim=query_dim,
            geometry_score_enabled=geometry_score_enabled,
            geometry_score_dim=geometry_score_dim,
            geometry_score_num_freqs=geometry_score_num_freqs,
            geometry_score_gate_init=geometry_score_gate_init,
            geometry_score_layers=geometry_score_layers,
            geometry_score_max_query_tokens=geometry_score_max_query_tokens,
            geometry_score_mode=geometry_score_mode,
            candidate_radius=candidate_radius,
            candidate_min_k=candidate_min_k,
            candidate_invalid_penalty=candidate_invalid_penalty,
            semantic_score_dim=semantic_score_dim,
            semantic_score_alpha=semantic_score_alpha,
            layer_name=layer_name,
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
        attention_alignment: Optional[Dict[str, Any]] = None,
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

        is_cross_attention = encoder_hidden_states is not None
        collect_alignment = self._should_collect_attention_alignment(
            attention_alignment,
            query_tokens=hidden_states.shape[1],
            is_cross_attention=is_cross_attention,
        )
        apply_geometry_score = self._should_apply_geometry_score(
            query_tokens=hidden_states.shape[1],
            is_cross_attention=is_cross_attention,
        )
        query = attn.to_q(hidden_states)
        query_pe_metrics: Dict[str, torch.Tensor] = {}
        if collect_alignment:
            query_norm = query.float().norm(dim=-1).mean()
            query_pe_metrics["query_content_norm"] = query_norm
        query_uv_tensor = None
        if is_cross_attention and (
            apply_geometry_score
            or collect_alignment
        ):
            query_uv_tensor = self._resolve_query_uv(
                hidden_states,
                batch_size=hidden_states.shape[0],
                query_base_hw=query_base_hw,
                query_uv=query_uv,
            ).to(dtype=query.dtype)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query_states_for_semantic = query
        key = attn.to_k(encoder_hidden_states)
        key_states_for_semantic = key
        value = attn.to_v(encoder_hidden_states)
        if collect_alignment:
            query_pe_metrics["key_content_norm"] = key.float().norm(dim=-1).mean()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        sat_uv = None
        sat_valid = None
        geometry_bias_for_metrics = None
        semantic_scores_for_metrics = None
        candidate_bool_for_metrics = None
        geometry_only_scores_for_metrics = None
        semantic_only_scores_for_metrics = None
        if (apply_geometry_score or collect_alignment) and is_cross_attention:
            sat_uv, sat_valid = self._resolve_sat_perspective_uv(
                hidden_states,
                batch_size=batch_size,
                key_length=key.shape[1],
                sat_perspective_uv=sat_perspective_uv,
                sat_perspective_valid=sat_perspective_valid,
            )
        use_geometry_first = apply_geometry_score and self._uses_geometry_first_mode()
        if apply_geometry_score and query_uv_tensor is not None and not use_geometry_first:
            geometry_score_bias, geometry_score_metrics = self._build_geometry_score_bias(
                query_uv_tensor,
                sat_uv,
                sat_valid,
                dtype=query.dtype,
            )
            if collect_alignment:
                query_pe_metrics.update(geometry_score_metrics)
            if geometry_score_bias is not None:
                geometry_score_bias = geometry_score_bias.repeat_interleave(attn.heads, dim=0)
                geometry_bias_for_metrics = self._reshape_sliced_attention_mask(
                    geometry_score_bias,
                    batch_size=batch_size,
                    heads=attn.heads,
                    query_tokens=query.shape[1],
                    key_tokens=key.shape[1],
                )
                attention_mask = (
                    geometry_score_bias
                    if attention_mask is None
                    else attention_mask + geometry_score_bias
                )
        batch_size_attention, query_tokens, dim = query.shape
        if use_geometry_first and query_uv_tensor is not None:
            query_4d = query.view(batch_size, attn.heads, query_tokens, dim)
            key_4d = key.view(batch_size, attn.heads, key.shape[1], dim)
            value_4d = value.view(batch_size, attn.heads, value.shape[1], dim)
            attention_mask_4d = self._reshape_sliced_attention_mask(
                attention_mask,
                batch_size=batch_size,
                heads=attn.heads,
                query_tokens=query_tokens,
                key_tokens=key.shape[1],
            ) if attention_mask is not None else None

            geometry_score_bias, geometry_score_metrics = self._build_geometry_score_bias(
                query_uv_tensor,
                sat_uv,
                sat_valid,
                dtype=query.dtype,
            )
            semantic_score_bias, semantic_score_metrics = self._build_semantic_score_bias(
                query_states_for_semantic,
                key_states_for_semantic,
                dtype=query.dtype,
            )
            candidate_mask, candidate_bool, candidate_metrics = self._build_candidate_mask(
                query_uv_tensor,
                sat_uv,
                sat_valid,
                dtype=query.dtype,
            )
            query_pe_metrics.update(geometry_score_metrics)
            query_pe_metrics.update(semantic_score_metrics)
            query_pe_metrics.update(candidate_metrics)

            content_scores = self._raw_content_scores(query_4d, key_4d)
            if candidate_mask is None:
                active_scores = torch.zeros_like(content_scores)
            else:
                active_scores = candidate_mask.unsqueeze(1).float()
            if geometry_score_bias is not None:
                geometry_bias_for_metrics = geometry_score_bias.unsqueeze(1)
                active_scores = active_scores + geometry_bias_for_metrics.float()
            if semantic_score_bias is not None:
                semantic_scores_for_metrics = semantic_score_bias.unsqueeze(1)
                active_scores = active_scores + semantic_scores_for_metrics.float()
            if attention_mask_4d is not None:
                active_scores = active_scores + attention_mask_4d.float()

            geometry_only_scores_for_metrics = (
                (candidate_mask.unsqueeze(1).float() if candidate_mask is not None else torch.zeros_like(content_scores))
                + (geometry_bias_for_metrics.float() if geometry_bias_for_metrics is not None else 0.0)
                + (attention_mask_4d.float() if attention_mask_4d is not None else 0.0)
            )
            semantic_only_scores_for_metrics = (
                (candidate_mask.unsqueeze(1).float() if candidate_mask is not None else torch.zeros_like(content_scores))
                + (semantic_scores_for_metrics.float() if semantic_scores_for_metrics is not None else 0.0)
                + (attention_mask_4d.float() if attention_mask_4d is not None else 0.0)
            )
            candidate_bool_for_metrics = candidate_bool
            hidden_4d, attention_probs = self._manual_attention_from_scores(value_4d, active_scores)
            attention_scores = active_scores
            hidden_states = hidden_4d.reshape(batch_size_attention, query_tokens, dim)
            if collect_alignment:
                self._record_attention_alignment(
                    attention_probs=attention_probs,
                    attention_scores=attention_scores,
                    content_scores=content_scores,
                    geometry_bias=geometry_bias_for_metrics,
                    semantic_scores=semantic_scores_for_metrics,
                    candidate_mask=candidate_bool_for_metrics,
                    geometry_only_scores=geometry_only_scores_for_metrics,
                    semantic_only_scores=semantic_only_scores_for_metrics,
                    query_uv=query_uv_tensor,
                    sat_perspective_uv=sat_uv,
                    sat_perspective_valid=sat_valid,
                    query_base_hw=query_base_hw,
                    attention_alignment=attention_alignment,
                    query_pe_metrics=query_pe_metrics,
                )
        elif collect_alignment:
            query_4d = query.view(batch_size, attn.heads, query_tokens, dim)
            key_4d = key.view(batch_size, attn.heads, key.shape[1], dim)
            value_4d = value.view(batch_size, attn.heads, value.shape[1], dim)
            attention_mask_4d = self._reshape_sliced_attention_mask(
                attention_mask,
                batch_size=batch_size,
                heads=attn.heads,
                query_tokens=query_tokens,
                key_tokens=key.shape[1],
            )
            hidden_4d, attention_probs, attention_scores, content_scores = self._manual_scaled_dot_product_attention(
                query_4d,
                key_4d,
                value_4d,
                attention_mask_4d,
            )
            hidden_states = hidden_4d.reshape(batch_size_attention, query_tokens, dim)
            if query_uv_tensor is not None:
                self._record_attention_alignment(
                    attention_probs=attention_probs,
                    attention_scores=attention_scores,
                    content_scores=content_scores,
                    geometry_bias=geometry_bias_for_metrics,
                    semantic_scores=semantic_scores_for_metrics,
                    candidate_mask=candidate_bool_for_metrics,
                    geometry_only_scores=geometry_only_scores_for_metrics,
                    semantic_only_scores=semantic_only_scores_for_metrics,
                    query_uv=query_uv_tensor,
                    sat_perspective_uv=sat_uv,
                    sat_perspective_valid=sat_valid,
                    query_base_hw=query_base_hw,
                    attention_alignment=attention_alignment,
                    query_pe_metrics=query_pe_metrics,
                )
        else:
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
