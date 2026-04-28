"""
Rejectable geometric transport reading for satellite-to-front conditioning.
"""

from __future__ import annotations

from typing import Dict, Optional

import math

import torch
import torch.nn as nn


class GeometricTransportReadingBlock(nn.Module):
    """
    Read satellite tokens with geometry-biased row-wise transport plus dustbin.

    The ground-plane projection supplies the main addressing prior. Plucker rays
    only modulate uncertainty and dustbin preference, so yaw control must pass
    through ``front_bev_xy``.
    """

    def __init__(
        self,
        front_dim: int,
        sat_in_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        lambda_geo: float = 8.0,
        compatibility_scale: float = 0.1,
        sigma_min: float = 0.03,
        sigma_max: float = 0.35,
        invalid_conf_loss_weight: float = 0.05,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError(f"num_heads/head_dim must be positive, got {num_heads}/{head_dim}")
        if sigma_min <= 0.0 or sigma_max <= sigma_min:
            raise ValueError(f"Expected 0 < sigma_min < sigma_max, got {sigma_min}/{sigma_max}")

        model_dim = int(num_heads) * int(head_dim)
        self.front_dim = int(front_dim)
        self.sat_in_dim = int(sat_in_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.model_dim = model_dim
        self.lambda_geo = float(lambda_geo)
        self.compatibility_scale = float(compatibility_scale)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.invalid_conf_loss_weight = float(invalid_conf_loss_weight)

        self.front_adapter = nn.Sequential(
            nn.Linear(front_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.sat_adapter = nn.Sequential(
            nn.Linear(sat_in_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

        self.plucker_adapter = nn.Sequential(
            nn.Linear(6, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.sigma_head = nn.Linear(model_dim, 1)
        self.dustbin_head = nn.Linear(model_dim, self.num_heads)
        self.out_proj = nn.Linear(model_dim, front_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # Start with a moderate dustbin preference; valid geometry can overcome it.
        nn.init.constant_(self.dustbin_head.bias, 0.0)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channel_count = tensor.shape
        expected = self.num_heads * self.head_dim
        if channel_count != expected:
            raise ValueError(f"Expected last dim {expected}, got {channel_count}")
        return tensor.reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    def forward(
        self,
        front_feat: torch.Tensor,
        sat_tokens: torch.Tensor,
        sat_xy: torch.Tensor,
        front_bev_xy: torch.Tensor,
        front_bev_valid_mask: Optional[torch.Tensor] = None,
        front_plucker: Optional[torch.Tensor] = None,
        return_attn_map: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if front_feat.ndim != 4:
            raise ValueError("front_feat must be [B,C,H,W]")
        if sat_tokens.ndim != 3 or sat_xy.ndim != 3 or front_bev_xy.ndim != 3:
            raise ValueError("sat_tokens, sat_xy, and front_bev_xy must be rank-3 tensors")
        if sat_xy.shape[-1] != 2 or front_bev_xy.shape[-1] != 2:
            raise ValueError("sat_xy/front_bev_xy last dim must be 2")

        batch_size, _, height, width = front_feat.shape
        token_count = height * width
        if front_bev_xy.shape[:2] != (batch_size, token_count):
            raise ValueError(
                f"front_bev_xy must be [B,{token_count},2], got {list(front_bev_xy.shape)}"
            )
        if sat_tokens.shape[0] != batch_size or sat_xy.shape[:2] != sat_tokens.shape[:2]:
            raise ValueError("sat_tokens/sat_xy batch or token count mismatch")

        front_flat = front_feat.flatten(2).transpose(1, 2)
        front_embed = self.front_adapter(front_flat)
        context = front_embed
        if front_plucker is not None:
            if front_plucker.ndim != 3 or front_plucker.shape[:2] != (batch_size, token_count) or front_plucker.shape[-1] != 6:
                raise ValueError(f"front_plucker must be [B,{token_count},6]")
            context = context + self.plucker_adapter(front_plucker)

        sat_embed = self.sat_adapter(sat_tokens)
        q = self._reshape_heads(self.q_proj(front_embed))
        k = self._reshape_heads(self.k_proj(sat_embed))
        v = self._reshape_heads(self.v_proj(sat_embed))

        compatibility = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        compatibility = self.compatibility_scale * compatibility

        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(
            self.sigma_head(context).squeeze(-1)
        )
        dist2 = (front_bev_xy.unsqueeze(2) - sat_xy.unsqueeze(1)).pow(2).sum(dim=-1)
        geo_logits = -self.lambda_geo * dist2 / sigma.unsqueeze(-1).pow(2).clamp_min(1e-6)
        sat_logits = compatibility + geo_logits.unsqueeze(1)

        if front_bev_valid_mask is None:
            valid_mask = torch.ones(batch_size, token_count, device=front_feat.device, dtype=torch.bool)
        else:
            if front_bev_valid_mask.ndim == 3 and front_bev_valid_mask.shape[-1] == 1:
                front_bev_valid_mask = front_bev_valid_mask.squeeze(-1)
            if front_bev_valid_mask.ndim != 2 or front_bev_valid_mask.shape != (batch_size, token_count):
                raise ValueError(
                    f"front_bev_valid_mask must be [B,{token_count}], got {list(front_bev_valid_mask.shape)}"
                )
            valid_mask = front_bev_valid_mask.to(device=front_feat.device, dtype=torch.bool)

        sat_logits = sat_logits.masked_fill(~valid_mask[:, None, :, None], -1e4)
        dustbin_logits = self.dustbin_head(context).transpose(1, 2).unsqueeze(-1)
        logits = torch.cat([sat_logits, dustbin_logits], dim=-1)

        probs = torch.softmax(logits, dim=-1)
        probs = self.attn_dropout(probs)
        sat_probs = probs[..., :-1]
        dustbin_prob = probs[..., -1]

        read = torch.matmul(sat_probs, v)
        read = read.transpose(1, 2).reshape(batch_size, token_count, self.model_dim)
        read_feat = self.out_proj(read).transpose(1, 2).reshape(batch_size, self.front_dim, height, width)

        confidence = (1.0 - dustbin_prob).mean(dim=1)
        confidence_feat = confidence.reshape(batch_size, 1, height, width)
        front_feat_out = front_feat + confidence_feat * read_feat

        valid_float = valid_mask.to(dtype=confidence.dtype)
        invalid_float = 1.0 - valid_float
        valid_conf = self._masked_mean(confidence, valid_float)
        invalid_conf = self._masked_mean(confidence, invalid_float)
        invalid_conf_loss = invalid_conf * self.invalid_conf_loss_weight

        stats = {
            "transport_confidence": confidence.float().mean().detach(),
            "transport_valid_confidence": valid_conf.detach(),
            "transport_invalid_confidence": invalid_conf.detach(),
            "transport_dustbin_prob": dustbin_prob.float().mean().detach(),
            "transport_sigma": sigma.float().mean().detach(),
            "transport_valid_ratio": valid_float.float().mean().detach(),
            "transport_compatibility_std": compatibility.float().std(unbiased=False).detach(),
            "transport_geo_logits_std": geo_logits.float().std(unbiased=False).detach(),
        }
        losses = {
            "transport_invalid_conf_loss": invalid_conf_loss,
        }

        return {
            "front_feat_out": front_feat_out,
            "read_feat": read_feat,
            "confidence": confidence,
            "attn_map": sat_probs.detach() if return_attn_map else None,
            "stats": stats,
            "losses": losses,
        }
