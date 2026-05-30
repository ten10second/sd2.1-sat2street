"""Pose-chain latent refiner for joint multi-view denoising."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossViewRefinerOutput:
    refined_x0: torch.Tensor
    consistency_loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class CrossViewLatentRefiner(nn.Module):
    """Refine predicted clean latents with adjacent pose-chain attention.

    The module is intentionally output-level: it receives predicted x0 latents
    shaped as [B, V, C, H, W] and returns a residual refinement with a small or
    zero initialized gate. Only adjacent views exchange messages.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 4,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
        bev_sigma: float = 0.25,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if bev_sigma <= 0:
            raise ValueError(f"bev_sigma must be positive, got {bev_sigma}")

        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.bev_sigma = float(bev_sigma)

        self.to_q = nn.Linear(self.latent_channels, self.hidden_dim)
        self.to_k = nn.Linear(self.latent_channels, self.hidden_dim)
        self.to_v = nn.Linear(self.latent_channels, self.hidden_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(7, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.latent_channels)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

        # The scalar gate controls startup identity. Keep the projection bias
        # neutral while leaving weights learnable from the first gate update.
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        pred_x0: torch.Tensor,
        *,
        target_latents: Optional[torch.Tensor] = None,
        vehicle_yaw_degs: Optional[torch.Tensor] = None,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
    ) -> CrossViewRefinerOutput:
        if pred_x0.ndim != 5:
            raise ValueError(f"pred_x0 must have shape [B,V,C,H,W], got {tuple(pred_x0.shape)}")
        batch_size, num_views, channels, height, width = pred_x0.shape
        if channels != self.latent_channels:
            raise ValueError(f"pred_x0 has {channels} channels, expected {self.latent_channels}")

        zero = pred_x0.sum() * 0.0
        if num_views < 2:
            return CrossViewRefinerOutput(
                refined_x0=pred_x0,
                consistency_loss=zero,
                metrics=self._empty_metrics(zero),
            )

        tokens = pred_x0.flatten(3).transpose(2, 3).contiguous()  # [B,V,N,C]
        num_tokens = tokens.shape[2]
        bev_tokens, valid_tokens = self._prepare_bev_tokens(
            front_bev_xy,
            front_ground_valid_mask,
            batch_size,
            num_views,
            height,
            width,
            pred_x0.device,
            pred_x0.dtype,
        )

        message_accum = torch.zeros_like(tokens)
        message_counts = torch.zeros(
            batch_size,
            num_views,
            1,
            1,
            device=pred_x0.device,
            dtype=pred_x0.dtype,
        )
        entropy_values = []
        match_distances = []
        valid_ratios = []

        for left in range(num_views - 1):
            right = left + 1
            for src_idx, dst_idx in ((left, right), (right, left)):
                message, entropy, match_distance, valid_ratio = self._attend_direction(
                    tokens[:, src_idx],
                    tokens[:, dst_idx],
                    src_idx=src_idx,
                    dst_idx=dst_idx,
                    vehicle_yaw_degs=vehicle_yaw_degs,
                    bev_tokens=bev_tokens,
                    valid_tokens=valid_tokens,
                )
                message_accum[:, dst_idx] = message_accum[:, dst_idx] + message
                message_counts[:, dst_idx] = message_counts[:, dst_idx] + 1.0
                entropy_values.append(entropy)
                match_distances.append(match_distance)
                valid_ratios.append(valid_ratio)

        averaged_message = message_accum / message_counts.clamp_min(1.0)
        delta = averaged_message.transpose(2, 3).reshape(batch_size, num_views, channels, height, width)
        active = (message_counts > 0).to(pred_x0.dtype).permute(0, 1, 3, 2).reshape(
            batch_size, num_views, 1, 1, 1
        )
        refined_x0 = pred_x0 + self.gate.to(dtype=pred_x0.dtype, device=pred_x0.device) * delta * active

        if target_latents is not None:
            target = target_latents.to(device=pred_x0.device, dtype=pred_x0.dtype)
            if target.shape != pred_x0.shape:
                raise ValueError(
                    f"target_latents shape {tuple(target.shape)} must match pred_x0 {tuple(pred_x0.shape)}"
                )
            consistency_loss = F.l1_loss(refined_x0 * active, target * active)
        else:
            consistency_loss = zero

        entropy_value = self._mean_or_zero(entropy_values, zero)
        match_distance_value = self._mean_or_zero(match_distances, zero)
        valid_ratio_value = self._mean_or_zero(valid_ratios, zero)
        metrics = {
            "joint_view_generation/source_refined_x0": torch.ones_like(zero),
            "joint_view_generation/refiner_gate": self.gate.detach().to(device=pred_x0.device),
            "joint_view_generation/attention_entropy": entropy_value.detach(),
            "joint_view_generation/bev_match_distance": match_distance_value.detach(),
            "joint_view_generation/valid_match_ratio": valid_ratio_value.detach(),
            "joint_view_generation/num_adjacent_directions": torch.tensor(
                float(len(entropy_values)), device=pred_x0.device, dtype=pred_x0.dtype
            ),
        }
        return CrossViewRefinerOutput(
            refined_x0=refined_x0,
            consistency_loss=consistency_loss,
            metrics=metrics,
        )

    def _attend_direction(
        self,
        src_tokens: torch.Tensor,
        dst_tokens: torch.Tensor,
        *,
        src_idx: int,
        dst_idx: int,
        vehicle_yaw_degs: Optional[torch.Tensor],
        bev_tokens: Optional[torch.Tensor],
        valid_tokens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = src_tokens.shape
        query = self._split_heads(self.to_q(dst_tokens))
        key = self._split_heads(self.to_k(src_tokens))
        value = self._split_heads(self.to_v(src_tokens))

        action = self._action_embedding(vehicle_yaw_degs, src_idx, dst_idx, batch_size, src_tokens.device, src_tokens.dtype)
        key = key + action[:, :, None, :]

        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        match_distance = src_tokens.sum() * 0.0
        valid_ratio = torch.ones((), device=src_tokens.device, dtype=src_tokens.dtype)

        if bev_tokens is not None:
            src_bev = bev_tokens[:, src_idx]  # [B,N,2]
            dst_bev = bev_tokens[:, dst_idx]
            diff = dst_bev[:, :, None, :] - src_bev[:, None, :, :]
            dist2 = diff.square().sum(dim=-1)
            local_bias = -dist2 / (2.0 * self.bev_sigma * self.bev_sigma)
            logits = logits + local_bias[:, None]

            if valid_tokens is not None:
                src_valid = valid_tokens[:, src_idx]
                dst_valid = valid_tokens[:, dst_idx]
                logits = logits.masked_fill(~src_valid[:, None, None, :], -1.0e4)
                logits = logits.masked_fill(~dst_valid[:, None, :, None], -1.0e4)
                valid_ratio = (src_valid.float().mean() * dst_valid.float().mean()).to(dtype=src_tokens.dtype)
            else:
                dst_valid = torch.ones(batch_size, num_tokens, device=src_tokens.device, dtype=torch.bool)

        attention = torch.softmax(logits.float(), dim=-1).to(dtype=value.dtype)
        attention = self.dropout(attention)
        message = torch.matmul(attention, value)
        message = self._merge_heads(message)
        message = self.out_proj(message)

        entropy = -(attention.float().clamp_min(1.0e-8).log() * attention.float()).sum(dim=-1)
        entropy = entropy.mean().to(dtype=src_tokens.dtype)

        if bev_tokens is not None:
            expected_distance = torch.sqrt(dist2.clamp_min(0.0))[:, None] * attention.float()
            expected_distance = expected_distance.sum(dim=-1).mean(dim=1)
            if valid_tokens is not None:
                valid_count = dst_valid.float().sum().clamp_min(1.0)
                match_distance = (expected_distance * dst_valid.float()).sum() / valid_count
            else:
                match_distance = expected_distance.mean()
            match_distance = match_distance.to(dtype=src_tokens.dtype)

        return message, entropy, match_distance, valid_ratio

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        tensor = tensor.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2).contiguous()

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_tokens, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, num_tokens, self.hidden_dim)

    def _action_embedding(
        self,
        vehicle_yaw_degs: Optional[torch.Tensor],
        src_idx: int,
        dst_idx: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if vehicle_yaw_degs is None:
            features = torch.zeros(batch_size, 7, device=device, dtype=dtype)
            features[:, -1] = float(dst_idx - src_idx)
        else:
            yaw = vehicle_yaw_degs.to(device=device, dtype=dtype)
            if yaw.ndim == 1:
                yaw = yaw.view(1, -1).expand(batch_size, -1)
            if yaw.shape[0] != batch_size:
                raise ValueError(f"vehicle_yaw_degs batch {yaw.shape[0]} does not match {batch_size}")
            if yaw.shape[1] <= max(src_idx, dst_idx):
                raise ValueError(
                    f"vehicle_yaw_degs has {yaw.shape[1]} views, need index {max(src_idx, dst_idx)}"
                )
            yaw = torch.nan_to_num(yaw, nan=0.0, posinf=0.0, neginf=0.0)
            src = torch.deg2rad(yaw[:, src_idx])
            dst = torch.deg2rad(yaw[:, dst_idx])
            delta = dst - src
            features = torch.stack(
                [
                    torch.sin(src),
                    torch.cos(src),
                    torch.sin(dst),
                    torch.cos(dst),
                    torch.sin(delta),
                    torch.cos(delta),
                    torch.full_like(delta, float(dst_idx - src_idx)),
                ],
                dim=-1,
            )
        embedding = self.action_mlp(features.float()).to(dtype=dtype)
        return embedding.view(batch_size, self.num_heads, self.head_dim)

    def _prepare_bev_tokens(
        self,
        front_bev_xy: Optional[torch.Tensor],
        front_ground_valid_mask: Optional[torch.Tensor],
        batch_size: int,
        num_views: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if front_bev_xy is None:
            return None, None
        if front_bev_xy.ndim != 5:
            raise ValueError(f"front_bev_xy must have shape [B,V,2,H,W], got {tuple(front_bev_xy.shape)}")
        if front_bev_xy.shape[0] != batch_size or front_bev_xy.shape[1] != num_views or front_bev_xy.shape[2] != 2:
            raise ValueError(
                f"front_bev_xy shape {tuple(front_bev_xy.shape)} is incompatible with [B={batch_size},V={num_views},2,H,W]"
            )
        bev = front_bev_xy.to(device=device, dtype=dtype).reshape(batch_size * num_views, 2, *front_bev_xy.shape[-2:])
        bev = F.interpolate(bev, size=(height, width), mode="bilinear", align_corners=False)
        bev = bev.reshape(batch_size, num_views, 2, height * width).transpose(2, 3).contiguous()

        if front_ground_valid_mask is None:
            valid = torch.ones(batch_size, num_views, height * width, device=device, dtype=torch.bool)
            return bev, valid
        if front_ground_valid_mask.ndim != 5:
            raise ValueError(
                "front_ground_valid_mask must have shape [B,V,1,H,W], "
                f"got {tuple(front_ground_valid_mask.shape)}"
            )
        valid = front_ground_valid_mask.to(device=device, dtype=torch.float32)
        valid = valid.reshape(batch_size * num_views, 1, *front_ground_valid_mask.shape[-2:])
        valid = F.interpolate(valid, size=(height, width), mode="nearest")
        valid = valid.reshape(batch_size, num_views, height * width) > 0.5
        return bev, valid

    def _empty_metrics(self, zero: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "joint_view_generation/source_refined_x0": zero.detach(),
            "joint_view_generation/refiner_gate": self.gate.detach().to(device=zero.device),
            "joint_view_generation/attention_entropy": zero.detach(),
            "joint_view_generation/bev_match_distance": zero.detach(),
            "joint_view_generation/valid_match_ratio": zero.detach(),
            "joint_view_generation/num_adjacent_directions": zero.detach(),
        }

    @staticmethod
    def _mean_or_zero(values, zero: torch.Tensor) -> torch.Tensor:
        if not values:
            return zero
        return torch.stack([value.to(device=zero.device, dtype=zero.dtype) for value in values]).mean()
