"""
Gated residual injection for read features into front features.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class GatedResidualInject(nn.Module):
    """
    Inject read features into current U-Net feature map.
    """

    def __init__(
        self,
        front_dim: int,
        read_dim: int,
        gate_hidden_ratio: float = 0.25,
        use_gated_residual: bool = True,
    ):
        super().__init__()
        hidden_dim = max(1, int(front_dim * gate_hidden_ratio))

        self.front_dim = front_dim
        self.read_dim = read_dim
        self.use_gated_residual = use_gated_residual

        self.out_proj = nn.Linear(read_dim, front_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(front_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, front_dim),
        )

        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -1.0)

    def _tokens_to_feat(self, read_tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, token_count, _ = read_tokens.shape
        expected_tokens = height * width
        if token_count != expected_tokens:
            raise ValueError(
                f"read_tokens token_count mismatch: expected {expected_tokens}, got {token_count}"
            )
        read_feat = self.out_proj(read_tokens)
        read_feat = read_feat.transpose(1, 2).reshape(batch_size, self.front_dim, height, width)
        return read_feat

    def forward(
        self,
        front_feat: torch.Tensor,
        read_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if front_feat.ndim != 4 or read_tokens.ndim != 3:
            raise ValueError("front_feat must be [B,C,H,W], read_tokens must be [B,N,C]")

        batch_size, channels, height, width = front_feat.shape
        if channels != self.front_dim:
            raise ValueError(f"front channel mismatch: expected {self.front_dim}, got {channels}")

        read_feat = self._tokens_to_feat(read_tokens, height, width)
        pooled = front_feat.mean(dim=(2, 3))
        gate = torch.sigmoid(self.gate_mlp(pooled)).view(batch_size, channels, 1, 1)

        if self.use_gated_residual:
            out = front_feat + gate * read_feat
        else:
            out = front_feat + read_feat

        return out, read_feat, gate
