"""
Lightweight decoupled satellite cross-attention branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecoupledSatCrossAttn(nn.Module):
    """
    Reuse the UNet cross-attention query and read satellite tokens with a small
    parallel K/V branch.
    """

    def __init__(
        self,
        sat_in_dim: int,
        inner_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.sat_norm = nn.LayerNorm(sat_in_dim)
        self.to_k_sat = nn.Linear(sat_in_dim, inner_dim, bias=False)
        self.to_v_sat = nn.Linear(sat_in_dim, inner_dim, bias=False)
        self.to_out_sat = nn.Identity() if inner_dim == out_dim else nn.Linear(inner_dim, out_dim, bias=False)
        nn.init.zeros_(self.to_v_sat.weight)
        if isinstance(self.to_out_sat, nn.Linear):
            nn.init.zeros_(self.to_out_sat.weight)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        sat_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sat_tokens = self.sat_norm(sat_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype))

        query = attn.to_q(hidden_states)
        key = self.to_k_sat(sat_tokens)
        value = self.to_v_sat(sat_tokens)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask=None)
        sat_output = torch.bmm(attention_probs, value)
        sat_output = attn.batch_to_head_dim(sat_output)
        sat_output = self.to_out_sat(sat_output)

        return self.alpha * sat_output
