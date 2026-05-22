import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from models.unet.geometry_masked_attention_processor import (
    GeometryMaskedAttnProcessor2_0,
    build_topk_mask,
)


class _DummyAttention(nn.Module):
    def __init__(self, hidden_dim: int = 32, heads: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.spatial_norm = None
        self.group_norm = None
        self.norm_q = None
        self.norm_k = None
        self.norm_cross = False
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Identity(), nn.Identity()])

    def prepare_attention_mask(self, attention_mask, key_length, batch_size):
        del key_length, batch_size
        return attention_mask


class GeometryMaskedAttentionProcessorTest(unittest.TestCase):
    def test_topk_mask_excludes_invalid_satellite_keys(self) -> None:
        front_xy = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
        sat_xy = torch.tensor([[[0.0, 0.0], [0.9, 0.0]]], dtype=torch.float32)
        key_mask = torch.tensor([[False, True]])

        mask = build_topk_mask(front_xy, sat_xy, topk=1, key_mask=key_mask)

        self.assertLess(float(mask[0, 0, 0]), -9999.0)
        self.assertEqual(float(mask[0, 0, 1]), 0.0)

    def test_geometry_bias_changes_cross_attention_output(self) -> None:
        torch.manual_seed(0)
        attn = _DummyAttention(hidden_dim=32, heads=2)
        processor = GeometryMaskedAttnProcessor2_0(
            site="mid",
            topk=1,
            context_provider=lambda: {
                "front_bev_xy": torch.tensor(
                    [[[-0.8, 0.8], [-0.4, 0.4], [-0.8, 0.8], [-0.4, 0.4]]],
                    dtype=torch.float32,
                ),
                "sat_xy": torch.tensor(
                    [[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]],
                    dtype=torch.float32,
                ),
                "condition_mask": torch.ones(1, dtype=torch.bool),
            },
        )
        hidden_states = torch.randn((1, 4, 32), dtype=torch.float32)
        encoder_hidden_states = torch.randn((1, 3, 32), dtype=torch.float32)

        with torch.no_grad():
            with_context = processor(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)

        processor_no_context = GeometryMaskedAttnProcessor2_0(
            site="mid",
            context_provider=lambda: None,
        )
        with torch.no_grad():
            without_context = processor_no_context(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)

        self.assertEqual(with_context.shape, hidden_states.shape)
        self.assertFalse(torch.allclose(with_context, without_context))

    def test_sdpa_cudnn_failure_falls_back_to_manual_attention(self) -> None:
        torch.manual_seed(0)
        processor = GeometryMaskedAttnProcessor2_0(
            site="mid",
            context_provider=lambda: None,
        )
        query = torch.randn((1, 2, 3, 8), dtype=torch.float32)
        key = torch.randn((1, 2, 4, 8), dtype=torch.float32)
        value = torch.randn((1, 2, 4, 8), dtype=torch.float32)
        expected = processor._manual_scaled_dot_product_attention(query, key, value, None)

        cudnn_error = RuntimeError(
            "cuDNN Frontend error: [cudnn_frontend] Error: No execution plans support the graph."
        )
        with patch(
            "models.unet.geometry_masked_attention_processor.F.scaled_dot_product_attention",
            side_effect=cudnn_error,
        ) as mock_sdpa:
            output = processor._scaled_dot_product_attention(query, key, value, None)

        self.assertEqual(mock_sdpa.call_count, 1)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
