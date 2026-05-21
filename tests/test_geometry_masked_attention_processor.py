import unittest

import torch
import torch.nn as nn

from models.unet.geometry_masked_attention_processor import (
    GeometryMaskedAttnProcessor2_0,
    apply_2d_rope,
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
    def test_apply_2d_rope_changes_rotated_channels(self) -> None:
        tensor = torch.zeros((1, 2, 2, 16), dtype=torch.float32)
        tensor[..., 0] = 1.0
        tensor[..., 2] = 1.0
        xy = torch.tensor([[[0.0, 0.0], [1.0, -1.0]]], dtype=torch.float32)

        rotated = apply_2d_rope(tensor, xy, num_freqs=2)

        self.assertFalse(torch.allclose(rotated[:, :, 0], rotated[:, :, 1]))
        self.assertTrue(torch.allclose(rotated[..., 8:], tensor[..., 8:]))

    def test_apply_2d_rope_uses_visual_octave_frequencies(self) -> None:
        tensor = torch.zeros((1, 1, 1, 16), dtype=torch.float32)
        tensor[..., 0] = 1.0
        xy = torch.tensor([[[0.25, 0.0]]], dtype=torch.float32)

        rotated = apply_2d_rope(tensor, xy, num_freqs=2)

        self.assertAlmostEqual(float(rotated[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(rotated[0, 0, 0, 1]), 1.0, places=5)

    def test_topk_mask_excludes_invalid_satellite_keys(self) -> None:
        front_xy = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
        sat_xy = torch.tensor([[[0.0, 0.0], [0.9, 0.0]]], dtype=torch.float32)
        key_mask = torch.tensor([[False, True]])

        mask = build_topk_mask(front_xy, sat_xy, topk=1, key_mask=key_mask)

        self.assertLess(float(mask[0, 0, 0]), -9999.0)
        self.assertEqual(float(mask[0, 0, 1]), 0.0)

    def test_geometry_context_rotates_q_and_k_before_sdpa(self) -> None:
        torch.manual_seed(0)
        attn = _DummyAttention(hidden_dim=32, heads=2)
        processor = GeometryMaskedAttnProcessor2_0(
            site="mid",
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


if __name__ == "__main__":
    unittest.main()
