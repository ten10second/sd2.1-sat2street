import math
import unittest

import torch

from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0, build_normalized_image_uv_grid


def _yaw_geometry(yaw_rad: float):
    K = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    K[:, 0, 0] = 10.0
    K[:, 1, 1] = 10.0
    K[:, 0, 2] = 5.0
    K[:, 1, 2] = 5.0

    T_cam_to_world = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    T_cam_to_world[:, 1, 1] = -1.0
    T_cam_to_world[:, 2, 2] = -1.0
    T_cam_to_world[:, 2, 3] = 132.85
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    Rz = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    T_cam_to_world[0, :3, :3] = Rz @ T_cam_to_world[0, :3, :3]

    T_imu_to_world = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    camera_height_m = torch.ones(1, dtype=torch.float32)
    return K, T_cam_to_world, T_imu_to_world, camera_height_m


class GeometryAddressingCleanTest(unittest.TestCase):
    def test_satellite_tokens_are_yaw_invariant_but_projected_uv_changes(self) -> None:
        torch.manual_seed(0)
        encoder = SatelliteConditionEncoder(
            embed_dim=4,
            patch_size=4,
            sat_size=8,
            num_heads=1,
            num_layers=1,
            attn_dropout=0.0,
        )
        encoder.eval()
        sat = torch.randn((1, 3, 8, 8), dtype=torch.float32)

        K0, T0, Ti0, h0 = _yaw_geometry(0.0)
        K1, T1, Ti1, h1 = _yaw_geometry(0.5)
        state0 = encoder(
            sat,
            K=K0,
            T_cam_to_world=T0,
            T_imu_to_world=Ti0,
            camera_height_m=h0,
            image_size=(8, 8),
        )
        state1 = encoder(
            sat,
            K=K1,
            T_cam_to_world=T1,
            T_imu_to_world=Ti1,
            camera_height_m=h1,
            image_size=(8, 8),
        )

        self.assertTrue(torch.allclose(state0.tokens, state1.tokens, atol=0.0, rtol=0.0))
        self.assertGreater(float((state0.perspective_uv - state1.perspective_uv).abs().max()), 0.1)
        self.assertFalse(any("perspective_pe" in name for name, _ in encoder.named_parameters()))
        self.assertFalse(any("perspective_pos_encoder" in name for name, _ in encoder.named_parameters()))

    def test_geometry_score_processor_has_no_additive_query_pe_params(self) -> None:
        processor = QueryUVAttnProcessor2_0(
            query_dim=16,
            query_uv_enabled=True,
            geometry_bias_enabled=True,
            geometry_score_enabled=True,
            geometry_score_dim=8,
        )

        parameter_names = [name for name, _ in processor.named_parameters()]
        self.assertIn("geometry_score_gate", parameter_names)
        self.assertIn("geometry_score_proj.weight", parameter_names)
        self.assertFalse(any("query_uv" in name for name in parameter_names))
        self.assertIsNone(processor.query_uv_encoder)
        self.assertIsNone(processor.query_uv_gate)
        self.assertFalse(processor.geometry_bias_enabled)

    def test_geometry_score_bias_depends_on_query_and_sat_uv(self) -> None:
        torch.manual_seed(0)
        processor = QueryUVAttnProcessor2_0(
            query_dim=16,
            geometry_score_enabled=True,
            geometry_score_dim=8,
            geometry_score_gate_init=1.0,
        )
        query_uv = build_normalized_image_uv_grid(
            2,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        sat_uv = query_uv.flip(1)
        score, metrics = processor._build_geometry_score_bias(
            query_uv,
            sat_uv,
            torch.ones((1, 4), dtype=torch.bool),
            dtype=torch.float32,
        )

        self.assertEqual(score.shape, (1, 4, 4))
        self.assertGreater(float(score.std()), 0.0)
        self.assertGreater(float(metrics["geometry_score_bias_std"]), 0.0)

    def test_geometry_score_runtime_scale_changes_effective_logit_bias(self) -> None:
        torch.manual_seed(0)
        processor = QueryUVAttnProcessor2_0(
            query_dim=16,
            geometry_score_enabled=True,
            geometry_score_dim=8,
            geometry_score_gate_init=2.0,
        )
        query_uv = build_normalized_image_uv_grid(
            2,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        sat_uv = query_uv.flip(1)
        processor.set_geometry_score_runtime_scale(0.5)
        scaled_score, metrics = processor._build_geometry_score_bias(
            query_uv,
            sat_uv,
            torch.ones((1, 4), dtype=torch.bool),
            dtype=torch.float32,
        )
        processor.set_geometry_score_runtime_scale(1.0)
        full_score, _ = processor._build_geometry_score_bias(
            query_uv,
            sat_uv,
            torch.ones((1, 4), dtype=torch.bool),
            dtype=torch.float32,
        )

        self.assertAlmostEqual(float(metrics["geometry_score_runtime_scale"]), 0.5)
        self.assertTrue(torch.allclose(scaled_score * 2.0, full_score, atol=1e-6, rtol=1e-6))

    def test_geometry_first_candidate_window_masks_far_tokens(self) -> None:
        processor = QueryUVAttnProcessor2_0(
            query_dim=16,
            geometry_score_enabled=True,
            candidate_radius=0.1,
            candidate_min_k=1,
        )
        query_uv = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
        sat_uv = torch.tensor([[[0.0, 0.0], [0.8, 0.0]]], dtype=torch.float32)
        mask, candidate, metrics = processor._build_candidate_mask(
            query_uv,
            sat_uv,
            torch.ones((1, 2), dtype=torch.bool),
            dtype=torch.float32,
        )

        self.assertTrue(bool(candidate[0, 0, 0]))
        self.assertFalse(bool(candidate[0, 0, 1]))
        self.assertEqual(float(mask[0, 0, 0]), 0.0)
        self.assertLess(float(mask[0, 0, 1]), -999.0)
        self.assertEqual(float(metrics["window_fallback_ratio"]), 0.0)

    def test_geometry_first_candidate_fallback_uses_nearest_valid_tokens(self) -> None:
        processor = QueryUVAttnProcessor2_0(
            query_dim=16,
            geometry_score_enabled=True,
            candidate_radius=0.01,
            candidate_min_k=2,
        )
        query_uv = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
        sat_uv = torch.tensor([[[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0]]], dtype=torch.float32)
        mask, candidate, metrics = processor._build_candidate_mask(
            query_uv,
            sat_uv,
            torch.ones((1, 4), dtype=torch.bool),
            dtype=torch.float32,
        )

        self.assertEqual(candidate[0, 0].tolist(), [True, True, False, False])
        self.assertEqual(float(mask[0, 0, 0]), 0.0)
        self.assertEqual(float(mask[0, 0, 1]), 0.0)
        self.assertLess(float(mask[0, 0, 2]), -999.0)
        self.assertEqual(float(metrics["window_fallback_ratio"]), 1.0)

    def test_semantic_alpha_zero_removes_semantic_from_active_scores(self) -> None:
        torch.manual_seed(0)
        processor = QueryUVAttnProcessor2_0(
            query_dim=8,
            geometry_score_enabled=True,
            semantic_score_dim=4,
            semantic_score_alpha=0.25,
        )
        query_states = torch.randn(1, 2, 8)
        key_states = torch.randn(1, 3, 8)

        processor.set_semantic_score_runtime_alpha(0.0)
        zero_score, zero_metrics = processor._build_semantic_score_bias(
            query_states,
            key_states,
            dtype=torch.float32,
        )
        processor.set_semantic_score_runtime_alpha(0.25)
        full_score, _ = processor._build_semantic_score_bias(
            query_states,
            key_states,
            dtype=torch.float32,
        )

        self.assertAlmostEqual(float(zero_metrics["semantic_score_alpha"]), 0.0)
        self.assertTrue(torch.allclose(zero_score, torch.zeros_like(zero_score), atol=1e-7, rtol=0.0))
        self.assertGreater(float(full_score.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
