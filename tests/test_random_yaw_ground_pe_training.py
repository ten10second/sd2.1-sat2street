import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from torch.utils.data import DataLoader, Dataset

from data.kitti360d_dataset import Kitti360dDataset, SampleIndex
from models.conditioning import SatelliteMemoryState
from models.encoders.perspective_position_encoder import compute_sat_patch_perspective_uv
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.sd_model import SatelliteConditionedSDModel, SatelliteConditionedUNet
from models.sd_trainer import SDTrainer
from models.unet.query_uv_attn_processor import (
    QueryUVSlicedAttnProcessor,
    build_normalized_image_uv_grid,
    infer_spatial_hw,
)


def _identity_geometry(batch_size: int = 1):
    K = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    K[:, 0, 0] = 10.0
    K[:, 1, 1] = 10.0
    K[:, 0, 2] = 5.0
    K[:, 1, 2] = 5.0
    T_cam_to_world = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    T_cam_to_world[:, 1, 1] = -1.0
    T_cam_to_world[:, 2, 2] = -1.0
    T_cam_to_world[:, 2, 3] = 132.85
    T_imu_to_world = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    return K, T_cam_to_world, T_imu_to_world


class _DummyLatentDistribution:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def sample(self) -> torch.Tensor:
        return self._latents


class _DummyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0, block_out_channels=[1, 1, 1])
        self.register_parameter("_dummy", nn.Parameter(torch.zeros(1), requires_grad=False))

    def encode(self, images: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=_DummyLatentDistribution(images))

    def decode(self, latents: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(sample=latents)


class _DummyScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=10, prediction_type="epsilon")
        self.alphas_cumprod = torch.linspace(0.95, 0.05, 10)
        self.timesteps = torch.arange(1)

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return latents + noise

    def get_last_lr(self):
        return [0.0]

    def get_velocity(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha = self.alphas_cumprod.to(device=latents.device, dtype=latents.dtype).index_select(
            0,
            timesteps.to(device=latents.device, dtype=torch.long),
        )
        while alpha.ndim < latents.ndim:
            alpha = alpha.unsqueeze(-1)
        beta = (1.0 - alpha).clamp_min(0.0)
        return alpha.sqrt() * noise - beta.sqrt() * latents

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1)

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor, generator=None):
        del timestep, generator
        return SimpleNamespace(prev_sample=latents - 0.1 * noise_pred)


class _DummyUNet(nn.Module):
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=3, sample_size=8, cross_attention_dim=embed_dim)
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.query_geometry_score_enabled = True
        self.call_sat_token_means = []
        self.extra_kwarg_keys = []
        self.cross_attention_kwargs = []

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        sat_tokens: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        **kwargs,
    ) -> SimpleNamespace:
        del timesteps
        tokens = sat_tokens if sat_tokens is not None else encoder_hidden_states
        self.call_sat_token_means.append(float(tokens.mean().detach().item()))
        self.extra_kwarg_keys.append(tuple(sorted(kwargs.keys())))
        self.cross_attention_kwargs.append(kwargs.get("cross_attention_kwargs"))
        return SimpleNamespace(sample=noisy_latents * self.scale)


class _SingleSampleDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [SampleIndex(drive_dir=Path("/tmp/drive"), frame_id=7, meta=None)]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        del idx
        K, T_cam_to_world, T_imu_to_world = _identity_geometry()
        return {
            "sat": torch.zeros((3, 8, 8), dtype=torch.float32),
            "image": torch.zeros((3, 8, 8), dtype=torch.float32),
            "K": K[0],
            "T_cam_to_world": T_cam_to_world[0],
            "T_imu_to_world": T_imu_to_world[0],
            "camera_height_m": torch.tensor(1.0, dtype=torch.float32),
            "front_bev_xy": torch.zeros((2, 8, 8), dtype=torch.float32),
            "front_ground_valid_mask": torch.ones((1, 8, 8), dtype=torch.float32),
            "frame_id": torch.tensor(7),
        }


class _SinglePoseChainDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        del idx
        K, T_cam_to_world, T_imu_to_world = _identity_geometry(batch_size=2)
        return {
            "sat": torch.zeros((3, 8, 8), dtype=torch.float32),
            "image": torch.zeros((2, 3, 8, 8), dtype=torch.float32),
            "K": K,
            "T_cam_to_world": T_cam_to_world,
            "T_imu_to_world": T_imu_to_world,
            "camera_height_m": torch.ones((2,), dtype=torch.float32),
            "front_bev_xy": torch.zeros((2, 2, 8, 8), dtype=torch.float32),
            "front_ground_valid_mask": torch.ones((2, 1, 8, 8), dtype=torch.float32),
            "frame_id": torch.tensor(7),
            "chain_name": "right",
            "view_names": ["front", "yaw_p60"],
            "vehicle_yaw_degs": torch.tensor([float("nan"), 60.0], dtype=torch.float32),
        }


class _SmallPerspectiveModel(SatelliteConditionedSDModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.unet = _DummyUNet(embed_dim=4)
        self.vae = _DummyVAE()
        self.noise_scheduler = _DummyScheduler()
        self.cond_drop_prob = 0.0
        self.perspective_geometry_enabled = True
        self.satellite_encoder = SatelliteConditionEncoder(
            embed_dim=4,
            patch_size=4,
            num_heads=1,
        )


class _TrainableLossModel(nn.Module):
    def __init__(self, nonfinite: bool = False) -> None:
        super().__init__()
        self.trainable_weight = nn.Parameter(torch.tensor(1.0))
        self.nonfinite = bool(nonfinite)

    def forward(self, sat_images: torch.Tensor, target_images: torch.Tensor, **geometry):
        del sat_images, target_images, geometry
        finite_anchor = self.trainable_weight * 0.0
        loss = finite_anchor + torch.tensor(float("nan"), device=finite_anchor.device) if self.nonfinite else self.trainable_weight.square()
        return {
            "loss": loss,
            "sat_state": SatelliteMemoryState(
                tokens=torch.zeros((1, 4, 3), dtype=loss.dtype, device=loss.device),
                xy=torch.zeros((1, 4, 2), dtype=loss.dtype, device=loss.device),
            ),
        }


class _RuntimeScaleUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scales = []
        self.semantic_alphas = []

    def set_query_geometry_score_runtime_scale(self, scale: float) -> None:
        self.scales.append(float(scale))

    def set_query_semantic_score_runtime_alpha(self, alpha: float) -> None:
        self.semantic_alphas.append(float(alpha))


class _GeometryScoreParamModel(_TrainableLossModel):
    def __init__(self) -> None:
        super().__init__()
        self.unet = nn.Module()
        self.unet.processor = nn.Module()
        self.unet.processor.geometry_score_gate = nn.Parameter(torch.tensor(1.0))
        self.unet.processor.geometry_score_proj = nn.Linear(1, 1, bias=False)
        self.unet.processor.semantic_query_proj = nn.Linear(1, 1, bias=False)
        self.unet.processor.semantic_key_proj = nn.Linear(1, 1, bias=False)


class _GeometryScoreWarmupModel(_TrainableLossModel):
    def __init__(self) -> None:
        super().__init__()
        self.unet = _RuntimeScaleUNet()


class _VisualizationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy_trainable = nn.Parameter(torch.tensor(0.0))
        self.joint_view_generation_enabled = True
        self.generate_pose_chain_calls = []

    def generate_pose_chain(
        self,
        sat_images: torch.Tensor,
        *,
        K: torch.Tensor,
        T_cam_to_world: torch.Tensor,
        T_imu_to_world: torch.Tensor,
        camera_height_m: torch.Tensor,
        vehicle_yaw_degs: torch.Tensor,
        front_bev_xy: Optional[torch.Tensor] = None,
        front_ground_valid_mask: Optional[torch.Tensor] = None,
        target_size=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator=None,
    ):
        del T_cam_to_world, T_imu_to_world, camera_height_m, front_bev_xy, front_ground_valid_mask
        del num_inference_steps, guidance_scale, generator
        self.generate_pose_chain_calls.append(
            {
                "sat_shape": tuple(sat_images.shape),
                "K_shape": tuple(K.shape),
                "vehicle_yaw_degs": vehicle_yaw_degs.detach().cpu().clone(),
            }
        )
        batch_size, num_views = int(K.shape[0]), int(K.shape[1])
        height, width = target_size
        generated = torch.ones(
            (batch_size, num_views, 3, height, width),
            dtype=sat_images.dtype,
            device=sat_images.device,
        )
        return generated


class RandomYawGroundPETrainingTest(unittest.TestCase):
    def test_front_sample_prob_mixes_real_front_without_overriding_explicit_views(self) -> None:
        dataset = Kitti360dDataset.__new__(Kitti360dDataset)
        dataset.mode = "fisheye_virtual"
        dataset.yaw_mode = "vehicle_relative"
        dataset.vehicle_yaw_sampling = "random_range"
        dataset.front_sample_prob = 1.0

        base_sample = SampleIndex(drive_dir=Path("/tmp/drive"), frame_id=1, meta=None)
        override_sample = SampleIndex(
            drive_dir=Path("/tmp/drive"),
            frame_id=1,
            meta={"mode_override": "fisheye_virtual"},
        )

        rng = np.random.RandomState(0)
        self.assertEqual(dataset._resolve_effective_sample_mode(base_sample, rng, idx=0), "front")
        self.assertEqual(dataset._resolve_effective_sample_mode(override_sample, rng, idx=0), "fisheye_virtual")

    def test_fixed_vehicle_yaw_sampler_uses_front_and_discrete_yaws(self) -> None:
        dataset = Kitti360dDataset.__new__(Kitti360dDataset)
        dataset.mode = "fisheye_virtual"
        dataset.yaw_mode = "vehicle_relative"
        dataset.vehicle_yaw_sampling = "fixed_list"
        dataset.vehicle_yaw_fixed_list = Kitti360dDataset._normalize_vehicle_yaw_fixed_list(
            ["front", -120.0, -90.0, -60.0, 60.0, 90.0, 120.0]
        )
        dataset.front_sample_prob = 0.0
        dataset.samples = [SampleIndex(drive_dir=Path("/tmp/drive"), frame_id=i, meta=None) for i in range(7)]
        dataset.epoch = 0

        rng = np.random.RandomState(0)
        self.assertEqual(dataset._resolve_effective_sample_mode(dataset.samples[0], rng, idx=0), "front")
        self.assertEqual(dataset._resolve_effective_sample_mode(dataset.samples[1], rng, idx=1), "fisheye_virtual")
        self.assertIsNone(dataset._choose_fixed_vehicle_yaw(0))
        self.assertEqual(dataset._choose_fixed_vehicle_yaw(1), -120.0)
        self.assertEqual(dataset._choose_fixed_vehicle_yaw(6), 120.0)

    def test_fixed_vehicle_yaw_sampler_expands_each_frame_to_all_fixed_views(self) -> None:
        dataset = Kitti360dDataset.__new__(Kitti360dDataset)
        dataset.vehicle_yaw_fixed_list = Kitti360dDataset._normalize_vehicle_yaw_fixed_list(
            ["front", -120.0, -90.0, -60.0, 60.0, 90.0, 120.0]
        )
        base_samples = [
            SampleIndex(drive_dir=Path("/tmp/drive"), frame_id=10, meta=None),
            SampleIndex(drive_dir=Path("/tmp/drive"), frame_id=11, meta=None),
        ]

        expanded = dataset._expand_samples_for_fixed_vehicle_yaws(base_samples)

        self.assertEqual(len(expanded), 14)
        self.assertEqual([sample.frame_id for sample in expanded[:7]], [10] * 7)
        self.assertEqual([sample.frame_id for sample in expanded[7:]], [11] * 7)
        self.assertEqual(expanded[0].meta["mode_override"], "front")
        self.assertEqual(expanded[0].meta["view_name"], "front")
        self.assertNotIn("vehicle_relative_yaw_deg_override", expanded[0].meta)
        self.assertEqual(expanded[1].meta["mode_override"], "fisheye_virtual")
        self.assertEqual(expanded[1].meta["vehicle_relative_yaw_deg_override"], -120.0)
        self.assertEqual(expanded[1].meta["view_name"], "yaw_m120")
        self.assertEqual(expanded[6].meta["vehicle_relative_yaw_deg_override"], 120.0)
        self.assertEqual(expanded[6].meta["view_name"], "yaw_p120")

    def test_pose_chain_sampler_expands_each_frame_to_ordered_overlap_chains(self) -> None:
        dataset = Kitti360dDataset.__new__(Kitti360dDataset)
        dataset.pose_chains = Kitti360dDataset._normalize_pose_chains(
            [
                {"name": "right", "yaws": ["front", 60.0, 90.0, 120.0]},
                {"name": "left", "yaws": ["front", -60.0, -90.0, -120.0]},
            ]
        )
        base_samples = [SampleIndex(drive_dir=Path("/tmp/drive"), frame_id=10, meta=None)]

        expanded = dataset._expand_samples_for_pose_chains(base_samples)

        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[0].meta["pose_chain_name"], "right")
        self.assertEqual(expanded[0].meta["pose_chain_view_names"], ["front", "yaw_p60", "yaw_p90", "yaw_p120"])
        self.assertEqual(expanded[0].meta["pose_chain_yaws"], [None, 60.0, 90.0, 120.0])
        self.assertEqual(expanded[1].meta["pose_chain_name"], "left")
        self.assertEqual(expanded[1].meta["pose_chain_view_names"], ["front", "yaw_m60", "yaw_m90", "yaw_m120"])
        self.assertEqual(expanded[1].meta["pose_chain_yaws"], [None, -60.0, -90.0, -120.0])

    def test_pose_chain_sampler_rejects_mismatched_chain_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of views"):
            Kitti360dDataset._normalize_pose_chains(
                [
                    {"name": "right", "yaws": ["front", 60.0, 90.0, 120.0]},
                    {"name": "left", "yaws": ["front", -60.0, -90.0]},
                ]
            )

    def test_pose_chain_sampler_rejects_duplicate_chain_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            Kitti360dDataset._normalize_pose_chains(
                [
                    {"name": "right", "yaws": ["front", 60.0]},
                    {"name": "right", "yaws": ["front", 90.0]},
                ]
            )

    def test_pose_chain_sampler_rejects_string_yaws(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a string"):
            Kitti360dDataset._normalize_pose_chains(
                [
                    {"name": "right", "yaws": "front,60,90"},
                ]
            )

    def test_pose_chain_item_stacks_views_while_sharing_satellite(self) -> None:
        dataset = Kitti360dDataset.__new__(Kitti360dDataset)
        dataset.mode = "fisheye_virtual"
        dataset.sat_m_per_px = 0.2
        dataset.samples = [
            SampleIndex(
                drive_dir=Path("/tmp/drive"),
                frame_id=10,
                meta={
                    "pose_chain_name": "right",
                    "pose_chain_yaws": [None, 60.0, 90.0],
                    "pose_chain_view_names": ["front", "yaw_p60", "yaw_p90"],
                },
            )
        ]

        def fake_get_single_view_item(idx: int):
            sample = dataset.samples[idx]
            meta = dict(sample.meta or {})
            yaw = meta.get("vehicle_relative_yaw_deg_override")
            value = 0.0 if yaw is None else float(yaw) / 120.0
            return {
                "image": torch.full((3, 4, 4), value, dtype=torch.float32),
                "sat": torch.ones((3, 8, 8), dtype=torch.float32),
                "sat_available": True,
                "sat_m_per_px": 0.2,
                "camera_height_m": torch.tensor(1.0, dtype=torch.float32),
                "front_bev_xy": torch.zeros((2, 4, 4), dtype=torch.float32),
                "front_ground_valid_mask": torch.ones((1, 4, 4), dtype=torch.float32),
                "K": torch.eye(3, dtype=torch.float32),
                "T_pose_cam": torch.eye(4, dtype=torch.float32),
                "T_imu_to_world": torch.eye(4, dtype=torch.float32),
                "T_cam0_to_world": torch.eye(4, dtype=torch.float32),
                "T_cam_to_world": torch.eye(4, dtype=torch.float32),
                "frame_id": 10,
                "drive": "drive",
                "meta": meta,
            }

        dataset._get_single_view_item = fake_get_single_view_item

        item = dataset._get_pose_chain_item(0)

        self.assertEqual(tuple(item["image"].shape), (3, 3, 4, 4))
        self.assertEqual(tuple(item["sat"].shape), (3, 8, 8))
        self.assertEqual(tuple(item["K"].shape), (3, 3, 3))
        self.assertEqual(item["view_names"], ["front", "yaw_p60", "yaw_p90"])
        self.assertTrue(torch.isnan(item["vehicle_yaw_degs"][0]))
        self.assertEqual(item["vehicle_yaw_degs"][1:].tolist(), [60.0, 90.0])
        self.assertEqual(item["meta"]["chain_name"], "right")

    def test_compute_sat_patch_perspective_uv_projects_ground_points(self) -> None:
        K, T_cam_to_world, T_imu_to_world = _identity_geometry()
        bev_coords = torch.tensor(
            [[[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [1.0, 0.0]]],
            dtype=torch.float32,
        )

        uv, valid = compute_sat_patch_perspective_uv(
            bev_coords=bev_coords,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=torch.tensor([1.0], dtype=torch.float32),
            image_w=10,
            image_h=10,
        )

        self.assertTrue(torch.allclose(uv[0, 0], torch.tensor([0.0, 0.0]), atol=1e-5))
        self.assertTrue(torch.allclose(uv[0, 1], torch.tensor([0.2, 0.0]), atol=1e-5))
        self.assertTrue(torch.allclose(uv[0, 2], torch.tensor([0.0, -0.2]), atol=1e-5))
        self.assertEqual(valid.tolist(), [[True, True, True, False]])

    def test_query_uv_helpers_use_the_same_normalized_pixel_convention(self) -> None:
        uv = build_normalized_image_uv_grid(
            2,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertTrue(
            torch.allclose(
                uv[0],
                torch.tensor(
                    [
                        [-0.5, -0.5],
                        [0.5, -0.5],
                        [-0.5, 0.5],
                        [0.5, 0.5],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertEqual(infer_spatial_hw(4, (8, 8)), (2, 2))
        self.assertEqual(infer_spatial_hw(640, (32, 80)), (16, 40))

    def test_query_uv_attn_processor_runs_on_flattened_cross_attention(self) -> None:
        from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0

        torch.manual_seed(0)
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4, bias=False)
        attn.set_processor(
            QueryUVAttnProcessor2_0(
                query_dim=int(attn.to_q.out_features),
                query_uv_enabled=True,
                geometry_bias_enabled=False,
            )
        )
        hidden_states = torch.randn(1, 4, 8)
        encoder_hidden_states = torch.randn(1, 3, 8)

        output = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_base_hw=(2, 2),
        )

        self.assertEqual(output.shape, hidden_states.shape)

    def test_query_uv_attn_processor_records_attention_alignment(self) -> None:
        from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0

        torch.manual_seed(0)
        layer_name = "test_block.attn2"
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4, bias=False)
        attn.set_processor(
            QueryUVAttnProcessor2_0(
                query_dim=int(attn.to_q.out_features),
                query_uv_enabled=False,
                geometry_bias_enabled=False,
                layer_name=layer_name,
            )
        )
        hidden_states = torch.randn(1, 4, 8)
        encoder_hidden_states = torch.randn(1, 4, 8)
        sat_perspective_uv = build_normalized_image_uv_grid(
            2,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        attention_alignment = {
            "enabled": True,
            "layers": [layer_name],
            "max_query_tokens": 4,
            "valid_radius": 0.75,
            "losses": [],
            "metrics": [],
            "debug_storage": {},
        }

        output = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_base_hw=(2, 2),
            sat_perspective_uv=sat_perspective_uv,
            sat_perspective_valid=torch.ones((1, 4), dtype=torch.bool),
            attention_alignment=attention_alignment,
        )

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(len(attention_alignment["losses"]), 1)
        self.assertEqual(len(attention_alignment["metrics"]), 1)
        self.assertTrue(torch.is_tensor(attention_alignment["losses"][0]))
        metric = attention_alignment["metrics"][0]
        self.assertIn("target_attention_mass", metric)
        self.assertIn("target_token_fraction", metric)
        self.assertIn("nearest_attention_mass", metric)
        self.assertIn("target_logit_gap", metric)
        self.assertIn(layer_name, attention_alignment["debug_storage"])
        payload = attention_alignment["debug_storage"][layer_name]
        self.assertEqual(payload["attention"].shape, (1, 4, 4))
        self.assertEqual(payload["query_hw"], (2, 2))

    def test_query_geometry_score_lifts_matching_attention_targets(self) -> None:
        from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0

        torch.manual_seed(0)
        layer_name = "test_block.attn2"
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4, bias=False)
        attn.set_processor(
            QueryUVAttnProcessor2_0(
                query_dim=int(attn.to_q.out_features),
                query_uv_enabled=False,
                geometry_bias_enabled=False,
                geometry_score_enabled=True,
                geometry_score_dim=8,
                geometry_score_gate_init=2.0,
                geometry_score_layers=(layer_name,),
                layer_name=layer_name,
            )
        )
        hidden_states = torch.zeros(1, 4, 8)
        encoder_hidden_states = torch.zeros(1, 4, 8)
        sat_perspective_uv = build_normalized_image_uv_grid(
            2,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        attention_alignment = {
            "enabled": True,
            "layers": [layer_name],
            "max_query_tokens": 4,
            "valid_radius": 0.2,
            "losses": [],
            "metrics": [],
        }

        _ = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_base_hw=(2, 2),
            sat_perspective_uv=sat_perspective_uv,
            sat_perspective_valid=torch.ones((1, 4), dtype=torch.bool),
            attention_alignment=attention_alignment,
        )

        metric = attention_alignment["metrics"][0]
        self.assertGreater(float(metric["target_attention_lift"]), 1.0)
        self.assertGreater(float(metric["target_logit_gap"]), 0.0)
        self.assertIn("content_logits_std", metric)
        self.assertIn("geometry_to_content_std_ratio", metric)
        self.assertIn("attention_geometry_kl", metric)
        self.assertIn("target_attention_lift_without_geometry", metric)
        self.assertGreater(float(metric["target_attention_lift_geometry_delta"]), 0.0)
        self.assertIn("geometry_score_gate", metric)

    def test_geometry_first_attention_blocks_far_tokens_when_alpha_zero(self) -> None:
        from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0

        torch.manual_seed(0)
        layer_name = "test_block.attn2"
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4, bias=False)
        processor = QueryUVAttnProcessor2_0(
            query_dim=int(attn.to_q.out_features),
            geometry_score_enabled=True,
            geometry_score_gate_init=0.0,
            geometry_score_layers=(layer_name,),
            candidate_radius=0.1,
            candidate_min_k=1,
            semantic_score_alpha=0.25,
            layer_name=layer_name,
        )
        processor.set_semantic_score_runtime_alpha(0.0)
        attn.set_processor(processor)
        hidden_states = torch.randn(1, 1, 8)
        encoder_hidden_states = torch.randn(1, 2, 8)
        attention_alignment = {
            "enabled": True,
            "layers": [layer_name],
            "max_query_tokens": 1,
            "valid_radius": 0.2,
            "losses": [],
            "metrics": [],
            "debug_storage": {},
        }

        _ = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_base_hw=(1, 1),
            query_uv=torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
            sat_perspective_uv=torch.tensor([[[0.0, 0.0], [0.8, 0.0]]], dtype=torch.float32),
            sat_perspective_valid=torch.ones((1, 2), dtype=torch.bool),
            attention_alignment=attention_alignment,
        )

        attention = attention_alignment["debug_storage"][layer_name]["attention"]
        self.assertGreater(float(attention[0, 0, 0]), 0.999)
        self.assertLess(float(attention[0, 0, 1]), 1e-4)
        metric = attention_alignment["metrics"][0]
        self.assertIn("raw_content_qk_std", metric)
        self.assertIn("semantic_logits_std", metric)
        self.assertIn("candidate_recall", metric)
        self.assertIn("target_attention_lift_geometry_only", metric)

    def test_removed_query_geometry_bias_is_ignored_without_additive_query_pe(self) -> None:
        from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0

        torch.manual_seed(0)
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4, bias=False)
        processor = QueryUVAttnProcessor2_0(
            query_dim=int(attn.to_q.out_features),
            query_uv_enabled=False,
            geometry_bias_enabled=True,
            geometry_bias_scale=3.0,
        )
        attn.set_processor(processor)
        hidden_states = torch.randn(1, 4, 8)
        encoder_hidden_states = torch.randn(1, 3, 8)
        sat_perspective_uv = torch.tensor(
            [[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        sat_perspective_valid = torch.tensor([[True, False, True]])

        output = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_base_hw=(2, 2),
            sat_perspective_uv=sat_perspective_uv,
            sat_perspective_valid=sat_perspective_valid,
        )

        self.assertEqual(output.shape, hidden_states.shape)

    def test_trainer_uses_separate_geometry_score_lr_group(self) -> None:
        model = _GeometryScoreParamModel()
        dataloader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=None,
                learning_rate=1e-4,
                geometry_score_lr_multiplier=3.0,
                num_train_epochs=1,
                warmup_epochs=0,
                output_dir=tmpdir,
                save_every=10,
                log_every=10,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=0,
            )

            lrs = {group.get("name"): group["lr"] for group in trainer.optimizer.param_groups}
            self.assertAlmostEqual(lrs["base"], 1e-4)
            self.assertAlmostEqual(lrs["geometry_score"], 3e-4)
            score_group = next(group for group in trainer.optimizer.param_groups if group.get("name") == "geometry_score")
            score_param_ids = {id(param) for param in score_group["params"]}
            self.assertIn(id(model.unet.processor.semantic_query_proj.weight), score_param_ids)
            self.assertIn(id(model.unet.processor.semantic_key_proj.weight), score_param_ids)

    def test_trainer_applies_geometry_score_gate_runtime_warmup(self) -> None:
        model = _GeometryScoreWarmupModel()
        dataloader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=None,
                geometry_score_gate_warmup_steps=10,
                geometry_score_gate_warmup_start_scale=0.25,
                geometry_score_gate_warmup_end_scale=1.0,
                num_train_epochs=1,
                output_dir=tmpdir,
                save_every=10,
                log_every=10,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=0,
            )

            self.assertAlmostEqual(model.unet.scales[-1], 0.25)
            self.assertAlmostEqual(trainer._apply_geometry_score_gate_warmup(5), 0.625)
            self.assertAlmostEqual(model.unet.scales[-1], 0.625)
            self.assertAlmostEqual(trainer._apply_geometry_score_gate_warmup(10), 1.0)

    def test_trainer_applies_semantic_score_alpha_schedule(self) -> None:
        model = _GeometryScoreWarmupModel()
        dataloader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=None,
                semantic_score_alpha_max=0.25,
                semantic_score_alpha_hold_steps=10,
                semantic_score_alpha_warmup_steps=20,
                num_train_epochs=1,
                output_dir=tmpdir,
                save_every=10,
                log_every=10,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=0,
            )

            self.assertAlmostEqual(model.unet.semantic_alphas[-1], 0.0)
            self.assertAlmostEqual(trainer._apply_semantic_score_alpha_schedule(10), 0.0)
            self.assertAlmostEqual(trainer._apply_semantic_score_alpha_schedule(20), 0.125)
            self.assertAlmostEqual(trainer._apply_semantic_score_alpha_schedule(30), 0.25)

    def test_query_uv_attn_processor_does_not_create_additive_query_encoder(self) -> None:
        from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0

        processor = QueryUVAttnProcessor2_0(
            query_dim=16,
            query_uv_enabled=True,
            geometry_bias_enabled=False,
        )

        self.assertIsNone(processor.query_uv_encoder)
        self.assertIsNone(processor.query_uv_gate)
        self.assertFalse(any("query_uv" in name for name, _ in processor.named_parameters()))

    def test_satellite_condition_encoder_uses_rope_self_attention(self) -> None:
        encoder = SatelliteConditionEncoder(
            embed_dim=32,
            patch_size=4,
            sat_resolution=0.2,
            sat_size=16,
            perspective_pe_enabled=False,
            num_heads=4,
            num_layers=2,
            attn_dropout=0.0,
        )

        self.assertFalse(hasattr(encoder, "grid_pos_embed"))
        layer0 = encoder.self_attn.layers[0]
        self.assertEqual(layer0.self_attn.head_dim, 8)
        self.assertGreater(layer0.self_attn.qkv.weight.abs().max().item(), 0.0)
        self.assertGreater(layer0.self_attn.out_proj.weight.abs().max().item(), 0.0)
        self.assertGreater(layer0.mlp[3].weight.abs().max().item(), 0.0)

    def test_group_forward_encodes_satellite_once_and_flattens_pose_views(self) -> None:
        torch.manual_seed(0)
        model = _SmallPerspectiveModel()
        original_forward = model.satellite_encoder.forward
        call_count = {"value": 0}

        def counted_forward(*args, **kwargs):
            call_count["value"] += 1
            return original_forward(*args, **kwargs)

        model.satellite_encoder.forward = counted_forward
        sat = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        target = torch.zeros((1, 2, 3, 8, 8), dtype=torch.float32)
        K, T_cam_to_world, T_imu_to_world = _identity_geometry(batch_size=2)
        outputs = model(
            sat,
            target,
            K=K.reshape(1, 2, 3, 3),
            T_cam_to_world=T_cam_to_world.reshape(1, 2, 4, 4),
            T_imu_to_world=T_imu_to_world.reshape(1, 2, 4, 4),
            camera_height_m=torch.ones((1, 2), dtype=torch.float32),
        )

        self.assertEqual(call_count["value"], 1)
        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(tuple(outputs["per_view_denoise_loss"].shape), (1, 2))
        self.assertEqual(float(outputs["chain_group_size"].item()), 2.0)
        self.assertEqual(outputs["sat_state"].tokens.shape[0], 2)

    def test_group_projection_uses_each_views_own_camera_intrinsics(self) -> None:
        model = _SmallPerspectiveModel()
        bev_coords = torch.tensor(
            [
                [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
                [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
            ],
            dtype=torch.float32,
        )
        sat_state = SatelliteMemoryState(
            tokens=torch.ones((2, 3, 4), dtype=torch.float32),
            xy=torch.zeros((2, 3, 2), dtype=torch.float32),
            bev_coords=bev_coords,
        )
        K, T_cam_to_world, T_imu_to_world = _identity_geometry(batch_size=2)
        K[1, 0, 2] = K[1, 0, 2] + 2.0

        projected = model._project_group_satellite_state(
            sat_state,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=torch.ones((2,), dtype=torch.float32),
            image_size=(10, 10),
        )

        self.assertTrue(torch.equal(projected.tokens, sat_state.tokens))
        self.assertEqual(tuple(projected.perspective_uv.shape), (2, 3, 2))
        self.assertEqual(tuple(projected.perspective_valid.shape), (2, 3))
        self.assertFalse(torch.allclose(projected.perspective_uv[0], projected.perspective_uv[1]))

    def test_group_condition_dropout_is_shared_within_each_pose_chain(self) -> None:
        torch.manual_seed(0)
        model = _SmallPerspectiveModel()
        model.train()

        def fixed_group_mask(batch_size: int, device: torch.device) -> torch.Tensor:
            self.assertEqual(batch_size, 2)
            return torch.tensor([True, False], device=device)

        model._sample_condition_mask = fixed_group_mask
        sat = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        target = torch.zeros((2, 3, 3, 8, 8), dtype=torch.float32)
        K, T_cam_to_world, T_imu_to_world = _identity_geometry(batch_size=6)

        outputs = model(
            sat,
            target,
            K=K.reshape(2, 3, 3, 3),
            T_cam_to_world=T_cam_to_world.reshape(2, 3, 4, 4),
            T_imu_to_world=T_imu_to_world.reshape(2, 3, 4, 4),
            camera_height_m=torch.ones((2, 3), dtype=torch.float32),
        )

        self.assertEqual(
            outputs["condition_mask"].tolist(),
            [True, True, True, False, False, False],
        )
        self.assertEqual(tuple(outputs["per_view_denoise_loss"].shape), (2, 3))
        self.assertEqual(tuple(outputs["chain_denoise_loss"].shape), (2,))

    def test_group_forward_reports_chain_mean_denoise_loss(self) -> None:
        model = _SmallPerspectiveModel()
        per_item = torch.tensor([1.0, 3.0, 2.0, 8.0])

        def fake_forward_denoising(target_images, sat_state, *, condition_mask=None, chain_group_size=None):
            self.assertEqual(tuple(target_images.shape[:1]), (4,))
            self.assertEqual(chain_group_size, 2)
            return {
                "loss": per_item.mean(),
                "per_item_denoise_loss": per_item.clone(),
                "sat_state": sat_state,
                "condition_mask": condition_mask,
            }

        model._forward_denoising_with_sat_state = fake_forward_denoising
        sat = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        target = torch.zeros((2, 2, 3, 8, 8), dtype=torch.float32)
        K, T_cam_to_world, T_imu_to_world = _identity_geometry(batch_size=4)

        outputs = model(
            sat,
            target,
            K=K.reshape(2, 2, 3, 3),
            T_cam_to_world=T_cam_to_world.reshape(2, 2, 4, 4),
            T_imu_to_world=T_imu_to_world.reshape(2, 2, 4, 4),
            camera_height_m=torch.ones((2, 2), dtype=torch.float32),
        )

        self.assertTrue(torch.equal(outputs["per_view_denoise_loss"], per_item.reshape(2, 2)))
        self.assertTrue(torch.equal(outputs["chain_denoise_loss"], torch.tensor([2.0, 5.0])))
        self.assertEqual(float(outputs["loss"].item()), 3.5)

    def test_predict_original_sample_matches_scheduler_formulas(self) -> None:
        model = _SmallPerspectiveModel()
        noisy = torch.tensor([[[[2.0]]], [[[3.0]]]], dtype=torch.float32)
        pred = torch.tensor([[[[0.5]]], [[[1.0]]]], dtype=torch.float32)
        timesteps = torch.tensor([0, 1], dtype=torch.long)
        model.noise_scheduler.alphas_cumprod = torch.tensor([0.25, 0.64], dtype=torch.float32)

        model.noise_scheduler.config.prediction_type = "epsilon"
        expected_epsilon = (noisy - torch.tensor([0.75, 0.36]).sqrt().reshape(2, 1, 1, 1) * pred) / torch.tensor(
            [0.25, 0.64]
        ).sqrt().reshape(2, 1, 1, 1)
        self.assertTrue(
            torch.allclose(
                model._predict_original_sample(noisy_latents=noisy, model_pred=pred, timesteps=timesteps),
                expected_epsilon,
            )
        )

        model.noise_scheduler.config.prediction_type = "v_prediction"
        expected_v = torch.tensor([0.25, 0.64]).sqrt().reshape(2, 1, 1, 1) * noisy - torch.tensor(
            [0.75, 0.36]
        ).sqrt().reshape(2, 1, 1, 1) * pred
        self.assertTrue(
            torch.allclose(
                model._predict_original_sample(noisy_latents=noisy, model_pred=pred, timesteps=timesteps),
                expected_v,
            )
        )

    def test_attention_alignment_kwargs_stay_enabled_for_validation_metrics(self) -> None:
        model = _SmallPerspectiveModel()
        model.attention_alignment_enabled = True
        model.attention_alignment_loss_weight = 0.5
        model.eval()
        sat_state = SatelliteMemoryState(
            tokens=torch.zeros((2, 4, 4), dtype=torch.float32),
            xy=torch.zeros((2, 4, 2), dtype=torch.float32),
            bev_coords=torch.zeros((2, 4, 2), dtype=torch.float32),
            perspective_uv=torch.zeros((2, 4, 2), dtype=torch.float32),
            perspective_valid=torch.ones((2, 4), dtype=torch.bool),
        )

        kwargs = model._build_cross_attention_kwargs(
            torch.zeros((2, 3, 8, 8), dtype=torch.float32),
            sat_state,
            chain_group_size=2,
        )

        self.assertIsNotNone(kwargs)
        self.assertIn("attention_alignment", kwargs)
        self.assertEqual(kwargs["attention_alignment"]["chain_group_size"], 2)

        outputs = model._forward_denoising_with_sat_state(
            torch.zeros((2, 3, 8, 8), dtype=torch.float32),
            sat_state,
            condition_mask=torch.ones((2,), dtype=torch.bool),
            chain_group_size=2,
        )

        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(float(outputs["attention_alignment_loss_weight"].item()), 0.0)

    def test_query_uv_attn_processor_records_alignment_metrics_in_eval_mode(self) -> None:
        from models.unet.query_uv_attn_processor import QueryUVAttnProcessor2_0

        torch.manual_seed(0)
        layer_name = "test_block.attn2"
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4, bias=False)
        attn.set_processor(
            QueryUVAttnProcessor2_0(
                query_dim=int(attn.to_q.out_features),
                geometry_score_enabled=True,
                geometry_score_dim=8,
                geometry_score_gate_init=2.0,
                geometry_score_layers=(layer_name,),
                layer_name=layer_name,
            )
        )
        attn.eval()
        attention_alignment = {
            "enabled": True,
            "layers": [layer_name],
            "max_query_tokens": 4,
            "valid_radius": 0.2,
            "losses": [],
            "metrics": [],
            "chain_group_size": 2,
        }

        with torch.no_grad():
            _ = attn(
                torch.zeros(2, 4, 8),
                encoder_hidden_states=torch.zeros(2, 4, 8),
                query_base_hw=(2, 2),
                sat_perspective_uv=build_normalized_image_uv_grid(
                    2,
                    2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                ).repeat(2, 1, 1),
                sat_perspective_valid=torch.ones((2, 4), dtype=torch.bool),
                attention_alignment=attention_alignment,
            )

        self.assertEqual(len(attention_alignment["metrics"]), 1)
        metric = attention_alignment["metrics"][0]
        self.assertIn("target_attention_lift_mixed", metric)
        self.assertIn("target_attention_lift_geometry_only", metric)
        self.assertIn("chain_attention_coverage_overlap", metric)
        self.assertIn("chain_attention_centroid_shift", metric)
        self.assertEqual(len(attention_alignment["losses"]), 1)

    def test_trainer_train_epoch_accepts_pose_chain_batches(self) -> None:
        torch.manual_seed(0)
        model = _SmallPerspectiveModel()
        dataloader = DataLoader(_SinglePoseChainDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=None,
                num_train_epochs=1,
                output_dir=tmpdir,
                save_every=10,
                log_every=1,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=0,
            )

            epoch_loss = trainer._train_epoch(0)

        self.assertTrue(np.isfinite(epoch_loss))
        self.assertAlmostEqual(model.unet.call_sat_token_means[-1], 0.0, places=6)

    def test_query_uv_sliced_attn_processor_preserves_query_base_hw(self) -> None:
        torch.manual_seed(0)
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4, bias=False)
        attn.set_processor(
            QueryUVSlicedAttnProcessor(
                query_dim=int(attn.to_q.out_features),
                slice_size=1,
                query_uv_enabled=True,
                geometry_bias_enabled=False,
            )
        )
        hidden_states = torch.randn(1, 4, 8)
        encoder_hidden_states = torch.randn(1, 3, 8)

        output = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_base_hw=(2, 2),
        )

        self.assertEqual(output.shape, hidden_states.shape)

    def test_unet_attention_slicing_keeps_geometry_score_processor(self) -> None:
        torch.manual_seed(0)
        model = SatelliteConditionedUNet(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            center_input_sample=False,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(16, 32),
            layers_per_block=1,
            cross_attention_dim=16,
            attention_head_dim=8,
            norm_num_groups=8,
            query_geometry_score_enabled=True,
            query_geometry_score_dim=8,
            query_geometry_score_max_query_tokens=64,
        )

        model.set_attention_slice("auto")
        self.assertTrue(
            any(
                isinstance(processor, QueryUVSlicedAttnProcessor) and processor.geometry_score_enabled
                for processor in model.attn_processors.values()
            )
        )

        latents = torch.randn(1, 4, 8, 8)
        encoder_hidden_states = torch.randn(1, 4, 16)
        output = model(
            latents,
            torch.tensor([0], dtype=torch.long),
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={
                "query_base_hw": (8, 8),
                "sat_perspective_uv": build_normalized_image_uv_grid(
                    2,
                    2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                ),
                "sat_perspective_valid": torch.ones((1, 4), dtype=torch.bool),
            },
        ).sample

        self.assertEqual(output.shape, latents.shape)

    def test_unet_geometry_score_runs_under_attention_slicing(self) -> None:
        torch.manual_seed(0)
        model = SatelliteConditionedUNet(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            center_input_sample=False,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(16, 32),
            layers_per_block=1,
            cross_attention_dim=16,
            attention_head_dim=8,
            norm_num_groups=8,
            query_geometry_score_enabled=True,
            query_geometry_score_dim=8,
            query_geometry_score_max_query_tokens=64,
        )

        model.set_attention_slice("auto")
        self.assertTrue(
            any(
                isinstance(processor, QueryUVSlicedAttnProcessor) and processor.geometry_bias_enabled
                or isinstance(processor, QueryUVSlicedAttnProcessor) and processor.geometry_score_enabled
                for processor in model.attn_processors.values()
            )
        )

        latents = torch.randn(1, 4, 8, 8)
        encoder_hidden_states = torch.randn(1, 4, 16)
        sat_perspective_uv = torch.tensor(
            [[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        sat_perspective_valid = torch.tensor([[True, True, False, True]])
        output = model(
            latents,
            torch.tensor([0], dtype=torch.long),
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={
                "query_base_hw": (8, 8),
                "sat_perspective_uv": sat_perspective_uv,
                "sat_perspective_valid": sat_perspective_valid,
            },
        ).sample

        self.assertEqual(output.shape, latents.shape)

    def test_unet_attention_debug_captures_selected_cross_attention_layer(self) -> None:
        torch.manual_seed(0)
        model = SatelliteConditionedUNet(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            center_input_sample=False,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(16, 32),
            layers_per_block=1,
            cross_attention_dim=16,
            attention_head_dim=8,
            norm_num_groups=8,
            query_geometry_score_enabled=True,
            query_geometry_score_dim=8,
            query_geometry_score_max_query_tokens=64,
        )
        layer_name = next(
            name.removesuffix(".processor")
            for name in model.attn_processors
            if name.endswith(".attn2.processor")
        )
        storage = {}
        model.enable_attention_debug(layers=[layer_name], storage=storage)

        latents = torch.randn(1, 4, 8, 8)
        encoder_hidden_states = torch.randn(1, 4, 16)
        sat_perspective_uv = build_normalized_image_uv_grid(
            2,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        output = model(
            latents,
            torch.tensor([0], dtype=torch.long),
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={
                "query_base_hw": (8, 8),
                "sat_perspective_uv": sat_perspective_uv,
                "sat_perspective_valid": torch.ones((1, 4), dtype=torch.bool),
            },
        ).sample
        model.disable_attention_debug()

        self.assertEqual(output.shape, latents.shape)
        self.assertIn(layer_name, storage)
        self.assertEqual(storage[layer_name]["attention"].shape[0], 1)

    def test_model_forward_stores_perspective_state_and_uses_only_sat_tokens(self) -> None:
        torch.manual_seed(0)
        model = _SmallPerspectiveModel()
        model.train()
        K, T_cam_to_world, T_imu_to_world = _identity_geometry()

        outputs = model(
            torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            torch.zeros((1, 3, 10, 10), dtype=torch.float32),
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=torch.tensor([1.0], dtype=torch.float32),
        )

        self.assertTrue(torch.is_tensor(outputs["loss"]))
        sat_state = outputs["sat_state"]
        self.assertIsNotNone(sat_state.perspective_uv)
        self.assertIsNotNone(sat_state.perspective_valid)
        self.assertEqual(sat_state.perspective_uv.shape, (1, 4, 2))
        self.assertEqual(model.unet.extra_kwarg_keys, [("cross_attention_kwargs",)])
        self.assertEqual(model.unet.cross_attention_kwargs[0]["query_base_hw"], (10, 10))

    def test_invalid_target_rank_is_rejected(self) -> None:
        model = _SmallPerspectiveModel()
        K, T_cam_to_world, T_imu_to_world = _identity_geometry()
        with self.assertRaisesRegex(ValueError, "target_images must be"):
            model(
                torch.zeros((1, 3, 8, 8), dtype=torch.float32),
                torch.zeros((1, 1, 2, 3, 8, 8), dtype=torch.float32),
                K=K,
                T_cam_to_world=T_cam_to_world,
                T_imu_to_world=T_imu_to_world,
                camera_height_m=torch.tensor([1.0], dtype=torch.float32),
            )

    def test_resume_checkpoint_restores_next_epoch_index(self) -> None:
        model = _TrainableLossModel()
        dataloader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=None,
                num_train_epochs=5,
                output_dir=tmpdir,
                save_every=10,
                log_every=10,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=0,
            )

            checkpoint_path = f"{tmpdir}/resume.pt"
            torch.save(
                {
                    "epoch": 2,
                    "model_state_dict": trainer.unwrapped_model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.lr_scheduler.state_dict(),
                },
                checkpoint_path,
            )

            resumed_epoch = trainer._load_checkpoint(checkpoint_path)
            self.assertEqual(resumed_epoch, 3)

            observed_epochs = []
            original_train_epoch = trainer._train_epoch

            def _record_epoch(epoch: int) -> float:
                observed_epochs.append(epoch)
                return original_train_epoch(epoch)

            trainer._train_epoch = _record_epoch
            trainer.train(resume_from=checkpoint_path)

            self.assertEqual(observed_epochs, [3, 4])

    def test_saved_checkpoint_includes_sanitized_run_config(self) -> None:
        model = _TrainableLossModel()
        dataloader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=None,
                num_train_epochs=1,
                output_dir=tmpdir,
                save_every=10,
                log_every=10,
                validate_every=10,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=0,
                run_config={
                    "view_set": "pose_chain",
                    "pose_chains": [{"name": "right", "yaws": ["front", 60.0]}],
                    "pose_chain_group_size": 2,
                    "effective_view_batch_size": 4,
                    "data_dir": Path("/tmp/kitti"),
                    "tuple_value": ("a", 1),
                },
            )

            trainer._save_checkpoint(0)
            checkpoint = torch.load(
                Path(tmpdir) / "checkpoints" / "checkpoint_epoch_1.pt",
                map_location="cpu",
            )

        self.assertEqual(checkpoint["run_config"]["view_set"], "pose_chain")
        self.assertEqual(checkpoint["run_config"]["data_dir"], "/tmp/kitti")
        self.assertEqual(checkpoint["run_config"]["pose_chain_group_size"], 2)
        self.assertEqual(checkpoint["run_config"]["effective_view_batch_size"], 4)
        self.assertEqual(checkpoint["run_config"]["tuple_value"], ["a", 1])
        self.assertEqual(checkpoint["trainer_metadata"]["checkpoint_epoch"], 1)
        self.assertEqual(checkpoint["trainer_metadata"]["validate_every"], 10)

    def test_nonfinite_loss_skips_optimizer_update(self) -> None:
        model = _TrainableLossModel(nonfinite=True)
        dataloader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=None,
                num_train_epochs=1,
                output_dir=tmpdir,
                save_every=10,
                log_every=1,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=0,
            )

            before = model.trainable_weight.detach().clone()
            epoch_loss = trainer._train_epoch(0)

            self.assertEqual(epoch_loss, 0.0)
            self.assertTrue(torch.allclose(model.trainable_weight.detach(), before))
            self.assertEqual(trainer.optimizer.state_dict()["state"], {})

    def test_train_step_logging_always_includes_epoch_tail(self) -> None:
        self.assertTrue(SDTrainer._should_log_train_step(step=52, num_batches=53, log_every=100))
        self.assertTrue(SDTrainer._should_log_train_step(step=99, num_batches=153, log_every=100))
        self.assertFalse(SDTrainer._should_log_train_step(step=51, num_batches=53, log_every=100))

    def test_validate_every_runs_on_interval_and_final_epoch(self) -> None:
        self.assertFalse(SDTrainer._should_validate_epoch(epoch=0, num_train_epochs=100, validate_every=10))
        self.assertTrue(SDTrainer._should_validate_epoch(epoch=9, num_train_epochs=100, validate_every=10))
        self.assertFalse(SDTrainer._should_validate_epoch(epoch=98, num_train_epochs=100, validate_every=0))
        self.assertTrue(SDTrainer._should_validate_epoch(epoch=99, num_train_epochs=100, validate_every=0))
        self.assertTrue(SDTrainer._should_validate_epoch(epoch=94, num_train_epochs=95, validate_every=10))

    def test_scalar_metrics_are_written_to_local_jsonl(self) -> None:
        model = _TrainableLossModel()
        dataloader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=dataloader,
                num_train_epochs=1,
                output_dir=tmpdir,
                save_every=10,
                log_every=10,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=0,
            )

            trainer._log_scalars(
                {
                    "val/loss": 0.5,
                    "val/attention_alignment_target_attention_lift_mixed": 1.25,
                    "skip_none": None,
                },
                step=7,
            )
            lines = trainer.scalar_log_path.read_text().splitlines()

        self.assertEqual(len(lines), 1)
        self.assertIn('"step": 7', lines[0])
        self.assertIn('"val/loss": 0.5', lines[0])
        self.assertIn('"val/attention_alignment_target_attention_lift_mixed": 1.25', lines[0])
        self.assertNotIn("skip_none", lines[0])

    def test_chain_coverage_metrics_use_adjacent_pose_pairs(self) -> None:
        batch = {
            "image": torch.zeros((1, 3, 3, 2, 2), dtype=torch.float32),
            "front_bev_xy": torch.tensor(
                [
                    [
                        [[[-0.5, 0.0], [-0.5, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                        [[[0.0, 0.5], [0.0, 0.5]], [[0.0, 0.0], [0.0, 0.0]]],
                        [[[0.5, 0.9], [0.5, 0.9]], [[0.0, 0.0], [0.0, 0.0]]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "front_ground_valid_mask": torch.ones((1, 3, 1, 2, 2), dtype=torch.float32),
        }

        metrics = SDTrainer._compute_chain_coverage_metrics(batch, grid_size=4)

        self.assertIn("chain/coverage_overlap", metrics)
        self.assertIn("chain/coverage_centroid_shift", metrics)
        self.assertEqual(metrics["chain/group_size"], 3.0)
        self.assertEqual(metrics["chain/valid_pair_ratio"], 1.0)

    def test_validation_output_scalar_metric_collection_filters_non_scalars(self) -> None:
        metrics = SDTrainer._collect_output_scalar_metrics(
            {
                "attention_alignment_chain_attention_coverage_overlap": torch.tensor(0.5),
                "attention_alignment_chain_attention_centroid_shift": torch.tensor(0.25),
                "per_view_denoise_loss": torch.ones((1, 4)),
                "missing": None,
            },
            (
                "attention_alignment_chain_attention_coverage_overlap",
                "attention_alignment_chain_attention_centroid_shift",
                "per_view_denoise_loss",
                "missing",
            ),
        )

        self.assertEqual(
            metrics,
            {
                "attention_alignment_chain_attention_coverage_overlap": 0.5,
                "attention_alignment_chain_attention_centroid_shift": 0.25,
            },
        )

    def test_visualization_generation_saves_joint_contact_sheets(self) -> None:
        model = _VisualizationModel()
        dataloader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)

        with TemporaryDirectory() as tmpdir:
            trainer = SDTrainer(
                model=model,
                train_dataloader=dataloader,
                val_dataloader=dataloader,
                num_train_epochs=1,
                output_dir=tmpdir,
                save_every=10,
                log_every=10,
                device="cpu",
                use_wandb=False,
                use_tensorboard=False,
                mixed_precision=None,
                visualize_every=1,
                num_visualizations=1,
                visualization_inference_steps=1,
            )

            trainer._save_visualizations(epoch=0)

            epoch_dir = trainer.visualization_dir / "epoch_0001"
            self.assertEqual(len(model.generate_pose_chain_calls), 2)
            self.assertTrue((epoch_dir / "joint_contact_sheet.png").is_file())
            self.assertTrue(any(path.name.startswith("joint_right") for path in epoch_dir.iterdir()))
            self.assertTrue(any(path.name.startswith("joint_left") for path in epoch_dir.iterdir()))
            self.assertEqual(model.generate_pose_chain_calls[0]["K_shape"][:2], (1, 4))
            self.assertEqual(model.generate_pose_chain_calls[1]["K_shape"][:2], (1, 4))
            self.assertTrue(torch.isnan(model.generate_pose_chain_calls[0]["vehicle_yaw_degs"][0, 0]))
            self.assertEqual(model.generate_pose_chain_calls[0]["vehicle_yaw_degs"][0, 1:].tolist(), [60.0, 90.0, 120.0])
            self.assertEqual(model.generate_pose_chain_calls[1]["vehicle_yaw_degs"][0, 1:].tolist(), [-60.0, -90.0, -120.0])

    def test_visualization_sample_override_sets_front_and_virtual_yaw(self) -> None:
        base_sample = SampleIndex(drive_dir=Path("/tmp/drive"), frame_id=7, meta=None)
        front_sample = SDTrainer._sample_with_visualization_view(base_sample, "front", None)
        yaw_sample = SDTrainer._sample_with_visualization_view(base_sample, "yaw_m90", -90.0)

        self.assertEqual(front_sample.meta["mode_override"], "front")
        self.assertNotIn("vehicle_relative_yaw_deg_override", front_sample.meta)
        self.assertEqual(yaw_sample.meta["mode_override"], "fisheye_virtual")
        self.assertEqual(yaw_sample.meta["vehicle_relative_yaw_deg_override"], -90.0)

    def test_satellite_visualization_draws_bev_coverage_overlay(self) -> None:
        sat_image = torch.zeros((3, 8, 8), dtype=torch.float32)
        front_bev_xy = torch.tensor(
            [
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, 0.5], [-0.5, -0.5]],
            ],
            dtype=torch.float32,
        )

        image = SDTrainer._draw_satellite_view_coverage(
            sat_image,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=None,
            view_label="yaw_p90",
            yaw_deg=90.0,
        )

        pixels = torch.from_numpy(np.array(image))
        self.assertGreater(int(pixels.sum().item()), 0)


if __name__ == "__main__":
    unittest.main()
