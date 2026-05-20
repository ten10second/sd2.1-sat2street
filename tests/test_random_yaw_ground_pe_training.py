import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from torch.utils.data import DataLoader, Dataset

from data.kitti360d_dataset import Kitti360dDataset, SampleIndex
from models.conditioning import SatelliteMemoryState
from models.encoders.perspective_position_encoder import compute_sat_patch_perspective_uv
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.sd_model import SatelliteConditionedSDModel
from models.sd_trainer import SDTrainer
from models.unet.query_uv_attn_processor import build_normalized_image_uv_grid, infer_spatial_hw


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
        self.timesteps = torch.arange(1)

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return latents + noise

    def get_last_lr(self):
        return [0.0]

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
        self.query_uv_pe_enabled = True
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


class _SmallPerspectiveModel(SatelliteConditionedSDModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.unet = _DummyUNet(embed_dim=4)
        self.vae = _DummyVAE()
        self.noise_scheduler = _DummyScheduler()
        self.cond_drop_prob = 0.0
        self.perspective_pe_enabled = True
        self.satellite_encoder = SatelliteConditionEncoder(
            embed_dim=4,
            patch_size=4,
            perspective_pe_enabled=True,
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


class _VisualizationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy_trainable = nn.Parameter(torch.tensor(0.0))
        self.generate_sat_token_means = []

    def encode_satellite(self, sat_images: torch.Tensor, **geometry) -> SatelliteMemoryState:
        del geometry
        batch_size = sat_images.shape[0]
        return SatelliteMemoryState(
            tokens=torch.zeros((batch_size, 4, 3), dtype=sat_images.dtype, device=sat_images.device),
            xy=torch.zeros((batch_size, 4, 2), dtype=sat_images.dtype, device=sat_images.device),
        )

    def generate_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        target_size=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator=None,
        sat_condition_mode: str = "normal",
    ):
        del num_inference_steps, guidance_scale, generator, sat_condition_mode
        self.generate_sat_token_means.append(float(sat_state.tokens.mean().item()))
        batch_size = sat_state.tokens.shape[0]
        height, width = target_size
        generated = torch.ones(
            (batch_size, 3, height, width),
            dtype=sat_state.tokens.dtype,
            device=sat_state.tokens.device,
        )
        return generated, sat_state


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

    def test_rank5_view_group_forward_is_rejected(self) -> None:
        model = _SmallPerspectiveModel()
        K, T_cam_to_world, T_imu_to_world = _identity_geometry()
        with self.assertRaisesRegex(ValueError, "one random-yaw street view per sample"):
            model(
                torch.zeros((1, 3, 8, 8), dtype=torch.float32),
                torch.zeros((1, 2, 3, 8, 8), dtype=torch.float32),
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

    def test_visualization_generation_uses_single_view_batch(self) -> None:
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

            self.assertEqual(model.generate_sat_token_means, [0.0])
            self.assertTrue((trainer.visualization_dir / "epoch_0001" / "sample_00_frame_0000000007.png").is_file())

    def test_visualization_view_specs_include_front_and_yaw_sweep(self) -> None:
        specs = SDTrainer._visualization_view_specs()
        self.assertEqual(
            specs,
            [
                ("front", None),
                ("yaw_m120", -120.0),
                ("yaw_m90", -90.0),
                ("yaw_m60", -60.0),
                ("yaw_p60", 60.0),
                ("yaw_p90", 90.0),
                ("yaw_p120", 120.0),
            ],
        )

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
