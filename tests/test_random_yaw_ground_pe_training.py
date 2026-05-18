import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.kitti360d_dataset import Kitti360dDataset, SampleIndex, _apply_virtual_pose_delta
from models.conditioning import SatelliteMemoryState
from models.sd_trainer import SDTrainer, SatelliteConditionedSDModel
from models.unet.cross_view_refinement_block import CrossViewRefinementBlock
from models.unet.street_to_satellite_attention import StreetToSatelliteAttention


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


class _DummyScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=10, prediction_type="epsilon")
        self.step_calls = 0

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return latents + noise

    def get_last_lr(self):
        return [0.0]

    def step(self) -> None:
        self.step_calls += 1


class _DummyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.supports_cross_view_refinement = True
        self.config = SimpleNamespace(in_channels=3, sample_size=8)
        self.call_sat_token_means = []
        self.last_satellite_state = None
        self.last_refinement_stats = {}

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states=None,
        sat_tokens: torch.Tensor = None,
        sat_xy: torch.Tensor = None,
        sat_bev_coords: torch.Tensor = None,
        front_bev_xy: torch.Tensor = None,
        front_ground_valid_mask: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        return_attn_map: bool = False,
    ) -> SimpleNamespace:
        del timesteps, encoder_hidden_states, front_ground_valid_mask, return_attn_map
        self.call_sat_token_means.append(float(sat_tokens.mean().item()))
        view_signal = front_bev_xy.mean(dim=(1, 2, 3), keepdim=True).to(sat_tokens.dtype)
        active = condition_mask.to(dtype=sat_tokens.dtype).view(-1, 1, 1)
        updated_tokens = sat_tokens + active * view_signal
        self.last_satellite_state = SatelliteMemoryState(
            tokens=updated_tokens,
            xy=sat_xy,
            bev_coords=sat_bev_coords,
        )
        signal_scalar = view_signal.mean()
        self.last_refinement_stats = {
            "mid": {
                "logits_sem_std": signal_scalar + 1.0,
                "logits_geom_std": signal_scalar + 2.0,
                "logits_geom_to_sem_ratio": signal_scalar + 3.0,
                "sat_update_norm": signal_scalar + 4.0,
                "adapter_residual_norm": signal_scalar + 5.0,
            }
        }
        return SimpleNamespace(sample=noisy_latents)


class _SingleSampleDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        del idx
        return {
            "sat": torch.zeros((3, 8, 8), dtype=torch.float32),
            "image": torch.zeros((3, 8, 8), dtype=torch.float32),
            "front_bev_xy": torch.zeros((2, 8, 8), dtype=torch.float32),
            "front_ground_valid_mask": torch.ones((1, 8, 8), dtype=torch.float32),
        }


class _TestSatelliteConditionedSDModel(SatelliteConditionedSDModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.unet = _DummyUNet()
        self.vae = _DummyVAE()
        self.noise_scheduler = _DummyScheduler()
        self.cond_drop_prob = 0.0

    def encode_satellite(self, sat_images: torch.Tensor) -> SatelliteMemoryState:
        batch_size = sat_images.shape[0]
        tokens = torch.zeros((batch_size, 4, 3), dtype=sat_images.dtype, device=sat_images.device)
        xy = torch.zeros((batch_size, 4, 2), dtype=sat_images.dtype, device=sat_images.device)
        return SatelliteMemoryState(tokens=tokens, xy=xy, bev_coords=None)


class _ResumeTestModel(_TestSatelliteConditionedSDModel):
    def __init__(self) -> None:
        super().__init__()
        self.trainable_weight = nn.Parameter(torch.tensor(1.0))

    def forward_view_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        target_images: torch.Tensor,
        front_bev_xy: torch.Tensor = None,
        front_ground_valid_mask: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
    ):
        del sat_state, target_images, front_bev_xy, front_ground_valid_mask, condition_mask
        loss = self.trainable_weight.square()
        return {
            "loss": loss,
            "sat_state": self.encode_satellite(torch.zeros((1, 3, 8, 8), dtype=loss.dtype, device=loss.device)),
            "refinement_stats": {},
            "refinement_stats_by_site": {},
        }


class _NonFiniteLossModel(_ResumeTestModel):
    def forward_view_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        target_images: torch.Tensor,
        front_bev_xy: torch.Tensor = None,
        front_ground_valid_mask: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
    ):
        del sat_state, target_images, front_bev_xy, front_ground_valid_mask, condition_mask
        finite_anchor = self.trainable_weight * 0.0
        return {
            "loss": finite_anchor + torch.tensor(float("nan"), device=finite_anchor.device),
            "sat_state": self.encode_satellite(torch.zeros((1, 3, 8, 8), dtype=finite_anchor.dtype, device=finite_anchor.device)),
            "refinement_stats": {},
            "refinement_stats_by_site": {},
        }


class _VisualizationSingleViewModel(_TestSatelliteConditionedSDModel):
    def __init__(self) -> None:
        super().__init__()
        self.generate_sat_token_means = []
        self.dummy_trainable = nn.Parameter(torch.tensor(0.0))

    def generate_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        front_bev_xy: torch.Tensor = None,
        front_ground_valid_mask: torch.Tensor = None,
        target_size=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator=None,
        sat_condition_mode: str = "normal",
    ):
        del front_ground_valid_mask, num_inference_steps, guidance_scale, generator, sat_condition_mode
        self.generate_sat_token_means.append(float(sat_state.tokens.mean().item()))
        batch_size = sat_state.tokens.shape[0]
        height, width = target_size
        generated = torch.full(
            (batch_size, 3, height, width),
            fill_value=float(len(self.generate_sat_token_means)),
            dtype=sat_state.tokens.dtype,
            device=sat_state.tokens.device,
        )
        view_signal = front_bev_xy.mean(dim=(1, 2, 3), keepdim=True).to(sat_state.tokens.dtype)
        updated_state = sat_state.replace(tokens=sat_state.tokens + view_signal)
        return generated, updated_state


class RandomYawGroundPETrainingTest(unittest.TestCase):
    def test_apply_virtual_pose_delta_changes_rotation_for_pitch(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        original_pose = pose.copy()

        zero_delta = _apply_virtual_pose_delta(pose, pitch_deg=0.0, roll_deg=0.0)
        pitched = _apply_virtual_pose_delta(pose, pitch_deg=10.0, roll_deg=0.0)

        self.assertTrue(np.allclose(zero_delta, pose))
        self.assertTrue(np.allclose(pose, original_pose))
        self.assertTrue(np.allclose(pitched[:3, 3], pose[:3, 3]))
        self.assertFalse(np.allclose(pitched[:3, :3], pose[:3, :3]))
        self.assertTrue(np.allclose(pitched[:3, :3] @ pitched[:3, :3].T, np.eye(3), atol=1e-6))

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

    def test_single_view_forward_initializes_satellite_state_per_batch(self) -> None:
        torch.manual_seed(0)
        model = _TestSatelliteConditionedSDModel()
        model.train()

        sat_images = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        target_images = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        front_bev_xy = torch.ones((1, 2, 8, 8), dtype=torch.float32)
        front_ground_valid_mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)

        outputs = model(
            sat_images,
            target_images,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
        )

        self.assertEqual(len(model.unet.call_sat_token_means), 1)
        self.assertAlmostEqual(model.unet.call_sat_token_means[0], 0.0, places=5)
        self.assertTrue(torch.is_tensor(outputs["loss"]))
        self.assertAlmostEqual(float(outputs["sat_state"].tokens.mean().item()), 1.0, places=5)
        self.assertIn("refinement_logits_geom_to_sem_ratio_mean", outputs)
        self.assertIn("refinement_sat_update_norm_mean", outputs)
        self.assertAlmostEqual(float(outputs["refinement_sat_update_norm_mean"].item()), 5.0, places=5)
        self.assertIn("refinement_adapter_residual_norm_mean", outputs)

    def test_rank5_view_group_forward_is_rejected(self) -> None:
        model = _TestSatelliteConditionedSDModel()
        model.train()

        sat_images = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        target_images = torch.zeros((1, 2, 3, 8, 8), dtype=torch.float32)
        front_bev_xy = torch.zeros((1, 2, 2, 8, 8), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "one random-yaw street view per sample"):
            model(
                sat_images,
                target_images,
                front_bev_xy=front_bev_xy,
            )

    def test_resume_checkpoint_restores_next_epoch_index(self) -> None:
        model = _ResumeTestModel()
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

    def test_street_to_sat_uses_ground_pe_only(self) -> None:
        torch.manual_seed(0)
        front_feat = torch.randn((1, 4, 2, 2), dtype=torch.float32)
        front_bev_xy = torch.tensor(
            [[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        sat_tokens = torch.randn((1, 3, 6), dtype=torch.float32)
        sat_xy = torch.tensor([[[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]], dtype=torch.float32)

        attention = StreetToSatelliteAttention(
            sat_in_dim=6,
            front_in_dim=4,
            num_heads=2,
            head_dim=8,
            geom_head_dim=2,
        )
        attention.eval()
        with torch.no_grad():
            output, _, stats = attention(
                front_feat,
                front_bev_xy,
                sat_tokens,
                sat_xy,
                front_ground_valid_mask=None,
            )
        self.assertEqual(output.shape, sat_tokens.shape)
        self.assertIn("logits_geom_to_sem_ratio", stats)

    def test_street_to_sat_out_proj_is_zero_initialized(self) -> None:
        torch.manual_seed(0)
        attention = StreetToSatelliteAttention(
            sat_in_dim=6,
            front_in_dim=4,
            num_heads=2,
            head_dim=8,
            geom_head_dim=2,
        )

        self.assertTrue(torch.allclose(attention.out_proj.weight, torch.zeros_like(attention.out_proj.weight)))
        self.assertTrue(torch.allclose(attention.out_proj.bias, torch.zeros_like(attention.out_proj.bias)))

        front_feat = torch.randn((1, 4, 2, 2), dtype=torch.float32)
        front_bev_xy = torch.tensor(
            [[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        sat_tokens = torch.randn((1, 3, 6), dtype=torch.float32)
        sat_xy = torch.tensor([[[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]], dtype=torch.float32)

        attention.eval()
        with torch.no_grad():
            _, _, stats = attention(
                front_feat,
                front_bev_xy,
                sat_tokens,
                sat_xy,
                front_ground_valid_mask=None,
            )

        self.assertTrue(torch.allclose(stats["sat_update_norm"], torch.tensor(0.0)))

    def test_cross_view_refinement_stacks_sat_update_layers(self) -> None:
        torch.manual_seed(0)
        block = CrossViewRefinementBlock(
            front_dim=4,
            sat_in_dim=6,
            num_heads=2,
            head_dim=8,
            geom_head_dim=2,
            sat_update_layers=2,
        )
        block.eval()

        front_feat = torch.randn((1, 4, 2, 2), dtype=torch.float32)
        front_bev_xy = torch.tensor(
            [[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        sat_state = SatelliteMemoryState(
            tokens=torch.randn((1, 3, 6), dtype=torch.float32),
            xy=torch.tensor([[[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]], dtype=torch.float32),
            bev_coords=None,
        )

        with torch.no_grad():
            output = block(front_feat, sat_state, front_bev_xy)

        self.assertEqual(len(block.street_to_sat_layers), 2)
        self.assertEqual(len(block.sat_self_refine_layers), 2)
        stats = output["stats"]
        self.assertIn("sat_update_l0_sat_update_norm", stats)
        self.assertIn("sat_update_l1_sat_update_norm", stats)
        self.assertIn("sat_update_norm", stats)
        self.assertIn("adapter_residual_norm", stats)
        self.assertTrue(torch.allclose(stats["sat_update_norm"], stats["sat_update_l1_sat_update_norm"]))
        self.assertEqual(output["satellite_state"].tokens.shape, sat_state.tokens.shape)
        self.assertIsNotNone(output["adapter_residual"])
        self.assertEqual(output["adapter_residual"].shape, front_feat.shape)
        self.assertTrue(torch.allclose(output["adapter_residual"], torch.zeros_like(front_feat)))

    def test_cross_view_refinement_adapter_residual_can_affect_front_feature(self) -> None:
        torch.manual_seed(0)
        block = CrossViewRefinementBlock(
            front_dim=4,
            sat_in_dim=6,
            num_heads=2,
            head_dim=8,
            geom_head_dim=2,
            sat_update_layers=1,
            adapter_residual=True,
        )
        block.eval()
        with torch.no_grad():
            block.adapter_out.weight.fill_(0.05)
            block.adapter_out.bias.fill_(0.01)

        front_feat = torch.randn((1, 4, 2, 2), dtype=torch.float32)
        front_bev_xy = torch.tensor(
            [[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        sat_state = SatelliteMemoryState(
            tokens=torch.randn((1, 3, 6), dtype=torch.float32),
            xy=torch.tensor([[[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]], dtype=torch.float32),
            bev_coords=None,
        )

        with torch.no_grad():
            output = block(front_feat, sat_state, front_bev_xy)

        residual = output["adapter_residual"]
        self.assertIsNotNone(residual)
        self.assertEqual(residual.shape, front_feat.shape)
        self.assertGreater(float(output["stats"]["adapter_residual_norm"].item()), 0.0)

    def test_cross_view_refinement_rejects_empty_sat_update_stack(self) -> None:
        with self.assertRaisesRegex(ValueError, "sat_update_layers must be positive"):
            CrossViewRefinementBlock(
                front_dim=4,
                sat_in_dim=6,
                num_heads=2,
                head_dim=8,
                sat_update_layers=0,
            )

    def test_nonfinite_loss_skips_optimizer_update(self) -> None:
        model = _NonFiniteLossModel()
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
        model = _VisualizationSingleViewModel()

        class _SingleVisualizationDataset(Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                del idx
                return {
                    "sat": torch.zeros((3, 8, 8), dtype=torch.float32),
                    "image": torch.zeros((3, 8, 8), dtype=torch.float32),
                    "front_bev_xy": torch.ones((2, 8, 8), dtype=torch.float32),
                    "front_ground_valid_mask": torch.ones((1, 8, 8), dtype=torch.float32),
                    "frame_id": torch.tensor(7),
                }

        dataloader = DataLoader(_SingleVisualizationDataset(), batch_size=1, shuffle=False)

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
