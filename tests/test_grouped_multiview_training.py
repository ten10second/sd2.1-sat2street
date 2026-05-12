import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.conditioning import SatelliteMemoryState
from models.sd_model import PluckerCameraTokenProjector, SatelliteConditionedUNet
from models.sd_trainer import SDTrainer, SatelliteConditionedSDModel
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
        front_plucker: torch.Tensor = None,
        front_ground_valid_mask: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        return_attn_map: bool = False,
    ) -> SimpleNamespace:
        del timesteps, encoder_hidden_states, front_plucker, front_ground_valid_mask, return_attn_map
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
            "plucker_map": torch.zeros((6, 8, 8), dtype=torch.float32),
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
        plucker_map: torch.Tensor = None,
        front_ground_valid_mask: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
    ):
        del sat_state, target_images, front_bev_xy, plucker_map, front_ground_valid_mask, condition_mask
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
        plucker_map: torch.Tensor = None,
        front_ground_valid_mask: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
    ):
        del sat_state, target_images, front_bev_xy, plucker_map, front_ground_valid_mask, condition_mask
        finite_anchor = self.trainable_weight * 0.0
        return {
            "loss": finite_anchor + torch.tensor(float("nan"), device=finite_anchor.device),
            "sat_state": self.encode_satellite(torch.zeros((1, 3, 8, 8), dtype=finite_anchor.dtype, device=finite_anchor.device)),
            "refinement_stats": {},
            "refinement_stats_by_site": {},
        }


class _VisualizationSequentialModel(_TestSatelliteConditionedSDModel):
    def __init__(self) -> None:
        super().__init__()
        self.generate_sat_token_means = []
        self.dummy_trainable = nn.Parameter(torch.tensor(0.0))

    def generate_with_satellite_state(
        self,
        sat_state: SatelliteMemoryState,
        front_bev_xy: torch.Tensor = None,
        plucker_map: torch.Tensor = None,
        front_ground_valid_mask: torch.Tensor = None,
        target_size=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator=None,
        sat_condition_mode: str = "normal",
    ):
        del plucker_map, front_ground_valid_mask, num_inference_steps, guidance_scale, generator, sat_condition_mode
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


class _CameraTokenHarness(nn.Module):
    _get_camera_plucker = SatelliteConditionedUNet._get_camera_plucker
    _camera_tokens = SatelliteConditionedUNet._camera_tokens

    def __init__(self, projector: nn.Module, token_scale: float) -> None:
        super().__init__()
        self.enable_camera_control = True
        self.camera_projector = projector
        self.camera_token_scale = token_scale


class GroupedMultiViewTrainingTest(unittest.TestCase):
    def test_grouped_forward_reuses_updated_satellite_state(self) -> None:
        torch.manual_seed(0)
        model = _TestSatelliteConditionedSDModel()
        model.train()

        sat_images = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        target_images = torch.zeros((1, 2, 3, 8, 8), dtype=torch.float32)
        front_bev_xy = torch.zeros((1, 2, 2, 8, 8), dtype=torch.float32)
        front_bev_xy[:, 0] = 1.0
        front_bev_xy[:, 1] = 2.0
        plucker_map = torch.zeros((1, 2, 6, 8, 8), dtype=torch.float32)
        front_ground_valid_mask = torch.ones((1, 2, 1, 8, 8), dtype=torch.float32)

        outputs = model(
            sat_images,
            target_images,
            front_bev_xy=front_bev_xy,
            plucker_map=plucker_map,
            front_ground_valid_mask=front_ground_valid_mask,
        )

        self.assertEqual(len(model.unet.call_sat_token_means), 2)
        self.assertAlmostEqual(model.unet.call_sat_token_means[0], 0.0, places=5)
        self.assertAlmostEqual(model.unet.call_sat_token_means[1], 1.0, places=5)
        self.assertTrue(torch.is_tensor(outputs["loss"]))
        self.assertEqual(tuple(outputs["per_view_loss"].shape), (2,))
        self.assertAlmostEqual(float(outputs["sat_state"].tokens.mean().item()), 3.0, places=5)
        self.assertIn("refinement_logits_geom_to_sem_ratio_mean", outputs)

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

    def test_street_to_sat_plucker_geometry_can_be_disabled(self) -> None:
        torch.manual_seed(0)
        front_feat = torch.randn((1, 4, 2, 2), dtype=torch.float32)
        front_bev_xy = torch.tensor(
            [[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        sat_tokens = torch.randn((1, 3, 6), dtype=torch.float32)
        sat_xy = torch.tensor([[[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]], dtype=torch.float32)
        zero_plucker = torch.zeros((1, 4, 6), dtype=torch.float32)
        shifted_plucker = torch.randn((1, 4, 6), dtype=torch.float32) * 10.0

        disabled = StreetToSatelliteAttention(
            sat_in_dim=6,
            front_in_dim=4,
            num_heads=2,
            head_dim=8,
            geom_head_dim=2,
            use_plucker_geom=False,
        )
        disabled.eval()
        with torch.no_grad():
            disabled_zero, _, _ = disabled(front_feat, front_bev_xy, sat_tokens, sat_xy, zero_plucker)
            disabled_shifted, _, _ = disabled(front_feat, front_bev_xy, sat_tokens, sat_xy, shifted_plucker)
        self.assertTrue(torch.allclose(disabled_zero, disabled_shifted, atol=1e-6))

        enabled = StreetToSatelliteAttention(
            sat_in_dim=6,
            front_in_dim=4,
            num_heads=2,
            head_dim=8,
            geom_head_dim=2,
            use_plucker_geom=True,
        )
        enabled.load_state_dict(disabled.state_dict())
        enabled.eval()
        with torch.no_grad():
            enabled_zero, _, _ = enabled(front_feat, front_bev_xy, sat_tokens, sat_xy, zero_plucker)
            enabled_shifted, _, _ = enabled(front_feat, front_bev_xy, sat_tokens, sat_xy, shifted_plucker)
        self.assertFalse(torch.allclose(enabled_zero, enabled_shifted, atol=1e-6))

    def test_camera_token_scale_is_applied(self) -> None:
        torch.manual_seed(0)
        projector = PluckerCameraTokenProjector(token_dim=4, patch_size=2, zero_init=False)
        unet = _CameraTokenHarness(projector=projector, token_scale=0.25)

        plucker_map = torch.randn((1, 6, 4, 4), dtype=torch.float32)
        reference = torch.zeros((1, 2, 4), dtype=torch.float32)

        expected = projector(plucker_map) * 0.25
        actual = unet._camera_tokens(plucker_map, reference, condition_mask=None)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

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

    def test_visualization_generation_reuses_updated_satellite_state(self) -> None:
        model = _VisualizationSequentialModel()

        class _GroupedVisualizationDataset(Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                del idx
                front_bev_xy = torch.zeros((2, 2, 8, 8), dtype=torch.float32)
                front_bev_xy[0] = 1.0
                front_bev_xy[1] = 2.0
                return {
                    "sat": torch.zeros((3, 8, 8), dtype=torch.float32),
                    "image": torch.zeros((2, 3, 8, 8), dtype=torch.float32),
                    "front_bev_xy": front_bev_xy,
                    "front_ground_valid_mask": torch.ones((2, 1, 8, 8), dtype=torch.float32),
                    "plucker_map": torch.zeros((2, 6, 8, 8), dtype=torch.float32),
                    "frame_id": torch.tensor(7),
                    "view_names": ["front", "left_side"],
                }

        dataloader = DataLoader(_GroupedVisualizationDataset(), batch_size=1, shuffle=False)

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

            self.assertEqual(model.generate_sat_token_means, [0.0, 1.0])
            self.assertTrue((trainer.visualization_dir / "epoch_0001" / "sample_00_frame_0000000007.png").is_file())


if __name__ == "__main__":
    unittest.main()
