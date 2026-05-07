import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.conditioning import SatelliteMemoryState
from models.sd_trainer import SDTrainer, SatelliteConditionedSDModel


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


if __name__ == "__main__":
    unittest.main()
