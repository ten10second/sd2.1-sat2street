import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.conditioning import SatelliteMemoryState
from models.sd_trainer import SatelliteConditionedSDModel


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

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return latents + noise


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


if __name__ == "__main__":
    unittest.main()
