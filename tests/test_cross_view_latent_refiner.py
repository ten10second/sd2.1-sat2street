from types import SimpleNamespace

import torch

from models.cross_view_refiner import CrossViewLatentRefiner
from models.sd_model import SatelliteConditionedSDModel


def test_cross_view_refiner_shape_and_finite_with_bev_bias():
    refiner = CrossViewLatentRefiner(
        latent_channels=4,
        hidden_dim=16,
        num_heads=4,
        gate_init=0.1,
        bev_sigma=0.5,
    )
    pred_x0 = torch.randn(2, 4, 4, 3, 5)
    target = torch.randn_like(pred_x0)
    yaw = torch.tensor(
        [
            [0.0, 60.0, 90.0, 120.0],
            [0.0, -60.0, -90.0, -120.0],
        ]
    )
    bev_x = torch.linspace(-1.0, 1.0, 6).view(1, 1, 1, 1, 6).expand(2, 4, 1, 4, 6)
    bev_y = torch.linspace(0.0, 1.0, 4).view(1, 1, 1, 4, 1).expand(2, 4, 1, 4, 6)
    front_bev_xy = torch.cat([bev_x, bev_y], dim=2)
    valid = torch.ones(2, 4, 1, 4, 6)

    output = refiner(
        pred_x0,
        target_latents=target,
        vehicle_yaw_degs=yaw,
        front_bev_xy=front_bev_xy,
        front_ground_valid_mask=valid,
    )

    assert output.refined_x0.shape == pred_x0.shape
    assert torch.isfinite(output.refined_x0).all()
    assert torch.isfinite(output.consistency_loss)
    assert output.metrics["joint_view_generation/num_adjacent_directions"].item() == 6.0
    assert output.metrics["joint_view_generation/valid_match_ratio"].item() > 0.0


def test_cross_view_refiner_adjacent_only():
    refiner = CrossViewLatentRefiner(latent_channels=4, hidden_dim=8, num_heads=2, gate_init=1.0)
    pred_x0 = torch.randn(1, 4, 4, 2, 2)
    calls = []

    def fake_attend(src_tokens, dst_tokens, *, src_idx, dst_idx, **kwargs):
        calls.append((src_idx, dst_idx, src_tokens.shape[0], dst_tokens.shape[0]))
        zero = src_tokens.sum() * 0.0
        return torch.zeros_like(dst_tokens), zero, zero, torch.ones_like(zero)

    refiner._attend_direction = fake_attend  # type: ignore[method-assign]
    output = refiner(pred_x0)

    assert output.refined_x0.shape == pred_x0.shape
    assert calls == [
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 2, 1, 1),
        (2, 1, 1, 1),
        (2, 3, 1, 1),
        (3, 2, 1, 1),
    ]


def test_cross_view_refiner_zero_gate_identity():
    refiner = CrossViewLatentRefiner(latent_channels=4, hidden_dim=8, num_heads=2, gate_init=0.0)
    pred_x0 = torch.randn(2, 3, 4, 2, 2)

    output = refiner(
        pred_x0,
        vehicle_yaw_degs=torch.tensor([[float("nan"), 60.0, 90.0], [float("nan"), -60.0, -90.0]]),
    )

    assert torch.allclose(output.refined_x0, pred_x0)
    assert torch.isfinite(output.metrics["joint_view_generation/attention_entropy"])


def test_joint_view_generation_samples_one_timestep_per_chain():
    model = SatelliteConditionedSDModel.__new__(SatelliteConditionedSDModel)
    model.joint_view_generation_enabled = True
    model.noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=100))

    timesteps = model._sample_training_timesteps(
        batch_size=8,
        device=torch.device("cpu"),
        chain_group_size=4,
    )

    assert timesteps.shape == (8,)
    assert torch.equal(timesteps[0:4], timesteps[0].expand(4))
    assert torch.equal(timesteps[4:8], timesteps[4].expand(4))


def test_refined_x0_converts_back_to_scheduler_prediction_types():
    model = SatelliteConditionedSDModel.__new__(SatelliteConditionedSDModel)
    alphas = torch.tensor([0.25, 0.64, 0.81], dtype=torch.float32)
    timesteps = torch.tensor([1, 2], dtype=torch.long)
    noisy = torch.randn(2, 4, 2, 2)
    x0 = torch.randn_like(noisy)

    model.noise_scheduler = SimpleNamespace(
        config=SimpleNamespace(prediction_type="epsilon"),
        alphas_cumprod=alphas,
    )
    eps = model._predict_model_output_from_original_sample(
        noisy_latents=noisy,
        pred_x0=x0,
        timesteps=timesteps,
    )
    recovered_x0 = model._predict_original_sample(
        noisy_latents=noisy,
        model_pred=eps,
        timesteps=timesteps,
    )
    assert torch.allclose(recovered_x0, x0, atol=1e-5)

    model.noise_scheduler.config.prediction_type = "v_prediction"
    velocity = model._predict_model_output_from_original_sample(
        noisy_latents=noisy,
        pred_x0=x0,
        timesteps=timesteps,
    )
    recovered_x0 = model._predict_original_sample(
        noisy_latents=noisy,
        model_pred=velocity,
        timesteps=timesteps,
    )
    assert torch.allclose(recovered_x0, x0, atol=1e-5)

    model.noise_scheduler.config.prediction_type = "sample"
    sample = model._predict_model_output_from_original_sample(
        noisy_latents=noisy,
        pred_x0=x0,
        timesteps=timesteps,
    )
    assert torch.equal(sample, x0)
