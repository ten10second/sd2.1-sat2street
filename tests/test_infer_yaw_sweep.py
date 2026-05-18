import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from PIL import Image

from scripts import infer


class InferYawSweepTest(unittest.TestCase):
    def test_default_single_yaw_sweep_specs(self) -> None:
        specs = infer._single_yaw_sweep_view_specs(
            SimpleNamespace(vehicle_yaws=None, include_front=True, yaw_sweep_preset="diagnostic")
        )

        self.assertEqual(
            specs,
            [
                ("front", None),
                ("yaw_m120", -120.0),
                ("yaw_m90", -90.0),
                ("yaw_m60", -60.0),
                ("yaw_m30", -30.0),
                ("yaw_p30", 30.0),
                ("yaw_p60", 60.0),
                ("yaw_p90", 90.0),
                ("yaw_p120", 120.0),
            ],
        )

    def test_train_fixed_single_yaw_sweep_specs(self) -> None:
        specs = infer._single_yaw_sweep_view_specs(
            SimpleNamespace(vehicle_yaws=None, include_front=True, yaw_sweep_preset="train_fixed")
        )

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

    def test_train_fixed_single_yaw_sweep_can_omit_front(self) -> None:
        specs = infer._single_yaw_sweep_view_specs(
            SimpleNamespace(vehicle_yaws=None, include_front=False, yaw_sweep_preset="train_fixed")
        )

        self.assertEqual(
            specs,
            [
                ("yaw_m120", -120.0),
                ("yaw_m90", -90.0),
                ("yaw_m60", -60.0),
                ("yaw_p60", 60.0),
                ("yaw_p90", 90.0),
                ("yaw_p120", 120.0),
            ],
        )

    def test_explicit_yaw_sweep_uses_signed_tokens(self) -> None:
        specs = infer._single_yaw_sweep_view_specs(
            SimpleNamespace(
                vehicle_yaws=[-30.0, 0.0, 30.0],
                include_front=False,
                yaw_sweep_preset="train_fixed",
            )
        )

        self.assertEqual(
            specs,
            [
                ("yaw_m30", -30.0),
                ("yaw_0", 0.0),
                ("yaw_p30", 30.0),
            ],
        )

    def test_pitch_token_uses_pitch_prefix(self) -> None:
        self.assertEqual(infer._pitch_token(0.0), "pitch_0")
        self.assertEqual(infer._pitch_token(5.0), "pitch_p5")
        self.assertEqual(infer._pitch_token(-2.5), "pitch_m2p5")

    def test_front_pitch_sweep_uses_front_samples_for_each_pitch(self) -> None:
        base_sample = SimpleNamespace(drive_dir=Path("/tmp/drive_0001_sync"), frame_id=42)
        dataset = SimpleNamespace(samples=[base_sample], pitch_deg=12.0, roll_deg=-1.0)
        sample_calls = []
        saved_views = []

        def fake_get_view_sample(dataset_arg, sample_index, view_name, yaw):
            self.assertIs(dataset_arg, dataset)
            self.assertEqual(sample_index, 0)
            sample_calls.append((view_name, yaw, float(dataset_arg.pitch_deg), float(dataset_arg.roll_deg)))
            return {
                "image": torch.zeros((3, 4, 4), dtype=torch.float32),
                "sat": torch.zeros((3, 4, 4), dtype=torch.float32),
                "front_bev_xy": torch.zeros((2, 4, 4), dtype=torch.float32),
                "front_ground_valid_mask": torch.ones((1, 4, 4), dtype=torch.float32),
                "drive": "drive_0001_sync",
                "frame_id": 42,
                "meta": {"view_name": view_name},
            }

        def fake_save_view_outputs(
            sample,
            generated,
            output_dir,
            view_name,
            yaw,
            ablation_name,
            sat_condition_mode,
            gt_override=None,
        ):
            del sample, generated, ablation_name, sat_condition_mode
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_views.append((view_name, yaw, gt_override is not None))
            return Image.new("RGB", (4, 4))

        with TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                pitch_values=[0.0, 5.0, -2.5],
                roll_deg=3.0,
                output_dir=tmpdir,
                mode="front_pitch_sweep",
                view_memory_mode="independent",
                checkpoint="checkpoint.pt",
            )

            with patch.object(infer, "_resolve_single_dataset", return_value=(dataset, 0)), \
                patch.object(infer, "_get_view_sample", side_effect=fake_get_view_sample), \
                patch.object(infer, "_load_model", return_value=(object(), {"epoch": 1})), \
                patch.object(infer, "_resolve_ablation_runs", return_value=[(None, "normal")]), \
                patch.object(infer, "_generate_one", return_value=torch.zeros((3, 4, 4), dtype=torch.float32)), \
                patch.object(infer, "_save_view_outputs", side_effect=fake_save_view_outputs), \
                patch.object(infer, "_stack_panel_rows", return_value=Image.new("RGB", (4, 12))):
                infer.run_front_pitch_sweep(args)

        self.assertEqual(
            [(view_name, yaw) for view_name, yaw, _, _ in sample_calls],
            [
                ("front", None),
                ("pitch_0", None),
                ("pitch_0", None),
                ("pitch_p5", None),
                ("pitch_m2p5", None),
            ],
        )
        self.assertEqual([pitch for _, _, pitch, _ in sample_calls], [12.0, 0.0, 0.0, 5.0, -2.5])
        self.assertTrue(all(yaw is None for _, yaw, _ in saved_views))
        self.assertTrue(all(has_gt_override for _, _, has_gt_override in saved_views))
        self.assertEqual(float(dataset.pitch_deg), 12.0)
        self.assertEqual(float(dataset.roll_deg), -1.0)

    def test_parse_args_accepts_independent_view_memory_mode(self) -> None:
        with patch(
            "sys.argv",
            [
                "infer.py",
                "--view_memory_mode",
                "independent",
            ],
        ):
            args = infer._parse_args()

        self.assertEqual(args.view_memory_mode, "independent")


if __name__ == "__main__":
    unittest.main()
