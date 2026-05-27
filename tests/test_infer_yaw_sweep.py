import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts import infer
from models.sd_trainer import SDTrainer


class InferYawSweepTest(unittest.TestCase):
    def test_default_single_yaw_sweep_specs(self) -> None:
        specs = infer._single_yaw_sweep_view_specs(
            SimpleNamespace(vehicle_yaws=None, include_front=False, yaw_sweep_preset="diagnostic")
        )

        self.assertEqual(
            specs,
            [
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

    def test_train_fixed_single_yaw_sweep_can_explicitly_include_front_for_diagnostics(self) -> None:
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

    def test_train_fixed_preset_matches_trainer_visualization_specs(self) -> None:
        specs = infer._single_yaw_sweep_view_specs(
            SimpleNamespace(vehicle_yaws=None, include_front=False, yaw_sweep_preset="train_fixed")
        )

        self.assertEqual(specs, SDTrainer._visualization_view_specs())

    def test_pose_zero_ablation_keeps_satellite_and_zeros_pose_only(self) -> None:
        runs = infer._resolve_ablation_runs(
            SimpleNamespace(ablation_modes=["normal", "sat_zero", "pose_zero"], sat_condition_mode="normal")
        )

        self.assertEqual(
            runs,
            [
                ("normal", "normal", "normal"),
                ("sat_zero", "zero", "normal"),
                ("pose_zero", "normal", "zero"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
