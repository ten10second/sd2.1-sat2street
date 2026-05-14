import unittest
from unittest.mock import patch
from types import SimpleNamespace

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
