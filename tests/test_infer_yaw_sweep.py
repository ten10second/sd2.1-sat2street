import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

from scripts import infer


class InferYawSweepTest(unittest.TestCase):
    def test_right_chain_yaw_sweep_specs(self) -> None:
        specs = infer._single_yaw_sweep_view_specs(
            SimpleNamespace(vehicle_yaws=None, include_front=True, yaw_sweep_preset="right_chain")
        )

        self.assertEqual(
            specs,
            [
                ("front", None),
                ("yaw_p60", 60.0),
                ("yaw_p90", 90.0),
                ("yaw_p120", 120.0),
            ],
        )

    def test_left_chain_yaw_sweep_specs(self) -> None:
        specs = infer._single_yaw_sweep_view_specs(
            SimpleNamespace(vehicle_yaws=None, include_front=True, yaw_sweep_preset="left_chain")
        )

        self.assertEqual(
            specs,
            [
                ("front", None),
                ("yaw_m60", -60.0),
                ("yaw_m90", -90.0),
                ("yaw_m120", -120.0),
            ],
        )

    def test_parse_args_rejects_independent_view_memory_mode(self) -> None:
        with patch(
            "sys.argv",
            [
                "infer.py",
                "--view_memory_mode",
                "independent",
            ],
        ):
            with self.assertRaises(SystemExit):
                infer._parse_args()

    def test_parse_args_defaults_to_joint_right_chain(self) -> None:
        with patch("sys.argv", ["infer.py"]):
            args = infer._parse_args()

        self.assertEqual(args.view_memory_mode, "joint_pose_chain")
        self.assertEqual(args.yaw_sweep_preset, "right_chain")

    def test_parse_args_accepts_split_yaw_sweep_mode(self) -> None:
        with patch(
            "sys.argv",
            [
                "infer.py",
                "--mode",
                "split_yaw_sweep",
                "--yaw_sweep_preset",
                "right_chain",
            ],
        ):
            args = infer._parse_args()

        self.assertEqual(args.mode, "split_yaw_sweep")
        self.assertEqual(args.yaw_sweep_preset, "right_chain")

    def test_parse_args_accepts_test_split_alias(self) -> None:
        with patch(
            "sys.argv",
            [
                "infer.py",
                "--dataset_split",
                "test",
            ],
        ):
            args = infer._parse_args()

        self.assertEqual(args.dataset_split, "test")

    def test_load_split_from_yaml_uses_test_entries_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            drive_dir = data_dir / "2013_05_28_drive_0003_sync"
            drive_dir.mkdir(parents=True)
            (drive_dir / "train_frames.txt").write_text("1\n2\n")
            (drive_dir / "val_frames.txt").write_text("10\n11\n")
            (drive_dir / "test_frames.txt").write_text("20\n21\n")
            split_yaml = Path(tmp) / "split.yaml"
            split_yaml.write_text(
                "\n".join(
                    [
                        "train:",
                        "  - drive: 2013_05_28_drive_0003_sync",
                        "    frames_file: train_frames.txt",
                        "val:",
                        "  - drive: 2013_05_28_drive_0003_sync",
                        "    frames_file: val_frames.txt",
                        "test:",
                        "  - drive: 2013_05_28_drive_0003_sync",
                        "    frames_file: test_frames.txt",
                    ]
                )
            )

            train_dirs, train_frames, eval_dirs, eval_frames = infer._load_split_from_yaml(
                data_dir,
                split_yaml,
                eval_split="test",
            )

        self.assertEqual([path.name for path in train_dirs], ["2013_05_28_drive_0003_sync"])
        self.assertEqual([path.name for path in eval_dirs], ["2013_05_28_drive_0003_sync"])
        self.assertEqual(train_frames, [[1, 2]])
        self.assertEqual(eval_frames, [[20, 21]])

    def test_load_split_from_yaml_rejects_missing_test_entries_when_test_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            drive_dir = data_dir / "2013_05_28_drive_0003_sync"
            drive_dir.mkdir(parents=True)
            (drive_dir / "train_frames.txt").write_text("1\n")
            (drive_dir / "val_frames.txt").write_text("10\n")
            split_yaml = Path(tmp) / "split.yaml"
            split_yaml.write_text(
                "\n".join(
                    [
                        "train:",
                        "  - drive: 2013_05_28_drive_0003_sync",
                        "    frames_file: train_frames.txt",
                        "val:",
                        "  - drive: 2013_05_28_drive_0003_sync",
                        "    frames_file: val_frames.txt",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "test"):
                infer._load_split_from_yaml(data_dir, split_yaml, eval_split="test")

    def test_parse_args_cli_values_override_config_even_when_equal_to_cli_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "inference.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  checkpoint_path: /tmp/config_checkpoint.pt",
                        "inference:",
                        "  num_inference_steps: 50",
                        "  guidance_scale: 7.5",
                        "output:",
                        "  output_dir: /tmp/config_output",
                    ]
                )
            )
            with patch(
                "sys.argv",
                [
                    "infer.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    "/tmp/cli_checkpoint.pt",
                    "--guidance_scale",
                    "1.0",
                    "--output_dir",
                    "./inference_results",
                ],
            ):
                args = infer._parse_args()

        self.assertEqual(args.checkpoint, "/tmp/cli_checkpoint.pt")
        self.assertEqual(args.guidance_scale, 1.0)
        self.assertEqual(args.output_dir, "./inference_results")
        self.assertEqual(args.num_inference_steps, 50)

    def test_checkpoint_gate_metadata_keeps_only_gate_relevant_config(self) -> None:
        meta = infer._checkpoint_gate_metadata(
            {
                "model_state_dict": {"large": object()},
                "trainer_metadata": {
                    "checkpoint_epoch": 100,
                    "validate_every": 10,
                    "ignored": object(),
                },
                "run_config": {
                    "view_set": "pose_chain",
                    "pose_chains": [{"name": "right", "yaws": ["front", 60.0]}],
                    "query_geometry_score_mode": "geometry_first_semantic_refine",
                    "unrelated": "skip",
                },
            }
        )

        self.assertEqual(meta["trainer_metadata"]["checkpoint_epoch"], 100)
        self.assertEqual(meta["trainer_metadata"]["validate_every"], 10)
        self.assertNotIn("ignored", meta["trainer_metadata"])
        self.assertEqual(meta["run_config"]["view_set"], "pose_chain")
        self.assertIn("pose_chains", meta["run_config"])
        self.assertNotIn("unrelated", meta["run_config"])
        self.assertNotIn("model_state_dict", meta)

    def test_inference_runtime_config_records_sampling_settings(self) -> None:
        runtime = infer._inference_runtime_config(
            SimpleNamespace(
                num_inference_steps=25,
                guidance_scale=1.0,
                seed=123,
                mixed_precision="bf16",
                view_memory_mode="joint_pose_chain",
                sat_condition_mode="normal",
            )
        )

        self.assertEqual(
            runtime,
            {
                "num_inference_steps": 25,
                "guidance_scale": 1.0,
                "seed": 123,
                "mixed_precision": "bf16",
                "view_memory_mode": "joint_pose_chain",
                "sat_condition_mode": "normal",
            },
        )

    def test_inference_runtime_config_uses_actual_ablation_sat_condition_mode(self) -> None:
        runtime = infer._inference_runtime_config(
            SimpleNamespace(
                num_inference_steps=25,
                guidance_scale=1.0,
                seed=123,
                mixed_precision="bf16",
                view_memory_mode="joint_pose_chain",
                sat_condition_mode="normal",
            ),
            sat_condition_mode="zero",
        )

        self.assertEqual(runtime["sat_condition_mode"], "zero")

    def test_checkpoint_inference_mismatch_reports_silent_gate_config_changes(self) -> None:
        mismatches = infer._checkpoint_inference_mismatches(
            {
                "run_config": {
                    "query_geometry_score_enabled": True,
                    "query_geometry_score_mode": "geometry_first_semantic_refine",
                    "query_geometry_candidate_radius": 0.35,
                    "query_semantic_score_alpha": 0.25,
                }
            },
            SimpleNamespace(
                query_geometry_score_enabled=True,
                query_geometry_score_dim=64,
                query_geometry_score_num_freqs=6,
                query_geometry_score_mode="geometry_first_semantic_refine",
                query_geometry_score_gate_init=2.0,
                query_geometry_candidate_radius=0.5,
                query_geometry_candidate_min_k=16,
                query_geometry_candidate_invalid_penalty=-10000.0,
                query_semantic_score_dim=64,
                query_semantic_score_alpha=0.25,
            ),
        )

        self.assertEqual(len(mismatches), 1)
        self.assertIn("query_geometry_candidate_radius", mismatches[0])

    def test_checkpoint_inference_mismatch_requires_checkpoint_run_config(self) -> None:
        mismatches = infer._checkpoint_inference_mismatches({}, SimpleNamespace())

        self.assertEqual(
            mismatches,
            ["checkpoint is missing run_config metadata for inference consistency checks"],
        )


if __name__ == "__main__":
    unittest.main()
