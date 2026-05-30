import runpy
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import yaml


_TRAIN_MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
_resolve_query_uv_config = _TRAIN_MODULE["_resolve_query_uv_config"]
_resolve_query_geometry_bias_config = _TRAIN_MODULE["_resolve_query_geometry_bias_config"]
_resolve_query_geometry_score_config = _TRAIN_MODULE["_resolve_query_geometry_score_config"]
_attach_query_geometry_score_args = _TRAIN_MODULE["_attach_query_geometry_score_args"]
_verify_query_geometry_score_model_config = _TRAIN_MODULE["_verify_query_geometry_score_model_config"]
_resolve_unet_attention_slicing_config = _TRAIN_MODULE["_resolve_unet_attention_slicing_config"]
_resolve_gradient_checkpointing_config = _TRAIN_MODULE["_resolve_gradient_checkpointing_config"]
_infer_pose_chain_group_size = _TRAIN_MODULE["_infer_pose_chain_group_size"]
_collect_cli_options = _TRAIN_MODULE["_collect_cli_options"]
_prefer_config = _TRAIN_MODULE["_prefer_config"]
_init_distributed = _TRAIN_MODULE["_init_distributed"]
DEFAULT_DISTRIBUTED_TIMEOUT = _TRAIN_MODULE["DEFAULT_DISTRIBUTED_TIMEOUT"]
_TRAIN_DIST = _TRAIN_MODULE["dist"]


class TrainQueryUVGateConfigTest(unittest.TestCase):
    def test_train_config_enables_pose_chain_view_set(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs" / "train.yaml").open("r") as f:
            config = yaml.safe_load(f)

        self.assertEqual(config["data"]["view_set"], "pose_chain")
        self.assertEqual(
            config["data"]["pose_chains"],
            [
                {"name": "right", "yaws": ["front", 60.0, 90.0, 120.0]},
                {"name": "left", "yaws": ["front", -60.0, -90.0, -120.0]},
            ],
        )
        self.assertEqual(config["data"]["batch_size"], 2)
        self.assertEqual(config["validation"]["validate_every"], 10)
        self.assertTrue(config["training"]["joint_view_generation"]["enable"])
        self.assertFalse(config["training"]["transition_aux"]["enable"])

    def test_query_uv_config_is_ignored_for_clean_geometry_addressing(self) -> None:
        enabled, gate_init = _resolve_query_uv_config({})
        self.assertFalse(enabled)
        self.assertEqual(gate_init, 0.0)

    def test_pose_chain_group_size_infers_view_count_for_effective_view_batch(self) -> None:
        self.assertEqual(_infer_pose_chain_group_size("single", None), 1)
        self.assertEqual(
            _infer_pose_chain_group_size(
                "pose_chain",
                [{"name": "right", "yaws": ["front", 60.0, 90.0, 120.0]}],
            ),
            4,
        )
        self.assertEqual(_infer_pose_chain_group_size("pose_chain", None), 4)

    def test_pose_chain_group_size_rejects_mismatched_chain_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of views"):
            _infer_pose_chain_group_size(
                "pose_chain",
                [
                    {"name": "right", "yaws": ["front", 60.0, 90.0, 120.0]},
                    {"name": "left", "yaws": ["front", -60.0, -90.0]},
                ],
            )

    def test_pose_chain_group_size_rejects_duplicate_chain_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            _infer_pose_chain_group_size(
                "pose_chain",
                [
                    {"name": "right", "yaws": ["front", 60.0]},
                    {"name": "right", "yaws": ["front", 90.0]},
                ],
            )

    def test_pose_chain_group_size_rejects_string_yaws(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a string"):
            _infer_pose_chain_group_size(
                "pose_chain",
                [
                    {"name": "right", "yaws": "front,60,90"},
                ],
            )

    def test_validate_every_cli_option_overrides_config_even_when_cli_value_is_default(self) -> None:
        cli_options = _collect_cli_options(["--validate_every", "1"])

        value = _prefer_config(
            1,
            1,
            10,
            cli_option="--validate_every",
            cli_options=cli_options,
        )

        self.assertEqual(value, 1)

    def test_distributed_init_uses_long_timeout_for_rank0_validation_waits(self) -> None:
        args = Namespace(device="cpu")

        with patch.dict(
            "os.environ",
            {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"},
            clear=False,
        ), patch.object(_TRAIN_DIST, "init_process_group") as init_process_group:
            distributed, rank, local_rank, world_size = _init_distributed(args)

        self.assertTrue(distributed)
        self.assertEqual((rank, local_rank, world_size), (0, 0, 2))
        init_process_group.assert_called_once_with(
            backend="gloo",
            timeout=DEFAULT_DISTRIBUTED_TIMEOUT,
        )

    def test_explicit_query_uv_config_still_resolves_to_disabled(self) -> None:
        enabled, gate_init = _resolve_query_uv_config(
            {
                "model": {
                    "query_position_encoding": {
                        "enable": False,
                        "gate_init": 0.25,
                    }
                }
            }
        )
        self.assertFalse(enabled)
        self.assertEqual(gate_init, 0.0)

    def test_geometry_bias_config_is_ignored(self) -> None:
        enabled, scale, invalid_penalty = _resolve_query_geometry_bias_config({})
        self.assertFalse(enabled)
        self.assertEqual(scale, 0.0)
        self.assertEqual(invalid_penalty, 0.0)

    def test_geometry_score_default_has_no_query_token_cutoff(self) -> None:
        config = _resolve_query_geometry_score_config({"model": {"query_geometry_score": {"enable": True}}})
        self.assertTrue(config["enabled"])
        self.assertIsNone(config["max_query_tokens"])

    def test_geometry_score_reads_layers_and_gate(self) -> None:
        config = _resolve_query_geometry_score_config(
            {
                "model": {
                    "query_geometry_score": {
                        "enable": True,
                        "dim": 32,
                        "num_freqs": 4,
                        "gate_init": 0.5,
                        "layers": ["mid.attn2"],
                        "max_query_tokens": None,
                        "mode": "geometry_first_semantic_refine",
                        "candidate_radius": 0.4,
                        "candidate_min_k": 12,
                        "candidate_invalid_penalty": -5000.0,
                        "semantic_score_dim": 48,
                        "semantic_alpha_max": 0.2,
                    }
                }
            }
        )
        self.assertTrue(config["enabled"])
        self.assertEqual(config["dim"], 32)
        self.assertEqual(config["num_freqs"], 4)
        self.assertEqual(config["gate_init"], 0.5)
        self.assertEqual(config["layers"], ["mid.attn2"])
        self.assertIsNone(config["max_query_tokens"])
        self.assertEqual(config["mode"], "geometry_first_semantic_refine")
        self.assertEqual(config["candidate_radius"], 0.4)
        self.assertEqual(config["candidate_min_k"], 12)
        self.assertEqual(config["candidate_invalid_penalty"], -5000.0)
        self.assertEqual(config["semantic_score_dim"], 48)
        self.assertEqual(config["semantic_alpha_max"], 0.2)

    def test_geometry_score_config_is_attached_to_training_args(self) -> None:
        args = Namespace()
        config = {
            "model": {
                "query_geometry_score": {
                    "enable": True,
                    "dim": 96,
                    "num_freqs": 5,
                    "gate_init": 0.75,
                    "layers": ["down.attn2", "mid.attn2"],
                    "max_query_tokens": 128,
                }
            }
        }

        resolved = _attach_query_geometry_score_args(args, config)

        self.assertTrue(resolved["enabled"])
        self.assertTrue(args.query_geometry_score_enabled)
        self.assertEqual(args.query_geometry_score_dim, 96)
        self.assertEqual(args.query_geometry_score_num_freqs, 5)
        self.assertEqual(args.query_geometry_score_gate_init, 0.75)
        self.assertEqual(args.query_geometry_score_layers, ["down.attn2", "mid.attn2"])
        self.assertEqual(args.query_geometry_score_max_query_tokens, 128)
        self.assertEqual(args.query_geometry_score_mode, "geometry_first_semantic_refine")
        self.assertEqual(args.query_geometry_candidate_radius, 0.35)
        self.assertEqual(args.query_geometry_candidate_min_k, 16)
        self.assertEqual(args.query_geometry_candidate_invalid_penalty, -1e4)
        self.assertEqual(args.query_semantic_score_dim, 64)
        self.assertEqual(args.query_semantic_score_alpha, 0.25)

    def test_geometry_score_model_config_verifier_rejects_unwired_model(self) -> None:
        config = {
            "enabled": True,
            "dim": 64,
            "num_freqs": 6,
            "gate_init": 1.0,
            "layers": ["mid.attn2"],
            "max_query_tokens": 256,
        }
        model = Namespace(
            unet=Namespace(
                query_geometry_score_enabled=False,
                query_geometry_score_dim=64,
                query_geometry_score_num_freqs=6,
                query_geometry_score_gate_init=1.0,
                query_geometry_score_layers=("mid.attn2",),
                query_geometry_score_max_query_tokens=256,
                query_geometry_score_mode="geometry_first_semantic_refine",
                query_geometry_candidate_radius=0.35,
                query_geometry_candidate_min_k=16,
                query_geometry_candidate_invalid_penalty=-1e4,
                query_semantic_score_dim=64,
                query_semantic_score_alpha=0.25,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "query_geometry_score config was not applied"):
            _verify_query_geometry_score_model_config(model, config)

    def test_geometry_score_model_config_verifier_accepts_wired_model(self) -> None:
        config = {
            "enabled": True,
            "dim": 64,
            "num_freqs": 6,
            "gate_init": 1.0,
            "layers": ["mid.attn2"],
            "max_query_tokens": 256,
        }
        model = Namespace(
            unet=Namespace(
                query_geometry_score_enabled=True,
                query_geometry_score_dim=64,
                query_geometry_score_num_freqs=6,
                query_geometry_score_gate_init=1.0,
                query_geometry_score_layers=("mid.attn2",),
                query_geometry_score_max_query_tokens=256,
                query_geometry_score_mode="geometry_first_semantic_refine",
                query_geometry_candidate_radius=0.35,
                query_geometry_candidate_min_k=16,
                query_geometry_candidate_invalid_penalty=-1e4,
                query_semantic_score_dim=64,
                query_semantic_score_alpha=0.25,
            )
        )

        _verify_query_geometry_score_model_config(model, config)

    def test_unet_attention_slicing_defaults_to_disabled(self) -> None:
        self.assertFalse(_resolve_unet_attention_slicing_config({}))
        self.assertTrue(_resolve_unet_attention_slicing_config({"attention_slicing": True}))

    def test_gradient_checkpointing_cli_override_wins(self) -> None:
        self.assertFalse(_resolve_gradient_checkpointing_config({"gradient_checkpointing": True}, False))
        self.assertTrue(_resolve_gradient_checkpointing_config({"gradient_checkpointing": False}, True))
        self.assertFalse(_resolve_gradient_checkpointing_config({"gradient_checkpointing": False}, None))

    def test_explicit_cli_option_wins_even_when_value_equals_parser_default(self) -> None:
        cli_options = _collect_cli_options(["--gradient_accumulation", "2"])

        value = _prefer_config(
            2,
            2,
            4,
            cli_option="--gradient_accumulation",
            cli_options=cli_options,
        )

        self.assertEqual(value, 2)

    def test_config_still_fills_when_cli_option_is_not_present(self) -> None:
        cli_options = _collect_cli_options(["--batch_size", "16"])

        value = _prefer_config(
            2,
            2,
            4,
            cli_option="--gradient_accumulation",
            cli_options=cli_options,
        )

        self.assertEqual(value, 4)
