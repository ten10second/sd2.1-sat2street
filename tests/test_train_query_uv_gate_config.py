import runpy
import unittest
from pathlib import Path


_TRAIN_MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
_resolve_query_uv_config = _TRAIN_MODULE["_resolve_query_uv_config"]
_resolve_query_geometry_bias_config = _TRAIN_MODULE["_resolve_query_geometry_bias_config"]
_resolve_query_geometry_score_config = _TRAIN_MODULE["_resolve_query_geometry_score_config"]
_resolve_unet_attention_slicing_config = _TRAIN_MODULE["_resolve_unet_attention_slicing_config"]
_resolve_gradient_checkpointing_config = _TRAIN_MODULE["_resolve_gradient_checkpointing_config"]
_collect_cli_options = _TRAIN_MODULE["_collect_cli_options"]
_prefer_config = _TRAIN_MODULE["_prefer_config"]


class TrainQueryUVGateConfigTest(unittest.TestCase):
    def test_defaults_to_disabled_query_uv_gate(self) -> None:
        enabled, gate_init = _resolve_query_uv_config({})
        self.assertFalse(enabled)
        self.assertEqual(gate_init, 0.05)

    def test_reads_explicit_gate_init(self) -> None:
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
        self.assertEqual(gate_init, 0.25)

    def test_geometry_bias_defaults_to_disabled(self) -> None:
        enabled, scale, invalid_penalty = _resolve_query_geometry_bias_config({})
        self.assertFalse(enabled)
        self.assertEqual(scale, 2.0)
        self.assertEqual(invalid_penalty, -1e4)

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
