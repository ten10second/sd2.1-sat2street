import runpy
import unittest
from pathlib import Path


_TRAIN_MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
_resolve_query_uv_config = _TRAIN_MODULE["_resolve_query_uv_config"]
_resolve_query_geometry_bias_config = _TRAIN_MODULE["_resolve_query_geometry_bias_config"]


class TrainQueryUVGateConfigTest(unittest.TestCase):
    def test_defaults_to_disabled_query_uv_gate(self) -> None:
        enabled, gate_init = _resolve_query_uv_config({})
        self.assertFalse(enabled)
        self.assertEqual(gate_init, 1.0)

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
