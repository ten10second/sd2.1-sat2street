import unittest
from pathlib import Path

import yaml


class SatelliteEncoderDimAlignmentTest(unittest.TestCase):
    def test_train_and_inference_configs_share_satellite_embed_dim(self) -> None:
        root = Path(__file__).resolve().parents[1]

        with (root / "configs" / "train.yaml").open("r") as f:
            train_cfg = yaml.safe_load(f)
        with (root / "configs" / "inference.yaml").open("r") as f:
            infer_cfg = yaml.safe_load(f)

        train_dim = train_cfg["model"]["satellite_encoder"]["embed_dim"]
        infer_dim = infer_cfg["model"]["satellite_encoder"]["embed_dim"]

        self.assertEqual(train_dim, 1024)
        self.assertEqual(infer_dim, 1024)
        self.assertEqual(train_dim, infer_dim)

    def test_train_config_is_stage1_yaw_only_without_front(self) -> None:
        root = Path(__file__).resolve().parents[1]

        with (root / "configs" / "train.yaml").open("r") as f:
            train_cfg = yaml.safe_load(f)
        with (root / "configs" / "inference.yaml").open("r") as f:
            infer_cfg = yaml.safe_load(f)

        data_cfg = train_cfg["data"]
        self.assertEqual(data_cfg["mode"], "fisheye_virtual")
        self.assertEqual(data_cfg["yaw_mode"], "vehicle_relative")
        self.assertEqual(data_cfg["vehicle_yaw_sampling"], "fixed_list")
        self.assertEqual(data_cfg["vehicle_yaw_fixed_list"], [-120.0, -90.0, -60.0, 60.0, 90.0, 120.0])
        self.assertEqual(data_cfg["pitch_fixed_list"], [0.0])
        self.assertEqual(data_cfg["roll_fixed_list"], [0.0])
        self.assertEqual(data_cfg["front_sample_prob"], 0.0)
        self.assertEqual(infer_cfg["data"]["mode"], "fisheye_virtual")
        self.assertEqual(infer_cfg["inference"]["yaw_sweep_preset"], "train_fixed")
        self.assertFalse(infer_cfg["inference"]["include_front"])


if __name__ == "__main__":
    unittest.main()
