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


if __name__ == "__main__":
    unittest.main()
