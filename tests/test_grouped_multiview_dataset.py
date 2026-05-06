import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from data.grouped_multiview_dataset import GroupedMultiViewDataset


@dataclass
class _SampleIndex:
    drive_dir: Path
    frame_id: int
    meta: Optional[Dict[str, Any]] = None


class _FlatDataset:
    def __init__(self) -> None:
        self.view_set = "fixed5"
        self.samples: List[_SampleIndex] = [
            _SampleIndex(Path("/tmp/drive_a"), 10, {"view_name": "front"}),
            _SampleIndex(Path("/tmp/drive_a"), 10, {"view_name": "left"}),
            _SampleIndex(Path("/tmp/drive_a"), 20, {"view_name": "front"}),
            _SampleIndex(Path("/tmp/drive_a"), 20, {"view_name": "right"}),
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        view_name = sample.meta["view_name"]
        view_index = {"front": 0.0, "left": 1.0, "right": 2.0}[view_name]
        return {
            "image": torch.full((3, 4, 4), fill_value=view_index, dtype=torch.float32),
            "sat": torch.full((3, 8, 8), fill_value=float(sample.frame_id), dtype=torch.float32),
            "front_bev_xy": torch.full((2, 4, 4), fill_value=view_index, dtype=torch.float32),
            "front_ground_valid_mask": torch.ones((1, 4, 4), dtype=torch.float32),
            "plucker_map": torch.full((6, 4, 4), fill_value=view_index + 10.0, dtype=torch.float32),
            "frame_id": sample.frame_id,
            "drive": sample.drive_dir.name,
            "meta": {"view_name": view_name, "frame_id": sample.frame_id},
        }

    def __len__(self) -> int:
        return len(self.samples)


class GroupedMultiViewDatasetTest(unittest.TestCase):
    def test_groups_flat_samples_by_drive_and_frame(self) -> None:
        dataset = GroupedMultiViewDataset(_FlatDataset())

        self.assertEqual(len(dataset), 2)

        sample0 = dataset[0]
        self.assertEqual(sample0["frame_id"], 10)
        self.assertEqual(sample0["num_views"], 2)
        self.assertEqual(sample0["view_names"], ["front", "left"])
        self.assertEqual(tuple(sample0["image"].shape), (2, 3, 4, 4))
        self.assertEqual(tuple(sample0["front_bev_xy"].shape), (2, 2, 4, 4))
        self.assertEqual(tuple(sample0["plucker_map"].shape), (2, 6, 4, 4))
        self.assertEqual(len(sample0["meta"]), 2)
        self.assertTrue(torch.equal(sample0["sat"], torch.full((3, 8, 8), 10.0)))

        sample1 = dataset[1]
        self.assertEqual(sample1["frame_id"], 20)
        self.assertEqual(sample1["view_names"], ["front", "right"])
        self.assertTrue(torch.equal(sample1["image"][1], torch.full((3, 4, 4), 2.0)))


if __name__ == "__main__":
    unittest.main()
