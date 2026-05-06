"""
Grouped multi-view dataset wrapper.

This wrapper regroups the flat per-view samples emitted by Kitti360dDataset into
per-site items so training can share one satellite memory state across multiple
street views from the same location.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class GroupedMultiViewDataset(Dataset):
    """
    Regroup a flat per-view dataset into one item per (drive, frame_id) site.

    Shared keys such as the satellite image are taken from the first view sample.
    View-dependent tensor fields are stacked along a new leading view dimension.
    Non-tensor metadata is kept as a per-view list.
    """

    _SHARED_KEYS = {
        "sat",
        "sat_available",
        "sat_m_per_px",
        "frame_id",
        "drive",
    }

    def __init__(self, base_dataset: Dataset):
        if not hasattr(base_dataset, "samples"):
            raise ValueError("GroupedMultiViewDataset expects a dataset with a 'samples' attribute")

        self.base_dataset = base_dataset
        self.samples = getattr(base_dataset, "samples")
        self.grouped_indices = self._build_groups(self.samples)
        self.view_set = getattr(base_dataset, "view_set", "single")

    @staticmethod
    def _sample_key(sample: Any) -> Tuple[str, int]:
        drive_dir = getattr(sample, "drive_dir", None)
        frame_id = getattr(sample, "frame_id", None)
        if drive_dir is None or frame_id is None:
            raise ValueError("Each sample must expose 'drive_dir' and 'frame_id' for grouping")
        return str(Path(drive_dir)), int(frame_id)

    @classmethod
    def _build_groups(cls, samples: Sequence[Any]) -> List[List[int]]:
        groups: "OrderedDict[Tuple[str, int], List[int]]" = OrderedDict()
        for index, sample in enumerate(samples):
            groups.setdefault(cls._sample_key(sample), []).append(index)
        return list(groups.values())

    def __len__(self) -> int:
        return len(self.grouped_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_indices = self.grouped_indices[idx]
        view_samples = [self.base_dataset[sample_index] for sample_index in sample_indices]
        first_view = view_samples[0]

        grouped: Dict[str, Any] = {
            key: first_view[key]
            for key in self._SHARED_KEYS
            if key in first_view
        }
        grouped["num_views"] = len(view_samples)
        grouped["view_names"] = [
            sample.get("meta", {}).get("view_name")
            if isinstance(sample.get("meta"), dict)
            else None
            for sample in view_samples
        ]

        for key in first_view.keys():
            if key in self._SHARED_KEYS:
                continue

            values = [sample.get(key) for sample in view_samples]
            first_value = values[0]

            if key == "meta" or any(value is None for value in values):
                grouped[key] = values
                continue

            if torch.is_tensor(first_value):
                grouped[key] = torch.stack(values, dim=0)
                continue

            grouped[key] = values

        return grouped
