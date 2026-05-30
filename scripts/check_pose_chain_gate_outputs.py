#!/usr/bin/env python3
"""Check pose-chain validation/test yaw-sweep outputs for gate review."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from PIL import Image, UnidentifiedImageError
import torch
import yaml


REQUIRED_FRAME_FILES: Tuple[str, ...] = ("summary.png",)
REQUIRED_VIEW_FILES: Tuple[str, ...] = (
    "generated.png",
    "gt.png",
    "satellite.png",
    "satellite_projected.png",
    "comparison.png",
    "metadata.yaml",
)

DEFAULT_PRESET_VIEWS: Mapping[str, Tuple[Tuple[str, Optional[float]], ...]] = {
    "train_fixed": (
        ("front", None),
        ("yaw_m120", -120.0),
        ("yaw_m90", -90.0),
        ("yaw_m60", -60.0),
        ("yaw_p60", 60.0),
        ("yaw_p90", 90.0),
        ("yaw_p120", 120.0),
    ),
    "heldout": (
        ("yaw_m105", -105.0),
        ("yaw_m75", -75.0),
        ("yaw_m45", -45.0),
        ("yaw_p45", 45.0),
        ("yaw_p75", 75.0),
        ("yaw_p105", 105.0),
    ),
}
EXPECTED_POSE_CHAINS: Mapping[str, Tuple[Optional[float], ...]] = {
    "right": (None, 60.0, 90.0, 120.0),
    "left": (None, -60.0, -90.0, -120.0),
}
EXPECTED_QUERY_GEOMETRY_SCORE_MODE = "geometry_first_semantic_refine"
REMOVED_ADDITIVE_PE_STATE_KEY_LIMIT = 10
VALID_GATE_SPLITS: Tuple[str, ...] = ("val", "test")
EXPECTED_INFERENCE_MODE = "split_yaw_sweep"
DEFAULT_EXPECTED_SAT_CONDITION_MODE = "normal"
REQUIRED_INFERENCE_RUNTIME_KEYS: Tuple[str, ...] = (
    "num_inference_steps",
    "guidance_scale",
    "seed",
    "mixed_precision",
    "view_memory_mode",
    "sat_condition_mode",
)
REQUIRED_VALIDATION_SCALAR_KEYS: Tuple[str, ...] = (
    "val/loss",
    "val/denoise_loss",
    "val/attention_alignment_target_attention_lift_mixed",
    "val/attention_alignment_target_attention_lift_geometry_only",
    "val/attention_alignment_target_attention_lift_semantic_only",
    "val/attention_alignment_target_attention_lift_without_geometry",
    "val/attention_alignment_semantic_to_geometry_ratio",
    "val/attention_alignment_chain_attention_coverage_overlap",
    "val/attention_alignment_chain_attention_centroid_shift",
    "val/attention_alignment_chain_attention_valid_pair_ratio",
    "val/chain/coverage_overlap",
    "val/chain/coverage_centroid_shift",
    "val/chain/valid_pair_ratio",
)
POSITIVE_LIFT_KEYS: Tuple[str, ...] = (
    "val/attention_alignment_target_attention_lift_mixed",
    "val/attention_alignment_target_attention_lift_geometry_only",
)
UNIT_INTERVAL_METRIC_KEYS: Tuple[str, ...] = (
    "val/attention_alignment_chain_attention_coverage_overlap",
    "val/attention_alignment_chain_attention_valid_pair_ratio",
    "val/chain/coverage_overlap",
    "val/chain/valid_pair_ratio",
)
NONNEGATIVE_METRIC_KEYS: Tuple[str, ...] = (
    "val/loss",
    "val/denoise_loss",
    "val/attention_alignment_target_attention_lift_semantic_only",
    "val/attention_alignment_target_attention_lift_without_geometry",
    "val/attention_alignment_semantic_to_geometry_ratio",
    "val/attention_alignment_chain_attention_centroid_shift",
    "val/chain/coverage_centroid_shift",
)


@dataclass(frozen=True, order=True)
class FrameKey:
    drive: str
    frame: str

    @property
    def label(self) -> str:
        return f"{self.drive}/{self.frame}"


@dataclass
class FrameScan:
    key: FrameKey
    path: Path
    view_tokens: Set[str]
    missing_frame_files: List[str] = field(default_factory=list)
    invalid_frame_files: List[str] = field(default_factory=list)
    missing_views: List[str] = field(default_factory=list)
    incomplete_views: Dict[str, List[str]] = field(default_factory=dict)
    invalid_view_files: Dict[str, List[str]] = field(default_factory=dict)
    metadata_errors: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return (
            not self.missing_frame_files
            and not self.invalid_frame_files
            and not self.missing_views
            and not self.incomplete_views
            and not self.invalid_view_files
            and not self.metadata_errors
        )


@dataclass
class PresetScan:
    preset: str
    root: Path
    preset_dir: Path
    metadata_path: Optional[Path]
    metadata: Mapping
    expected_views: List[str]
    frames: Dict[FrameKey, FrameScan]
    errors: List[str] = field(default_factory=list)

    @property
    def complete_frames(self) -> int:
        return sum(1 for frame in self.frames.values() if frame.is_complete)

    @property
    def incomplete_frames(self) -> int:
        return len(self.frames) - self.complete_frames

    @property
    def is_complete(self) -> bool:
        return bool(self.frames) and not self.errors and self.incomplete_frames == 0


@dataclass
class GateScan:
    train_fixed: PresetScan
    heldout: PresetScan
    common_frames: List[FrameKey]
    train_only_frames: List[FrameKey]
    heldout_only_frames: List[FrameKey]
    min_common_frames: int = 1
    cross_errors: List[str] = field(default_factory=list)
    latest_scalars: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_output_complete(self) -> bool:
        return (
            self.train_fixed.is_complete
            and self.heldout.is_complete
            and len(self.common_frames) >= max(1, int(self.min_common_frames))
            and not self.train_only_frames
            and not self.heldout_only_frames
            and not self.cross_errors
        )


def _load_yaml(path: Path) -> Mapping:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        return {}
    return data


def _view_token(view_name: str, yaw: Optional[float]) -> str:
    if yaw is None:
        return view_name
    prefix = "p" if yaw > 0 else "m" if yaw < 0 else ""
    return f"yaw_{prefix}{abs(float(yaw)):g}".replace(".", "p")


def _default_preset_view_tokens(preset: str) -> List[str]:
    return [
        _view_token(view_name, yaw)
        for view_name, yaw in DEFAULT_PRESET_VIEWS.get(preset, ())
    ]


def _expected_view_specs_by_token(preset: str) -> Dict[str, Tuple[str, Optional[float]]]:
    return {
        _view_token(view_name, yaw): (view_name, yaw)
        for view_name, yaw in DEFAULT_PRESET_VIEWS.get(preset, ())
    }


def _metadata_view_tokens(metadata: Mapping) -> Optional[List[str]]:
    views = metadata.get("views")
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)):
        return None

    tokens: List[str] = []
    for view in views:
        if not isinstance(view, Mapping):
            continue
        view_name = view.get("view_name")
        yaw = view.get("vehicle_yaw_deg")
        if not isinstance(view_name, str):
            continue
        tokens.append(_view_token(view_name, None if yaw is None else float(yaw)))
    return tokens if tokens else None


def _expected_views_from_metadata(metadata: Mapping, preset: str) -> List[str]:
    default_tokens = _default_preset_view_tokens(preset)
    if default_tokens:
        return default_tokens
    return _metadata_view_tokens(metadata) or []


def _normalize_pose_chain_yaw(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "front":
        return None
    return float(value)


def _pose_chain_yaws_match(
    actual: Sequence[Optional[float]],
    expected: Sequence[Optional[float]],
) -> bool:
    if len(actual) != len(expected):
        return False
    for actual_yaw, expected_yaw in zip(actual, expected):
        if expected_yaw is None:
            if actual_yaw is not None:
                return False
            continue
        if actual_yaw is None or abs(float(actual_yaw) - float(expected_yaw)) > 1e-6:
            return False
    return True


def _validate_checkpoint_pose_chains(pose_chains: object) -> List[str]:
    if (
        not isinstance(pose_chains, Sequence)
        or isinstance(pose_chains, (str, bytes))
        or not pose_chains
    ):
        return ["checkpoint run_config.pose_chains must be a non-empty sequence"]

    actual_by_name: Dict[str, Tuple[Optional[float], ...]] = {}
    errors: List[str] = []
    for index, chain in enumerate(pose_chains):
        if not isinstance(chain, Mapping):
            errors.append(f"checkpoint run_config.pose_chains[{index}] must be a mapping")
            continue
        name = chain.get("name")
        yaw_values = chain.get("yaws", chain.get("views"))
        if not isinstance(name, str):
            errors.append(f"checkpoint run_config.pose_chains[{index}].name must be a string")
            continue
        if not isinstance(yaw_values, Sequence) or isinstance(yaw_values, (str, bytes)):
            errors.append(f"checkpoint run_config.pose_chains[{name!r}].yaws must be a sequence")
            continue
        try:
            actual_by_name[name] = tuple(_normalize_pose_chain_yaw(value) for value in yaw_values)
        except (TypeError, ValueError) as exc:
            errors.append(f"checkpoint run_config.pose_chains[{name!r}].yaws contains invalid yaw: {exc}")

    for name, expected_yaws in EXPECTED_POSE_CHAINS.items():
        actual_yaws = actual_by_name.get(name)
        if actual_yaws is None:
            errors.append(f"checkpoint run_config.pose_chains must include {name!r} chain")
            continue
        if not _pose_chain_yaws_match(actual_yaws, expected_yaws):
            errors.append(
                f"checkpoint run_config.pose_chains[{name!r}] must be "
                f"{list(expected_yaws)!r}, got {list(actual_yaws)!r}"
            )
    return errors


def _validate_checkpoint_algorithm_config(run_config: Mapping) -> List[str]:
    errors: List[str] = []
    if run_config.get("query_geometry_score_enabled") is not True:
        errors.append(
            "checkpoint run_config.query_geometry_score_enabled must be true "
            "for geometry-first semantic-refine"
        )
    mode = run_config.get("query_geometry_score_mode")
    if mode != EXPECTED_QUERY_GEOMETRY_SCORE_MODE:
        errors.append(
            "checkpoint run_config.query_geometry_score_mode must be "
            f"{EXPECTED_QUERY_GEOMETRY_SCORE_MODE!r}, got {mode!r}"
        )
    if run_config.get("attention_alignment_enabled") is not True:
        errors.append(
            "checkpoint run_config.attention_alignment_enabled must be true "
            "so target-lift and chain diagnostics have a supervised training path"
        )
    return errors


def _validate_inference_algorithm_config(inference_config: Mapping) -> List[str]:
    errors: List[str] = []
    if inference_config.get("query_geometry_score_enabled") is not True:
        errors.append(
            "inference_gate_config.query_geometry_score_enabled must be true "
            "for geometry-first semantic-refine"
        )
    mode = inference_config.get("query_geometry_score_mode")
    if mode != EXPECTED_QUERY_GEOMETRY_SCORE_MODE:
        errors.append(
            "inference_gate_config.query_geometry_score_mode must be "
            f"{EXPECTED_QUERY_GEOMETRY_SCORE_MODE!r}, got {mode!r}"
        )
    return errors


def _is_removed_additive_pe_state_key(key: str) -> bool:
    return (
        key == "satellite_encoder.perspective_pe_gate"
        or key.startswith("satellite_encoder.perspective_pos_encoder.")
        or ".processor.query_uv_gate" in key
        or ".processor.query_uv_encoder." in key
    )


def _inspect_checkpoint_state_dict(checkpoint_path: object) -> List[str]:
    errors: List[str] = []
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        return ["checkpoint path is required for checkpoint state inspection"]
    path = Path(checkpoint_path)
    if not path.exists():
        return [f"checkpoint file does not exist for state inspection: {path}"]
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive for corrupt external files
        return [f"failed to load checkpoint for state inspection: {path} ({exc})"]

    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state_dict, Mapping):
        return [f"checkpoint does not contain a model_state_dict mapping: {path}"]

    removed_keys = sorted(
        str(key)
        for key in state_dict.keys()
        if _is_removed_additive_pe_state_key(str(key))
    )
    if removed_keys:
        sample = removed_keys[:REMOVED_ADDITIVE_PE_STATE_KEY_LIMIT]
        suffix = "" if len(removed_keys) <= len(sample) else f" ... (+{len(removed_keys) - len(sample)} more)"
        errors.append(
            "checkpoint state_dict contains removed additive PE keys: "
            + ", ".join(sample)
            + suffix
        )
    return errors


def _validate_run_metadata(metadata: Mapping, preset: str) -> List[str]:
    errors: List[str] = []
    if not metadata:
        return errors

    mode = metadata.get("mode")
    if mode != EXPECTED_INFERENCE_MODE:
        errors.append(
            f"metadata mode must be {EXPECTED_INFERENCE_MODE!r}, got {mode!r}"
        )

    yaw_sweep_preset = metadata.get("yaw_sweep_preset")
    if yaw_sweep_preset != preset:
        errors.append(
            f"metadata yaw_sweep_preset must be {preset!r}, got {yaw_sweep_preset!r}"
        )

    expected_views = _default_preset_view_tokens(preset)
    metadata_views = _metadata_view_tokens(metadata)
    if expected_views and metadata_views != expected_views:
        errors.append(
            "metadata views must match the "
            f"{preset!r} preset exactly: expected {expected_views!r}, got {metadata_views!r}"
        )

    dataset_split = metadata.get("dataset_split")
    if dataset_split not in VALID_GATE_SPLITS:
        errors.append(
            f"metadata dataset_split must be one of {list(VALID_GATE_SPLITS)}, got {dataset_split!r}"
        )

    checkpoint_gate_metadata = metadata.get("checkpoint_gate_metadata")
    if not isinstance(checkpoint_gate_metadata, Mapping):
        errors.append("missing checkpoint_gate_metadata from split yaw-sweep run")
    else:
        run_config = checkpoint_gate_metadata.get("run_config")
        if not isinstance(run_config, Mapping):
            errors.append("missing checkpoint_gate_metadata.run_config")
        else:
            view_set = run_config.get("view_set")
            if view_set != "pose_chain":
                errors.append(f"checkpoint run_config.view_set must be 'pose_chain', got {view_set!r}")
            errors.extend(_validate_checkpoint_pose_chains(run_config.get("pose_chains")))
            errors.extend(_validate_checkpoint_algorithm_config(run_config))

    inference_gate_config = metadata.get("inference_gate_config")
    if not isinstance(inference_gate_config, Mapping):
        errors.append("missing inference_gate_config from split yaw-sweep run")
    else:
        errors.extend(_validate_inference_algorithm_config(inference_gate_config))

    runtime_config = metadata.get("inference_runtime_config")
    if not isinstance(runtime_config, Mapping):
        errors.append("missing inference_runtime_config from split yaw-sweep run")
    else:
        missing_runtime_keys = [
            key
            for key in REQUIRED_INFERENCE_RUNTIME_KEYS
            if key not in runtime_config
        ]
        if missing_runtime_keys:
            errors.append(
                "inference_runtime_config missing required keys: "
                + ", ".join(missing_runtime_keys)
            )

    mismatches = metadata.get("checkpoint_inference_mismatches")
    if isinstance(mismatches, Sequence) and not isinstance(mismatches, (str, bytes)):
        mismatch_messages = [str(item) for item in mismatches]
        if mismatch_messages:
            errors.append(
                "checkpoint/inference config mismatches: "
                + "; ".join(mismatch_messages)
            )
    elif mismatches is not None:
        errors.append("checkpoint_inference_mismatches must be a list")
    return errors


def _resolve_preset_paths(path: Path, preset: str) -> Tuple[Path, Path, Optional[Path]]:
    direct_metadata = path / f"run_metadata_{preset}.yaml"
    if direct_metadata.exists():
        return path, path / preset, direct_metadata

    parent_metadata = path.parent / f"run_metadata_{preset}.yaml"
    if path.name == preset and parent_metadata.exists():
        return path.parent, path, parent_metadata

    fallback_preset_dir = path / preset
    if fallback_preset_dir.exists():
        return path, fallback_preset_dir, direct_metadata if direct_metadata.exists() else None

    return path, path, direct_metadata if direct_metadata.exists() else None


def _validate_image_file(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return f"{path.name}: invalid image ({exc})"
    if int(width) <= 0 or int(height) <= 0:
        return f"{path.name}: invalid image size {width}x{height}"
    return None


def _yaw_values_match(actual: object, expected: Optional[float]) -> bool:
    if expected is None:
        return actual is None
    if not isinstance(actual, (int, float)):
        return False
    return abs(float(actual) - float(expected)) <= 1e-6


def _drive_values_match(actual: object, expected: str) -> bool:
    if not isinstance(actual, str):
        return False
    return actual == expected or Path(actual).name == expected


def _frame_id_from_label(frame: str) -> Optional[int]:
    if not frame.startswith("frame_"):
        return None
    try:
        return int(frame.removeprefix("frame_"))
    except ValueError:
        return None


def _validate_view_metadata(
    metadata_path: Path,
    *,
    expected_drive: str,
    expected_frame_id: Optional[int],
    expected_view_name: str,
    expected_yaw: Optional[float],
    expected_sat_condition_mode: Optional[str] = DEFAULT_EXPECTED_SAT_CONDITION_MODE,
) -> List[str]:
    metadata = _load_yaml(metadata_path)
    if not metadata:
        return ["metadata.yaml is empty or invalid"]

    errors: List[str] = []
    drive = metadata.get("drive")
    if not _drive_values_match(drive, expected_drive):
        errors.append(
            f"metadata drive must be {expected_drive!r}, got {drive!r}"
        )
    frame_id = metadata.get("frame_id")
    if expected_frame_id is not None and frame_id != expected_frame_id:
        errors.append(
            f"metadata frame_id must be {expected_frame_id!r}, got {frame_id!r}"
        )
    view_name = metadata.get("view_name")
    if view_name != expected_view_name:
        errors.append(
            f"metadata view_name must be {expected_view_name!r}, got {view_name!r}"
        )
    yaw = metadata.get("vehicle_yaw_deg")
    if not _yaw_values_match(yaw, expected_yaw):
        errors.append(
            f"metadata vehicle_yaw_deg must be {expected_yaw!r}, got {yaw!r}"
        )
    if expected_sat_condition_mode is not None:
        sat_condition_mode = metadata.get("sat_condition_mode")
        if sat_condition_mode != expected_sat_condition_mode:
            errors.append(
                "metadata sat_condition_mode must be "
                f"{expected_sat_condition_mode!r}, got {sat_condition_mode!r}"
            )
    sample_meta = metadata.get("meta")
    if isinstance(sample_meta, Mapping):
        actual_view_name = sample_meta.get("view_name")
        if actual_view_name is not None and actual_view_name != expected_view_name:
            errors.append(
                f"metadata meta.view_name must be {expected_view_name!r}, got {actual_view_name!r}"
            )
        actual_yaw = sample_meta.get("vehicle_yaw_deg_used")
        if actual_yaw is None and "vehicle_relative_yaw_deg" in sample_meta:
            actual_yaw = sample_meta.get("vehicle_relative_yaw_deg")
        if not _yaw_values_match(actual_yaw, expected_yaw):
            errors.append(
                f"metadata meta.vehicle_yaw_deg_used must be {expected_yaw!r}, got {actual_yaw!r}"
            )
    return errors


def _iter_frame_dirs(preset_dir: Path) -> Iterable[Tuple[FrameKey, Path]]:
    if not preset_dir.exists():
        return
    for drive_dir in sorted(p for p in preset_dir.iterdir() if p.is_dir()):
        for frame_dir in sorted(p for p in drive_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")):
            yield FrameKey(drive=drive_dir.name, frame=frame_dir.name), frame_dir


def scan_preset_outputs(
    path: Path,
    preset: str,
    *,
    expected_sat_condition_mode: Optional[str] = DEFAULT_EXPECTED_SAT_CONDITION_MODE,
) -> PresetScan:
    root, preset_dir, metadata_path = _resolve_preset_paths(path, preset)
    metadata = _load_yaml(metadata_path) if metadata_path is not None else {}
    expected_views = _expected_views_from_metadata(metadata, preset)
    errors: List[str] = []
    if metadata_path is None or not metadata_path.exists():
        errors.append(f"missing run_metadata_{preset}.yaml")
    else:
        errors.extend(_validate_run_metadata(metadata, preset))
    if not preset_dir.exists():
        errors.append(f"missing preset directory: {preset_dir}")

    frames: Dict[FrameKey, FrameScan] = {}
    expected_specs = _expected_view_specs_by_token(preset)
    view_image_files = [name for name in REQUIRED_VIEW_FILES if name.endswith(".png")]
    for key, frame_dir in _iter_frame_dirs(preset_dir):
        view_dirs = {p.name for p in frame_dir.iterdir() if p.is_dir()}
        expected_frame_id = _frame_id_from_label(key.frame)
        missing_frame_files = [name for name in REQUIRED_FRAME_FILES if not (frame_dir / name).exists()]
        invalid_frame_files = [
            error
            for name in REQUIRED_FRAME_FILES
            if (frame_dir / name).exists()
            for error in [_validate_image_file(frame_dir / name)]
            if error is not None
        ]
        missing_views = [view for view in expected_views if view not in view_dirs]
        incomplete_views: Dict[str, List[str]] = {}
        invalid_view_files: Dict[str, List[str]] = {}
        metadata_errors: Dict[str, List[str]] = {}
        for view in expected_views:
            view_dir = frame_dir / view
            if not view_dir.exists():
                continue
            missing_files = [name for name in REQUIRED_VIEW_FILES if not (view_dir / name).exists()]
            if missing_files:
                incomplete_views[view] = missing_files
                continue
            invalid_files = [
                error
                for name in view_image_files
                for error in [_validate_image_file(view_dir / name)]
                if error is not None
            ]
            if invalid_files:
                invalid_view_files[view] = invalid_files
            expected_spec = expected_specs.get(view)
            if expected_spec is not None:
                view_metadata_errors = _validate_view_metadata(
                    view_dir / "metadata.yaml",
                    expected_drive=key.drive,
                    expected_frame_id=expected_frame_id,
                    expected_view_name=expected_spec[0],
                    expected_yaw=expected_spec[1],
                    expected_sat_condition_mode=expected_sat_condition_mode,
                )
                if view_metadata_errors:
                    metadata_errors[view] = view_metadata_errors
        frames[key] = FrameScan(
            key=key,
            path=frame_dir,
            view_tokens=view_dirs,
            missing_frame_files=missing_frame_files,
            invalid_frame_files=invalid_frame_files,
            missing_views=missing_views,
            incomplete_views=incomplete_views,
            invalid_view_files=invalid_view_files,
            metadata_errors=metadata_errors,
        )

    if not frames:
        errors.append(f"no frame outputs found under {preset_dir}")
    metadata_num_frames = metadata.get("num_frames")
    if isinstance(metadata_num_frames, int):
        if int(metadata_num_frames) != len(frames):
            errors.append(
                f"metadata num_frames must match output frame directories, "
                f"got metadata={metadata_num_frames} outputs={len(frames)}"
            )
    elif metadata and metadata_num_frames is not None:
        errors.append(f"metadata num_frames must be an integer, got {metadata_num_frames!r}")

    return PresetScan(
        preset=preset,
        root=root,
        preset_dir=preset_dir,
        metadata_path=metadata_path if metadata_path and metadata_path.exists() else None,
        metadata=metadata,
        expected_views=expected_views,
        frames=frames,
        errors=errors,
    )


def _load_latest_scalar_metrics(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}

    latest_val_record: Dict[str, float] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping):
                continue
            scalar_record: Dict[str, float] = {}
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    scalar_record[str(key)] = float(value)
            if any(key.startswith("val/") for key in scalar_record):
                latest_val_record = scalar_record
    return latest_val_record


def _metadata_checkpoint_epoch(metadata: Mapping) -> Optional[float]:
    checkpoint_gate_metadata = metadata.get("checkpoint_gate_metadata")
    if isinstance(checkpoint_gate_metadata, Mapping):
        trainer_metadata = checkpoint_gate_metadata.get("trainer_metadata")
        if isinstance(trainer_metadata, Mapping):
            checkpoint_epoch = trainer_metadata.get("checkpoint_epoch")
            if isinstance(checkpoint_epoch, (int, float)):
                return float(checkpoint_epoch)
    checkpoint_epoch = metadata.get("checkpoint_epoch")
    if isinstance(checkpoint_epoch, (int, float)):
        return float(checkpoint_epoch)
    return None


def _latest_scalar_epoch(scalars: Mapping[str, float]) -> Optional[float]:
    for key in ("val/epoch", "train/epoch", "epoch"):
        value = scalars.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _missing_required_validation_scalar_keys(scalars: Mapping[str, float]) -> List[str]:
    return [
        key
        for key in REQUIRED_VALIDATION_SCALAR_KEYS
        if key not in scalars
    ]


def _validate_metric_sanity(scalars: Mapping[str, float]) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_VALIDATION_SCALAR_KEYS:
        value = scalars.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"validation metric {key} must be finite, got {value!r}")

    for key in POSITIVE_LIFT_KEYS:
        value = scalars.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) <= 1.0:
            errors.append(f"validation metric {key} must be > 1.0, got {float(value):g}")

    for key in UNIT_INTERVAL_METRIC_KEYS:
        value = scalars.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            value_f = float(value)
            if value_f < 0.0 or value_f > 1.0:
                errors.append(f"validation metric {key} must be in [0, 1], got {value_f:g}")

    for key in NONNEGATIVE_METRIC_KEYS:
        value = scalars.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) < 0.0:
            errors.append(f"validation metric {key} must be non-negative, got {float(value):g}")

    mixed = scalars.get("val/attention_alignment_target_attention_lift_mixed")
    without_geometry = scalars.get("val/attention_alignment_target_attention_lift_without_geometry")
    if (
        isinstance(mixed, (int, float))
        and isinstance(without_geometry, (int, float))
        and math.isfinite(float(mixed))
        and math.isfinite(float(without_geometry))
        and float(mixed) <= float(without_geometry)
    ):
        errors.append(
            "validation target lift with geometry must exceed without-geometry lift, "
            f"got mixed={float(mixed):g} without_geometry={float(without_geometry):g}"
        )
    return errors


def scan_gate_outputs(
    train_fixed_dir: Path,
    heldout_dir: Path,
    *,
    scalars_jsonl: Optional[Path] = None,
    min_common_frames: int = 1,
    expected_sat_condition_mode: Optional[str] = DEFAULT_EXPECTED_SAT_CONDITION_MODE,
    require_scalars: bool = False,
    require_checkpoint_state: bool = False,
    require_metric_sanity: bool = False,
) -> GateScan:
    train_fixed = scan_preset_outputs(
        train_fixed_dir,
        "train_fixed",
        expected_sat_condition_mode=expected_sat_condition_mode,
    )
    heldout = scan_preset_outputs(
        heldout_dir,
        "heldout",
        expected_sat_condition_mode=expected_sat_condition_mode,
    )
    train_keys = set(train_fixed.frames)
    heldout_keys = set(heldout.frames)
    common = sorted(train_keys & heldout_keys)
    min_common_frames = max(1, int(min_common_frames))
    cross_errors: List[str] = []
    if len(common) < min_common_frames:
        cross_errors.append(
            f"paired val/test frames must be at least {min_common_frames}, got {len(common)}"
        )
    train_split = train_fixed.metadata.get("dataset_split")
    heldout_split = heldout.metadata.get("dataset_split")
    if train_split in VALID_GATE_SPLITS and heldout_split in VALID_GATE_SPLITS and train_split != heldout_split:
        cross_errors.append(
            f"train_fixed and heldout must use the same dataset_split, got {train_split!r} vs {heldout_split!r}"
        )
    train_checkpoint = train_fixed.metadata.get("checkpoint")
    heldout_checkpoint = heldout.metadata.get("checkpoint")
    if train_checkpoint is not None and heldout_checkpoint is not None and str(train_checkpoint) != str(heldout_checkpoint):
        cross_errors.append(
            f"train_fixed and heldout must use the same checkpoint, got {train_checkpoint!r} vs {heldout_checkpoint!r}"
        )
    if require_checkpoint_state:
        cross_errors.extend(_inspect_checkpoint_state_dict(train_checkpoint))
    train_split_yaml = train_fixed.metadata.get("split_yaml")
    heldout_split_yaml = heldout.metadata.get("split_yaml")
    if train_split_yaml is not None and heldout_split_yaml is not None and str(train_split_yaml) != str(heldout_split_yaml):
        cross_errors.append(
            f"train_fixed and heldout must use the same split_yaml, got {train_split_yaml!r} vs {heldout_split_yaml!r}"
        )
    train_runtime = train_fixed.metadata.get("inference_runtime_config")
    heldout_runtime = heldout.metadata.get("inference_runtime_config")
    if isinstance(train_runtime, Mapping) and isinstance(heldout_runtime, Mapping):
        if dict(train_runtime) != dict(heldout_runtime):
            cross_errors.append(
                "train_fixed and heldout must use the same inference_runtime_config, "
                f"got {dict(train_runtime)!r} vs {dict(heldout_runtime)!r}"
            )
    if expected_sat_condition_mode is not None:
        for label, metadata in (("train_fixed", train_fixed.metadata), ("heldout", heldout.metadata)):
            runtime_config = metadata.get("inference_runtime_config")
            if isinstance(runtime_config, Mapping):
                sat_condition_mode = runtime_config.get("sat_condition_mode")
                if sat_condition_mode != expected_sat_condition_mode:
                    cross_errors.append(
                        f"{label} inference_runtime_config.sat_condition_mode must be "
                        f"{expected_sat_condition_mode!r}, got {sat_condition_mode!r}"
                    )
            top_level_sat_condition_mode = metadata.get("sat_condition_mode")
            if top_level_sat_condition_mode is not None and top_level_sat_condition_mode != expected_sat_condition_mode:
                cross_errors.append(
                    f"{label} sat_condition_mode must be "
                    f"{expected_sat_condition_mode!r}, got {top_level_sat_condition_mode!r}"
                )
    train_epoch = _metadata_checkpoint_epoch(train_fixed.metadata)
    heldout_epoch = _metadata_checkpoint_epoch(heldout.metadata)
    if train_epoch is not None and heldout_epoch is not None and abs(train_epoch - heldout_epoch) > 1e-6:
        cross_errors.append(
            f"train_fixed and heldout must use the same checkpoint epoch, got {train_epoch:g} vs {heldout_epoch:g}"
        )
    if require_scalars and scalars_jsonl is None:
        cross_errors.append("scalars_jsonl is required for the formal pose-chain gate")
    latest_scalars = _load_latest_scalar_metrics(scalars_jsonl)
    if scalars_jsonl is not None:
        if not scalars_jsonl.exists():
            cross_errors.append(f"missing scalars_jsonl: {scalars_jsonl}")
        elif not latest_scalars:
            cross_errors.append(
                f"scalars_jsonl has no validation record with val/ metrics: {scalars_jsonl}"
            )
        else:
            scalar_epoch = _latest_scalar_epoch(latest_scalars)
            checkpoint_epoch = train_epoch if train_epoch is not None else heldout_epoch
            if checkpoint_epoch is not None and scalar_epoch is None:
                cross_errors.append(
                    "latest validation scalars must include val/epoch or train/epoch "
                    "so metrics can be tied to the checkpoint epoch"
                )
            elif (
                checkpoint_epoch is not None
                and scalar_epoch is not None
                and abs(float(scalar_epoch) - float(checkpoint_epoch)) > 1e-6
            ):
                cross_errors.append(
                    f"latest validation scalar epoch must match checkpoint epoch, "
                    f"got scalars={scalar_epoch:g} checkpoint={checkpoint_epoch:g}"
                )
            missing_scalar_keys = _missing_required_validation_scalar_keys(latest_scalars)
            if missing_scalar_keys:
                cross_errors.append(
                    "latest validation scalars missing required gate metrics: "
                    + ", ".join(missing_scalar_keys)
                )
            elif require_metric_sanity:
                cross_errors.extend(_validate_metric_sanity(latest_scalars))
    return GateScan(
        train_fixed=train_fixed,
        heldout=heldout,
        common_frames=common,
        train_only_frames=sorted(train_keys - heldout_keys),
        heldout_only_frames=sorted(heldout_keys - train_keys),
        min_common_frames=min_common_frames,
        cross_errors=cross_errors,
        latest_scalars=latest_scalars,
    )


def _format_frame_list(frames: Sequence[FrameKey], limit: int = 20) -> str:
    if not frames:
        return "none"
    labels = [frame.label for frame in frames[:limit]]
    suffix = "" if len(frames) <= limit else f" ... (+{len(frames) - limit} more)"
    return ", ".join(labels) + suffix


def _format_runtime_config(config: Mapping) -> str:
    ordered_keys = [key for key in REQUIRED_INFERENCE_RUNTIME_KEYS if key in config]
    ordered_keys.extend(
        sorted(str(key) for key in config.keys() if key not in ordered_keys)
    )
    return ", ".join(f"{key}={config[key]!r}" for key in ordered_keys)


def _format_preset_section(scan: PresetScan) -> List[str]:
    lines = [
        f"## {scan.preset}",
        f"- root: `{scan.root}`",
        f"- preset_dir: `{scan.preset_dir}`",
        f"- metadata: `{scan.metadata_path}`" if scan.metadata_path else "- metadata: missing",
        f"- expected_views: {', '.join(scan.expected_views) if scan.expected_views else 'unknown'}",
        f"- frames: {len(scan.frames)} total, {scan.complete_frames} complete, {scan.incomplete_frames} incomplete",
    ]
    dataset_split = scan.metadata.get("dataset_split")
    if dataset_split is not None:
        lines.append(f"- dataset_split: `{dataset_split}`")
    checkpoint = scan.metadata.get("checkpoint")
    if checkpoint is not None:
        lines.append(f"- checkpoint: `{checkpoint}`")
    checkpoint_epoch = _metadata_checkpoint_epoch(scan.metadata)
    if checkpoint_epoch is not None:
        lines.append(f"- checkpoint_epoch: {checkpoint_epoch:g}")
    runtime_config = scan.metadata.get("inference_runtime_config")
    if isinstance(runtime_config, Mapping):
        lines.append(f"- inference_runtime_config: `{_format_runtime_config(runtime_config)}`")
    if scan.errors:
        lines.append("- errors:")
        lines.extend(f"  - {error}" for error in scan.errors)

    incomplete = [frame for frame in sorted(scan.frames.values(), key=lambda item: item.key) if not frame.is_complete]
    if incomplete:
        lines.append("- incomplete frame details:")
        for frame in incomplete[:20]:
            bits = []
            if frame.missing_frame_files:
                bits.append(f"missing frame files={frame.missing_frame_files}")
            if frame.invalid_frame_files:
                bits.append(f"invalid frame files={frame.invalid_frame_files}")
            if frame.missing_views:
                bits.append(f"missing views={frame.missing_views}")
            if frame.incomplete_views:
                bits.append(f"incomplete views={frame.incomplete_views}")
            if frame.invalid_view_files:
                bits.append(f"invalid view files={frame.invalid_view_files}")
            if frame.metadata_errors:
                bits.append(f"metadata errors={frame.metadata_errors}")
            lines.append(f"  - {frame.key.label}: {'; '.join(bits)}")
        if len(incomplete) > 20:
            lines.append(f"  - ... +{len(incomplete) - 20} more")
    return lines


def _format_derived_scalar_lines(scalars: Mapping[str, float]) -> List[str]:
    lines: List[str] = []
    mixed = scalars.get("val/attention_alignment_target_attention_lift_mixed")
    geometry_only = scalars.get("val/attention_alignment_target_attention_lift_geometry_only")
    without_geometry = scalars.get("val/attention_alignment_target_attention_lift_without_geometry")
    if isinstance(mixed, (int, float)) and isinstance(geometry_only, (int, float)):
        lines.append(
            "- `val/derived_target_lift_mixed_minus_geometry_only`: "
            f"{float(mixed) - float(geometry_only):.6g}"
        )
    if isinstance(mixed, (int, float)) and isinstance(without_geometry, (int, float)):
        lines.append(
            "- `val/derived_target_lift_mixed_minus_without_geometry`: "
            f"{float(mixed) - float(without_geometry):.6g}"
        )
    return lines


def render_markdown_report(scan: GateScan) -> str:
    status = "READY_FOR_MANUAL_TEST_GATE" if scan.is_output_complete else "INCOMPLETE_OUTPUTS"
    lines = [
        "# Pose-Chain Test Gate Report",
        "",
        f"- status: `{status}`",
        f"- paired val/test frames: {len(scan.common_frames)}",
        f"- required paired frames: {max(1, int(scan.min_common_frames))}",
        f"- train_fixed-only frames: {_format_frame_list(scan.train_only_frames)}",
        f"- heldout-only frames: {_format_frame_list(scan.heldout_only_frames)}",
    ]
    runtime_config = scan.train_fixed.metadata.get("inference_runtime_config")
    if isinstance(runtime_config, Mapping):
        lines.append(f"- inference runtime: `{_format_runtime_config(runtime_config)}`")
    lines.extend(
        [
            "",
            "## Gate Rule",
            "- Use validation/test split outputs only; do not accept train-split visuals as evidence.",
            "- Pass v1/v2 only if train_fixed test views keep continuous road/layout geometry and heldout yaw views do not collapse or rotate inconsistently.",
            "- If attention/coverage metrics look correct but generated road direction is still wrong on test frames, move to v3 latent-action auxiliary.",
            "- This script checks completeness and pairing; the final visual/metric decision still needs human review of the generated comparison grids and validation metrics.",
            "",
            "## Manual Review Checklist",
            "- Open the same drive/frame in train_fixed and heldout reports.",
            "- Check front -> +/-60 -> +/-90 -> +/-120 continuity in train_fixed.",
            "- Check +/-45, +/-75, +/-105 interpolation behavior in heldout.",
            "- Prefer test frames where satellite roads are not fully ambiguous under tree shadow before deciding the algorithm failed.",
            "- Compare validation scalars: target lift should stay positive, attention coverage should move continuously, and semantic-to-geometry ratio should not dominate early addressing.",
            "",
        ]
    )
    if scan.latest_scalars:
        lines.extend(["## Latest Validation Scalars"])
        scalar_epoch = _latest_scalar_epoch(scan.latest_scalars)
        if scalar_epoch is not None:
            lines.append(f"- scalar epoch: {scalar_epoch:g}")
        for key in REQUIRED_VALIDATION_SCALAR_KEYS:
            if key in scan.latest_scalars:
                lines.append(f"- `{key}`: {scan.latest_scalars[key]:.6g}")
        lines.extend(_format_derived_scalar_lines(scan.latest_scalars))
        lines.append("")
    if scan.cross_errors:
        lines.append("## Cross-Run Errors")
        lines.extend(f"- {error}" for error in scan.cross_errors)
        lines.append("")
    lines.extend(_format_preset_section(scan.train_fixed))
    lines.append("")
    lines.extend(_format_preset_section(scan.heldout))
    lines.append("")
    lines.extend(
        [
            "## Paired Frames",
            f"- common: {_format_frame_list(scan.common_frames)}",
            f"- train_fixed-only: {_format_frame_list(scan.train_only_frames)}",
            f"- heldout-only: {_format_frame_list(scan.heldout_only_frames)}",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check pose-chain train_fixed/heldout yaw-sweep outputs before manual test-gate review."
    )
    parser.add_argument("--train_fixed_dir", type=Path, required=True)
    parser.add_argument("--heldout_dir", type=Path, required=True)
    parser.add_argument(
        "--scalars_jsonl",
        type=Path,
        default=None,
        help="Optional output/.../logs/scalars.jsonl from training.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown report path.")
    parser.add_argument(
        "--min_common_frames",
        type=int,
        default=1,
        help="Minimum number of paired train_fixed/heldout validation or test frames required for readiness.",
    )
    parser.add_argument(
        "--expected_sat_condition_mode",
        type=str,
        default=DEFAULT_EXPECTED_SAT_CONDITION_MODE,
        help=(
            "Expected satellite conditioning mode for the formal gate. "
            "Use an empty string to disable this check for ablation-only diagnostics."
        ),
    )
    parser.add_argument(
        "--require_scalars",
        action="store_true",
        help="Require --scalars_jsonl with validation diagnostics for formal gate readiness.",
    )
    parser.add_argument(
        "--require_checkpoint_state",
        action="store_true",
        help=(
            "Load the checkpoint and reject removed additive PE parameters. "
            "Use this for the formal gate."
        ),
    )
    parser.add_argument(
        "--require_metric_sanity",
        action="store_true",
        help=(
            "Require conservative validation metric sanity checks, including "
            "positive target lift and valid chain pair ratios."
        ),
    )
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 when outputs are incomplete.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scan = scan_gate_outputs(
        args.train_fixed_dir,
        args.heldout_dir,
        scalars_jsonl=args.scalars_jsonl,
        min_common_frames=args.min_common_frames,
        expected_sat_condition_mode=args.expected_sat_condition_mode or None,
        require_scalars=bool(args.require_scalars),
        require_checkpoint_state=bool(args.require_checkpoint_state),
        require_metric_sanity=bool(args.require_metric_sanity),
    )
    report = render_markdown_report(scan)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"Wrote gate report to {args.output}")
    else:
        print(report)
    if args.strict and not scan.is_output_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
