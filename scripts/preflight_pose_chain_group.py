#!/usr/bin/env python3
"""Preflight checks for pose-chain group training and gate inference."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml


SD21_CACHE_DIR = "models--sd2-community--stable-diffusion-2-1-base"
EXPECTED_POSE_CHAINS = {
    "right": [None, 60.0, 90.0, 120.0],
    "left": [None, -60.0, -90.0, -120.0],
}
REQUIRED_SNAPSHOT_FILES: tuple[tuple[str, ...], ...] = (
    ("model_index.json",),
    ("vae/config.json",),
    ("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.bin"),
    ("unet/config.json",),
    ("unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.bin"),
    ("scheduler/scheduler_config.json",),
    ("text_encoder/config.json",),
    ("text_encoder/model.safetensors", "text_encoder/pytorch_model.bin"),
    ("tokenizer/tokenizer_config.json",),
    ("tokenizer/vocab.json",),
)


@dataclass
class PreflightResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_yaml_mapping(path: Path, errors: list[str], label: str) -> Mapping:
    if not path.is_file():
        errors.append(f"{label} not found: {path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        errors.append(f"{label} could not be parsed: {path}: {exc}")
        return {}
    if not isinstance(data, Mapping):
        errors.append(f"{label} must be a mapping: {path}")
        return {}
    return data


def _normalize_yaw(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "front":
        return None
    return float(value)


def _validate_pose_chain_config(config: Mapping, result: PreflightResult) -> None:
    data_config = config.get("data")
    if not isinstance(data_config, Mapping):
        result.errors.append("config.data must be a mapping")
        return

    view_set = data_config.get("view_set")
    if view_set != "pose_chain":
        result.errors.append(f"config.data.view_set must be 'pose_chain', got {view_set!r}")

    pose_chains = data_config.get("pose_chains")
    if not isinstance(pose_chains, Sequence) or isinstance(pose_chains, (str, bytes)) or not pose_chains:
        result.errors.append("config.data.pose_chains must be a non-empty YAML list")
        return

    actual: dict[str, list[Optional[float]]] = {}
    lengths: set[int] = set()
    for index, chain in enumerate(pose_chains):
        if not isinstance(chain, Mapping):
            result.errors.append(f"config.data.pose_chains[{index}] must be a mapping")
            continue
        name = chain.get("name")
        yaws = chain.get("yaws", chain.get("views"))
        if not isinstance(name, str):
            result.errors.append(f"config.data.pose_chains[{index}].name must be a string")
            continue
        if name in actual:
            result.errors.append(f"config.data.pose_chains has duplicate chain name {name!r}")
            continue
        if isinstance(yaws, (str, bytes)) or not isinstance(yaws, Sequence):
            result.errors.append(f"config.data.pose_chains[{name!r}].yaws must be a YAML list")
            continue
        try:
            normalized = [_normalize_yaw(yaw) for yaw in yaws]
        except Exception as exc:
            result.errors.append(f"config.data.pose_chains[{name!r}].yaws contains invalid yaw: {exc}")
            continue
        actual[name] = normalized
        lengths.add(len(normalized))

    if len(lengths) > 1:
        result.errors.append("all config.data.pose_chains must have the same number of views")

    for name, expected_yaws in EXPECTED_POSE_CHAINS.items():
        actual_yaws = actual.get(name)
        if actual_yaws is None:
            result.errors.append(f"config.data.pose_chains must include {name!r}")
            continue
        if actual_yaws != expected_yaws:
            result.errors.append(
                f"config.data.pose_chains[{name!r}] must be {expected_yaws!r}, got {actual_yaws!r}"
            )


def _validate_geometry_config(config: Mapping, result: PreflightResult) -> None:
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        result.errors.append("config.model must be a mapping")
        return
    geometry_config = model_config.get("query_geometry_score")
    if not isinstance(geometry_config, Mapping):
        result.errors.append("config.model.query_geometry_score must be a mapping")
        return
    if not bool(geometry_config.get("enable", False)):
        result.errors.append("config.model.query_geometry_score.enable must be true")
    mode = geometry_config.get("mode")
    if mode != "geometry_first_semantic_refine":
        result.errors.append(
            "config.model.query_geometry_score.mode must be "
            f"'geometry_first_semantic_refine', got {mode!r}"
        )

    training_config = config.get("training")
    if not isinstance(training_config, Mapping):
        result.errors.append("config.training must be a mapping")
        return
    alignment_config = training_config.get("attention_alignment")
    if not isinstance(alignment_config, Mapping) or not bool(alignment_config.get("enable", False)):
        result.errors.append("config.training.attention_alignment.enable must be true")

    validation_config = config.get("validation")
    if not isinstance(validation_config, Mapping):
        result.errors.append("config.validation must be a mapping")
        return
    validate_every = validation_config.get("validate_every")
    if not isinstance(validate_every, int) or validate_every <= 0:
        result.errors.append(f"config.validation.validate_every must be a positive integer, got {validate_every!r}")


def _validate_split_file(
    *,
    data_dir: Path,
    split_yaml: Path,
    gate_split: str,
    result: PreflightResult,
) -> None:
    split_config = _load_yaml_mapping(split_yaml, result.errors, "split yaml")
    if not split_config:
        return

    train_entries = split_config.get("train")
    if not isinstance(train_entries, list) or not train_entries:
        result.errors.append("split yaml must contain a non-empty 'train' list")

    eval_entries = None
    eval_label = gate_split
    if gate_split == "test":
        eval_entries = split_config.get("test")
        if not isinstance(eval_entries, list) or not eval_entries:
            result.errors.append("gate_split='test' requires a non-empty explicit 'test' list in split yaml")
    elif gate_split == "val":
        eval_entries = split_config.get("val")
        if eval_entries is None:
            eval_entries = split_config.get("test")
            eval_label = "test"
            if isinstance(eval_entries, list) and eval_entries:
                result.warnings.append("gate_split='val' will use split yaml 'test' entries because 'val' is absent")
        if not isinstance(eval_entries, list) or not eval_entries:
            result.errors.append("gate_split='val' requires a non-empty 'val' list or fallback 'test' list")
    else:
        result.errors.append(f"gate_split must be 'val' or 'test', got {gate_split!r}")

    for entries, label in ((train_entries, "train"), (eval_entries, eval_label)):
        if not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                result.errors.append(f"split yaml {label}[{index}] must be a mapping")
                continue
            drive = entry.get("drive")
            frames_file = entry.get("frames_file")
            if not drive or not frames_file:
                result.errors.append(f"split yaml {label}[{index}] must contain drive and frames_file")
                continue
            drive_dir = data_dir / str(drive)
            if not drive_dir.is_dir():
                result.errors.append(f"drive directory not found for split {label}[{index}]: {drive_dir}")
                continue
            frames_path = Path(str(frames_file))
            if not frames_path.is_absolute():
                frames_path = drive_dir / frames_path
            if not frames_path.is_file():
                result.errors.append(f"frames_file not found for split {label}[{index}]: {frames_path}")
                continue
            frame_ids = [line.strip() for line in frames_path.read_text().splitlines() if line.strip()]
            if not frame_ids:
                result.errors.append(f"frames_file is empty for split {label}[{index}]: {frames_path}")


def _snapshot_has_required_files(snapshot: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for alternatives in REQUIRED_SNAPSHOT_FILES:
        if not any((snapshot / relative).is_file() for relative in alternatives):
            missing.append(" or ".join(alternatives))
    return not missing, missing


def _validate_hf_cache(hf_home: Path, result: PreflightResult) -> None:
    cache_root = hf_home / "hub" / SD21_CACHE_DIR
    if not cache_root.is_dir():
        result.errors.append(f"SD2.1 cache directory not found: {cache_root}")
        return
    snapshots_dir = cache_root / "snapshots"
    snapshots = sorted(path for path in snapshots_dir.glob("*") if path.is_dir()) if snapshots_dir.is_dir() else []
    if not snapshots:
        result.errors.append(f"SD2.1 cache has no snapshots under: {snapshots_dir}")
        return
    missing_by_snapshot: list[str] = []
    for snapshot in snapshots:
        ok, missing = _snapshot_has_required_files(snapshot)
        if ok:
            return
        missing_by_snapshot.append(f"{snapshot.name}: {', '.join(missing)}")
    result.errors.append(
        "no SD2.1 cache snapshot contains the required diffusers files; "
        + "; ".join(missing_by_snapshot[:3])
    )


def _visible_cuda_count(env: Mapping[str, str]) -> Optional[int]:
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return None
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    return len(tokens)


def _validate_environment(
    *,
    expected_num_gpus: int,
    require_offline_env: bool,
    env: Mapping[str, str],
    result: PreflightResult,
) -> None:
    if expected_num_gpus > 0:
        count = _visible_cuda_count(env)
        if count is None:
            result.errors.append(
                "CUDA_VISIBLE_DEVICES is not set; cannot verify expected GPU count "
                f"{expected_num_gpus}"
            )
        elif count != expected_num_gpus:
            result.errors.append(
                f"CUDA_VISIBLE_DEVICES exposes {count} GPU(s), expected {expected_num_gpus}"
            )

    if require_offline_env:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
            value = env.get(name)
            if str(value).lower() not in {"1", "true", "yes"}:
                result.errors.append(f"{name} must be set to 1/true for offline server startup")


def run_preflight(
    *,
    config_path: Path,
    data_dir: Path,
    split_yaml: Path,
    hf_home: Path,
    gate_split: str = "val",
    expected_num_gpus: int = 0,
    require_offline_env: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> PreflightResult:
    result = PreflightResult()
    config = _load_yaml_mapping(config_path, result.errors, "config")
    if config:
        _validate_pose_chain_config(config, result)
        _validate_geometry_config(config, result)

    if not data_dir.is_dir():
        result.errors.append(f"data_dir not found: {data_dir}")
    else:
        _validate_split_file(
            data_dir=data_dir,
            split_yaml=split_yaml,
            gate_split=gate_split,
            result=result,
        )

    _validate_hf_cache(hf_home, result)
    _validate_environment(
        expected_num_gpus=expected_num_gpus,
        require_offline_env=require_offline_env,
        env=env if env is not None else os.environ,
        result=result,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check pose-chain group server prerequisites before training/inference."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split_yaml", type=Path, default=None)
    parser.add_argument("--hf_home", type=Path, default=Path(".hf-home"))
    parser.add_argument("--gate_split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--expected_num_gpus", type=int, default=0)
    parser.add_argument("--require_offline_env", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    split_yaml = args.split_yaml if args.split_yaml is not None else args.data_dir / "train_test_split_config.yaml"
    result = run_preflight(
        config_path=args.config,
        data_dir=args.data_dir,
        split_yaml=split_yaml,
        hf_home=args.hf_home,
        gate_split=args.gate_split,
        expected_num_gpus=max(0, int(args.expected_num_gpus)),
        require_offline_env=bool(args.require_offline_env),
    )

    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")
    if result.ok:
        print("POSE_CHAIN_PREFLIGHT_OK")
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
