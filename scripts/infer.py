#!/usr/bin/env python3
"""
Inference utilities for satellite-to-street generation.

Supported modes:
  - single_yaw_sweep: render one frame at front plus vehicle-relative yaw values.
  - split_yaw_sweep: render split frames with a yaw sweep preset.
  - split_fixed_views: render split frames at fixed views and save GT comparisons.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.kitti360d_dataset import Kitti360dDataset, SampleIndex
from models.sd_model import create_sd_model, load_model_checkpoint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DEFAULT_SD21_BASE_REPO = "sd2-community/stable-diffusion-2-1-base"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_HOME = _project_root / ".hf-home"

FIXED_VIEW_SPECS: Sequence[Tuple[str, Optional[float]]] = (
    ("front", None),
    ("left_forward_45", -45.0),
    ("left_side", -90.0),
    ("right_forward_45", 45.0),
    ("right_side", 90.0),
)

DEFAULT_YAW_SWEEP_SPECS: Sequence[Tuple[str, Optional[float]]] = (
    ("front", None),
    ("yaw_m120", -120.0),
    ("yaw_m90", -90.0),
    ("yaw_m60", -60.0),
    ("yaw_m30", -30.0),
    ("yaw_p30", 30.0),
    ("yaw_p60", 60.0),
    ("yaw_p90", 90.0),
    ("yaw_p120", 120.0),
)

TRAIN_FIXED_YAW_SWEEP_SPECS: Sequence[Tuple[str, Optional[float]]] = (
    ("front", None),
    ("yaw_m120", -120.0),
    ("yaw_m90", -90.0),
    ("yaw_m60", -60.0),
    ("yaw_p60", 60.0),
    ("yaw_p90", 90.0),
    ("yaw_p120", 120.0),
)

JOINT_POSE_CHAIN_SPECS: Sequence[Sequence[Tuple[str, Optional[float]]]] = (
    (("front", None), ("yaw_p60", 60.0), ("yaw_p90", 90.0), ("yaw_p120", 120.0)),
    (("front", None), ("yaw_m60", -60.0), ("yaw_m90", -90.0), ("yaw_m120", -120.0)),
)

HELDOUT_YAW_SWEEP_SPECS: Sequence[Tuple[str, Optional[float]]] = (
    ("yaw_m105", -105.0),
    ("yaw_m75", -75.0),
    ("yaw_m45", -45.0),
    ("yaw_p45", 45.0),
    ("yaw_p75", 75.0),
    ("yaw_p105", 105.0),
)

YAW_SWEEP_PRESETS: Dict[str, Sequence[Tuple[str, Optional[float]]]] = {
    "diagnostic": DEFAULT_YAW_SWEEP_SPECS,
    "heldout": HELDOUT_YAW_SWEEP_SPECS,
    "left_chain": JOINT_POSE_CHAIN_SPECS[1],
    "right_chain": JOINT_POSE_CHAIN_SPECS[0],
    "train_fixed": TRAIN_FIXED_YAW_SWEEP_SPECS,
}

ABLATION_MODE_CONFIGS: Dict[str, str] = {
    "normal": "normal",
    "sat_zero": "zero",
}

CHECKPOINT_RUN_CONFIG_KEYS: Tuple[str, ...] = (
    "view_set",
    "pose_chains",
    "pose_chain_group_size",
    "effective_view_batch_size",
    "query_geometry_score_enabled",
    "query_geometry_score_dim",
    "query_geometry_score_num_freqs",
    "query_geometry_score_mode",
    "query_geometry_score_gate_init",
    "query_geometry_candidate_radius",
    "query_geometry_candidate_min_k",
    "query_geometry_candidate_invalid_penalty",
    "query_semantic_score_dim",
    "query_semantic_score_alpha",
    "attention_alignment_enabled",
    "attention_alignment_loss_weight",
    "attention_alignment_valid_radius",
    "joint_view_generation_enabled",
    "joint_view_generation_loss_weight",
)
INFERENCE_GATE_CONFIG_KEYS: Tuple[str, ...] = (
    "query_geometry_score_enabled",
    "query_geometry_score_dim",
    "query_geometry_score_num_freqs",
    "query_geometry_score_mode",
    "query_geometry_score_gate_init",
    "query_geometry_candidate_radius",
    "query_geometry_candidate_min_k",
    "query_geometry_candidate_invalid_penalty",
    "query_semantic_score_dim",
    "query_semantic_score_alpha",
    "joint_view_generation_enabled",
    "joint_view_generation_loss_weight",
)


def _load_frame_ids(frames_file: Path) -> List[int]:
    frame_ids: List[int] = []
    for line in frames_file.read_text().splitlines():
        token = line.strip()
        if token:
            frame_ids.append(int(token))
    return frame_ids


def _checkpoint_gate_metadata(checkpoint_meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(checkpoint_meta, dict):
        return {}
    payload: Dict[str, Any] = {}
    trainer_metadata = checkpoint_meta.get("trainer_metadata")
    if isinstance(trainer_metadata, dict):
        payload["trainer_metadata"] = {
            str(key): value
            for key, value in trainer_metadata.items()
            if isinstance(value, (bool, int, float, str)) or value is None
        }
    run_config = checkpoint_meta.get("run_config")
    if isinstance(run_config, dict):
        payload["run_config"] = {
            key: run_config[key]
            for key in CHECKPOINT_RUN_CONFIG_KEYS
            if key in run_config
        }
    return payload


def _checkpoint_display_epoch(checkpoint_meta: Dict[str, Any]) -> Optional[int]:
    if not isinstance(checkpoint_meta, dict):
        return None
    trainer_metadata = checkpoint_meta.get("trainer_metadata")
    if isinstance(trainer_metadata, dict):
        checkpoint_epoch = trainer_metadata.get("checkpoint_epoch")
        if isinstance(checkpoint_epoch, (int, float)):
            return int(checkpoint_epoch)
    raw_epoch = checkpoint_meta.get("epoch")
    if isinstance(raw_epoch, (int, float)):
        return int(raw_epoch) + 1
    return None


def _inference_gate_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: getattr(args, key)
        for key in INFERENCE_GATE_CONFIG_KEYS
        if hasattr(args, key)
    }


def _inference_runtime_config(
    args: argparse.Namespace,
    *,
    sat_condition_mode: Optional[str] = None,
) -> Dict[str, Any]:
    active_sat_condition_mode = (
        sat_condition_mode
        if sat_condition_mode is not None
        else getattr(args, "sat_condition_mode")
    )
    return {
        "num_inference_steps": int(getattr(args, "num_inference_steps")),
        "guidance_scale": float(getattr(args, "guidance_scale")),
        "seed": int(getattr(args, "seed")),
        "mixed_precision": str(getattr(args, "mixed_precision")),
        "view_memory_mode": str(getattr(args, "view_memory_mode")),
        "sat_condition_mode": str(active_sat_condition_mode),
    }


def _gate_values_match(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-6
    return left == right


def _checkpoint_inference_mismatches(
    checkpoint_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    checkpoint_gate = _checkpoint_gate_metadata(checkpoint_meta)
    run_config = checkpoint_gate.get("run_config")
    if not isinstance(run_config, dict):
        return ["checkpoint is missing run_config metadata for inference consistency checks"]

    inference_config = _inference_gate_config(args)
    mismatches: List[str] = []
    for key, inference_value in inference_config.items():
        if key not in run_config:
            continue
        checkpoint_value = run_config[key]
        if not _gate_values_match(checkpoint_value, inference_value):
            mismatches.append(
                f"{key}: checkpoint={checkpoint_value!r} inference={inference_value!r}"
            )
    return mismatches


def _load_split_from_yaml(
    data_dir: Path,
    split_yaml: Path,
    eval_split: str = "val",
) -> Tuple[List[Path], List[List[int]], List[Path], List[List[int]]]:
    if not split_yaml.exists():
        raise FileNotFoundError(f"Split yaml not found: {split_yaml}")

    with open(split_yaml, "r") as f:
        split_cfg = yaml.safe_load(f)

    if not isinstance(split_cfg, dict):
        raise ValueError(f"Invalid split yaml format: {split_yaml}")

    train_entries = split_cfg.get("train")
    eval_split = str(eval_split)
    if eval_split == "test":
        eval_entries = split_cfg.get("test")
        eval_label = "test"
    elif eval_split == "val":
        eval_entries = split_cfg.get("val", split_cfg.get("test"))
        eval_label = "val/test"
    else:
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")
    if not isinstance(train_entries, list):
        raise ValueError("Split yaml must contain list entries for 'train'")
    if not isinstance(eval_entries, list):
        raise ValueError(f"Split yaml must contain list entries for '{eval_split}'")

    def parse_entries(entries: List[dict], split_name: str) -> Tuple[List[Path], List[List[int]]]:
        drives: List[Path] = []
        frames_per_drive: List[List[int]] = []

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{split_name} entry must be a dict, got: {entry}")
            drive_name = entry.get("drive")
            frames_file_name = entry.get("frames_file")
            if not drive_name or not frames_file_name:
                raise ValueError(f"{split_name} entry must contain 'drive' and 'frames_file': {entry}")

            drive_dir = data_dir / str(drive_name)
            if not drive_dir.is_dir():
                raise FileNotFoundError(f"Drive directory not found: {drive_dir}")

            frames_file = drive_dir / str(frames_file_name)
            if not frames_file.is_file():
                raise FileNotFoundError(f"Frames file not found: {frames_file}")

            frame_ids = _load_frame_ids(frames_file)
            if not frame_ids:
                raise ValueError(f"No frame ids found in {frames_file}")

            drives.append(drive_dir)
            frames_per_drive.append(frame_ids)

        return drives, frames_per_drive

    train_dirs, train_frames = parse_entries(train_entries, "train")
    eval_dirs, eval_frames = parse_entries(eval_entries, eval_label)
    return train_dirs, train_frames, eval_dirs, eval_frames


def _eval_split_for_dataset_split(dataset_split: str) -> str:
    return "test" if str(dataset_split) == "test" else "val"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for satellite-to-street generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional inference YAML.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single_yaw_sweep",
        choices=["single_yaw_sweep", "split_yaw_sweep", "split_fixed_views", "front_pitch_sweep"],
        help="Inference mode.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/user/574b4a05-57d2-424d-bb82-763098cbf0a4/shizhm/KITTI-360",
        help="Path to KITTI-360 data root.",
    )
    parser.add_argument("--split_yaml", type=str, default=None, help="Split yaml for split-based inference.")
    parser.add_argument("--dataset_split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--drive", type=str, default=None, help="Drive name for single_yaw_sweep.")
    parser.add_argument("--drive_dir", type=str, default=None, help="Drive path for single_yaw_sweep.")
    parser.add_argument("--frame_id", type=int, default=None, help="Exact frame id for single_yaw_sweep.")
    parser.add_argument("--start_frame", type=int, default=None, help="Minimum frame id for split_fixed_views.")
    parser.add_argument("--end_frame", type=int, default=None, help="Maximum frame id for split_fixed_views.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of split frames.")
    parser.add_argument(
        "--vehicle_yaws",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Yaw values for single_yaw_sweep. When omitted, uses the default "
            "front/-120/-90/-60/-30/+30/+60/+90/+120 diagnostic sweep. "
            "Front is included by default; pass --no_include_front to omit it."
        ),
    )
    parser.add_argument(
        "--yaw_sweep_preset",
        type=str,
        default="right_chain",
        choices=sorted(YAW_SWEEP_PRESETS.keys()),
        help=(
            "Preset yaw list for single_yaw_sweep when --vehicle_yaws is omitted. "
            "right_chain/left_chain are the joint-generation fixed chains; "
            "diagnostic includes +/-30; train_fixed uses front/-120/-90/-60/+60/+90/+120; "
            "heldout uses +/-45, +/-75, +/-105."
        ),
    )
    front_group = parser.add_mutually_exclusive_group()
    front_group.add_argument("--include_front", dest="include_front", action="store_true")
    front_group.add_argument("--no_include_front", dest="include_front", action="store_false")
    parser.set_defaults(include_front=True)
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pitch_deg",
        type=float,
        default=0.0,
        help="Virtual camera pitch in degrees for fisheye remap and BEV projection.",
    )
    parser.add_argument(
        "--pitch_values",
        type=float,
        nargs="+",
        default=None,
        help="Pitch values for front_pitch_sweep. Defaults to 0 5 10 15 20 25 30.",
    )
    parser.add_argument(
        "--roll_deg",
        type=float,
        default=0.0,
        help="Virtual camera roll in degrees for fisheye remap and BEV projection.",
    )
    parser.add_argument(
        "--view_memory_mode",
        type=str,
        default="joint_pose_chain",
        choices=["joint_pose_chain"],
        help=(
            "Jointly sample front/+60/+90/+120 or front/-60/-90/-120. "
            "Single-view fallback is intentionally disabled."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--base_model", type=str, default=DEFAULT_SD21_BASE_REPO)
    parser.add_argument("--base_model_revision", type=str, default=None)
    parser.add_argument("--hf_endpoint", type=str, default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--hf_home", type=str, default=str(DEFAULT_HF_HOME))
    parser.add_argument(
        "--sat_condition_mode",
        type=str,
        default="normal",
        choices=["normal", "zero"],
        help="Use zero for satellite-conditioning ablation.",
    )
    parser.add_argument(
        "--ablation_modes",
        type=str,
        nargs="+",
        default=None,
        choices=list(ABLATION_MODE_CONFIGS.keys()),
        help=(
            "Optional suite of ablations to run in one command. "
            "When set, this overrides --sat_condition_mode and writes each ablation "
            "under a separate output subdirectory."
        ),
    )
    args = parser.parse_args()
    cli_options = _collect_cli_options(sys.argv[1:])
    config = _load_runtime_config(Path(args.config)) if args.config is not None else {}
    _apply_config_defaults(args, config, cli_options=cli_options)
    return args


def _load_runtime_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {config_path}")
    return config


def _config_get(config: Dict[str, Any], path: Tuple[str, ...], default: Any = None) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def _collect_cli_options(argv: Sequence[str]) -> Set[str]:
    options: Set[str] = set()
    for arg in argv:
        if arg.startswith("--"):
            options.add(arg.split("=", 1)[0])
    return options


def _prefer_config(
    current: Any,
    cli_default: Any,
    config_value: Any,
    *,
    cli_option: Optional[str] = None,
    cli_options: Optional[Set[str]] = None,
) -> Any:
    if cli_option is not None and cli_options is not None and cli_option in cli_options:
        return current
    if current == cli_default and config_value is not None:
        return config_value
    return current


def _apply_config_defaults(
    args: argparse.Namespace,
    config: Dict[str, Any],
    *,
    cli_options: Optional[Set[str]] = None,
) -> None:
    if not config:
        args.satellite_encoder_config = {}
        args.query_uv_pe_enabled = False
        args.query_geometry_score_enabled = False
        args.query_geometry_score_dim = 64
        args.query_geometry_score_num_freqs = 6
        args.query_geometry_score_gate_init = 1.0
        args.query_geometry_score_layers = None
        args.query_geometry_score_max_query_tokens = None
        args.query_geometry_score_mode = "geometry_first_semantic_refine"
        args.query_geometry_candidate_radius = 0.35
        args.query_geometry_candidate_min_k = 16
        args.query_geometry_candidate_invalid_penalty = -1e4
        args.query_semantic_score_dim = 64
        args.query_semantic_score_alpha = 0.25
        args.joint_view_generation_enabled = False
        args.joint_view_generation_loss_weight = 0.0
        args.joint_view_generation_hidden_dim = 32
        args.joint_view_generation_num_heads = 4
        args.joint_view_generation_dropout = 0.0
        args.joint_view_generation_bev_sigma = 0.25
        args.joint_view_generation_gate_init = 0.0
        return

    args.seed = int(_prefer_config(args.seed, 42, _config_get(config, ("seed",)), cli_option="--seed", cli_options=cli_options))
    args.device = str(_prefer_config(args.device, "cuda", _config_get(config, ("device",)), cli_option="--device", cli_options=cli_options))
    args.data_dir = str(
        _prefer_config(
            args.data_dir,
            "/media/user/574b4a05-57d2-424d-bb82-763098cbf0a4/shizhm/KITTI-360",
            _config_get(config, ("data", "data_dir")),
            cli_option="--data_dir",
            cli_options=cli_options,
        )
    )
    args.base_model = str(
        _prefer_config(
            args.base_model,
            DEFAULT_SD21_BASE_REPO,
            _config_get(config, ("model", "base_model")),
            cli_option="--base_model",
            cli_options=cli_options,
        )
    )
    args.base_model_revision = _prefer_config(
        args.base_model_revision,
        None,
        _config_get(config, ("model", "base_model_revision")),
        cli_option="--base_model_revision",
        cli_options=cli_options,
    )
    checkpoint_path = _prefer_config(
        args.checkpoint,
        None,
        _config_get(config, ("model", "checkpoint_path")),
        cli_option="--checkpoint",
        cli_options=cli_options,
    )
    args.checkpoint = None if checkpoint_path is None else str(checkpoint_path)
    args.num_inference_steps = int(
        _prefer_config(
            args.num_inference_steps,
            30,
            _config_get(config, ("inference", "num_inference_steps")),
            cli_option="--num_inference_steps",
            cli_options=cli_options,
        )
    )
    args.guidance_scale = float(
        _prefer_config(
            args.guidance_scale,
            1.0,
            _config_get(config, ("inference", "guidance_scale")),
            cli_option="--guidance_scale",
            cli_options=cli_options,
        )
    )
    args.pitch_deg = float(
        _prefer_config(
            args.pitch_deg,
            0.0,
            _config_get(config, ("data", "pitch_deg")),
            cli_option="--pitch_deg",
            cli_options=cli_options,
        )
    )
    args.roll_deg = float(
        _prefer_config(
            args.roll_deg,
            0.0,
            _config_get(config, ("data", "roll_deg")),
            cli_option="--roll_deg",
            cli_options=cli_options,
        )
    )
    args.output_dir = str(
        _prefer_config(
            args.output_dir,
            "./inference_results",
            _config_get(config, ("output", "output_dir")),
            cli_option="--output_dir",
            cli_options=cli_options,
        )
    )

    args.satellite_encoder_config = dict(_config_get(config, ("model", "satellite_encoder"), {}) or {})
    args.query_uv_pe_enabled = False
    score_config = dict(_config_get(config, ("model", "query_geometry_score"), {}) or {})
    score_layers = score_config.get("layers")
    if score_layers is not None:
        score_layers = [str(layer) for layer in score_layers]
    score_max_query_tokens = score_config.get("max_query_tokens", None)
    if score_max_query_tokens is not None:
        score_max_query_tokens = int(score_max_query_tokens)
    args.query_geometry_score_enabled = bool(score_config.get("enable", False))
    args.query_geometry_score_dim = int(score_config.get("dim", 64) or 64)
    args.query_geometry_score_num_freqs = int(score_config.get("num_freqs", 6) or 6)
    args.query_geometry_score_gate_init = float(score_config.get("gate_init", 1.0) or 1.0)
    args.query_geometry_score_layers = score_layers
    args.query_geometry_score_max_query_tokens = score_max_query_tokens
    args.query_geometry_score_mode = str(score_config.get("mode", "geometry_first_semantic_refine"))
    args.query_geometry_candidate_radius = float(score_config.get("candidate_radius", 0.35) or 0.35)
    args.query_geometry_candidate_min_k = int(score_config.get("candidate_min_k", 16) or 16)
    args.query_geometry_candidate_invalid_penalty = float(score_config.get("candidate_invalid_penalty", -1e4) or -1e4)
    args.query_semantic_score_dim = int(score_config.get("semantic_score_dim", 64) or 64)
    args.query_semantic_score_alpha = float(score_config.get("semantic_alpha_max", 0.25) or 0.25)
    joint_config = dict(_config_get(config, ("training", "joint_view_generation"), {}) or {})
    args.joint_view_generation_enabled = bool(joint_config.get("enable", False))
    args.joint_view_generation_loss_weight = float(joint_config.get("loss_weight", 0.0) or 0.0)
    args.joint_view_generation_hidden_dim = int(joint_config.get("hidden_dim", 32) or 32)
    args.joint_view_generation_num_heads = int(joint_config.get("num_heads", 4) or 4)
    args.joint_view_generation_dropout = float(joint_config.get("dropout", 0.0) or 0.0)
    args.joint_view_generation_bev_sigma = float(joint_config.get("bev_sigma", 0.25) or 0.25)
    args.joint_view_generation_gate_init = float(joint_config.get("gate_init", 0.0) or 0.0)
    default_memory_mode = "joint_pose_chain"
    args.view_memory_mode = str(
        _prefer_config(
            args.view_memory_mode,
            "joint_pose_chain",
            _config_get(config, ("inference", "view_memory_mode"), default_memory_mode),
            cli_option="--view_memory_mode",
            cli_options=cli_options,
        )
    )
    if args.view_memory_mode != "joint_pose_chain":
        raise ValueError("Only inference.view_memory_mode=joint_pose_chain is supported in this branch")


def _view_token(view_name: str, yaw: Optional[float]) -> str:
    if yaw is None:
        return view_name
    prefix = "p" if yaw > 0 else "m" if yaw < 0 else ""
    token = f"yaw_{prefix}{abs(float(yaw)):g}".replace(".", "p")
    return token


def _single_yaw_sweep_view_specs(args: argparse.Namespace) -> List[Tuple[str, Optional[float]]]:
    if args.vehicle_yaws is None:
        specs = list(YAW_SWEEP_PRESETS[args.yaw_sweep_preset])
        if not args.include_front:
            specs = [(name, yaw) for name, yaw in specs if yaw is not None]
        return specs

    specs: List[Tuple[str, Optional[float]]] = []
    if args.include_front:
        specs.append(("front", None))
    specs.extend((_view_token("yaw", yaw), float(yaw)) for yaw in args.vehicle_yaws)
    return specs


def _resolve_ablation_runs(args: argparse.Namespace) -> List[Tuple[Optional[str], str]]:
    """Return (output_subdir, sat_condition_mode)."""
    if args.ablation_modes is None:
        return [(None, args.sat_condition_mode)]

    runs: List[Tuple[Optional[str], str]] = []
    for mode_name in args.ablation_modes:
        runs.append((mode_name, ABLATION_MODE_CONFIGS[mode_name]))
    return runs


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    if image.ndim == 3:
        image = image.permute(1, 2, 0)
    image = (image * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(image)


def _add_label(image: Image.Image, label: str) -> Image.Image:
    label_height = 24
    canvas = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(0, 0, 0))
    return canvas


def _compose_panels(panels: Sequence[Tuple[str, torch.Tensor]]) -> Image.Image:
    labeled = [_add_label(_tensor_to_pil(image), label) for label, image in panels]
    total_width = sum(image.width for image in labeled)
    max_height = max(image.height for image in labeled)
    canvas = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    x_offset = 0
    for image in labeled:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas


def _stack_panel_rows(rows: Sequence[Image.Image], spacing: int = 8) -> Image.Image:
    width = max(image.width for image in rows)
    height = sum(image.height for image in rows) + spacing * max(0, len(rows) - 1)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    y_offset = 0
    for image in rows:
        canvas.paste(image, (0, y_offset))
        y_offset += image.height + spacing
    return canvas


def _resize_satellite_for_front(sat_image: torch.Tensor, target_h: int) -> torch.Tensor:
    return F.interpolate(
        sat_image.unsqueeze(0),
        size=(target_h, target_h),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _project_satellite_to_perspective(
    sat_image: torch.Tensor,
    front_bev_xy: Optional[torch.Tensor],
    front_ground_valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if front_bev_xy is None or not torch.is_tensor(front_bev_xy):
        return torch.zeros_like(sat_image)

    coords = front_bev_xy.detach().cpu().to(torch.float32)
    if coords.ndim == 3 and coords.shape[0] == 2:
        coords_hw = coords.permute(1, 2, 0)
    elif coords.ndim == 3 and coords.shape[-1] == 2:
        coords_hw = coords
    else:
        return torch.zeros_like(sat_image)

    grid = coords_hw.clone()
    grid[..., 1] = -grid[..., 1]
    projected = F.grid_sample(
        sat_image.detach().cpu().to(torch.float32).unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)

    valid_mask = _front_mask_to_hw(front_ground_valid_mask)
    if valid_mask is not None and tuple(valid_mask.shape) == tuple(projected.shape[-2:]):
        projected = projected * valid_mask.to(torch.float32).unsqueeze(0)
    return projected.clamp(0, 1)


def _coords_to_satellite_pixels(
    coords: torch.Tensor,
    sat_width: int,
    sat_height: int,
) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
    if coords.numel() == 0:
        return [], None

    valid = (
        torch.isfinite(coords).all(dim=-1)
        & ~((coords[:, 0].abs() < 1e-6) & (coords[:, 1].abs() < 1e-6))
        & (coords[:, 0] >= -1.0)
        & (coords[:, 0] <= 1.0)
        & (coords[:, 1] >= -1.0)
        & (coords[:, 1] <= 1.0)
    )
    coords = coords[valid]
    if coords.numel() == 0:
        return [], None

    x_px = (coords[:, 0] + 1.0) * 0.5 * float(max(1, sat_width - 1))
    y_px = (1.0 - (coords[:, 1] + 1.0) * 0.5) * float(max(1, sat_height - 1))
    points = list(zip(x_px.tolist(), y_px.tolist()))
    bbox = (
        float(x_px.min().item()),
        float(y_px.min().item()),
        float(x_px.max().item()),
        float(y_px.max().item()),
    )
    return points, bbox


def _front_bev_xy_to_satellite_pixels(
    front_bev_xy: Optional[torch.Tensor],
    sat_width: int,
    sat_height: int,
) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
    if front_bev_xy is None or not torch.is_tensor(front_bev_xy):
        return [], None

    coords = front_bev_xy.detach().cpu().to(torch.float32)
    if coords.ndim == 3 and coords.shape[0] == 2:
        coords = coords.permute(1, 2, 0).reshape(-1, 2)
    elif coords.ndim == 3 and coords.shape[-1] == 2:
        coords = coords.reshape(-1, 2)
    elif coords.ndim == 2 and coords.shape[-1] == 2:
        pass
    else:
        return [], None

    return _coords_to_satellite_pixels(coords, sat_width, sat_height)


def _convex_hull_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) <= 3:
        return points

    unique_points = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique_points) <= 3:
        return unique_points

    def cross(
        origin: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: List[Tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _front_mask_to_hw(front_ground_valid_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if front_ground_valid_mask is None or not torch.is_tensor(front_ground_valid_mask):
        return None

    mask = front_ground_valid_mask.detach().cpu().to(torch.float32)
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0] > 0.5
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask[..., 0] > 0.5
    if mask.ndim == 2:
        return mask > 0.5
    return None


def _front_bev_xy_to_fov_polygon(
    front_bev_xy: Optional[torch.Tensor],
    sat_width: int,
    sat_height: int,
    front_ground_valid_mask: Optional[torch.Tensor] = None,
) -> List[Tuple[float, float]]:
    if front_bev_xy is None or not torch.is_tensor(front_bev_xy):
        return []

    coords = front_bev_xy.detach().cpu().to(torch.float32)
    if coords.ndim == 3 and coords.shape[0] == 2:
        coords_hw = coords.permute(1, 2, 0)
    elif coords.ndim == 3 and coords.shape[-1] == 2:
        coords_hw = coords
    else:
        return []

    height, width = int(coords_hw.shape[0]), int(coords_hw.shape[1])
    if height < 2 or width < 2:
        return []

    valid_mask = _front_mask_to_hw(front_ground_valid_mask)
    if valid_mask is not None and tuple(valid_mask.shape) == (height, width):
        valid_coords = coords_hw[valid_mask]
        points, bbox = _front_bev_xy_to_satellite_pixels(valid_coords, sat_width, sat_height)
        if len(points) >= 3:
            max_hull_points = 6000
            if len(points) > max_hull_points:
                step = max(1, len(points) // max_hull_points)
                points = points[::step]
            hull = _convex_hull_points(points)
            if len(hull) >= 3:
                return hull
        if bbox is not None:
            left_px, top_px, right_px, bottom_px = bbox
            return [
                (left_px, top_px),
                (right_px, top_px),
                (right_px, bottom_px),
                (left_px, bottom_px),
            ]

    top = coords_hw[0, :, :]
    right = coords_hw[:, width - 1, :]
    bottom = torch.flip(coords_hw[height - 1, :, :], dims=[0])
    left = torch.flip(coords_hw[:, 0, :], dims=[0])
    boundary = torch.cat([top, right, bottom, left], dim=0)
    points, _ = _coords_to_satellite_pixels(boundary, sat_width, sat_height)

    if len(points) < 3:
        all_points, bbox = _front_bev_xy_to_satellite_pixels(coords_hw, sat_width, sat_height)
        if bbox is None:
            return []
        left_px, top_px, right_px, bottom_px = bbox
        return [
            (left_px, top_px),
            (right_px, top_px),
            (right_px, bottom_px),
            (left_px, bottom_px),
        ]
    return points


def _draw_satellite_coverage(
    sat_image: torch.Tensor,
    front_bev_xy: Optional[torch.Tensor],
    front_ground_valid_mask: Optional[torch.Tensor],
    view_name: str,
    yaw: Optional[float],
) -> Image.Image:
    image = _tensor_to_pil(sat_image).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    polygon = _front_bev_xy_to_fov_polygon(front_bev_xy, width, height, front_ground_valid_mask)

    # Ego vehicle is at the satellite crop center by construction.
    center = (width / 2.0, height / 2.0)
    cross = 7
    draw.line((center[0] - cross, center[1], center[0] + cross, center[1]), fill=(255, 255, 255, 230), width=3)
    draw.line((center[0], center[1] - cross, center[0], center[1] + cross), fill=(255, 255, 255, 230), width=3)
    draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=(255, 64, 64, 240))

    if polygon:
        draw.polygon(polygon, fill=(0, 180, 255, 45), outline=(255, 230, 0, 235))
        draw.line(polygon + [polygon[0]], fill=(255, 230, 0, 235), width=3)

    label = view_name if yaw is None else f"{view_name} yaw={yaw:g}"
    label_bg = (0, 0, 0, 150)
    draw.rectangle((4, 4, 260, 27), fill=label_bg)
    draw.text((9, 9), label, fill=(255, 255, 255, 255))
    return image


def _build_base_dataset(
    drives,
    frames,
    seed: int,
    *,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> Kitti360dDataset:
    return Kitti360dDataset(
        drives=drives,
        frames=frames,
        mode="fisheye_virtual",
        yaw_mode="vehicle_relative",
        vehicle_relative_yaw_deg=90.0,
        random_vehicle_relative_yaw=False,
        virtual_size=(640, 256),
        front_resize=(640, 256),
        front_center_crop=None,
        random_fisheye_relative_yaw=False,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        seed=seed,
        return_bgr=False,
    )


def _make_override_sample(base_sample: SampleIndex, view_name: str, yaw: Optional[float]) -> SampleIndex:
    if yaw is None:
        meta = {
            "view_name": view_name,
            "mode_override": "front",
        }
    else:
        meta = {
            "view_name": view_name,
            "mode_override": "fisheye_virtual",
            "vehicle_relative_yaw_deg_override": float(yaw),
        }
    return SampleIndex(
        drive_dir=base_sample.drive_dir,
        frame_id=base_sample.frame_id,
        meta=meta,
    )


def _get_view_sample(
    dataset: Kitti360dDataset,
    sample_index: int,
    view_name: str,
    yaw: Optional[float],
) -> Dict:
    original_sample = dataset.samples[sample_index]
    dataset.samples[sample_index] = _make_override_sample(original_sample, view_name, yaw)
    try:
        return dataset[sample_index]
    finally:
        dataset.samples[sample_index] = original_sample


def _resolve_sample_index(
    dataset: Kitti360dDataset,
    frame_id: int,
    drive: Optional[str],
) -> int:
    matches = []
    for idx, sample in enumerate(dataset.samples):
        if sample.frame_id != frame_id:
            continue
        if drive is not None and sample.drive_dir.name != drive:
            continue
        matches.append(idx)

    if not matches:
        drive_suffix = f" in drive {drive}" if drive is not None else ""
        raise ValueError(f"Could not find frame_id={frame_id}{drive_suffix}")
    if len(matches) > 1 and drive is None:
        drives = sorted({dataset.samples[idx].drive_dir.name for idx in matches})
        raise ValueError(f"frame_id={frame_id} appears in multiple drives {drives}; pass --drive")
    return matches[0]


def _filter_sample_indices(
    dataset: Kitti360dDataset,
    start_frame: Optional[int],
    end_frame: Optional[int],
    max_frames: Optional[int],
) -> List[int]:
    indices: List[int] = []
    for idx, sample in enumerate(dataset.samples):
        frame_id = int(sample.frame_id)
        if start_frame is not None and frame_id < start_frame:
            continue
        if end_frame is not None and frame_id > end_frame:
            continue
        indices.append(idx)

    indices.sort(key=lambda i: (str(dataset.samples[i].drive_dir), int(dataset.samples[i].frame_id)))
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    if not indices:
        raise ValueError("No frames selected for inference")
    return indices


def _batched_camera_height(sample: Dict, device: str) -> Optional[torch.Tensor]:
    value = sample.get("camera_height_m")
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor.to(device=device, dtype=torch.float32)


@torch.no_grad()
def _materialize_lazy_modules(
    model,
    sample: Dict,
    device: str,
) -> None:
    sat_images = sample["sat"].unsqueeze(0).to(device)
    target_size = tuple(int(x) for x in sample["image"].shape[-2:])

    K = sample.get("K")
    K = K.unsqueeze(0).to(device) if K is not None else None
    T_cam_to_world = sample.get("T_cam_to_world")
    T_cam_to_world = T_cam_to_world.unsqueeze(0).to(device) if T_cam_to_world is not None else None
    T_imu_to_world = sample.get("T_imu_to_world")
    T_imu_to_world = T_imu_to_world.unsqueeze(0).to(device) if T_imu_to_world is not None else None
    camera_height_m = _batched_camera_height(sample, device)

    sat_state = model.encode_satellite(
        sat_images,
        K=K,
        T_cam_to_world=T_cam_to_world,
        T_imu_to_world=T_imu_to_world,
        camera_height_m=camera_height_m,
        image_size=target_size,
    )

    vae_scale_factor = model._get_vae_scale_factor()
    latent_h = max(1, (target_size[0] + vae_scale_factor - 1) // vae_scale_factor)
    latent_w = max(1, (target_size[1] + vae_scale_factor - 1) // vae_scale_factor)
    unet_param = next(model.unet.parameters(), None)
    latent_dtype = unet_param.dtype if unet_param is not None else sat_state.tokens.dtype
    latents = torch.randn(
        (sat_images.shape[0], model.unet.config.in_channels, latent_h, latent_w),
        device=sat_images.device,
        dtype=latent_dtype,
    )
    timestep = torch.zeros((sat_images.shape[0],), device=sat_images.device, dtype=torch.long)

    amp_dtype = latent_dtype if latent_dtype in {torch.float16, torch.bfloat16} else torch.float32
    cross_attention_kwargs = model._build_cross_attention_kwargs(latents, sat_state)
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=str(device).startswith("cuda") and amp_dtype in {torch.float16, torch.bfloat16},
    ):
        model.unet(
            latents,
            timestep,
            sat_tokens=sat_state.tokens,
            cross_attention_kwargs=cross_attention_kwargs,
        )


def _load_model(args: argparse.Namespace, materialize_sample: Dict):
    model_torch_dtype = None
    if args.device.startswith("cuda") and args.mixed_precision == "fp16":
        model_torch_dtype = torch.float16
    elif args.device.startswith("cuda") and args.mixed_precision == "bf16":
        model_torch_dtype = torch.bfloat16

    logger.info("Loading model")
    model = create_sd_model(
        base_model=args.base_model,
        freeze_base=True,
        revision=args.base_model_revision,
        torch_dtype=model_torch_dtype,
        cond_drop_prob=0.0,
        query_geometry_score_enabled=getattr(args, "query_geometry_score_enabled", False),
        query_geometry_score_dim=getattr(args, "query_geometry_score_dim", 64),
        query_geometry_score_num_freqs=getattr(args, "query_geometry_score_num_freqs", 6),
        query_geometry_score_gate_init=getattr(args, "query_geometry_score_gate_init", 1.0),
        query_geometry_score_layers=getattr(args, "query_geometry_score_layers", None),
        query_geometry_score_max_query_tokens=getattr(args, "query_geometry_score_max_query_tokens", None),
        query_geometry_score_mode=getattr(args, "query_geometry_score_mode", "geometry_first_semantic_refine"),
        query_geometry_candidate_radius=getattr(args, "query_geometry_candidate_radius", 0.35),
        query_geometry_candidate_min_k=getattr(args, "query_geometry_candidate_min_k", 16),
        query_geometry_candidate_invalid_penalty=getattr(args, "query_geometry_candidate_invalid_penalty", -1e4),
        query_semantic_score_dim=getattr(args, "query_semantic_score_dim", 64),
        query_semantic_score_alpha=getattr(args, "query_semantic_score_alpha", 0.25),
        joint_view_generation_enabled=getattr(args, "joint_view_generation_enabled", False),
        joint_view_generation_loss_weight=getattr(args, "joint_view_generation_loss_weight", 0.0),
        joint_view_generation_hidden_dim=getattr(args, "joint_view_generation_hidden_dim", 32),
        joint_view_generation_num_heads=getattr(args, "joint_view_generation_num_heads", 4),
        joint_view_generation_dropout=getattr(args, "joint_view_generation_dropout", 0.0),
        joint_view_generation_bev_sigma=getattr(args, "joint_view_generation_bev_sigma", 0.25),
        joint_view_generation_gate_init=getattr(args, "joint_view_generation_gate_init", 0.0),
        satellite_encoder_config=getattr(args, "satellite_encoder_config", None),
    )
    if hasattr(model.unet, "set_attention_slice"):
        model.unet.set_attention_slice("auto")
    if hasattr(model.vae, "enable_slicing"):
        model.vae.enable_slicing()
    model.to(args.device)
    model.eval()

    logger.info("Materializing lazy condition modules before loading checkpoint")
    _materialize_lazy_modules(model, materialize_sample, args.device)
    if args.checkpoint is None:
        raise ValueError("Pass --checkpoint or set model.checkpoint_path in --config")
    checkpoint_meta = load_model_checkpoint(model, Path(args.checkpoint), args.device)
    mismatches = _checkpoint_inference_mismatches(checkpoint_meta, args)
    if mismatches:
        logger.warning(
            "Checkpoint/inference gate config mismatch detected: %s",
            "; ".join(mismatches),
        )
    model.eval()
    return model, checkpoint_meta


@torch.no_grad()
def _generate_one(
    model,
    sample: Dict,
    args: argparse.Namespace,
    sat_condition_mode: str,
) -> torch.Tensor:
    sat_image = sample["sat"].unsqueeze(0).to(args.device)
    target_size = tuple(int(x) for x in sample["image"].shape[-2:])

    K = sample.get("K")
    K = K.unsqueeze(0).to(args.device) if K is not None else None
    T_cam_to_world = sample.get("T_cam_to_world")
    T_cam_to_world = T_cam_to_world.unsqueeze(0).to(args.device) if T_cam_to_world is not None else None
    T_imu_to_world = sample.get("T_imu_to_world")
    T_imu_to_world = T_imu_to_world.unsqueeze(0).to(args.device) if T_imu_to_world is not None else None
    camera_height_m = _batched_camera_height(sample, args.device)

    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(args.seed))

    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=args.device.startswith("cuda") and amp_dtype in {torch.float16, torch.bfloat16},
    ):
        return model.generate(
            sat_image,
            target_size=target_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            sat_condition_mode=sat_condition_mode,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
        )[0].cpu()


def _camera_height_scalar(sample: Dict) -> torch.Tensor:
    value = sample.get("camera_height_m")
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[0].to(dtype=torch.float32)
    if value is None:
        return torch.tensor(1.6, dtype=torch.float32)
    return torch.tensor(float(value), dtype=torch.float32)


@torch.no_grad()
def _generate_pose_chain(
    model,
    view_samples: Sequence[Dict],
    view_specs: Sequence[Tuple[str, Optional[float]]],
    args: argparse.Namespace,
    sat_condition_mode: str,
) -> List[torch.Tensor]:
    if not hasattr(model, "generate_pose_chain"):
        raise RuntimeError("Model does not expose generate_pose_chain")
    if not view_samples:
        return []

    sat_image = view_samples[0]["sat"].unsqueeze(0).to(args.device)
    target_size = tuple(int(x) for x in view_samples[0]["image"].shape[-2:])
    K = torch.stack([sample["K"] for sample in view_samples], dim=0).unsqueeze(0).to(args.device)
    T_cam_to_world = torch.stack(
        [sample["T_cam_to_world"] for sample in view_samples],
        dim=0,
    ).unsqueeze(0).to(args.device)
    T_imu_to_world = torch.stack(
        [sample["T_imu_to_world"] for sample in view_samples],
        dim=0,
    ).unsqueeze(0).to(args.device)
    camera_height_m = torch.stack(
        [_camera_height_scalar(sample) for sample in view_samples],
        dim=0,
    ).unsqueeze(0).to(args.device)
    vehicle_yaw_degs = torch.tensor(
        [[float("nan") if yaw is None else float(yaw) for _, yaw in view_specs]],
        device=args.device,
        dtype=torch.float32,
    )

    front_bev_xy = None
    if all(torch.is_tensor(sample.get("front_bev_xy")) for sample in view_samples):
        front_bev_xy = torch.stack([sample["front_bev_xy"] for sample in view_samples], dim=0).unsqueeze(0).to(args.device)
    front_ground_valid_mask = None
    if all(torch.is_tensor(sample.get("front_ground_valid_mask")) for sample in view_samples):
        front_ground_valid_mask = torch.stack(
            [sample["front_ground_valid_mask"] for sample in view_samples],
            dim=0,
        ).unsqueeze(0).to(args.device)

    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(args.seed))

    amp_dtype = (
        torch.float16
        if args.mixed_precision == "fp16"
        else torch.bfloat16
        if args.mixed_precision == "bf16"
        else torch.float32
    )
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=args.device.startswith("cuda") and amp_dtype in {torch.float16, torch.bfloat16},
    ):
        generated = model.generate_pose_chain(
            sat_image,
            target_size=target_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            sat_condition_mode=sat_condition_mode,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            camera_height_m=camera_height_m,
            vehicle_yaw_degs=vehicle_yaw_degs,
            front_bev_xy=front_bev_xy,
            front_ground_valid_mask=front_ground_valid_mask,
        )[0].cpu()
    return [generated[index] for index in range(generated.shape[0])]


def _matching_chain_indices(
    view_specs: Sequence[Tuple[str, Optional[float]]],
    chain_specs: Sequence[Tuple[str, Optional[float]]],
) -> Optional[List[int]]:
    indices: List[int] = []
    used: Set[int] = set()
    for _, chain_yaw in chain_specs:
        matched_index = None
        for index, (_, yaw) in enumerate(view_specs):
            if index in used:
                continue
            if chain_yaw is None and yaw is None:
                matched_index = index
                break
            if chain_yaw is not None and yaw is not None and abs(float(chain_yaw) - float(yaw)) < 1e-4:
                matched_index = index
                break
        if matched_index is None:
            return None
        used.add(matched_index)
        indices.append(matched_index)
    return indices


def _generate_views(
    model,
    view_samples: Sequence[Dict],
    view_specs: Sequence[Tuple[str, Optional[float]]],
    args: argparse.Namespace,
    sat_condition_mode: str,
) -> List[torch.Tensor]:
    if args.view_memory_mode != "joint_pose_chain":
        raise ValueError("Only joint_pose_chain inference is supported; single-view fallback has been removed")
    if not bool(getattr(model, "joint_view_generation_enabled", False)):
        raise ValueError("Checkpoint/model was not created with joint_view_generation enabled")

    for chain_specs in JOINT_POSE_CHAIN_SPECS:
        indices = _matching_chain_indices(view_specs, chain_specs)
        if indices is None or len(indices) != len(view_specs):
            continue
        if set(indices) != set(range(len(view_specs))):
            continue
        chain_samples = [view_samples[index] for index in indices]
        chain_view_specs = [view_specs[index] for index in indices]
        chain_outputs = _generate_pose_chain(model, chain_samples, chain_view_specs, args, sat_condition_mode)
        generated: List[Optional[torch.Tensor]] = [None] * len(view_samples)
        for index, output in zip(indices, chain_outputs):
            generated[index] = output
        return [item for item in generated if item is not None]

    requested = ", ".join(
        "front" if yaw is None else f"{float(yaw):g}"
        for _, yaw in view_specs
    )
    raise ValueError(
        "joint_pose_chain inference requires exactly one fixed chain: "
        "front,+60,+90,+120 or front,-60,-90,-120. "
        f"Requested views were: {requested}"
    )


def _save_view_outputs(
    sample: Dict,
    generated: torch.Tensor,
    output_dir: Path,
    view_name: str,
    yaw: Optional[float],
    ablation_mode: Optional[str],
    sat_condition_mode: str,
    gt_override: Optional[torch.Tensor] = None,
) -> Image.Image:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_image = gt_override if gt_override is not None else sample["image"]
    sat_resized = _resize_satellite_for_front(sample["sat"], int(gt_image.shape[-2]))
    sat_projected = _project_satellite_to_perspective(
        sample["sat"],
        sample.get("front_bev_xy"),
        sample.get("front_ground_valid_mask"),
    )
    sat_overlay = _draw_satellite_coverage(
        sample["sat"],
        sample.get("front_bev_xy"),
        sample.get("front_ground_valid_mask"),
        view_name,
        yaw,
    ).resize((sat_resized.shape[-1], sat_resized.shape[-2]), resample=Image.BILINEAR)

    _tensor_to_pil(generated).save(output_dir / "generated.png")
    _tensor_to_pil(gt_image).save(output_dir / "gt.png")
    sat_overlay.save(output_dir / "satellite.png")
    _tensor_to_pil(sat_projected).save(output_dir / "satellite_projected.png")
    comparison = _compose_panels([
        ("sat coverage", torch.from_numpy(np.array(sat_overlay)).permute(2, 0, 1).to(torch.float32) / 255.0),
        ("sat projected", sat_projected),
        (f"gen {view_name}", generated),
        ("gt", gt_image),
    ])
    comparison.save(output_dir / "comparison.png")

    metadata = {
        "drive": str(sample["drive"]),
        "frame_id": int(sample["frame_id"]),
        "view_name": view_name,
        "vehicle_yaw_deg": None if yaw is None else float(yaw),
        "ablation_mode": ablation_mode,
        "sat_condition_mode": sat_condition_mode,
        "gt_override": gt_override is not None,
        "meta": sample.get("meta", {}),
    }
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return comparison


def _resolve_single_dataset(args: argparse.Namespace) -> Tuple[Kitti360dDataset, int]:
    if args.frame_id is None:
        raise ValueError("--frame_id is required for single_yaw_sweep")

    if args.drive_dir is not None:
        dataset = _build_base_dataset(
            Path(args.drive_dir),
            [int(args.frame_id)],
            args.seed,
            pitch_deg=args.pitch_deg,
            roll_deg=args.roll_deg,
        )
        return dataset, 0

    if args.split_yaml is not None:
        train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(
            Path(args.data_dir),
            Path(args.split_yaml),
            eval_split=_eval_split_for_dataset_split(args.dataset_split),
        )
        drives = train_dirs if args.dataset_split == "train" else val_dirs
        frames = train_frames if args.dataset_split == "train" else val_frames
        dataset = _build_base_dataset(
            drives,
            frames,
            args.seed,
            pitch_deg=args.pitch_deg,
            roll_deg=args.roll_deg,
        )
        return dataset, _resolve_sample_index(dataset, int(args.frame_id), args.drive)

    if args.drive is None:
        raise ValueError("Pass --drive_dir, or pass --split_yaml with optional --drive")
    dataset = _build_base_dataset(
        Path(args.data_dir) / args.drive,
        [int(args.frame_id)],
        args.seed,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
    )
    return dataset, 0


def run_single_yaw_sweep(args: argparse.Namespace) -> None:
    dataset, sample_index = _resolve_single_dataset(args)
    view_specs = _single_yaw_sweep_view_specs(args)

    materialize_sample = _get_view_sample(dataset, sample_index, *view_specs[0])
    model, checkpoint_meta = _load_model(args, materialize_sample)

    output_root = Path(args.output_dir)
    base_sample = dataset.samples[sample_index]
    ablation_runs = _resolve_ablation_runs(args)

    for ablation_name, sat_mode in ablation_runs:
        active_output_root = output_root / ablation_name if ablation_name is not None else output_root
        sample_dir = active_output_root / f"{base_sample.drive_dir.name}_frame_{base_sample.frame_id:010d}_yaw_sweep"
        summary_rows: List[Image.Image] = []
        view_samples = [
            _get_view_sample(dataset, sample_index, view_name, yaw)
            for view_name, yaw in view_specs
        ]
        generated_views = _generate_views(model, view_samples, view_specs, args, sat_mode)

        for (view_name, yaw), sample, generated in zip(view_specs, view_samples, generated_views):
            logger.info(
                "Saving frame=%s, view=%s, yaw=%s, ablation=%s",
                f"{base_sample.frame_id:010d}",
                view_name,
                yaw,
                ablation_name or "custom",
            )
            comparison = _save_view_outputs(
                sample,
                generated,
                sample_dir / _view_token(view_name, yaw),
                view_name,
                yaw,
                ablation_name,
                sat_mode,
            )
            summary_rows.append(comparison)

        with open(sample_dir / "run_metadata.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_epoch": _checkpoint_display_epoch(checkpoint_meta),
                    "checkpoint_gate_metadata": _checkpoint_gate_metadata(checkpoint_meta),
                    "inference_gate_config": _inference_gate_config(args),
                    "inference_runtime_config": _inference_runtime_config(
                        args,
                        sat_condition_mode=sat_mode,
                    ),
                    "checkpoint_inference_mismatches": _checkpoint_inference_mismatches(checkpoint_meta, args),
                    "mode": args.mode,
                    "memory_mode": args.view_memory_mode,
                    "ablation_mode": ablation_name,
                    "sat_condition_mode": sat_mode,
                    "virtual_pitch_deg": float(args.pitch_deg),
                    "virtual_roll_deg": float(args.roll_deg),
                    "views": [{"view_name": name, "vehicle_yaw_deg": yaw} for name, yaw in view_specs],
                },
                f,
                sort_keys=False,
            )
        if summary_rows:
            _stack_panel_rows(summary_rows).save(sample_dir / "summary.png")
        logger.info(f"Saved single-frame yaw sweep to: {sample_dir}")


def run_split_fixed_views(args: argparse.Namespace) -> None:
    if args.split_yaml is None:
        raise ValueError("--split_yaml is required for split_fixed_views")

    train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(
        Path(args.data_dir),
        Path(args.split_yaml),
        eval_split=_eval_split_for_dataset_split(args.dataset_split),
    )
    drives = train_dirs if args.dataset_split == "train" else val_dirs
    frames = train_frames if args.dataset_split == "train" else val_frames
    dataset = _build_base_dataset(
        drives,
        frames,
        args.seed,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
    )
    sample_indices = _filter_sample_indices(dataset, args.start_frame, args.end_frame, args.max_frames)

    materialize_sample = _get_view_sample(dataset, sample_indices[0], *FIXED_VIEW_SPECS[0])
    model, checkpoint_meta = _load_model(args, materialize_sample)

    output_root = Path(args.output_dir)
    ablation_runs = _resolve_ablation_runs(args)
    for ablation_name, sat_mode in ablation_runs:
        active_output_root = output_root / ablation_name if ablation_name is not None else output_root
        progress = tqdm(sample_indices, desc=f"Split fixed-view inference [{ablation_name or 'custom'}]")
        for sample_index in progress:
            base_sample = dataset.samples[sample_index]
            frame_dir = active_output_root / base_sample.drive_dir.name / f"frame_{base_sample.frame_id:010d}"
            summary_rows: List[Image.Image] = []
            view_samples = [
                _get_view_sample(dataset, sample_index, view_name, yaw)
                for view_name, yaw in FIXED_VIEW_SPECS
            ]
            generated_views = _generate_views(model, view_samples, view_specs, args, sat_mode)

            for (view_name, yaw), sample, generated in zip(FIXED_VIEW_SPECS, view_samples, generated_views):
                progress.set_postfix(
                    frame=f"{base_sample.frame_id:010d}",
                    view=view_name,
                    ablation=ablation_name or "custom",
                )
                comparison = _save_view_outputs(
                    sample,
                    generated,
                    frame_dir / view_name,
                    view_name,
                    yaw,
                    ablation_name,
                    sat_mode,
                )
                summary_rows.append(comparison)
            if summary_rows:
                _stack_panel_rows(summary_rows).save(frame_dir / "summary.png")

        with open(active_output_root / "run_metadata.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_epoch": _checkpoint_display_epoch(checkpoint_meta),
                    "checkpoint_gate_metadata": _checkpoint_gate_metadata(checkpoint_meta),
                    "inference_gate_config": _inference_gate_config(args),
                    "inference_runtime_config": _inference_runtime_config(
                        args,
                        sat_condition_mode=sat_mode,
                    ),
                    "checkpoint_inference_mismatches": _checkpoint_inference_mismatches(checkpoint_meta, args),
                    "mode": args.mode,
                    "memory_mode": args.view_memory_mode,
                    "dataset_split": args.dataset_split,
                    "split_yaml": str(Path(args.split_yaml)),
                    "start_frame": args.start_frame,
                    "end_frame": args.end_frame,
                    "max_frames": args.max_frames,
                    "num_frames": len(sample_indices),
                    "ablation_mode": ablation_name,
                    "sat_condition_mode": sat_mode,
                    "virtual_pitch_deg": float(args.pitch_deg),
                    "virtual_roll_deg": float(args.roll_deg),
                    "fixed_views": [
                        {"view_name": view_name, "vehicle_yaw_deg": yaw}
                        for view_name, yaw in FIXED_VIEW_SPECS
                    ],
                },
                f,
                sort_keys=False,
            )
        logger.info(f"Saved split fixed-view inference to: {active_output_root}")


def run_split_yaw_sweep(args: argparse.Namespace) -> None:
    if args.split_yaml is None:
        raise ValueError("--split_yaml is required for split_yaw_sweep")

    train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(
        Path(args.data_dir),
        Path(args.split_yaml),
        eval_split=_eval_split_for_dataset_split(args.dataset_split),
    )
    drives = train_dirs if args.dataset_split == "train" else val_dirs
    frames = train_frames if args.dataset_split == "train" else val_frames
    dataset = _build_base_dataset(
        drives,
        frames,
        args.seed,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
    )
    sample_indices = _filter_sample_indices(dataset, args.start_frame, args.end_frame, args.max_frames)
    view_specs = _single_yaw_sweep_view_specs(args)

    materialize_sample = _get_view_sample(dataset, sample_indices[0], *view_specs[0])
    model, checkpoint_meta = _load_model(args, materialize_sample)

    output_root = Path(args.output_dir)
    ablation_runs = _resolve_ablation_runs(args)
    for ablation_name, sat_mode in ablation_runs:
        active_output_root = output_root / ablation_name if ablation_name is not None else output_root
        progress = tqdm(sample_indices, desc=f"Split yaw-sweep inference [{ablation_name or 'custom'}]")
        for sample_index in progress:
            base_sample = dataset.samples[sample_index]
            frame_dir = active_output_root / args.yaw_sweep_preset / base_sample.drive_dir.name / f"frame_{base_sample.frame_id:010d}"
            summary_rows: List[Image.Image] = []
            view_samples = [
                _get_view_sample(dataset, sample_index, view_name, yaw)
                for view_name, yaw in view_specs
            ]
            generated_views = [
                _generate_one(model, sample, args, sat_mode)
                for sample in view_samples
            ]

            for (view_name, yaw), sample, generated in zip(view_specs, view_samples, generated_views):
                progress.set_postfix(
                    frame=f"{base_sample.frame_id:010d}",
                    view=view_name,
                    preset=args.yaw_sweep_preset,
                    ablation=ablation_name or "custom",
                )
                comparison = _save_view_outputs(
                    sample,
                    generated,
                    frame_dir / _view_token(view_name, yaw),
                    view_name,
                    yaw,
                    ablation_name,
                    sat_mode,
                )
                summary_rows.append(comparison)
            if summary_rows:
                _stack_panel_rows(summary_rows).save(frame_dir / "summary.png")

        active_output_root.mkdir(parents=True, exist_ok=True)
        with open(active_output_root / f"run_metadata_{args.yaw_sweep_preset}.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_epoch": _checkpoint_display_epoch(checkpoint_meta),
                    "checkpoint_gate_metadata": _checkpoint_gate_metadata(checkpoint_meta),
                    "inference_gate_config": _inference_gate_config(args),
                    "inference_runtime_config": _inference_runtime_config(
                        args,
                        sat_condition_mode=sat_mode,
                    ),
                    "checkpoint_inference_mismatches": _checkpoint_inference_mismatches(checkpoint_meta, args),
                    "mode": args.mode,
                    "memory_mode": args.view_memory_mode,
                    "dataset_split": args.dataset_split,
                    "split_yaml": str(Path(args.split_yaml)),
                    "yaw_sweep_preset": args.yaw_sweep_preset,
                    "include_front": bool(args.include_front),
                    "start_frame": args.start_frame,
                    "end_frame": args.end_frame,
                    "max_frames": args.max_frames,
                    "num_frames": len(sample_indices),
                    "ablation_mode": ablation_name,
                    "sat_condition_mode": sat_mode,
                    "virtual_pitch_deg": float(args.pitch_deg),
                    "virtual_roll_deg": float(args.roll_deg),
                    "views": [
                        {"view_name": view_name, "vehicle_yaw_deg": yaw}
                        for view_name, yaw in view_specs
                    ],
                },
                f,
                sort_keys=False,
            )
        logger.info(f"Saved split yaw-sweep inference to: {active_output_root}")


def run_front_pitch_sweep(args: argparse.Namespace) -> None:
    dataset, sample_index = _resolve_single_dataset(args)
    pitch_values = args.pitch_values if args.pitch_values is not None else [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    pitch_values = [float(value) for value in pitch_values]

    front_sample = _get_view_sample(dataset, sample_index, "front", None)
    original_pitch = float(dataset.pitch_deg)
    original_roll = float(dataset.roll_deg)
    dataset.pitch_deg = pitch_values[0]
    dataset.roll_deg = float(args.roll_deg)
    materialize_sample = _get_view_sample(dataset, sample_index, _view_token("pitch", pitch_values[0]), 0.0)
    model, checkpoint_meta = _load_model(args, materialize_sample)

    output_root = Path(args.output_dir)
    base_sample = dataset.samples[sample_index]
    ablation_runs = _resolve_ablation_runs(args)

    try:
        for ablation_name, sat_mode in ablation_runs:
            active_output_root = output_root / ablation_name if ablation_name is not None else output_root
            sample_dir = active_output_root / f"{base_sample.drive_dir.name}_frame_{base_sample.frame_id:010d}_front_pitch_sweep"
            summary_rows: List[Image.Image] = []

            for pitch in pitch_values:
                dataset.pitch_deg = float(pitch)
                dataset.roll_deg = float(args.roll_deg)
                view_name = _view_token("pitch", pitch)
                sample = _get_view_sample(dataset, sample_index, view_name, 0.0)
                generated = _generate_one(model, sample, args, sat_mode)
                comparison = _save_view_outputs(
                    sample,
                    generated,
                    sample_dir / view_name,
                    view_name,
                    0.0,
                    ablation_name,
                    sat_mode,
                    gt_override=front_sample["image"],
                )
                summary_rows.append(comparison)

            with open(sample_dir / "run_metadata.yaml", "w") as f:
                yaml.safe_dump(
                    {
                        "checkpoint": str(Path(args.checkpoint).resolve()),
                        "checkpoint_epoch": _checkpoint_display_epoch(checkpoint_meta),
                        "checkpoint_gate_metadata": _checkpoint_gate_metadata(checkpoint_meta),
                        "inference_gate_config": _inference_gate_config(args),
                        "inference_runtime_config": _inference_runtime_config(
                            args,
                            sat_condition_mode=sat_mode,
                        ),
                        "checkpoint_inference_mismatches": _checkpoint_inference_mismatches(checkpoint_meta, args),
                        "mode": args.mode,
                        "memory_mode": args.view_memory_mode,
                        "ablation_mode": ablation_name,
                        "sat_condition_mode": sat_mode,
                        "vehicle_yaw_deg": 0.0,
                        "fixed_gt": "front",
                        "virtual_roll_deg": float(args.roll_deg),
                        "pitch_values": pitch_values,
                    },
                    f,
                    sort_keys=False,
                )
            if summary_rows:
                _stack_panel_rows(summary_rows).save(sample_dir / "summary.png")
            logger.info(f"Saved front pitch sweep to: {sample_dir}")
    finally:
        dataset.pitch_deg = original_pitch
        dataset.roll_deg = original_roll


def main() -> None:
    args = _parse_args()
    if args.checkpoint is None:
        raise ValueError("Pass --checkpoint or set model.checkpoint_path in --config")
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    logger.info(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    logger.info(f"HF_HOME={os.environ['HF_HOME']}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
        args.mixed_precision = "no"

    if args.mode == "single_yaw_sweep":
        run_single_yaw_sweep(args)
    elif args.mode == "split_yaw_sweep":
        run_split_yaw_sweep(args)
    elif args.mode == "split_fixed_views":
        run_split_fixed_views(args)
    elif args.mode == "front_pitch_sweep":
        run_front_pitch_sweep(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
