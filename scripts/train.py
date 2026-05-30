#!/usr/bin/env python3
"""
Training script for satellite-to-frontview generation using Stable Diffusion.

This script uses the simplified trainer interface.
"""

import sys
from pathlib import Path
from datetime import timedelta

# Add project root to Python path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import os
import torch
import torch.distributed as dist
import numpy as np
import random
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple

from models.sd_model import create_sd_model, load_model_checkpoint
from models.sd_trainer import SDTrainer
from data import Kitti360dDataset
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)

DEFAULT_SD21_BASE_REPO = "sd2-community/stable-diffusion-2-1-base"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_HOME = _project_root / ".hf-home"
DEFAULT_DISTRIBUTED_TIMEOUT = timedelta(hours=12)


def _worker_init_fn(_worker_id: int) -> None:
    try:
        import cv2  # type: ignore

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def _init_distributed(args) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if not distributed:
        return False, rank, local_rank, world_size

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA training requested, but CUDA is not available")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    dist.init_process_group(
        backend="nccl" if args.device.startswith("cuda") else "gloo",
        timeout=DEFAULT_DISTRIBUTED_TIMEOUT,
    )
    return True, rank, local_rank, world_size


def _cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _load_frame_ids(frames_file: Path) -> List[int]:
    frame_ids: List[int] = []
    for line in frames_file.read_text().splitlines():
        token = line.strip()
        if not token:
            continue
        frame_ids.append(int(token))
    return frame_ids


def _load_split_from_yaml(data_dir: Path, split_yaml: Path) -> Tuple[List[Path], List[List[int]], List[Path], List[List[int]]]:
    if not split_yaml.exists():
        raise FileNotFoundError(f"Split yaml not found: {split_yaml}")

    with open(split_yaml, "r") as f:
        split_cfg = yaml.safe_load(f)

    if not isinstance(split_cfg, dict):
        raise ValueError(f"Invalid split yaml format: {split_yaml}")

    train_entries = split_cfg.get("train")
    val_entries = split_cfg.get("val", split_cfg.get("test"))
    if not isinstance(train_entries, list) or not isinstance(val_entries, list):
        raise ValueError("Split yaml must contain list entries for 'train' and 'val' or 'test'")

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
            if len(frame_ids) == 0:
                raise ValueError(f"No frame ids found in {frames_file}")

            drives.append(drive_dir)
            frames_per_drive.append(frame_ids)

        return drives, frames_per_drive

    train_dirs, train_frames = parse_entries(train_entries, "train")
    val_dirs, val_frames = parse_entries(val_entries, "val/test")
    return train_dirs, train_frames, val_dirs, val_frames


def _safe_collate(batch):
    """Keep non-tensor metadata as raw lists so default_collate doesn't choke on None."""
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in {"meta", "view_names", "view_metas"} or any(value is None for value in values):
            collated[key] = values
            continue
        collated[key] = default_collate(values)
    return collated


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


def _collect_cli_options(argv: List[str]) -> set[str]:
    options: set[str] = set()
    for arg in argv:
        if arg.startswith("--"):
            options.add(arg.split("=", 1)[0])
    return options


def _prefer_config(
    current: Any,
    cli_default: Any,
    config_value: Any,
    *,
    cli_option: str | None = None,
    cli_options: set[str] | None = None,
) -> Any:
    if cli_option is not None and cli_options is not None and cli_option in cli_options:
        return current
    if current == cli_default and config_value is not None:
        return config_value
    return current


def _config_bool(
    current: bool,
    cli_default: bool,
    config_value: Any,
) -> bool:
    if current == cli_default and config_value is not None:
        return bool(config_value)
    return bool(current)


def _resolve_output_dir(
    current: str,
    cli_default: str,
    config: Dict[str, Any],
) -> str:
    if current != cli_default:
        return current

    configured_output_dir = _config_get(config, ("output_dir",))
    if configured_output_dir:
        return str(configured_output_dir)

    checkpoint_save_dir = _config_get(config, ("checkpoint", "save_dir"))
    if checkpoint_save_dir:
        checkpoint_path = Path(str(checkpoint_save_dir))
        if checkpoint_path.name == "checkpoints":
            parent = checkpoint_path.parent
            return str(parent if str(parent) else Path("."))
        return str(checkpoint_path)

    return current


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _resolve_query_uv_config(config: Dict[str, Any]) -> Tuple[bool, float]:
    del config
    return False, 0.0


def _resolve_query_geometry_bias_config(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    del config
    return False, 0.0, 0.0


def _resolve_query_geometry_score_config(config: Dict[str, Any]) -> Dict[str, Any]:
    score_config = dict(_config_get(config, ("model", "query_geometry_score"), {}) or {})
    layers = score_config.get("layers")
    if layers is not None:
        layers = [str(layer) for layer in layers]
    max_query_tokens = score_config.get("max_query_tokens", None)
    if max_query_tokens is not None:
        max_query_tokens = int(max_query_tokens)
    return {
        "enabled": bool(score_config.get("enable", False)),
        "dim": int(score_config.get("dim", 64) or 64),
        "num_freqs": int(score_config.get("num_freqs", 6) or 6),
        "gate_init": float(score_config.get("gate_init", 1.0) or 1.0),
        "layers": layers,
        "max_query_tokens": max_query_tokens,
        "mode": str(score_config.get("mode", "geometry_first_semantic_refine")),
        "candidate_radius": float(score_config.get("candidate_radius", 0.35) or 0.35),
        "candidate_min_k": int(score_config.get("candidate_min_k", 16) or 16),
        "candidate_invalid_penalty": float(score_config.get("candidate_invalid_penalty", -1e4) or -1e4),
        "semantic_score_dim": int(score_config.get("semantic_score_dim", 64) or 64),
        "semantic_alpha_max": float(score_config.get("semantic_alpha_max", 0.25) or 0.25),
    }


def _attach_query_geometry_score_args(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    score_config = _resolve_query_geometry_score_config(config)
    args.query_geometry_score_enabled = score_config["enabled"]
    args.query_geometry_score_dim = score_config["dim"]
    args.query_geometry_score_num_freqs = score_config["num_freqs"]
    args.query_geometry_score_gate_init = score_config["gate_init"]
    args.query_geometry_score_layers = score_config["layers"]
    args.query_geometry_score_max_query_tokens = score_config["max_query_tokens"]
    args.query_geometry_score_mode = score_config["mode"]
    args.query_geometry_candidate_radius = score_config["candidate_radius"]
    args.query_geometry_candidate_min_k = score_config["candidate_min_k"]
    args.query_geometry_candidate_invalid_penalty = score_config["candidate_invalid_penalty"]
    args.query_semantic_score_dim = score_config["semantic_score_dim"]
    args.query_semantic_score_alpha = score_config["semantic_alpha_max"]
    return score_config


def _verify_query_geometry_score_model_config(model: Any, score_config: Dict[str, Any]) -> None:
    if not bool(score_config.get("enabled", False)):
        return

    unet = getattr(model, "unet", None)
    if unet is None:
        raise RuntimeError("query_geometry_score is enabled, but the created model has no UNet")

    expected_layers = score_config.get("layers")
    actual_layers = getattr(unet, "query_geometry_score_layers", None)
    actual_layers_list = None if actual_layers is None else [str(layer) for layer in actual_layers]
    expected_layers_list = None if expected_layers is None else [str(layer) for layer in expected_layers]
    checks = {
        "query_geometry_score_enabled": bool(score_config["enabled"]),
        "query_geometry_score_dim": int(score_config["dim"]),
        "query_geometry_score_num_freqs": int(score_config["num_freqs"]),
        "query_geometry_score_gate_init": float(score_config["gate_init"]),
        "query_geometry_score_layers": expected_layers_list,
        "query_geometry_score_max_query_tokens": score_config["max_query_tokens"],
        "query_geometry_score_mode": str(score_config.get("mode", "geometry_first_semantic_refine")),
        "query_geometry_candidate_radius": float(score_config.get("candidate_radius", 0.35)),
        "query_geometry_candidate_min_k": int(score_config.get("candidate_min_k", 16)),
        "query_geometry_candidate_invalid_penalty": float(score_config.get("candidate_invalid_penalty", -1e4)),
        "query_semantic_score_dim": int(score_config.get("semantic_score_dim", 64)),
        "query_semantic_score_alpha": float(score_config.get("semantic_alpha_max", 0.25)),
    }
    actual = {
        "query_geometry_score_enabled": bool(getattr(unet, "query_geometry_score_enabled", False)),
        "query_geometry_score_dim": int(getattr(unet, "query_geometry_score_dim", -1)),
        "query_geometry_score_num_freqs": int(getattr(unet, "query_geometry_score_num_freqs", -1)),
        "query_geometry_score_gate_init": float(getattr(unet, "query_geometry_score_gate_init", float("nan"))),
        "query_geometry_score_layers": actual_layers_list,
        "query_geometry_score_max_query_tokens": getattr(unet, "query_geometry_score_max_query_tokens", None),
        "query_geometry_score_mode": str(getattr(unet, "query_geometry_score_mode", "")),
        "query_geometry_candidate_radius": float(getattr(unet, "query_geometry_candidate_radius", float("nan"))),
        "query_geometry_candidate_min_k": int(getattr(unet, "query_geometry_candidate_min_k", -1)),
        "query_geometry_candidate_invalid_penalty": float(getattr(unet, "query_geometry_candidate_invalid_penalty", float("nan"))),
        "query_semantic_score_dim": int(getattr(unet, "query_semantic_score_dim", -1)),
        "query_semantic_score_alpha": float(getattr(unet, "query_semantic_score_alpha", float("nan"))),
    }
    mismatches = {
        name: (expected, actual[name])
        for name, expected in checks.items()
        if actual[name] != expected
    }
    if mismatches:
        raise RuntimeError(
            "query_geometry_score config was not applied to the created UNet: "
            + ", ".join(
                f"{name} expected {expected!r}, got {got!r}"
                for name, (expected, got) in mismatches.items()
            )
        )


def _resolve_attention_alignment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    alignment_config = dict(_config_get(config, ("training", "attention_alignment"), {}) or {})
    layers = alignment_config.get("layers")
    if layers is not None:
        layers = [str(layer) for layer in layers]

    max_query_tokens = alignment_config.get("max_query_tokens", 256)
    if max_query_tokens is not None:
        max_query_tokens = int(max_query_tokens)

    return {
        "enabled": bool(alignment_config.get("enable", False)),
        "loss_weight": float(alignment_config.get("loss_weight", 0.0) or 0.0),
        "layers": layers,
        "max_query_tokens": max_query_tokens,
        "valid_radius": float(alignment_config.get("valid_radius", 0.25) or 0.25),
        "invalid_attention_weight": float(
            alignment_config.get("invalid_attention_weight", 0.1) or 0.0
        ),
    }


def _resolve_transition_aux_config(config: Dict[str, Any]) -> Dict[str, Any]:
    aux_config = dict(_config_get(config, ("training", "transition_aux"), {}) or {})

    def _value(name: str, default: Any) -> Any:
        value = aux_config.get(name, default)
        return default if value is None else value

    source = str(_value("source", _value("mode", "gt_latent"))).strip().lower().replace("-", "_")
    if source in {"gt", "gt_latents", "latent", "vae_latent"}:
        source = "gt_latent"
    elif source in {"x0", "pred_x0", "pred_original_sample"}:
        source = "predicted_x0"
    if source not in {"gt_latent", "predicted_x0"}:
        raise ValueError(f"training.transition_aux.source must be 'gt_latent' or 'predicted_x0', got {source!r}")

    return {
        "enabled": bool(aux_config.get("enable", False)),
        "source": source,
        "loss_weight": float(_value("loss_weight", 0.0)),
        "warmup_steps": int(_value("warmup_steps", 0)),
        "cycle_weight": float(_value("cycle_weight", 0.1)),
        "composition_weight": float(_value("composition_weight", 0.05)),
        "mse_weight": float(_value("mse_weight", 0.1)),
        "hidden_channels": int(_value("hidden_channels", 128)),
        "action_dim": int(_value("action_dim", 128)),
        "lr_multiplier": float(_value("lr_multiplier", 1.0)),
    }


def _resolve_joint_view_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    joint_config = dict(_config_get(config, ("training", "joint_view_generation"), {}) or {})

    def _value(name: str, default: Any) -> Any:
        value = joint_config.get(name, default)
        return default if value is None else value

    return {
        "enabled": bool(joint_config.get("enable", False)),
        "loss_weight": float(_value("loss_weight", 0.0)),
        "hidden_dim": int(_value("hidden_dim", 32)),
        "num_heads": int(_value("num_heads", 4)),
        "dropout": float(_value("dropout", 0.0)),
        "bev_sigma": float(_value("bev_sigma", 0.25)),
        "gate_init": float(_value("gate_init", 0.0)),
    }


def _resolve_geometry_score_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    geometry_config = dict(_config_get(config, ("training", "geometry_score"), {}) or {})
    def _value(name: str, default: Any) -> Any:
        value = geometry_config.get(name, default)
        return default if value is None else value

    return {
        "lr_multiplier": float(_value("lr_multiplier", 1.0)),
        "gate_warmup_steps": int(_value("gate_warmup_steps", 0)),
        "gate_warmup_start_scale": float(_value("gate_warmup_start_scale", 1.0)),
        "gate_warmup_end_scale": float(_value("gate_warmup_end_scale", 1.0)),
        "semantic_alpha_hold_steps": int(_value("semantic_alpha_hold_steps", 1000)),
        "semantic_alpha_warmup_steps": int(_value("semantic_alpha_warmup_steps", 2000)),
    }


def _resolve_unet_attention_slicing_config(config: Dict[str, Any]) -> bool:
    return bool(_config_get(config, ("attention_slicing",), False))


def _resolve_gradient_checkpointing_config(config: Dict[str, Any], cli_value: bool | None) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    return bool(_config_get(config, ("gradient_checkpointing",), True))


def _infer_pose_chain_group_size(view_set: str, pose_chains: Any) -> int:
    if str(view_set) != "pose_chain":
        return 1
    if not isinstance(pose_chains, list) or not pose_chains:
        return 4
    lengths: List[int] = []
    seen_names: set[str] = set()
    for chain_index, chain in enumerate(pose_chains):
        if isinstance(chain, dict):
            name = str(chain.get("name", f"chain_{chain_index}"))
            yaw_values = chain.get("yaws", chain.get("views"))
        else:
            name = f"chain_{chain_index}"
            yaw_values = chain
        if name in seen_names:
            raise ValueError(f"pose chain names must be unique, got duplicate '{name}'")
        seen_names.add(name)
        if yaw_values is None:
            raise ValueError(f"pose chain {chain_index} must define yaws/views")
        if isinstance(yaw_values, (str, bytes)):
            raise ValueError(f"pose chain {chain_index} yaws/views must be a sequence, not a string")
        lengths.append(len(list(yaw_values)))
    if len(set(lengths)) > 1:
        raise ValueError(
            "all pose chains must contain the same number of views for "
            f"batched pose-chain training, got lengths={lengths}"
        )
    return max(1, lengths[0])


def main():
    parser = argparse.ArgumentParser(
        description="Train Stable Diffusion for satellite-to-frontview generation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/media/user/574b4a05-57d2-424d-bb82-763098cbf0a4/shizhm/KITTI-360",
        help="Path to KITTI-360 data",
    )
    parser.add_argument(
        "--split_yaml", type=str, default=None,
        help="Path to split yaml (defaults to <data_dir>/train_test_split_config.yaml; val may be named test)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Learning rate warmup epochs",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--gradient_accumulation", type=int, default=2,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="Gradient clipping threshold. Set <=0 to disable.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable UNet gradient checkpointing. Defaults to the config value. "
            "Use --no-gradient_checkpointing for auxiliary attention-alignment loss runs."
        ),
    )
    parser.add_argument(
        "--base_model", type=str, default=DEFAULT_SD21_BASE_REPO,
        help="Base diffusers model repo id or local path",
    )
    parser.add_argument(
        "--base_model_revision", type=str, default=None,
        help="Optional model revision/branch, e.g. fp16",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode for training",
    )
    parser.add_argument(
        "--cond_drop_prob", type=float, default=0.1,
        help="Probability of dropping satellite conditioning during training.",
    )
    parser.add_argument(
        "--init_checkpoint", type=str, default=None,
        help="Optional checkpoint used to initialize model weights before training.",
    )
    parser.add_argument(
        "--dataset_mode", type=str, default="front",
        choices=["front", "fisheye_virtual"],
        help="Dataset view mode used for training/validation.",
    )
    parser.add_argument(
        "--yaw_mode", type=str, default="fisheye_relative",
        choices=["fisheye_relative", "vehicle_relative"],
        help="Yaw semantics for virtual fisheye views.",
    )
    parser.add_argument(
        "--view_set", type=str, default="single",
        choices=["single", "pose_chain"],
        help=(
            "single trains one target yaw per sample; pose_chain trains ordered "
            "overlapping yaw chains per frame."
        ),
    )
    parser.add_argument(
        "--vehicle_yaw_min_deg", type=float, default=60.0,
        help="Minimum absolute vehicle-relative yaw sampled for virtual-view training; random training samples from [-max,-min] U [min,max].",
    )
    parser.add_argument(
        "--vehicle_yaw_max_deg", type=float, default=120.0,
        help="Maximum absolute vehicle-relative yaw sampled for virtual-view training; random training samples from [-max,-min] U [min,max].",
    )
    parser.add_argument(
        "--vehicle_yaw_sampling",
        type=str,
        default="random_range",
        choices=["random_range", "fixed_list"],
        help="Vehicle-relative yaw sampler for fisheye_virtual training.",
    )
    parser.add_argument(
        "--vehicle_yaw_fixed_list",
        type=str,
        nargs="+",
        default=None,
        help="Fixed yaw list for vehicle_yaw_sampling=fixed_list. Use 'front' for image_00.",
    )
    parser.add_argument(
        "--front_sample_prob", type=float, default=0.0,
        help="Probability that a training item uses the real image_00 front view instead of a random virtual fisheye yaw.",
    )
    parser.add_argument(
        "--pitch_deg",
        type=float,
        default=0.0,
        help="Virtual camera pitch in degrees for fisheye remap and BEV projection.",
    )
    parser.add_argument(
        "--roll_deg",
        type=float,
        default=0.0,
        help="Virtual camera roll in degrees for fisheye remap and BEV projection.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.0,
        help="Guidance scale used for training visualizations. 1.0 disables CFG.",
    )
    parser.add_argument(
        "--visualize_every", type=int, default=10,
        help="Save fixed-sample visualization comparisons every N epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--validate_every", type=int, default=1,
        help="Run validation every N epochs and always on the final epoch. Set 0 for final epoch only.",
    )
    parser.add_argument(
        "--num_visualizations", type=int, default=4,
        help="Number of fixed validation samples to visualize per save.",
    )
    parser.add_argument(
        "--visualization_inference_steps", type=int, default=20,
        help="Number of denoising steps used for visualization generation.",
    )
    parser.add_argument(
        "--visualization_seed", type=int, default=42,
        help="Random seed for reproducible visualization generations.",
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Log training metrics and visualization images to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="kitti360_sd",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="Optional Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--wandb_mode", type=str, default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    tensorboard_group = parser.add_mutually_exclusive_group()
    tensorboard_group.add_argument(
        "--use_tensorboard", dest="use_tensorboard", action="store_true",
        help="Log training metrics and visualization images to TensorBoard (default: enabled).",
    )
    tensorboard_group.add_argument(
        "--no_tensorboard", dest="use_tensorboard", action="store_false",
        help="Disable TensorBoard logging.",
    )
    parser.set_defaults(use_tensorboard=True)
    parser.add_argument(
        "--tensorboard_log_dir", type=str, default=None,
        help="TensorBoard log directory. Defaults to <output_dir>/tensorboard.",
    )
    parser.add_argument(
        "--hf_endpoint", type=str, default=DEFAULT_HF_ENDPOINT,
        help="Hugging Face endpoint. Defaults to hf-mirror for first-time downloads.",
    )
    parser.add_argument(
        "--hf_home", type=str, default=str(DEFAULT_HF_HOME),
        help="Local Hugging Face cache directory.",
    )

    args = parser.parse_args()
    cli_options = _collect_cli_options(sys.argv[1:])
    config = _load_runtime_config(Path(args.config))

    args.seed = int(_prefer_config(args.seed, 42, _config_get(config, ("seed",))))
    args.device = str(_prefer_config(args.device, "cuda", _config_get(config, ("device",))))
    args.mixed_precision = str(
        _prefer_config(args.mixed_precision, "fp16", _config_get(config, ("mixed_precision",)))
    )
    args.data_dir = str(
        _prefer_config(
            args.data_dir,
            "/media/user/574b4a05-57d2-424d-bb82-763098cbf0a4/shizhm/KITTI-360",
            _config_get(config, ("data", "data_dir")),
        )
    )
    args.output_dir = _resolve_output_dir(args.output_dir, "./output", config)
    args.batch_size = int(_prefer_config(args.batch_size, 2, _config_get(config, ("data", "batch_size"))))
    args.epochs = int(_prefer_config(args.epochs, 50, _config_get(config, ("training", "epochs"))))
    args.lr = float(_prefer_config(args.lr, 1e-4, _config_get(config, ("training", "learning_rate"))))
    args.warmup = int(_prefer_config(args.warmup, 5, _config_get(config, ("training", "warmup_epochs"))))
    args.resume = args.resume or _config_get(config, ("checkpoint", "resume_from"))
    args.num_workers = int(_prefer_config(args.num_workers, 8, _config_get(config, ("data", "num_workers"))))
    args.gradient_accumulation = int(
        _prefer_config(
            args.gradient_accumulation,
            2,
            _config_get(config, ("training", "gradient_accumulation_steps")),
            cli_option="--gradient_accumulation",
            cli_options=cli_options,
        )
    )
    args.max_grad_norm = float(
        _prefer_config(args.max_grad_norm, 1.0, _config_get(config, ("training", "gradient_clip_val")))
    )
    args.base_model = str(
        _prefer_config(args.base_model, DEFAULT_SD21_BASE_REPO, _config_get(config, ("model", "base_model")))
    )
    args.base_model_revision = _prefer_config(
        args.base_model_revision,
        None,
        _config_get(config, ("model", "base_model_revision")),
    )
    args.cond_drop_prob = float(
        _prefer_config(args.cond_drop_prob, 0.1, _config_get(config, ("training", "cond_drop_prob")))
    )
    args.init_checkpoint = _prefer_config(
        args.init_checkpoint,
        None,
        _config_get(config, ("checkpoint", "init_checkpoint")),
    )
    args.use_wandb = _config_bool(args.use_wandb, False, _config_get(config, ("logging", "use_wandb")))
    args.wandb_project = str(
        _prefer_config(args.wandb_project, "kitti360_sd", _config_get(config, ("logging", "project_name")))
    )
    args.wandb_run_name = _prefer_config(args.wandb_run_name, None, _config_get(config, ("logging", "run_name")))
    args.wandb_entity = _prefer_config(args.wandb_entity, None, _config_get(config, ("logging", "entity")))
    args.wandb_mode = str(_prefer_config(args.wandb_mode, "online", _config_get(config, ("logging", "wandb_mode"))))
    args.use_tensorboard = _config_bool(
        args.use_tensorboard,
        True,
        _config_get(config, ("logging", "use_tensorboard")),
    )
    args.tensorboard_log_dir = _prefer_config(
        args.tensorboard_log_dir,
        None,
        _config_get(config, ("logging", "tensorboard_log_dir")),
    )
    args.dataset_mode = str(_prefer_config(args.dataset_mode, "front", _config_get(config, ("data", "mode"))))
    args.yaw_mode = str(_prefer_config(args.yaw_mode, "fisheye_relative", _config_get(config, ("data", "yaw_mode"))))
    args.view_set = str(_prefer_config(args.view_set, "single", _config_get(config, ("data", "view_set"))))
    args.pose_chains = _config_get(config, ("data", "pose_chains"))
    args.vehicle_yaw_min_deg = float(
        _prefer_config(
            args.vehicle_yaw_min_deg,
            60.0,
            _config_get(config, ("data", "vehicle_yaw_min_deg")),
        )
    )
    args.vehicle_yaw_max_deg = float(
        _prefer_config(
            args.vehicle_yaw_max_deg,
            120.0,
            _config_get(config, ("data", "vehicle_yaw_max_deg")),
        )
    )
    args.vehicle_yaw_sampling = str(
        _prefer_config(
            args.vehicle_yaw_sampling,
            "random_range",
            _config_get(config, ("data", "vehicle_yaw_sampling")),
        )
    )
    args.vehicle_yaw_fixed_list = _prefer_config(
        args.vehicle_yaw_fixed_list,
        None,
        _config_get(config, ("data", "vehicle_yaw_fixed_list")),
    )
    args.front_sample_prob = float(
        _prefer_config(
            args.front_sample_prob,
            0.0,
            _config_get(config, ("data", "front_sample_prob")),
        )
    )
    args.pitch_deg = float(_prefer_config(args.pitch_deg, 0.0, _config_get(config, ("data", "pitch_deg"))))
    args.roll_deg = float(_prefer_config(args.roll_deg, 0.0, _config_get(config, ("data", "roll_deg"))))
    args.guidance_scale = float(
        _prefer_config(args.guidance_scale, 3.0, _config_get(config, ("validation", "guidance_scale")))
    )
    args.visualize_every = int(
        _prefer_config(args.visualize_every, 10, _config_get(config, ("validation", "visualize_every")))
    )
    args.validate_every = int(
        _prefer_config(
            args.validate_every,
            1,
            _config_get(config, ("validation", "validate_every")),
            cli_option="--validate_every",
            cli_options=cli_options,
        )
    )
    args.num_visualizations = int(
        _prefer_config(
            args.num_visualizations,
            4,
            _config_get(config, ("validation", "num_validation_samples")),
        )
    )
    args.visualization_inference_steps = int(
        _prefer_config(
            args.visualization_inference_steps,
            20,
            _config_get(config, ("validation", "visualization_inference_steps")),
        )
    )
    args.visualization_seed = int(
        _prefer_config(args.visualization_seed, 42, _config_get(config, ("validation", "visualization_seed")))
    )

    freeze_base = bool(_config_get(config, ("model", "freeze_base"), True))
    gradient_checkpointing = _resolve_gradient_checkpointing_config(config, args.gradient_checkpointing)
    args.gradient_checkpointing = gradient_checkpointing
    unet_attention_slicing = _resolve_unet_attention_slicing_config(config)
    weight_decay = float(_config_get(config, ("training", "weight_decay"), 1e-4))
    lr_scheduler_type = str(_config_get(config, ("training", "scheduler"), "cosine"))
    save_every = int(_config_get(config, ("checkpoint", "save_every"), 50))
    log_every = int(_config_get(config, ("logging", "log_every"), 100))
    front_resize_cfg = _config_get(config, ("data", "front_resize"), [640, 256])
    front_resize = tuple(int(x) for x in front_resize_cfg)
    log_dir = Path(str(_config_get(config, ("logging", "log_dir"), args.output_dir)))
    satellite_encoder_config = dict(_config_get(config, ("model", "satellite_encoder"), {}) or {})
    query_geometry_score_config = _attach_query_geometry_score_args(args, config)
    attention_alignment_config = _resolve_attention_alignment_config(config)
    transition_aux_config = _resolve_transition_aux_config(config)
    joint_view_generation_config = _resolve_joint_view_generation_config(config)
    geometry_score_training_config = _resolve_geometry_score_training_config(config)
    args.attention_alignment_enabled = attention_alignment_config["enabled"]
    args.attention_alignment_loss_weight = attention_alignment_config["loss_weight"]
    args.attention_alignment_layers = attention_alignment_config["layers"]
    args.attention_alignment_max_query_tokens = attention_alignment_config["max_query_tokens"]
    args.attention_alignment_valid_radius = attention_alignment_config["valid_radius"]
    args.attention_alignment_invalid_attention_weight = attention_alignment_config["invalid_attention_weight"]
    args.transition_aux_enabled = transition_aux_config["enabled"]
    args.transition_aux_source = transition_aux_config["source"]
    args.transition_aux_loss_weight = transition_aux_config["loss_weight"]
    args.transition_aux_warmup_steps = transition_aux_config["warmup_steps"]
    args.transition_aux_cycle_weight = transition_aux_config["cycle_weight"]
    args.transition_aux_composition_weight = transition_aux_config["composition_weight"]
    args.transition_aux_mse_weight = transition_aux_config["mse_weight"]
    args.transition_aux_hidden_channels = transition_aux_config["hidden_channels"]
    args.transition_aux_action_dim = transition_aux_config["action_dim"]
    args.transition_aux_lr_multiplier = transition_aux_config["lr_multiplier"]
    args.joint_view_generation_enabled = joint_view_generation_config["enabled"]
    args.joint_view_generation_loss_weight = joint_view_generation_config["loss_weight"]
    args.joint_view_generation_hidden_dim = joint_view_generation_config["hidden_dim"]
    args.joint_view_generation_num_heads = joint_view_generation_config["num_heads"]
    args.joint_view_generation_dropout = joint_view_generation_config["dropout"]
    args.joint_view_generation_bev_sigma = joint_view_generation_config["bev_sigma"]
    args.joint_view_generation_gate_init = joint_view_generation_config["gate_init"]
    args.geometry_score_lr_multiplier = geometry_score_training_config["lr_multiplier"]
    args.geometry_score_gate_warmup_steps = geometry_score_training_config["gate_warmup_steps"]
    args.geometry_score_gate_warmup_start_scale = geometry_score_training_config["gate_warmup_start_scale"]
    args.geometry_score_gate_warmup_end_scale = geometry_score_training_config["gate_warmup_end_scale"]

    _configure_logging(log_dir)

    distributed, rank, local_rank, world_size = _init_distributed(args)
    args.distributed = distributed
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.effective_batch_size = int(args.batch_size) * int(world_size) * int(args.gradient_accumulation)
    args.pose_chain_group_size = _infer_pose_chain_group_size(args.view_set, args.pose_chains)
    args.effective_view_batch_size = int(args.effective_batch_size) * int(args.pose_chain_group_size)
    is_main_process = rank == 0

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    if is_main_process:
        logger.info(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
        logger.info(f"HF_HOME={os.environ['HF_HOME']}")
        if distributed:
            logger.info(f"Distributed training enabled: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
        args.mixed_precision = "no"

    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init_checkpoint are mutually exclusive")

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if attention_alignment_config["enabled"] and attention_alignment_config["loss_weight"] > 0.0 and gradient_checkpointing:
        raise ValueError(
            "training.attention_alignment.loss_weight > 0 requires gradient_checkpointing=false. "
            "With UNet gradient checkpointing the captured attention tensors are non-differentiable, "
            "so the auxiliary alignment loss would be logged but not trained. "
            "Use --no-gradient_checkpointing and reduce --batch_size if needed."
        )
    if joint_view_generation_config["enabled"] and args.view_set != "pose_chain":
        raise ValueError("training.joint_view_generation.enable=true requires data.view_set=pose_chain")
    if is_main_process:
        logger.info(f"Training configuration: {args}")

    # Load data
    if is_main_process:
        logger.info(f"Loading data from: {args.data_dir}")

    data_path = Path(args.data_dir)
    split_yaml = Path(args.split_yaml) if args.split_yaml is not None else data_path / "train_test_split_config.yaml"
    train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(data_path, split_yaml)

    if is_main_process:
        logger.info(f"Loaded split file: {split_yaml}")
        logger.info(f"Training on {len(train_dirs)} drives, validating on {len(val_dirs)} drives")

    common_dataset_kwargs = dict(
        mode=args.dataset_mode,
        yaw_mode=args.yaw_mode,
        view_set=args.view_set,
        pose_chains=args.pose_chains,
        virtual_size=front_resize,
        front_resize=front_resize,
        front_center_crop=None,
        random_fisheye_relative_yaw=False,
        random_vehicle_relative_yaw=False,
        front_sample_prob=0.0,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
        seed=args.seed,
        return_bgr=False,
    )
    train_dataset_kwargs = dict(common_dataset_kwargs)
    val_dataset_kwargs = dict(common_dataset_kwargs)

    if args.dataset_mode != "front" and args.yaw_mode == "vehicle_relative":
        random_vehicle_relative_yaw = args.vehicle_yaw_sampling == "random_range"
        train_dataset_kwargs.update({
            "random_vehicle_relative_yaw": random_vehicle_relative_yaw,
            "vehicle_yaw_min_deg": args.vehicle_yaw_min_deg,
            "vehicle_yaw_max_deg": args.vehicle_yaw_max_deg,
            "vehicle_yaw_sampling": args.vehicle_yaw_sampling,
            "vehicle_yaw_fixed_list": args.vehicle_yaw_fixed_list,
            "front_sample_prob": args.front_sample_prob,
        })
        val_dataset_kwargs.update({
            "vehicle_relative_yaw_deg": 0.5 * (args.vehicle_yaw_min_deg + args.vehicle_yaw_max_deg),
            "vehicle_yaw_min_deg": args.vehicle_yaw_min_deg,
            "vehicle_yaw_max_deg": args.vehicle_yaw_max_deg,
            "vehicle_yaw_sampling": "fixed_list" if args.vehicle_yaw_sampling == "fixed_list" else "random_range",
            "vehicle_yaw_fixed_list": args.vehicle_yaw_fixed_list,
        })

    train_dataset = Kitti360dDataset(
        drives=train_dirs,
        frames=train_frames,
        **train_dataset_kwargs,
    )

    val_dataset = Kitti360dDataset(
        drives=val_dirs,
        frames=val_frames,
        **val_dataset_kwargs,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    ) if distributed else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_safe_collate,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_safe_collate,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    if is_main_process:
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Load model
    if is_main_process:
        logger.info("Loading model...")
    model = create_sd_model(
        base_model=args.base_model,
        freeze_base=freeze_base,
        revision=args.base_model_revision,
        torch_dtype=None,
        cond_drop_prob=args.cond_drop_prob,
        query_geometry_score_enabled=query_geometry_score_config["enabled"],
        query_geometry_score_dim=query_geometry_score_config["dim"],
        query_geometry_score_num_freqs=query_geometry_score_config["num_freqs"],
        query_geometry_score_gate_init=query_geometry_score_config["gate_init"],
        query_geometry_score_layers=query_geometry_score_config["layers"],
        query_geometry_score_max_query_tokens=query_geometry_score_config["max_query_tokens"],
        query_geometry_score_mode=query_geometry_score_config["mode"],
        query_geometry_candidate_radius=query_geometry_score_config["candidate_radius"],
        query_geometry_candidate_min_k=query_geometry_score_config["candidate_min_k"],
        query_geometry_candidate_invalid_penalty=query_geometry_score_config["candidate_invalid_penalty"],
        query_semantic_score_dim=query_geometry_score_config["semantic_score_dim"],
        query_semantic_score_alpha=query_geometry_score_config["semantic_alpha_max"],
        attention_alignment_enabled=attention_alignment_config["enabled"],
        attention_alignment_loss_weight=attention_alignment_config["loss_weight"],
        attention_alignment_layers=attention_alignment_config["layers"],
        attention_alignment_max_query_tokens=attention_alignment_config["max_query_tokens"],
        attention_alignment_valid_radius=attention_alignment_config["valid_radius"],
        attention_alignment_invalid_attention_weight=attention_alignment_config["invalid_attention_weight"],
        transition_aux_enabled=transition_aux_config["enabled"],
        transition_aux_loss_weight=transition_aux_config["loss_weight"],
        transition_aux_cycle_weight=transition_aux_config["cycle_weight"],
        transition_aux_composition_weight=transition_aux_config["composition_weight"],
        transition_aux_mse_weight=transition_aux_config["mse_weight"],
        transition_aux_hidden_channels=transition_aux_config["hidden_channels"],
        transition_aux_action_dim=transition_aux_config["action_dim"],
        transition_aux_source=transition_aux_config["source"],
        joint_view_generation_enabled=joint_view_generation_config["enabled"],
        joint_view_generation_loss_weight=joint_view_generation_config["loss_weight"],
        joint_view_generation_hidden_dim=joint_view_generation_config["hidden_dim"],
        joint_view_generation_num_heads=joint_view_generation_config["num_heads"],
        joint_view_generation_dropout=joint_view_generation_config["dropout"],
        joint_view_generation_bev_sigma=joint_view_generation_config["bev_sigma"],
        joint_view_generation_gate_init=joint_view_generation_config["gate_init"],
        satellite_encoder_config=satellite_encoder_config,
    )
    _verify_query_geometry_score_model_config(model, query_geometry_score_config)
    if args.device.startswith("cuda") and args.mixed_precision != "no":
        if is_main_process:
            logger.info(
            "Training keeps model weights in fp32; mixed precision is applied via autocast only"
            )
    if gradient_checkpointing and hasattr(model.unet, "enable_gradient_checkpointing"):
        model.unet.enable_gradient_checkpointing()
        if is_main_process:
            logger.info("Enabled UNet gradient checkpointing")
    if unet_attention_slicing and hasattr(model.unet, "set_attention_slice"):
        model.unet.set_attention_slice("auto")
        if is_main_process:
            logger.info("Enabled UNet attention slicing")
    elif is_main_process:
        logger.info("UNet attention slicing disabled; using SDPA attention processors")
    if hasattr(model.vae, "enable_slicing"):
        model.vae.enable_slicing()
        if is_main_process:
            logger.info("Enabled VAE slicing")

    # Create trainer
    if is_main_process:
        logger.info("Creating trainer...")
    trainer = SDTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=args.lr,
        weight_decay=weight_decay,
        num_train_epochs=args.epochs,
        lr_scheduler_type=lr_scheduler_type,
        warmup_epochs=args.warmup,
        geometry_score_lr_multiplier=geometry_score_training_config["lr_multiplier"],
        transition_aux_lr_multiplier=transition_aux_config["lr_multiplier"],
        transition_aux_warmup_steps=transition_aux_config["warmup_steps"],
        geometry_score_gate_warmup_steps=geometry_score_training_config["gate_warmup_steps"],
        geometry_score_gate_warmup_start_scale=geometry_score_training_config["gate_warmup_start_scale"],
        geometry_score_gate_warmup_end_scale=geometry_score_training_config["gate_warmup_end_scale"],
        semantic_score_alpha_max=query_geometry_score_config["semantic_alpha_max"],
        semantic_score_alpha_hold_steps=geometry_score_training_config["semantic_alpha_hold_steps"],
        semantic_score_alpha_warmup_steps=geometry_score_training_config["semantic_alpha_warmup_steps"],
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.output_dir,
        save_every=save_every,
        log_every=log_every,
        validate_every=args.validate_every,
        device=args.device,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        use_tensorboard=args.use_tensorboard,
        tensorboard_log_dir=args.tensorboard_log_dir,
        run_config=vars(args),
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        max_grad_norm=args.max_grad_norm,
        visualize_every=args.visualize_every,
        num_visualizations=args.num_visualizations,
        visualization_inference_steps=args.visualization_inference_steps,
        visualization_guidance_scale=args.guidance_scale,
        visualization_seed=args.visualization_seed,
        distributed=distributed,
        local_rank=local_rank,
    )

    if args.init_checkpoint is not None:
        if is_main_process:
            logger.info(f"Initializing model weights from checkpoint: {args.init_checkpoint}")
        load_model_checkpoint(
            trainer.unwrapped_model,
            Path(args.init_checkpoint),
            args.device,
        )

    # Start training
    if is_main_process:
        logger.info("Starting training...")
    try:
        trainer.train(resume_from=args.resume)
    finally:
        _cleanup_distributed(distributed)

    if is_main_process:
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
