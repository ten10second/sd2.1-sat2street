#!/usr/bin/env python3
"""Run checkpoint-time geometry diagnostics without relying on W&B history."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data import Kitti360dDataset
from models.conditioning import SatelliteMemoryState
from models.sd_model import create_sd_model, load_model_checkpoint


_train_mod = runpy.run_path(str(_project_root / "scripts" / "train.py"))


def _config_get(config: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def _scalar(outputs: Dict[str, Any], key: str) -> Optional[float]:
    value = outputs.get(key)
    if not torch.is_tensor(value):
        return None
    return float(value.detach().float().cpu().item())


def _meta_value(meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default


def _move_geometry(batch: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    keys = ("K", "T_cam_to_world", "T_imu_to_world", "camera_height_m")
    moved = {}
    for key in keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device)
    return moved


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _build_dataset(args: argparse.Namespace, config: Dict[str, Any]) -> Kitti360dDataset:
    data_path = Path(args.data_dir)
    split_yaml = Path(args.split_yaml)
    train_dirs, train_frames, val_dirs, val_frames = _train_mod["_load_split_from_yaml"](data_path, split_yaml)
    drives = train_dirs if args.dataset_split == "train" else val_dirs
    frames = train_frames if args.dataset_split == "train" else val_frames

    data_cfg = dict(config.get("data", {}) or {})
    front_resize_cfg = data_cfg.get("front_resize", [640, 256])
    front_resize = tuple(int(x) for x in front_resize_cfg)
    fixed_list = data_cfg.get("vehicle_yaw_fixed_list")
    if fixed_list is None:
        fixed_list = ["front", -120.0, -90.0, -60.0, 60.0, 90.0, 120.0]

    return Kitti360dDataset(
        drives=drives,
        frames=frames,
        mode=str(data_cfg.get("mode", "fisheye_virtual")),
        yaw_mode=str(data_cfg.get("yaw_mode", "vehicle_relative")),
        view_set=str(data_cfg.get("view_set", "single")),
        virtual_size=front_resize,
        front_resize=front_resize,
        front_center_crop=None,
        random_fisheye_relative_yaw=False,
        random_vehicle_relative_yaw=False,
        vehicle_yaw_min_deg=float(data_cfg.get("vehicle_yaw_min_deg", 60.0)),
        vehicle_yaw_max_deg=float(data_cfg.get("vehicle_yaw_max_deg", 120.0)),
        vehicle_yaw_sampling=str(data_cfg.get("vehicle_yaw_sampling", "fixed_list")),
        vehicle_yaw_fixed_list=fixed_list,
        front_sample_prob=float(data_cfg.get("front_sample_prob", 0.0)),
        pitch_deg=float(data_cfg.get("pitch_deg", 0.0)),
        roll_deg=float(data_cfg.get("roll_deg", 0.0)),
        seed=int(config.get("seed", args.seed)),
        return_bgr=False,
    )


def _build_model(args: argparse.Namespace, config: Dict[str, Any]):
    model_cfg = dict(config.get("model", {}) or {})
    satellite_encoder_config = dict(model_cfg.get("satellite_encoder", {}) or {})
    query_geometry_score_config = _train_mod["_resolve_query_geometry_score_config"](config)
    attention_alignment_config = _train_mod["_resolve_attention_alignment_config"](config)

    model = create_sd_model(
        base_model=str(model_cfg.get("base_model", args.base_model)),
        freeze_base=bool(model_cfg.get("freeze_base", True)),
        revision=args.base_model_revision,
        torch_dtype=None,
        cond_drop_prob=0.0,
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
        satellite_encoder_config=satellite_encoder_config,
    )
    _train_mod["_verify_query_geometry_score_model_config"](model, query_geometry_score_config)
    return model


def _summarize(rows: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_rows": len(rows), "overall": {}, "by_view": {}}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            summary["overall"][key] = {
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("view_name") or "unknown")].append(row)
    for view_name, view_rows in sorted(grouped.items()):
        view_summary: Dict[str, Any] = {"num_rows": len(view_rows)}
        for key in metric_keys:
            values = [float(row[key]) for row in view_rows if row.get(key) is not None]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                view_summary[key] = {
                    "mean": float(arr.mean()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        summary["by_view"][view_name] = view_summary
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose attention geometry metrics from a training checkpoint"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_yaml", type=str, required=True)
    parser.add_argument("--dataset_split", choices=["train", "val"], default="val")
    parser.add_argument("--output_dir", type=str, default="output/checkpoint_runtime_diagnostics")
    parser.add_argument("--base_model", type=str, default="sd2-community/stable-diffusion-2-1-base")
    parser.add_argument("--base_model_revision", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", choices=["no", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--hf_home", type=str, default=str(_project_root / ".hf-home"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"
        args.mixed_precision = "no"
    if int(args.batch_size) != 1:
        print(
            "WARNING: batch_size > 1 aggregates attention metrics across samples; "
            "use --batch_size 1 for per-view diagnosis."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = _load_config(Path(args.config))
    dataset = _build_dataset(args, config)
    if args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(len(dataset), args.max_samples))))
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=args.device.startswith("cuda"),
        collate_fn=_train_mod["_safe_collate"],
    )

    model = _build_model(args, config)
    model.to(args.device)
    load_model_checkpoint(model, Path(args.checkpoint), args.device)
    model.train()
    model.vae.eval()
    model.zero_grad(set_to_none=True)

    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(args.mixed_precision)
    use_amp = args.device.startswith("cuda") and amp_dtype is not None
    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"

    metric_keys = [
        "loss",
        "denoise_loss",
        "attention_alignment_loss",
        "attention_alignment_loss_weight",
        "attention_alignment_loss_is_differentiable",
        "attention_alignment_mean_error",
        "attention_alignment_valid_query_ratio",
        "attention_alignment_valid_attention_mass",
        "attention_alignment_target_attention_mass",
        "attention_alignment_target_token_fraction",
        "attention_alignment_target_attention_lift",
        "attention_alignment_nearest_attention_mass",
        "attention_alignment_target_logit_gap",
        "attention_alignment_geometry_score_gate",
        "attention_alignment_geometry_score_raw_std",
        "attention_alignment_geometry_score_bias_std",
    ]
    rows: List[Dict[str, Any]] = []

    for batch_index, batch in enumerate(tqdm(dataloader, desc="Runtime diagnostics")):
        sat_images = batch["sat"].to(args.device)
        target_images = batch["image"].to(args.device)
        geometry_kwargs = _move_geometry(batch, args.device)
        with torch.enable_grad():
            with torch.autocast(
                device_type=autocast_device,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(
                    sat_images,
                    target_images,
                    **geometry_kwargs,
                )

        sat_state = outputs.get("sat_state")
        perspective_valid_ratio = None
        if isinstance(sat_state, SatelliteMemoryState) and sat_state.perspective_valid is not None:
            perspective_valid_ratio = float(sat_state.perspective_valid.detach().float().mean().cpu().item())

        metas = batch.get("meta", [{} for _ in range(sat_images.shape[0])])
        if not isinstance(metas, list):
            metas = [{} for _ in range(sat_images.shape[0])]
        frame_ids = batch.get("frame_id")
        drives = batch.get("drive")
        if torch.is_tensor(frame_ids):
            frame_ids_list = [int(x) for x in frame_ids.detach().cpu().tolist()]
        elif isinstance(frame_ids, list):
            frame_ids_list = [int(x) for x in frame_ids]
        else:
            frame_ids_list = [None for _ in range(sat_images.shape[0])]
        if isinstance(drives, list):
            drives_list = [str(x) for x in drives]
        else:
            drives_list = [None for _ in range(sat_images.shape[0])]

        row: Dict[str, Any] = {
            "batch_index": int(batch_index),
            "batch_size": int(sat_images.shape[0]),
            "drive": drives_list[0] if drives_list else None,
            "frame_id": frame_ids_list[0] if frame_ids_list else None,
            "view_name": _meta_value(metas[0] if metas else {}, "view_name", "unknown"),
            "vehicle_yaw_deg": _meta_value(metas[0] if metas else {}, "vehicle_yaw_deg_used"),
            "perspective_valid_ratio": perspective_valid_ratio,
        }
        for key in metric_keys:
            row[key] = _scalar(outputs, key)
        rows.append(row)

        del outputs, sat_images, target_images
        model.zero_grad(set_to_none=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runtime_metrics.csv"
    json_path = output_dir / "runtime_metrics.json"
    summary_path = output_dir / "runtime_summary.json"

    fieldnames = [
        "batch_index",
        "batch_size",
        "drive",
        "frame_id",
        "view_name",
        "vehicle_yaw_deg",
        "perspective_valid_ratio",
        *metric_keys,
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    summary = _summarize(rows, ["perspective_valid_ratio", *metric_keys])
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved runtime metrics: {csv_path}")
    print(f"Saved runtime summary: {summary_path}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
