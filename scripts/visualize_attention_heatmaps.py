#!/usr/bin/env python3
"""Save generation comparisons and cross-attention heatmaps for selected frames/views."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data import Kitti360dDataset
from models.conditioning import SatelliteMemoryState
from models.sd_model import create_sd_model, load_model_checkpoint
from models.sd_trainer import SDTrainer


_train_mod = runpy.run_path(str(_project_root / "scripts" / "train.py"))

VIEW_TO_YAW = {
    "front": None,
    "yaw_m120": -120.0,
    "yaw_m90": -90.0,
    "yaw_m60": -60.0,
    "yaw_p60": 60.0,
    "yaw_p90": 90.0,
    "yaw_p120": 120.0,
}


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
    train_dirs, train_frames, val_dirs, val_frames = _train_mod["_load_split_from_yaml"](
        data_path,
        Path(args.split_yaml),
    )
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


def _find_sample_index(dataset: Kitti360dDataset, drive: str, frame_id: int, view_name: str) -> int:
    matches = []
    for idx, sample in enumerate(dataset.samples):
        if sample.drive_dir.name != drive or int(sample.frame_id) != int(frame_id):
            continue
        meta = sample.meta if isinstance(sample.meta, dict) else {}
        if str(meta.get("view_name")) == view_name:
            matches.append(idx)
    if not matches:
        raise ValueError(f"No sample found for drive={drive}, frame={frame_id}, view={view_name}")
    return int(matches[0])


def _to_batch(sample: Dict[str, Any], key: str, device: str) -> Optional[torch.Tensor]:
    value = sample.get(key)
    if value is None or not torch.is_tensor(value):
        return None
    return value.unsqueeze(0).to(device)


def _batched_camera_height(sample: Dict[str, Any], device: str) -> Optional[torch.Tensor]:
    value = sample.get("camera_height_m")
    if value is None:
        return None
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor.to(device=device, dtype=torch.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cross-attention heatmaps from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_yaml", type=str, required=True)
    parser.add_argument("--dataset_split", choices=["train", "val"], default="val")
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--frame_ids", type=int, nargs="+", required=True)
    parser.add_argument("--views", type=str, nargs="+", default=["front", "yaw_p60", "yaw_p120"])
    parser.add_argument("--output_dir", type=str, default="output/attention_heatmaps")
    parser.add_argument("--layers", type=str, nargs="+", default=[
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2",
        "mid_block.attentions.0.transformer_blocks.0.attn2",
    ])
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default="sd2-community/stable-diffusion-2-1-base")
    parser.add_argument("--base_model_revision", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--hf_home", type=str, default=str(_project_root / ".hf-home"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home

    unknown_views = [view for view in args.views if view not in VIEW_TO_YAW]
    if unknown_views:
        raise ValueError(f"Unknown views: {unknown_views}; choices={sorted(VIEW_TO_YAW)}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = _load_config(Path(args.config))
    dataset = _build_dataset(args, config)
    model = _build_model(args, config)
    model.to(args.device)
    load_model_checkpoint(model, Path(args.checkpoint), args.device)
    model.eval()
    if hasattr(model.vae, "enable_slicing"):
        model.vae.enable_slicing()

    visualizer = object.__new__(SDTrainer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_device = args.device if args.device.startswith("cuda") else "cpu"

    for frame_id in args.frame_ids:
        for view_name in args.views:
            sample_index = _find_sample_index(dataset, args.drive, int(frame_id), view_name)
            sample = dataset[sample_index]
            sat_image = sample["sat"].unsqueeze(0).to(args.device)
            real_image = sample["image"]
            target_size = tuple(int(x) for x in real_image.shape[-2:])

            K = _to_batch(sample, "K", args.device)
            T_cam_to_world = _to_batch(sample, "T_cam_to_world", args.device)
            T_imu_to_world = _to_batch(sample, "T_imu_to_world", args.device)
            camera_height_m = _batched_camera_height(sample, args.device)

            with torch.no_grad():
                sat_state = model.encode_satellite(
                    sat_image,
                    K=K,
                    T_cam_to_world=T_cam_to_world,
                    T_imu_to_world=T_imu_to_world,
                    camera_height_m=camera_height_m,
                    image_size=target_size,
                )
                view_sat_state = SatelliteMemoryState(
                    tokens=sat_state.tokens[:1],
                    xy=sat_state.xy[:1],
                    bev_coords=sat_state.bev_coords[:1] if sat_state.bev_coords is not None else None,
                    perspective_uv=sat_state.perspective_uv[:1] if sat_state.perspective_uv is not None else None,
                    perspective_valid=sat_state.perspective_valid[:1] if sat_state.perspective_valid is not None else None,
                )
                attention_debug: Dict[str, Any] = {}
                model.unet.enable_attention_debug(layers=args.layers, storage=attention_debug)
                try:
                    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
                    generated, _ = model.generate_with_satellite_state(
                        view_sat_state,
                        target_size=target_size,
                        num_inference_steps=int(args.inference_steps),
                        guidance_scale=float(args.guidance_scale),
                        generator=generator,
                    )
                finally:
                    model.unet.disable_attention_debug()

            yaw_deg = VIEW_TO_YAW[view_name]
            prefix = f"{args.drive}_frame_{int(frame_id):010d}_{view_name}"
            comparison = visualizer._compose_visualization(
                sample["sat"],
                generated[0].detach().cpu(),
                real_image,
                front_bev_xy=sample.get("front_bev_xy"),
                front_ground_valid_mask=sample.get("front_ground_valid_mask"),
                view_label=view_name,
                yaw_deg=yaw_deg,
            )
            comparison.save(output_dir / f"{prefix}_comparison.png")
            visualizer._save_attention_debug_visualizations(
                attention_debug=attention_debug,
                sat_image=sample["sat"],
                gt_image=real_image,
                front_bev_xy=sample.get("front_bev_xy"),
                front_ground_valid_mask=sample.get("front_ground_valid_mask"),
                output_dir=output_dir / "attention",
                prefix=prefix,
            )
            print(f"saved {prefix}")

    print(f"Saved heatmap diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
