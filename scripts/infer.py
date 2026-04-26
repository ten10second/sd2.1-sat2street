#!/usr/bin/env python3
"""
Inference utilities for satellite-to-street generation.

Supported modes:
  - single_yaw_sweep: render one frame at front plus vehicle-relative yaw values.
  - split_fixed_views: render split frames at fixed views and save GT comparisons.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from models.sd_trainer import create_sd_model, load_model_checkpoint


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
    ("yaw_m120", -120.0),
    ("yaw_m90", -90.0),
    ("yaw_m40", -40.0),
    ("yaw_p40", 40.0),
    ("yaw_p90", 90.0),
    ("yaw_p120", 120.0),
)


def _load_frame_ids(frames_file: Path) -> List[int]:
    frame_ids: List[int] = []
    for line in frames_file.read_text().splitlines():
        token = line.strip()
        if token:
            frame_ids.append(int(token))
    return frame_ids


def _load_split_from_yaml(
    data_dir: Path,
    split_yaml: Path,
) -> Tuple[List[Path], List[List[int]], List[Path], List[List[int]]]:
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
            if not frame_ids:
                raise ValueError(f"No frame ids found in {frames_file}")

            drives.append(drive_dir)
            frames_per_drive.append(frame_ids)

        return drives, frames_per_drive

    train_dirs, train_frames = parse_entries(train_entries, "train")
    val_dirs, val_frames = parse_entries(val_entries, "val/test")
    return train_dirs, train_frames, val_dirs, val_frames


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for satellite-to-street generation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single_yaw_sweep",
        choices=["single_yaw_sweep", "split_fixed_views"],
        help="Inference mode.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/user/574b4a05-57d2-424d-bb82-763098cbf0a4/shizhm/KITTI-360",
        help="Path to KITTI-360 data root.",
    )
    parser.add_argument("--split_yaml", type=str, default=None, help="Split yaml for split-based inference.")
    parser.add_argument("--dataset_split", type=str, default="val", choices=["train", "val"])
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
        help="Yaw values for single_yaw_sweep. Front is included by default; pass --no_include_front to omit it.",
    )
    parser.add_argument("--vehicle_yaw_min_deg", type=float, default=60.0)
    parser.add_argument("--vehicle_yaw_max_deg", type=float, default=120.0)
    front_group = parser.add_mutually_exclusive_group()
    front_group.add_argument("--include_front", dest="include_front", action="store_true")
    front_group.add_argument("--no_include_front", dest="include_front", action="store_false")
    parser.set_defaults(include_front=True)
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
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
    return parser.parse_args()


def _training_range_yaws(min_abs: float, max_abs: float) -> List[float]:
    min_abs = abs(float(min_abs))
    max_abs = abs(float(max_abs))
    if max_abs < min_abs:
        min_abs, max_abs = max_abs, min_abs
    mid = 0.5 * (min_abs + max_abs)
    values = [-max_abs, -mid, -min_abs, min_abs, mid, max_abs]
    deduped: List[float] = []
    for value in values:
        if not any(abs(value - seen) < 1e-6 for seen in deduped):
            deduped.append(value)
    return deduped


def _view_token(view_name: str, yaw: Optional[float]) -> str:
    if yaw is None:
        return view_name
    token = f"yaw_{yaw:g}".replace("-", "m").replace(".", "p")
    return token


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


def _resize_satellite_for_front(sat_image: torch.Tensor, target_h: int) -> torch.Tensor:
    return F.interpolate(
        sat_image.unsqueeze(0),
        size=(target_h, target_h),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


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


def _coords_map_to_satellite_pixels(
    coords_map: Optional[torch.Tensor],
    sat_width: int,
    sat_height: int,
) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
    if coords_map is None or not torch.is_tensor(coords_map):
        return [], None

    coords = coords_map.detach().cpu().to(torch.float32)
    if coords.ndim == 3 and coords.shape[0] == 2:
        coords = coords.permute(1, 2, 0).reshape(-1, 2)
    elif coords.ndim == 3 and coords.shape[-1] == 2:
        coords = coords.reshape(-1, 2)
    elif coords.ndim == 2 and coords.shape[-1] == 2:
        pass
    else:
        return [], None

    return _coords_to_satellite_pixels(coords, sat_width, sat_height)


def _coords_map_to_fov_polygon(
    coords_map: Optional[torch.Tensor],
    sat_width: int,
    sat_height: int,
) -> List[Tuple[float, float]]:
    if coords_map is None or not torch.is_tensor(coords_map):
        return []

    coords = coords_map.detach().cpu().to(torch.float32)
    if coords.ndim == 3 and coords.shape[0] == 2:
        coords_hw = coords.permute(1, 2, 0)
    elif coords.ndim == 3 and coords.shape[-1] == 2:
        coords_hw = coords
    else:
        return []

    height, width = int(coords_hw.shape[0]), int(coords_hw.shape[1])
    if height < 2 or width < 2:
        return []

    top = coords_hw[0, :, :]
    right = coords_hw[:, width - 1, :]
    bottom = torch.flip(coords_hw[height - 1, :, :], dims=[0])
    left = torch.flip(coords_hw[:, 0, :], dims=[0])
    boundary = torch.cat([top, right, bottom, left], dim=0)
    points, _ = _coords_to_satellite_pixels(boundary, sat_width, sat_height)

    if len(points) < 3:
        all_points, bbox = _coords_map_to_satellite_pixels(coords_hw, sat_width, sat_height)
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
    coords_map: Optional[torch.Tensor],
    view_name: str,
    yaw: Optional[float],
) -> Image.Image:
    image = _tensor_to_pil(sat_image).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    polygon = _coords_map_to_fov_polygon(coords_map, width, height)

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


def _build_base_dataset(drives, frames, seed: int) -> Kitti360dDataset:
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


@torch.no_grad()
def _materialize_lazy_modules(
    model,
    sample: Dict,
    device: str,
) -> None:
    sat_images = sample["sat"].unsqueeze(0).to(device)
    coords_map = sample.get("coords_map")
    coords_map = coords_map.unsqueeze(0).to(device) if coords_map is not None else None
    plucker_map = sample.get("plucker_map")
    plucker_map = plucker_map.unsqueeze(0).to(device) if plucker_map is not None else None
    target_size = tuple(int(x) for x in sample["image"].shape[-2:])

    sat_encoded = model.encode_satellite(sat_images, coords_map)
    if isinstance(sat_encoded, tuple):
        sat_tokens, sat_xy = sat_encoded
    else:
        sat_tokens = sat_encoded
        sat_xy = None

    vae_scale_factor = model._get_vae_scale_factor()
    latent_h = max(1, (target_size[0] + vae_scale_factor - 1) // vae_scale_factor)
    latent_w = max(1, (target_size[1] + vae_scale_factor - 1) // vae_scale_factor)
    latents = torch.randn(
        (sat_images.shape[0], model.unet.config.in_channels, latent_h, latent_w),
        device=sat_images.device,
        dtype=sat_tokens.dtype,
    )
    timestep = torch.zeros((sat_images.shape[0],), device=sat_images.device, dtype=torch.long)

    model.unet(
        latents,
        timestep,
        encoder_hidden_states=None,
        sat_tokens=sat_tokens,
        sat_xy=sat_xy,
        front_bev_xy=coords_map,
        front_plucker=plucker_map,
        return_attn_map=False,
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
        reading_block_config={"enable": True},
        revision=args.base_model_revision,
        torch_dtype=model_torch_dtype,
        cond_drop_prob=0.0,
    )
    if hasattr(model.unet, "set_attention_slice"):
        model.unet.set_attention_slice("auto")
    if hasattr(model.vae, "enable_slicing"):
        model.vae.enable_slicing()
    model.to(args.device)
    model.eval()

    logger.info("Materializing lazy reading blocks before loading checkpoint")
    _materialize_lazy_modules(model, materialize_sample, args.device)
    checkpoint_meta = load_model_checkpoint(model, Path(args.checkpoint), args.device)
    model.eval()
    return model, checkpoint_meta


@torch.no_grad()
def _generate_one(
    model,
    sample: Dict,
    args: argparse.Namespace,
) -> torch.Tensor:
    sat_image = sample["sat"].unsqueeze(0).to(args.device)
    coords_map = sample.get("coords_map")
    coords_map = coords_map.unsqueeze(0).to(args.device) if coords_map is not None else None
    plucker_map = sample.get("plucker_map")
    plucker_map = plucker_map.unsqueeze(0).to(args.device) if plucker_map is not None else None
    target_size = tuple(int(x) for x in sample["image"].shape[-2:])

    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(args.seed))

    return model.generate(
        sat_image,
        coords_map=coords_map,
        plucker_map=plucker_map,
        target_size=target_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        sat_condition_mode=args.sat_condition_mode,
    )[0].cpu()


def _save_view_outputs(
    sample: Dict,
    generated: torch.Tensor,
    output_dir: Path,
    view_name: str,
    yaw: Optional[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_image = sample["image"]
    sat_resized = _resize_satellite_for_front(sample["sat"], int(gt_image.shape[-2]))
    sat_overlay = _draw_satellite_coverage(
        sample["sat"],
        sample.get("coords_map"),
        view_name,
        yaw,
    ).resize((sat_resized.shape[-1], sat_resized.shape[-2]), resample=Image.BILINEAR)

    _tensor_to_pil(generated).save(output_dir / "generated.png")
    _tensor_to_pil(gt_image).save(output_dir / "gt.png")
    sat_overlay.save(output_dir / "satellite.png")
    comparison = _compose_panels([
        ("sat coverage", torch.from_numpy(np.array(sat_overlay)).permute(2, 0, 1).to(torch.float32) / 255.0),
        (f"gen {view_name}", generated),
        ("gt", gt_image),
    ])
    comparison.save(output_dir / "comparison.png")

    metadata = {
        "drive": str(sample["drive"]),
        "frame_id": int(sample["frame_id"]),
        "view_name": view_name,
        "vehicle_yaw_deg": None if yaw is None else float(yaw),
        "meta": sample.get("meta", {}),
    }
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def _resolve_single_dataset(args: argparse.Namespace) -> Tuple[Kitti360dDataset, int]:
    if args.frame_id is None:
        raise ValueError("--frame_id is required for single_yaw_sweep")

    if args.drive_dir is not None:
        dataset = _build_base_dataset(Path(args.drive_dir), [int(args.frame_id)], args.seed)
        return dataset, 0

    if args.split_yaml is not None:
        train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(
            Path(args.data_dir),
            Path(args.split_yaml),
        )
        drives = train_dirs if args.dataset_split == "train" else val_dirs
        frames = train_frames if args.dataset_split == "train" else val_frames
        dataset = _build_base_dataset(drives, frames, args.seed)
        return dataset, _resolve_sample_index(dataset, int(args.frame_id), args.drive)

    if args.drive is None:
        raise ValueError("Pass --drive_dir, or pass --split_yaml with optional --drive")
    dataset = _build_base_dataset(Path(args.data_dir) / args.drive, [int(args.frame_id)], args.seed)
    return dataset, 0


def run_single_yaw_sweep(args: argparse.Namespace) -> None:
    dataset, sample_index = _resolve_single_dataset(args)
    yaws = args.vehicle_yaws if args.vehicle_yaws is not None else _training_range_yaws(
        args.vehicle_yaw_min_deg,
        args.vehicle_yaw_max_deg,
    )
    view_specs: List[Tuple[str, Optional[float]]] = []
    if args.include_front:
        view_specs.append(("front", None))
    view_specs.extend((_view_token("yaw", yaw), float(yaw)) for yaw in yaws)

    materialize_sample = _get_view_sample(dataset, sample_index, *view_specs[0])
    model, checkpoint_meta = _load_model(args, materialize_sample)

    output_root = Path(args.output_dir)
    base_sample = dataset.samples[sample_index]
    sample_dir = output_root / f"{base_sample.drive_dir.name}_frame_{base_sample.frame_id:010d}_yaw_sweep"

    for view_name, yaw in view_specs:
        logger.info(f"Generating frame={base_sample.frame_id:010d}, view={view_name}, yaw={yaw}")
        sample = _get_view_sample(dataset, sample_index, view_name, yaw)
        generated = _generate_one(model, sample, args)
        _save_view_outputs(sample, generated, sample_dir / _view_token(view_name, yaw), view_name, yaw)

    with open(sample_dir / "run_metadata.yaml", "w") as f:
        yaml.safe_dump(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_epoch": checkpoint_meta.get("epoch"),
                "mode": args.mode,
                "views": [{"view_name": name, "vehicle_yaw_deg": yaw} for name, yaw in view_specs],
            },
            f,
            sort_keys=False,
        )
    logger.info(f"Saved single-frame yaw sweep to: {sample_dir}")


def run_split_fixed_views(args: argparse.Namespace) -> None:
    if args.split_yaml is None:
        raise ValueError("--split_yaml is required for split_fixed_views")

    train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(
        Path(args.data_dir),
        Path(args.split_yaml),
    )
    drives = train_dirs if args.dataset_split == "train" else val_dirs
    frames = train_frames if args.dataset_split == "train" else val_frames
    dataset = _build_base_dataset(drives, frames, args.seed)
    sample_indices = _filter_sample_indices(dataset, args.start_frame, args.end_frame, args.max_frames)

    materialize_sample = _get_view_sample(dataset, sample_indices[0], *FIXED_VIEW_SPECS[0])
    model, checkpoint_meta = _load_model(args, materialize_sample)

    output_root = Path(args.output_dir)
    progress = tqdm(sample_indices, desc="Split fixed-view inference")
    for sample_index in progress:
        base_sample = dataset.samples[sample_index]
        frame_dir = output_root / base_sample.drive_dir.name / f"frame_{base_sample.frame_id:010d}"
        for view_name, yaw in FIXED_VIEW_SPECS:
            progress.set_postfix(frame=f"{base_sample.frame_id:010d}", view=view_name)
            sample = _get_view_sample(dataset, sample_index, view_name, yaw)
            generated = _generate_one(model, sample, args)
            _save_view_outputs(sample, generated, frame_dir / _view_token(view_name, yaw), view_name, yaw)

    with open(output_root / "run_metadata.yaml", "w") as f:
        yaml.safe_dump(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_epoch": checkpoint_meta.get("epoch"),
                "mode": args.mode,
                "dataset_split": args.dataset_split,
                "split_yaml": str(Path(args.split_yaml)),
                "start_frame": args.start_frame,
                "end_frame": args.end_frame,
                "max_frames": args.max_frames,
                "num_frames": len(sample_indices),
                "fixed_views": [
                    {"view_name": view_name, "vehicle_yaw_deg": yaw}
                    for view_name, yaw in FIXED_VIEW_SPECS
                ],
            },
            f,
            sort_keys=False,
        )
    logger.info(f"Saved split fixed-view inference to: {output_root}")


def main() -> None:
    args = _parse_args()
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
    elif args.mode == "split_fixed_views":
        run_split_fixed_views(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
