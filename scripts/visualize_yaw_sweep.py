#!/usr/bin/env python3
"""
Visualize one fixed sample while sweeping vehicle-relative yaw values.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

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


def _load_frame_ids(frames_file: Path) -> List[int]:
    frame_ids: List[int] = []
    for line in frames_file.read_text().splitlines():
        token = line.strip()
        if not token:
            continue
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
        description="Visualize one frame while sweeping vehicle-relative yaw values"
    )
    parser.add_argument(
        "--data_dir", type=str, default="/media/user/574b4a05-57d2-424d-bb82-763098cbf0a4/shizhm/KITTI-360",
        help="Path to KITTI-360 data root",
    )
    parser.add_argument(
        "--split_yaml", type=str, required=True,
        help="Path to split yaml",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="val", choices=["train", "val"],
        help="Which split to draw the sample from",
    )
    parser.add_argument(
        "--drive", type=str, required=True,
        help="Drive name used together with --frame_id",
    )
    parser.add_argument(
        "--frame_id", type=int, required=True,
        help="Exact frame id to render",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to training checkpoint",
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
        "--output_dir", type=str, default="output/yaw_sweep_visualizations",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--vehicle_yaws", type=float, nargs="+", default=[60.0, 75.0, 90.0, 105.0, 120.0],
        help="Vehicle-relative yaw values in degrees",
    )
    parser.add_argument(
        "--inference_steps", type=int, default=30,
        help="Number of denoising steps for each visualization",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0,
        help="Guidance scale for generation. 1.0 disables CFG.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed reused for every yaw",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Precision mode used to load the model",
    )
    parser.add_argument(
        "--hf_endpoint", type=str, default=DEFAULT_HF_ENDPOINT,
        help="Hugging Face endpoint",
    )
    parser.add_argument(
        "--hf_home", type=str, default=str(DEFAULT_HF_HOME),
        help="Local Hugging Face cache directory",
    )
    return parser.parse_args()


def _build_dataset(args: argparse.Namespace) -> Kitti360dDataset:
    data_path = Path(args.data_dir)
    split_yaml = Path(args.split_yaml)
    train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(data_path, split_yaml)

    drives = train_dirs if args.dataset_split == "train" else val_dirs
    frames = train_frames if args.dataset_split == "train" else val_frames

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
        seed=args.seed,
        return_bgr=False,
    )


def _resolve_sample_index(
    dataset: Kitti360dDataset,
    frame_id: int,
    drive: str,
) -> int:
    matches = []
    for idx, sample in enumerate(dataset.samples):
        if sample.frame_id == frame_id and sample.drive_dir.name == drive:
            matches.append(idx)

    if not matches:
        raise ValueError(f"Could not find frame_id={frame_id} in drive {drive}")
    if len(matches) > 1:
        raise ValueError(f"frame_id={frame_id} appears multiple times in drive {drive}")
    return matches[0]


def _get_sample_with_vehicle_yaw(
    dataset: Kitti360dDataset,
    sample_index: int,
    vehicle_yaw_deg: float,
) -> Dict:
    base_sample = dataset.samples[sample_index]
    override_meta = dict(base_sample.meta or {})
    override_meta["vehicle_relative_yaw_deg_override"] = float(vehicle_yaw_deg)
    override_sample = SampleIndex(
        drive_dir=base_sample.drive_dir,
        frame_id=base_sample.frame_id,
        meta=override_meta,
    )

    original_sample = dataset.samples[sample_index]
    dataset.samples[sample_index] = override_sample
    try:
        return dataset[sample_index]
    finally:
        dataset.samples[sample_index] = original_sample


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


@torch.no_grad()
def _materialize_lazy_modules(
    model,
    sat_images: torch.Tensor,
    coords_map: Optional[torch.Tensor],
    target_size: Tuple[int, int],
) -> None:
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
        return_attn_map=False,
    )


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

    dataset = _build_dataset(args)
    sample_index = _resolve_sample_index(dataset, args.frame_id, args.drive)
    sample_90 = _get_sample_with_vehicle_yaw(dataset, sample_index, 90.0)

    sat_image = sample_90["sat"].unsqueeze(0).to(args.device)
    target_size = tuple(int(x) for x in sample_90["image"].shape[-2:])

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

    coords_map_90 = sample_90["coords_map"].unsqueeze(0).to(args.device)
    logger.info("Materializing lazy reading blocks before loading checkpoint")
    _materialize_lazy_modules(model, sat_image, coords_map_90, target_size)

    load_model_checkpoint(
        model,
        Path(args.checkpoint),
        args.device,
    )
    model.eval()

    output_root = Path(args.output_dir)
    sample_dir = output_root / f"{args.drive}_frame_{args.frame_id:010d}_seed_{args.seed}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sat_resized = _resize_satellite_for_front(sample_90["sat"], target_size[0])
    _tensor_to_pil(sat_resized).save(sample_dir / "satellite.png")

    summary_rows: List[Image.Image] = []
    summary_rows.append(_compose_panels([("satellite", sat_resized)]))

    generator_device = args.device if args.device.startswith("cuda") else "cpu"

    for yaw in args.vehicle_yaws:
        logger.info(f"Generating yaw={yaw:g}")
        sample = _get_sample_with_vehicle_yaw(dataset, sample_index, yaw)
        real_image = sample["image"]
        coords_map = sample["coords_map"].unsqueeze(0).to(args.device)
        plucker_map = sample["plucker_map"].unsqueeze(0).to(args.device)

        generator = torch.Generator(device=generator_device)
        generator.manual_seed(args.seed)
        generated = model.generate(
            sat_image,
            coords_map=coords_map,
            plucker_map=plucker_map,
            target_size=target_size,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )[0].cpu()

        yaw_token = str(yaw).replace("-", "m").replace(".", "p")
        _tensor_to_pil(real_image).save(sample_dir / f"real_yaw_{yaw_token}.png")
        _tensor_to_pil(generated).save(sample_dir / f"generated_yaw_{yaw_token}.png")

        summary_rows.append(_compose_panels([
            (f"real yaw={yaw:g}", real_image),
            (f"generated yaw={yaw:g}", generated),
        ]))

    summary = _stack_panel_rows(summary_rows)
    summary.save(sample_dir / "summary.png")

    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_model": args.base_model,
        "dataset_split": args.dataset_split,
        "drive": args.drive,
        "frame_id": int(args.frame_id),
        "vehicle_yaws": [float(yaw) for yaw in args.vehicle_yaws],
        "seed": int(args.seed),
        "inference_steps": int(args.inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "sample_index": int(sample_index),
    }
    with open(sample_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    logger.info(f"Saved yaw sweep visualizations to: {sample_dir}")


if __name__ == "__main__":
    main()
