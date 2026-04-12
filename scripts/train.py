#!/usr/bin/env python3
"""
Training script for satellite-to-frontview generation using Stable Diffusion.

This script uses the simplified trainer interface.
"""

import sys
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import os
import torch
import numpy as np
import random
import logging
import yaml
from pathlib import Path
from typing import List, Tuple

from models.sd_trainer import create_sd_model, SDTrainer
from data.kitti360d_dataset import Kitti360dDataset
from torch.utils.data import DataLoader, default_collate


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler(),
    ],
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
        if key == "meta" or any(value is None for value in values):
            collated[key] = values
            continue
        collated[key] = default_collate(values)
    return collated


def main():
    parser = argparse.ArgumentParser(
        description="Train Stable Diffusion for satellite-to-frontview generation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/media/zhimiao/Lenovo/KITTI-360",
        help="Path to KITTI-360 data",
    )
    parser.add_argument(
        "--split_yaml", type=str, default=None,
        help="Path to split yaml (defaults to <data_dir>/train_test_split_config.yaml)",
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
        "--text_anchor_prompt", type=str, default="",
        help="Fixed text prompt used for the main UNet cross-attention branch.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.0,
        help="Guidance scale used for training visualizations. 1.0 disables CFG.",
    )
    parser.add_argument(
        "--visualize_every", type=int, default=1,
        help="Save fixed-sample visualization comparisons every N epochs. Set 0 to disable.",
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
        "--hf_endpoint", type=str, default=DEFAULT_HF_ENDPOINT,
        help="Hugging Face endpoint. Defaults to hf-mirror for first-time downloads.",
    )
    parser.add_argument(
        "--hf_home", type=str, default=str(DEFAULT_HF_HOME),
        help="Local Hugging Face cache directory.",
    )

    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    logger.info(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    logger.info(f"HF_HOME={os.environ['HF_HOME']}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
        args.mixed_precision = "no"

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Training configuration: {args}")

    # Load data
    logger.info(f"Loading data from: {args.data_dir}")

    data_path = Path(args.data_dir)
    split_yaml = Path(args.split_yaml) if args.split_yaml is not None else data_path / "train_test_split_config.yaml"
    train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(data_path, split_yaml)

    logger.info(f"Loaded split file: {split_yaml}")
    logger.info(f"Training on {len(train_dirs)} drives, validating on {len(val_dirs)} drives")

    train_dataset = Kitti360dDataset(
        drives=train_dirs,
        frames=train_frames,
        mode="front",
        virtual_size=(640, 256),
        front_resize=(640, 256),
        front_center_crop=None,
        random_fisheye_relative_yaw=False,
        seed=args.seed,
        return_bgr=False,
    )

    val_dataset = Kitti360dDataset(
        drives=val_dirs,
        frames=val_frames,
        mode="front",
        virtual_size=(640, 256),
        front_resize=(640, 256),
        front_center_crop=None,
        random_fisheye_relative_yaw=False,
        seed=args.seed,
        return_bgr=False,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_safe_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_safe_collate,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Load model
    logger.info("Loading model...")
    model_torch_dtype = None
    if args.device.startswith("cuda") and args.mixed_precision == "fp16":
        model_torch_dtype = torch.float16
    elif args.device.startswith("cuda") and args.mixed_precision == "bf16":
        model_torch_dtype = torch.bfloat16

    model = create_sd_model(
        base_model=args.base_model,
        freeze_base=True,
        reading_block_config={"enable": True},
        revision=args.base_model_revision,
        torch_dtype=model_torch_dtype,
        cond_drop_prob=args.cond_drop_prob,
        text_anchor_prompt=args.text_anchor_prompt,
    )
    if hasattr(model.unet, "enable_gradient_checkpointing"):
        model.unet.enable_gradient_checkpointing()
        logger.info("Enabled UNet gradient checkpointing")
    if hasattr(model.unet, "set_attention_slice"):
        model.unet.set_attention_slice("auto")
        logger.info("Enabled UNet attention slicing")
    if hasattr(model.vae, "enable_slicing"):
        model.vae.enable_slicing()
        logger.info("Enabled VAE slicing")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = SDTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=args.lr,
        weight_decay=1e-4,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_epochs=args.warmup,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.output_dir,
        save_every=10,
        log_every=100,
        device=args.device,
        use_wandb=False,
        project_name="kitti360_sd",
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        visualize_every=args.visualize_every,
        num_visualizations=args.num_visualizations,
        visualization_inference_steps=args.visualization_inference_steps,
        visualization_guidance_scale=args.guidance_scale,
        visualization_seed=args.visualization_seed,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from=args.resume)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
