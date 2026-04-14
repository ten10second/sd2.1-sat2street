#!/usr/bin/env python3
"""
Inference script for satellite-to-frontview generation.

This script allows you to generate frontview images from satellite images using
a trained model.
"""

import argparse
import torch
import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from models.sd_trainer import create_sd_model, load_model_checkpoint, SatelliteConditionedSDModel
from data.kitti360d_dataset import Kitti360dDataset


def main():
    parser = argparse.ArgumentParser(
        description="Inference for satellite-to-frontview generation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/media/zhimiao/Lenovo/KITTI-360",
        help="Path to KITTI-360 dataset",
    )
    parser.add_argument(
        "--drive_dir", type=str,
        help="Specific drive directory to process (optional)",
    )
    parser.add_argument(
        "--frame_id", type=int,
        help="Specific frame to process (optional)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./inference_results",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--num_inference_steps", type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed", type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save_real", type=bool,
        default=True,
        help="Whether to save real images alongside generated",
    )

    args = parser.parse_args()

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated").mkdir(exist_ok=True)
    if args.save_real:
        (output_dir / "real").mkdir(exist_ok=True)
    (output_dir / "satellite").mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = create_sd_model(enable_plucker_guider=True)

    # Load checkpoint
    load_model_checkpoint(
        model,
        Path(args.checkpoint),
        device,
        allow_missing_prefixes=("plucker_guider.",),
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Prepare data
    if args.drive_dir and args.frame_id:
        # Process single frame
        print(f"Processing single frame: {args.drive_dir}, frame {args.frame_id}")
        process_single_frame(
            model,
            args.drive_dir,
            args.frame_id,
            output_dir,
            device,
            args.num_inference_steps,
            args.save_real,
        )
    elif args.drive_dir:
        # Process entire drive
        print(f"Processing drive: {args.drive_dir}")
        process_drive(
            model,
            args.drive_dir,
            output_dir,
            device,
            args.num_inference_steps,
            args.batch_size,
            args.save_real,
        )
    else:
        # Process validation split
        print(f"Processing validation split")
        process_validation(
            model,
            args.data_dir,
            output_dir,
            device,
            args.num_inference_steps,
            args.batch_size,
            args.save_real,
        )

    print("Inference completed!")
    print(f"Results saved to: {args.output_dir}")


def process_single_frame(
    model, drive_dir, frame_id, output_dir, device,
    num_inference_steps, save_real,
):
    """Process a single frame."""

    dataset = Kitti360dDataset(
        drives=drive_dir,
        frames=[frame_id],
        mode="front",
        virtual_size=(640, 256),
        front_resize=(640, 256),
        random_fisheye_relative_yaw=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    for batch in dataloader:
        # Get satellite image
        sat_image = batch['sat'].to(device)
        real_image = batch['image']
        coords_map = batch.get('coords_map', None)
        if coords_map is not None:
            coords_map = coords_map.to(device)
        plucker_map = batch.get('plucker_map', None)
        if plucker_map is not None:
            plucker_map = plucker_map.to(device)

        # Generate frontview
        with torch.no_grad():
            generated = model.generate(
                sat_image,
                coords_map=coords_map,
                plucker_map=plucker_map,
                target_size=tuple(real_image.shape[-2:]),
                num_inference_steps=num_inference_steps,
            )

        # Post-process
        generated = generated.cpu().squeeze(0)
        generated = (generated * 255).clamp(0, 255).byte()

        # Convert to PIL Image
        generated_img = Image.fromarray(generated.permute(1, 2, 0).numpy())
        generated_path = output_dir / "generated" / f"frame_{frame_id:010d}.png"
        generated_img.save(generated_path)

        # Save satellite image
        sat_img = sat_image.cpu().squeeze(0)
        sat_img = (sat_img * 255).clamp(0, 255).byte()
        sat_img = Image.fromarray(sat_img.permute(1, 2, 0).numpy())
        sat_path = output_dir / "satellite" / f"frame_{frame_id:010d}.png"
        sat_img.save(sat_path)

        # Save real image if requested
        if save_real:
            real_img = (real_image.squeeze(0) * 255).clamp(0, 255).byte()
            real_img = Image.fromarray(real_img.permute(1, 2, 0).numpy())
            real_path = output_dir / "real" / f"frame_{frame_id:010d}.png"
            real_img.save(real_path)

        print(f"Generated image saved to: {generated_path}")


def process_drive(
    model, drive_dir, output_dir, device,
    num_inference_steps, batch_size, save_real,
):
    """Process an entire drive directory."""

    dataset = Kitti360dDataset(
        drives=drive_dir,
        mode="front",
        virtual_size=(640, 256),
        front_resize=(640, 256),
        random_fisheye_relative_yaw=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    progress_bar = tqdm(dataloader, desc="Processing drive")

    for batch in progress_bar:
        # Get satellite images and metadata
        sat_images = batch['sat'].to(device)
        real_images = batch['image']
        frame_ids = batch['frame_id']
        coords_map = batch.get('coords_map', None)
        if coords_map is not None:
            coords_map = coords_map.to(device)
        plucker_map = batch.get('plucker_map', None)
        if plucker_map is not None:
            plucker_map = plucker_map.to(device)

        # Generate frontviews
        with torch.no_grad():
            generated = model.generate(
                sat_images,
                coords_map=coords_map,
                plucker_map=plucker_map,
                target_size=tuple(real_images.shape[-2:]),
                num_inference_steps=num_inference_steps,
            )

        # Post-process and save
        for i in range(generated.shape[0]):
            # Generated image
            gen_img = generated[i].cpu()
            gen_img = (gen_img * 255).clamp(0, 255).byte()
            gen_img = Image.fromarray(gen_img.permute(1, 2, 0).numpy())
            gen_path = output_dir / "generated" / f"frame_{frame_ids[i]:010d}.png"
            gen_img.save(gen_path)

            # Satellite image
            sat_img = sat_images[i].cpu()
            sat_img = (sat_img * 255).clamp(0, 255).byte()
            sat_img = Image.fromarray(sat_img.permute(1, 2, 0).numpy())
            sat_path = output_dir / "satellite" / f"frame_{frame_ids[i]:010d}.png"
            sat_img.save(sat_path)

            # Real image if requested
            if save_real:
                real_img = real_images[i]
                real_img = (real_img * 255).clamp(0, 255).byte()
                real_img = Image.fromarray(real_img.permute(1, 2, 0).numpy())
                real_path = output_dir / "real" / f"frame_{frame_ids[i]:010d}.png"
                real_img.save(real_path)


def process_validation(
    model, data_dir, output_dir, device,
    num_inference_steps, batch_size, save_real,
):
    """Process validation split."""

    # Find drive directories
    data_path = Path(data_dir)
    drive_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("2013_")])

    # Use last 10% for validation
    num_val = int(0.1 * len(drive_dirs))
    val_dirs = drive_dirs[-num_val:]

    dataset = Kitti360dDataset(
        drives=val_dirs,
        mode="front",
        virtual_size=(640, 256),
        front_resize=(640, 256),
        random_fisheye_relative_yaw=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    progress_bar = tqdm(dataloader, desc="Processing validation split")

    for batch in progress_bar:
        # Get satellite images and metadata
        sat_images = batch['sat'].to(device)
        real_images = batch['image']
        frame_ids = batch['frame_id']
        drive_names = batch['drive']
        coords_map = batch.get('coords_map', None)
        if coords_map is not None:
            coords_map = coords_map.to(device)
        plucker_map = batch.get('plucker_map', None)
        if plucker_map is not None:
            plucker_map = plucker_map.to(device)

        # Generate frontviews
        with torch.no_grad():
            generated = model.generate(
                sat_images,
                coords_map=coords_map,
                plucker_map=plucker_map,
                target_size=tuple(real_images.shape[-2:]),
                num_inference_steps=num_inference_steps,
            )

        # Post-process and save
        for i in range(generated.shape[0]):
            # Create drive-specific directory
            drive_dir = output_dir / "generated" / drive_names[i]
            drive_dir.mkdir(exist_ok=True)

            # Generated image
            gen_img = generated[i].cpu()
            gen_img = (gen_img * 255).clamp(0, 255).byte()
            gen_img = Image.fromarray(gen_img.permute(1, 2, 0).numpy())
            gen_path = drive_dir / f"frame_{frame_ids[i]:010d}.png"
            gen_img.save(gen_path)

            # Satellite image
            sat_img = sat_images[i].cpu()
            sat_img = (sat_img * 255).clamp(0, 255).byte()
            sat_img = Image.fromarray(sat_img.permute(1, 2, 0).numpy())
            sat_dir = output_dir / "satellite" / drive_names[i]
            sat_dir.mkdir(exist_ok=True)
            sat_path = sat_dir / f"frame_{frame_ids[i]:010d}.png"
            sat_img.save(sat_path)

            # Real image if requested
            if save_real:
                real_dir = output_dir / "real" / drive_names[i]
                real_dir.mkdir(exist_ok=True)

                real_img = real_images[i]
                real_img = (real_img * 255).clamp(0, 255).byte()
                real_img = Image.fromarray(real_img.permute(1, 2, 0).numpy())
                real_path = real_dir / f"frame_{frame_ids[i]:010d}.png"
                real_img.save(real_path)


if __name__ == "__main__":
    main()
