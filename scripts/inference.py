#!/usr/bin/env python3
"""
Inference script for satellite-to-frontview generation with Stable Diffusion.
"""

import argparse
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import logging

from models.sd_model import load_sd_model, SatelliteConditionedSDPipeline
from data.kitti360_dataset import Kitti360Dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/inference.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split (train/val/test)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def setup_logging(log_dir: str):
    """Set up logging."""
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(log_dir)/'inference.log'),
            logging.StreamHandler(),
        ],
    )


def save_image(image: torch.Tensor, output_path: Path):
    """Save tensor image to file."""
    # (C, H, W) -> (H, W, C) -> [0, 255]
    if image.dim() == 3:
        image = image.permute(1, 2, 0)
    image = (image * 255).clamp(0, 255).to(torch.uint8)
    img_pil = Image.fromarray(image.cpu().numpy())
    img_pil.save(output_path)


def main():
    """Main inference function."""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'generated').mkdir(exist_ok=True)
    (output_dir / 'satellite').mkdir(exist_ok=True)
    (output_dir / 'real').mkdir(exist_ok=True)

    # Set up logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model: {config['model']['base_model']}")
    model = load_sd_model(
        base_model=config['model']['base_model'],
        freeze_base=config['model']['freeze_base'],
    )
    model.to(args.device)

    # Load checkpoint if provided
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        # Handle possible keys
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # Create dataset
    logger.info("Creating dataset")
    dataset = Kitti360Dataset(
        data_dir=config['data']['data_dir'],
        split=args.split,
        image_size=config['data']['image_size'],
        sat_size=config['data']['sat_size'],
        resolution=config['data']['resolution'],
        transform=None,
        use_cache=False,
    )

    # Take subset of samples
    if args.num_samples > 0 and args.num_samples < len(dataset):
        import random
        random.seed(42)
        subset_indices = random.sample(range(len(dataset)), args.num_samples)
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )

    # Run inference
    logger.info(f"Running inference on {len(dataset)} samples")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logger.info(f"Processing batch {i+1}/{len(dataloader)}")

            # Move data to device
            sat_images = batch['sat_image'].to(args.device)
            front_images = batch['front_image'].to(args.device)

            # Generate images
            output = model(
                sat_images=sat_images,
                num_inference_steps=config['inference']['num_inference_steps'],
                guidance_scale=config['inference']['guidance_scale'],
                negative_prompt=config['inference'].get('negative_prompt', None),
            )

            # Save images
            for j in range(output.images.shape[0]):
                sample_idx = i * args.batch_size + j
                # Generated image
                save_image(
                    output.images[j],
                    output_dir / 'generated' / f'sample_{sample_idx:04d}.png',
                )
                # Satellite image
                save_image(
                    sat_images[j],
                    output_dir / 'satellite' / f'sample_{sample_idx:04d}.png',
                )
                # Real image
                save_image(
                    front_images[j],
                    output_dir / 'real' / f'sample_{sample_idx:04d}.png',
                )

    # Compute metrics if requested
    if config['validation']['compute_metrics']:
        logger.info("Computing metrics")
        compute_metrics(output_dir, config['validation']['metrics'])

    logger.info("Inference completed")
    logger.info(f"Generated images saved to: {output_dir / 'generated'}")
    logger.info(f"Satellite images saved to: {output_dir / 'satellite'}")
    logger.info(f"Real images saved to: {output_dir / 'real'}")


def compute_metrics(output_dir: Path, metrics: list):
    """Compute evaluation metrics."""
    from metrics.psnr import compute_psnr
    from metrics.ssim import compute_ssim
    from metrics.lpips import compute_lpips

    real_dir = output_dir / 'real'
    generated_dir = output_dir / 'generated'

    real_paths = sorted(list(real_dir.glob('*.png')))
    generated_paths = sorted(list(generated_dir.glob('*.png')))

    assert len(real_paths) == len(generated_paths), "Number of real and generated images must match"

    metrics_dict = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
    }

    for real_path, gen_path in zip(real_paths, generated_paths):
        real_img = cv2.imread(str(real_path))
        gen_img = cv2.imread(str(gen_path))

        if 'psnr' in metrics:
            metrics_dict['psnr'].append(compute_psnr(real_img, gen_img))
        if 'ssim' in metrics:
            metrics_dict['ssim'].append(compute_ssim(real_img, gen_img))
        if 'lpips' in metrics:
            # Need to load images as tensors
            from torchvision.transforms import ToTensor
            to_tensor = ToTensor()
            real_tensor = to_tensor(Image.open(real_path)).unsqueeze(0)
            gen_tensor = to_tensor(Image.open(gen_path)).unsqueeze(0)
            metrics_dict['lpips'].append(compute_lpips(real_tensor, gen_tensor))

    # Compute average metrics
    for metric_name in metrics:
        if metric_name in metrics_dict and len(metrics_dict[metric_name]) > 0:
            avg_val = np.mean(metrics_dict[metric_name])
            print(f"Average {metric_name.upper()}: {avg_val:.4f}")
            with open(output_dir / 'metrics.txt', 'a') as f:
                f.write(f"{metric_name.upper()}: {avg_val:.4f}\n")


if __name__ == '__main__':
    main()
