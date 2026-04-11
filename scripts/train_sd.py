#!/usr/bin/env python3
"""
Training script for KITTI-360 Satellite-to-Frontview Generation with Stable Diffusion.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
from pathlib import Path

from data.kitti360_dataset import Kitti360Dataset
from models.sd_model import load_sd_model
from utils.geometry.pose_encoding import PoseEncoder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def setup_logging(log_dir: str):
    """Set up logging."""
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(log_dir)/'train.log'),
            logging.StreamHandler(),
        ],
    )


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set up logging
    setup_logging(config['logging']['log_dir'])
    logger = logging.getLogger(__name__)
    logger.info(f"Training configuration: {yaml.dump(config)}")

    # Set random seed
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Create dataset and dataloader
    logger.info("Creating dataset and dataloader")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['data']['image_size'][0], config['data']['image_size'][1])),
        transforms.ToTensor(),
    ])

    dataset = Kitti360Dataset(
        data_dir=config['data']['data_dir'],
        split='train',
        image_size=config['data']['image_size'],
        sat_size=config['data']['sat_size'],
        resolution=config['data']['resolution'],
        transform=transform,
        use_cache=config['data'].get('use_cache', False),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )

    # Load model
    logger.info(f"Loading model: {config['model']['base_model']}")
    model = load_sd_model(
        base_model=config['model']['base_model'],
        freeze_base=config['model']['freeze_base'],
    )
    model.to(args.device)

    # Create optimizer and scheduler
    logger.info("Creating optimizer and scheduler")
    # Only optimize unfrozen parameters
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Create scheduler
    if config['training']['scheduler'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    elif config['training']['scheduler'] == 'linear':
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=config['training']['epochs'])
    else:
        raise ValueError(f"Unknown scheduler: {config['training']['scheduler']}")

    # Resume training if needed
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # Training loop
    logger.info("Starting training")
    model.train()

    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_loss = 0.0
        recon_loss_total = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # Move data to device
            sat_images = batch['sat_image'].to(args.device)
            front_images = batch['front_image'].to(args.device)

            # Generate images
            output = model(
                sat_images=sat_images,
                num_inference_steps=config.get('inference_steps', 20),
            )

            # Compute reconstruction loss
            recon_loss = nn.functional.mse_loss(output.images, front_images)

            # Compute total loss
            total_loss = config['training']['loss']['lambda_recon'] * recon_loss

            # Backward pass
            total_loss.backward()

            # Gradient clip
            if config['training'].get('gradient_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])

            optimizer.step()

            # Update metrics
            epoch_loss += total_loss.item()
            recon_loss_total += recon_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
            })

        # Step scheduler
        scheduler.step()

        # Log epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = recon_loss_total / len(dataloader)

        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']}: "
            f"Loss={avg_loss:.4f}, "
            f"Recon={avg_recon:.4f}"
        )

        # Save checkpoint
        if (epoch + 1) % config['checkpoint']['save_every'] == 0 or (epoch + 1) == config['training']['epochs']:
            checkpoint_dir = Path(config['checkpoint']['save_dir'])
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, checkpoint_path)

            logger.info(f"Checkpoint saved: {checkpoint_path}")


if __name__ == '__main__':
    main()
