"""Quick diagnostic: generate with and without sat conditioning to verify injection."""
from __future__ import annotations

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import argparse
import numpy as np
from PIL import Image

from models.sd_model import create_sd_model, load_model_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/media/shizhm/Lenovo/KITTI-360")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="debug_condition.png")
    args = parser.parse_args()

    device = args.device

    # Create model
    print("Creating model...")
    model = create_sd_model(
        base_model="sd2-community/stable-diffusion-2-1-base",
        freeze_base=True,
        torch_dtype=torch.float32,
        perspective_pe_enabled=True,
        query_uv_pe_enabled=False,
        query_geometry_bias_enabled=True,
    )
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_model_checkpoint(model, Path(args.checkpoint), device)

    # Load a single sample from the dataset
    print("Loading dataset sample...")
    from data.kitti360d_dataset import Kitti360dDataset
    from torch.utils.data import DataLoader

    data_path = Path(args.data_dir)
    dataset = Kitti360dDataset(
        drives=str(data_path / "2013_05_28_drive_0003_sync"),
        frames=list(range(0, 100)),
        mode="fisheye_virtual",
        yaw_mode="vehicle_relative",
        vehicle_yaw_sampling="fixed_list",
        vehicle_yaw_fixed_list=["front", 60.0],
        view_set="single",
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    sat_images = batch["sat"].to(device)  # (B, 3, 512, 512)
    target_images = batch["image"].to(device)  # (B, 3, 256, 256)
    target_size = (int(target_images.shape[2]), int(target_images.shape[3]))
    K = batch["K"].to(device)
    T_cam_to_world = batch["T_cam_to_world"].to(device)
    T_imu_to_world = batch["T_imu_to_world"].to(device)
    camera_height_m = batch["camera_height_m"].to(device)

    model.eval()
    with torch.no_grad():
        # Encode satellite
        sat_state = model.encode_satellite(
            sat_images, K=K, T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world, camera_height_m=camera_height_m,
            image_size=target_size,
        )

        print(f"\nSat tokens stats:")
        print(f"  shape: {sat_state.tokens.shape}")
        print(f"  mean: {sat_state.tokens.mean().item():.4f}")
        print(f"  std:  {sat_state.tokens.std().item():.4f}")
        print(f"  min:  {sat_state.tokens.min().item():.4f}")
        print(f"  max:  {sat_state.tokens.max().item():.4f}")
        print(f"  has_nan: {torch.isnan(sat_state.tokens).any().item()}")
        print(f"  has_inf: {torch.isinf(sat_state.tokens).any().item()}")
        if sat_state.perspective_valid is not None:
            print(f"  perspective_valid ratio: {sat_state.perspective_valid.float().mean().item():.3f}")

        # Generate WITH conditioning (guidance_scale=1.0 to see pure conditional)
        print("\nGenerating WITH conditioning (guidance_scale=1.0)...")
        gen_cond, _ = model.generate_with_satellite_state(
            sat_state, target_size=target_size,
            num_inference_steps=20, guidance_scale=1.0,
            generator=torch.Generator(device=device).manual_seed(42),
        )

        # Generate WITHOUT conditioning (zero tokens)
        print("Generating WITHOUT conditioning (zero satellite conditioning)...")
        gen_uncond, _ = model.generate_with_satellite_state(
            sat_state, target_size=target_size,
            num_inference_steps=20, guidance_scale=1.0,
            generator=torch.Generator(device=device).manual_seed(42),
            sat_condition_mode="zero",
        )

        # Compute difference
        diff = (gen_cond.float() - gen_uncond.float()).abs()
        print(f"\nL1 diff between cond and uncond: {diff.mean().item():.4f}")
        print(f"  (if > 0.01, condition IS injecting; if ~0, condition has NO effect)")

        # Save comparison image
        def to_pil(t):
            t = t.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
            return Image.fromarray((t * 255).astype(np.uint8))

        h, w = target_size
        panel = Image.new("RGB", (w * 5, h))
        panel.paste(to_pil(sat_images[0:1, :, :256, :256].cpu()), (0, 0))  # sat crop
        panel.paste(to_pil(target_images[0:1]), (w, 0))  # ground truth
        panel.paste(to_pil(gen_cond[0:1]), (w * 2, 0))  # conditioned gen
        panel.paste(to_pil(gen_uncond[0:1]), (w * 3, 0))  # unconditioned gen
        panel.paste(to_pil(diff[0:1].div(diff.max() + 1e-8)), (w * 4, 0))  # diff map
        panel.save(args.output)
        print(f"\nSaved: {args.output} (sat | GT | cond | uncond | diff)")


if __name__ == "__main__":
    main()
