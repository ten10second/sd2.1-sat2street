"""Diagnose perspective PE: check LayerNorm weights & perspective_valid ratio."""
from __future__ import annotations
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import argparse
import numpy as np
from models.sd_model import create_sd_model, load_model_checkpoint
from data.kitti360d_dataset import Kitti360dDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/media/shizhm/Lenovo/KITTI-360")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # 1. Check LayerNorm weights in checkpoint
    print("=" * 60)
    print("1. Checking checkpoint LayerNorm weights in perspective_pos_encoder")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    pe_keys = [k for k in state_dict.keys() if "perspective_pos_encoder" in k]
    if not pe_keys:
        print("  NO perspective_pos_encoder keys found in checkpoint!")
    for k in sorted(pe_keys):
        w = state_dict[k]
        is_layer_norm_weight = "mlp.1.weight" in k or "mlp.4.weight" in k
        is_layer_norm_bias = "mlp.1.bias" in k or "mlp.4.bias" in k
        marker = ""
        if is_layer_norm_weight:
            diff_from_ones = (w - 1.0).abs().max().item()
            marker = f"  ← LayerNorm weight, max|w-1|={diff_from_ones:.8f}"
        elif is_layer_norm_bias:
            diff_from_zero = w.abs().max().item()
            marker = f"  ← LayerNorm bias, max|b|={diff_from_zero:.8f}"
        print(f"  {k}: shape={tuple(w.shape)}, mean={w.float().mean():.6f}, std={w.float().std():.6f}{marker}")

    # Also check other MLP layers
    print()
    for k in sorted(pe_keys):
        w = state_dict[k]
        if "mlp.0.weight" in k:
            print(f"  {k}: mean={w.float().mean():.6f}, std={w.float().std():.6f}  ← Linear(2→D)")
        elif "mlp.3.weight" in k:
            print(f"  {k}: mean={w.float().mean():.6f}, std={w.float().std():.6f}  ← Linear(D→D)")

    # 2. Check perspective_valid ratio on training data
    print()
    print("=" * 60)
    print("2. Checking perspective_valid ratio on training samples")

    data_path = Path(args.data_dir)
    dataset = Kitti360dDataset(
        drives=str(data_path / "2013_05_28_drive_0003_sync"),
        frames=list(range(0, 200)),  # first 200 training frames
        mode="fisheye_virtual",
        yaw_mode="vehicle_relative",
        vehicle_yaw_sampling="fixed_list",
        vehicle_yaw_fixed_list=["front", 60.0, 90.0, 120.0, -60.0, -90.0, -120.0],
        view_set="single",
        seed=42,
    )

    valid_ratios = []
    num_zero = 0
    shown_nonzero = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        sat = sample["sat"]
        K = sample["K"]
        T_cam_to_world = sample["T_cam_to_world"]
        T_imu_to_world = sample["T_imu_to_world"]
        camera_height_m = sample["camera_height_m"]
        target_size = tuple(sample["image"].shape[-2:])

        # Compute perspective_uv on CPU
        from models.encoders.perspective_position_encoder import compute_sat_patch_perspective_uv

        # Manual BEV coords
        patch_size = 16
        sat_resolution = 0.2
        sat_size = 512
        H, W = sat.shape[-2:]
        patch_h, patch_w = H // patch_size, W // patch_size
        patch_pixel_h = torch.arange(patch_h, dtype=torch.float32) * patch_size + patch_size / 2
        patch_pixel_w = torch.arange(patch_w, dtype=torch.float32) * patch_size + patch_size / 2
        w_grid, h_grid = torch.meshgrid(patch_pixel_w, patch_pixel_h, indexing="xy")
        half_w, half_h = float(W) / 2.0, float(H) / 2.0
        x_meters = (w_grid - half_w) * sat_resolution
        y_meters = (half_h - h_grid) * sat_resolution
        bev_coords = torch.stack([x_meters.reshape(-1), y_meters.reshape(-1)], dim=-1).unsqueeze(0)

        _, perspective_valid = compute_sat_patch_perspective_uv(
            bev_coords=bev_coords,
            K=K.unsqueeze(0),
            T_cam_to_world=T_cam_to_world.unsqueeze(0),
            T_imu_to_world=T_imu_to_world.unsqueeze(0),
            camera_height_m=torch.tensor([float(camera_height_m)], dtype=torch.float32),
            image_w=target_size[1],
            image_h=target_size[0],
        )
        ratio = perspective_valid.float().mean().item()
        valid_ratios.append(ratio)
        if ratio == 0.0:
            num_zero += 1

        if idx < 3:
            ground_z = float(T_cam_to_world[2, 3]) - float(camera_height_m)
            print(
                f"  sample[{idx}]: valid_ratio={ratio:.4f}  "
                f"({perspective_valid.sum().item()}/{perspective_valid.numel()} patches), "
                f"ground_z={ground_z:.3f}"
            )
        if ratio > 0.0 and shown_nonzero < 3:
            ground_z = float(T_cam_to_world[2, 3]) - float(camera_height_m)
            print(
                f"  nonzero sample[{idx}]: valid_ratio={ratio:.4f}  "
                f"({perspective_valid.sum().item()}/{perspective_valid.numel()} patches), "
                f"ground_z={ground_z:.3f}"
            )
            shown_nonzero += 1
        
    valid_ratios = np.array(valid_ratios)
    print(f"\n  Summary over {len(valid_ratios)} samples:")
    print(f"    mean valid ratio: {valid_ratios.mean():.4f}")
    print(f"    min  valid ratio: {valid_ratios.min():.4f}")
    print(f"    max  valid ratio: {valid_ratios.max():.4f}")
    print(f"    std  valid ratio: {valid_ratios.std():.4f}")
    print(f"    samples with 0 valid: {num_zero}/{len(valid_ratios)}")

    if valid_ratios.mean() < 0.01:
        print("\n  ⚠️  perspective_valid is nearly all ZERO! UV PE gets no gradient at all!")
    elif valid_ratios.mean() < 0.1:
        print("\n  ⚠️  Very low valid ratio. UV PE signal is very weak.")
    else:
        print("\n  ✓ perspective_valid ratio looks reasonable.")


if __name__ == "__main__":
    main()
