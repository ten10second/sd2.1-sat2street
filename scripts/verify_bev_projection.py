#!/usr/bin/env python3
"""Visualize camera-dependent BEV coordinates for one dataset sample."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.kitti360d_dataset import Kitti360dDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify BEV projection for one sample")
    parser.add_argument("--drive", type=str, required=True, help="Path to one KITTI-360 drive")
    parser.add_argument("--frame", type=int, default=1, help="Frame id to inspect")
    parser.add_argument(
        "--mode",
        type=str,
        default="front",
        choices=["front", "fisheye_virtual"],
        help="Dataset view mode",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        choices=["image_02", "image_03"],
        help="Fisheye camera to use in virtual mode",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="Relative yaw in degrees for fisheye_virtual mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="verify_bev_projection.png",
        help="Path to save the visualization",
    )
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> Kitti360dDataset:
    return Kitti360dDataset(
        drives=args.drive,
        frames=[args.frame],
        mode=args.mode,
        fisheye_camera=args.camera,
        fisheye_relative_yaw_deg=args.yaw,
        front_resize=(640, 256),
        virtual_size=(640, 256),
        random_fisheye_relative_yaw=False,
    )


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    sample = dataset[0]

    bev_xy = sample["front_bev_xy"]
    image = sample["image"].permute(1, 2, 0).cpu()
    valid_mask = ((bev_xy[0] != 0) | (bev_xy[1] != 0)).to(torch.float32)

    height, width = bev_xy.shape[1:]
    bottom_center = bev_xy[:, height - 1, width // 2]
    physical_camera = sample["meta"].get("physical_camera")
    camera_height_m = sample["meta"].get("camera_height_m")

    print(f"mode={args.mode}, camera={args.camera}, yaw={args.yaw}")
    print(f"frame={args.frame}, size={width}x{height}")
    print(f"dummy={sample['meta'].get('dummy', False)}")
    print(f"physical_camera={physical_camera}, camera_height_m={camera_height_m}")
    print(f"bottom-center bev = x={float(bottom_center[0]):.4f}, y={float(bottom_center[1]):.4f}")
    print(f"valid ratio = {float(valid_mask.mean()):.4f}")
    print("interpretation:")
    print("  front view: expect y > 0")
    print("  left view: expect x < 0")
    print("  right view: expect x > 0")
    print("reference heights:")
    print("  image_00/front  = 1.55m")
    print("  image_02/image_03 fisheye = 1.95m")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].imshow(image.numpy())
    axes[0, 0].set_title("Target image")
    axes[0, 0].axis("off")

    im_x = axes[0, 1].imshow(bev_xy[0].cpu().numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes[0, 1].set_title("BEV X")
    axes[0, 1].axis("off")
    plt.colorbar(im_x, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_y = axes[1, 0].imshow(bev_xy[1].cpu().numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes[1, 0].set_title("BEV Y")
    axes[1, 0].axis("off")
    plt.colorbar(im_y, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(valid_mask.cpu().numpy(), cmap="gray")
    axes[1, 1].set_title(f"Valid mask ({100.0 * float(valid_mask.mean()):.1f}%)")
    axes[1, 1].axis("off")

    plt.suptitle(
        f"BEV Projection: mode={args.mode}, camera={args.camera}, yaw={args.yaw}, "
        f"height={camera_height_m}"
    )
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
