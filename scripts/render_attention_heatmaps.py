#!/usr/bin/env python3
"""
Render saved attention heatmap tensors onto satellite images.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render attention heatmaps saved during training")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing attention_heatmaps/epoch_xxxx/*.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for rendered PNGs. Defaults to <input_dir>_rendered",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of .pt files to render",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=8,
        help="Maximum number of query tokens to render per file",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay alpha in [0, 1]",
    )
    return parser.parse_args()


def _iter_pt_files(root: Path, max_files: Optional[int]) -> List[Path]:
    files = sorted(root.rglob("*.pt"))
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    return files


def _tensor_chw_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError(f"Expected CHW image, got {list(image.shape)}")
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    array = image.permute(1, 2, 0).cpu().numpy()
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _normalize_heatmap(heatmap: torch.Tensor) -> np.ndarray:
    heatmap = heatmap.float().cpu()
    heatmap = heatmap - heatmap.min()
    denom = float(heatmap.max().item())
    if denom > 1e-8:
        heatmap = heatmap / denom
    return heatmap.numpy()


def _colormap_red_yellow(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.full_like(values, 255.0)
    green = 255.0 * values
    blue = np.zeros_like(values)
    rgb = np.stack([red, green, blue], axis=-1)
    return rgb.astype(np.uint8)


def _xy_to_pixel(xy: Sequence[float], width: int, height: int) -> Tuple[float, float]:
    x = float(xy[0])
    y = float(xy[1])
    px = (x + 1.0) * 0.5 * float(max(1, width - 1))
    py = (1.0 - (y + 1.0) * 0.5) * float(max(1, height - 1))
    return px, py


def _overlay_single_query(
    sat_image: Image.Image,
    heatmap: torch.Tensor,
    view_xy: Optional[torch.Tensor],
    title: str,
    alpha: float,
) -> Image.Image:
    base = sat_image.convert("RGB")
    heatmap_np = _normalize_heatmap(heatmap)
    heatmap_img = Image.fromarray(_colormap_red_yellow(heatmap_np)).resize(base.size, resample=Image.BILINEAR)
    blended = Image.blend(base, heatmap_img, alpha=max(0.0, min(1.0, alpha)))
    draw = ImageDraw.Draw(blended, "RGBA")

    if view_xy is not None and torch.is_tensor(view_xy) and view_xy.numel() >= 2:
        px, py = _xy_to_pixel(view_xy.tolist(), blended.width, blended.height)
        r = 5
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(0, 255, 255, 220), outline=(0, 0, 0, 255))

    draw.rectangle((4, 4, min(blended.width - 4, 220), 26), fill=(0, 0, 0, 160))
    draw.text((8, 8), title, fill=(255, 255, 255, 255))
    return blended


def _compose_grid(images: Sequence[Image.Image], columns: int = 4) -> Image.Image:
    if not images:
        raise ValueError("No images to compose")
    columns = max(1, min(columns, len(images)))
    rows = int(math.ceil(len(images) / columns))
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (columns * width, rows * height), color=(255, 255, 255))
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        canvas.paste(image, (col * width, row * height))
    return canvas


def _render_one(pt_path: Path, output_dir: Path, max_queries: int, alpha: float) -> None:
    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(pt_path, map_location="cpu")
    sat_image = payload.get("sat_image")
    attention_grid = payload.get("attention_grid")
    view_xy = payload.get("view_xy")
    frame_ids = payload.get("frame_id")
    global_step = payload.get("global_step")
    epoch = payload.get("epoch")

    if sat_image is None or attention_grid is None:
        raise ValueError(f"{pt_path} does not contain sat_image and attention_grid")
    if not torch.is_tensor(sat_image) or not torch.is_tensor(attention_grid):
        raise ValueError(f"{pt_path} contains invalid sat_image/attention_grid")

    batch = int(attention_grid.shape[0])
    num_queries = int(attention_grid.shape[1])
    queries_to_render = min(max_queries, num_queries)

    rel_parent = pt_path.parent.name
    stem = pt_path.stem
    file_output_dir = output_dir / rel_parent / stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(batch):
        sat_pil = _tensor_chw_to_pil(sat_image[batch_idx])
        overlays = []
        for query_idx in range(queries_to_render):
            query_xy = None
            if torch.is_tensor(view_xy) and batch_idx < view_xy.shape[0] and query_idx < view_xy.shape[1]:
                query_xy = view_xy[batch_idx, query_idx]
            title = f"q{query_idx:02d}"
            overlay = _overlay_single_query(
                sat_image=sat_pil,
                heatmap=attention_grid[batch_idx, query_idx],
                view_xy=query_xy,
                title=title,
                alpha=alpha,
            )
            overlay.save(file_output_dir / f"batch_{batch_idx:02d}_query_{query_idx:02d}.png")
            overlays.append(overlay)

        sheet = _compose_grid(overlays, columns=min(4, max(1, queries_to_render)))
        draw = ImageDraw.Draw(sheet, "RGBA")
        title = f"{rel_parent}/{stem} batch={batch_idx}"
        if frame_ids is not None and batch_idx < len(frame_ids):
            title += f" frame={frame_ids[batch_idx]}"
        if epoch is not None:
            title += f" epoch={epoch}"
        if global_step is not None:
            title += f" step={global_step}"
        draw.rectangle((4, 4, min(sheet.width - 4, 600), 28), fill=(0, 0, 0, 160))
        draw.text((8, 8), title, fill=(255, 255, 255, 255))
        sheet.save(file_output_dir / f"batch_{batch_idx:02d}_grid.png")


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir.parent / f"{input_dir.name}_rendered"
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = _iter_pt_files(input_dir, args.max_files)
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {input_dir}")

    for pt_path in pt_files:
        _render_one(
            pt_path=pt_path,
            output_dir=output_dir,
            max_queries=max(1, int(args.max_queries)),
            alpha=float(args.alpha),
        )


if __name__ == "__main__":
    main()
