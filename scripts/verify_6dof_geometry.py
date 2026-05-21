#!/usr/bin/env python3
"""Render virtual poses and overlay projected vehicle-local BEV ground points."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import cv2  # type: ignore
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.kitti360d_dataset import Kitti360dDataset, _make_virtual_rectify_rotation


DEFAULT_POSES: Sequence[Tuple[float, float, float]] = (
    (0.0, 0.0, 0.0),
    (0.0, 10.0, 0.0),
    (0.0, -10.0, 0.0),
    (0.0, 0.0, 10.0),
    (0.0, 0.0, -10.0),
    (30.0, 0.0, 0.0),
    (-30.0, 0.0, 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Geometry-only 6DoF validation. Render one KITTI-360 frame at several "
            "vehicle-relative virtual camera poses, project known vehicle-local "
            "ground points into each perspective image, and save overlays."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/shizhm/Lenovo/KITTI-360",
        help="KITTI-360 root directory.",
    )
    parser.add_argument(
        "--drive",
        type=str,
        default="2013_05_28_drive_0003_sync",
        help="Drive name under --data_dir or an absolute drive path.",
    )
    parser.add_argument("--frame", type=int, default=978, help="Frame id to inspect.")
    parser.add_argument(
        "--source",
        type=str,
        default="front",
        choices=["front", "fisheye"],
        help=(
            "Image source for rendering virtual poses. 'front' uses fixed image_00 "
            "pinhole camera; 'fisheye' uses the fisheye virtual renderer."
        ),
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="image_02",
        choices=["image_02", "image_03"],
        help="Physical fisheye camera used to render the virtual perspective.",
    )
    parser.add_argument(
        "--virtual_size",
        type=int,
        nargs=2,
        default=(640, 256),
        metavar=("W", "H"),
        help="Virtual perspective size.",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=100.0,
        help="Virtual perspective horizontal field of view in degrees.",
    )
    parser.add_argument(
        "--poses",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional pose triples as yaw,pitch,roll degrees. "
            "Default: 0,0,0 0,10,0 0,-10,0 0,0,10 0,0,-10 30,0,0 -30,0,0"
        ),
    )
    parser.add_argument(
        "--forward_m",
        type=float,
        nargs="+",
        default=(5.0, 10.0),
        help="Vehicle-forward distances for projected ground points, in meters.",
    )
    parser.add_argument(
        "--lateral_m",
        type=float,
        nargs="+",
        default=(-2.0, 0.0, 2.0),
        help="Vehicle-right lateral offsets for projected ground points, in meters.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tmp/verify_6dof_geometry",
        help="Directory for per-pose overlays, contact sheet, and summary.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Dataset RNG seed.")
    return parser.parse_args()


def parse_pose_triples(values: Sequence[str] | None) -> List[Tuple[float, float, float]]:
    if not values:
        return list(DEFAULT_POSES)

    poses: List[Tuple[float, float, float]] = []
    for raw in values:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Pose must be yaw,pitch,roll, got: {raw!r}")
        poses.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return poses


def resolve_drive_path(data_dir: Path, drive: str) -> Path:
    drive_path = Path(drive).expanduser()
    if drive_path.is_absolute():
        return drive_path
    return data_dir.expanduser() / drive


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.permute(1, 2, 0)
    array = (image.numpy() * 255.0).round().astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def normalize_xy(vector: np.ndarray, name: str) -> np.ndarray:
    xy = np.asarray(vector[:2], dtype=np.float64)
    norm = float(np.linalg.norm(xy))
    if norm < 1e-8:
        raise ValueError(f"{name} has near-zero XY norm: {xy}")
    return xy / norm


def build_vehicle_ground_points(
    T_imu_to_world: np.ndarray,
    *,
    forward_m: Iterable[float],
    lateral_m: Iterable[float],
    ground_z: float,
) -> List[Dict[str, Any]]:
    """Build vehicle-local ground points and convert them to north-up BEV offsets."""
    imu_xy = T_imu_to_world[:2, 3].astype(np.float64)
    R_imu_to_world = T_imu_to_world[:3, :3].astype(np.float64)

    # KITTI-360 GPS/IMU convention supplied by the user:
    # x = forward, y = right, z = down.
    forward_xy = normalize_xy(R_imu_to_world[:, 0], "IMU forward axis")
    right_xy = normalize_xy(R_imu_to_world[:, 1], "IMU right axis")

    points: List[Dict[str, Any]] = []
    for fwd in forward_m:
        for lat in lateral_m:
            fwd_f = float(fwd)
            lat_f = float(lat)
            bev_offset_xy = forward_xy * fwd_f + right_xy * lat_f
            world_xy = imu_xy + bev_offset_xy
            points.append(
                {
                    "label": f"F{fwd_f:g}/R{lat_f:+g}",
                    "forward_m": fwd_f,
                    "right_m": lat_f,
                    "bev_xy_m": bev_offset_xy.tolist(),
                    "world_xyz": [float(world_xy[0]), float(world_xy[1]), float(ground_z)],
                }
            )
    return points


def project_world_points(
    points: Sequence[Dict[str, Any]],
    *,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    T_world_to_cam = np.linalg.inv(T_cam_to_world.astype(np.float64))
    K = K.astype(np.float64)

    projected: List[Dict[str, Any]] = []
    for point in points:
        world_xyz = np.asarray(point["world_xyz"], dtype=np.float64)
        world_h = np.concatenate([world_xyz, np.ones(1, dtype=np.float64)])
        cam_h = T_world_to_cam @ world_h
        cam_xyz = cam_h[:3]
        pixel_h = K @ cam_xyz

        z = float(cam_xyz[2])
        if abs(z) < 1e-8:
            u = math.inf
            v = math.inf
        else:
            u = float(pixel_h[0] / z)
            v = float(pixel_h[1] / z)

        valid = bool(z > 0.01 and 0.0 <= u < image_w and 0.0 <= v < image_h)
        record = dict(point)
        record.update(
            {
                "cam_xyz": cam_xyz.tolist(),
                "pixel_uv": [u, v],
                "valid": valid,
            }
        )
        projected.append(record)
    return projected


def point_color(point: Dict[str, Any]) -> Tuple[int, int, int]:
    lat = float(point["right_m"])
    if lat < -1e-6:
        return (64, 160, 255)
    if lat > 1e-6:
        return (255, 80, 180)
    return (255, 220, 0)


def draw_overlay(
    image: Image.Image,
    *,
    pose: Tuple[float, float, float],
    projected_points: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    yaw, pitch, roll = pose

    source = meta.get("geometry_source", meta.get("mode", "unknown"))
    camera = meta.get("physical_camera", meta.get("fisheye_camera_used"))
    fisheye_yaw = meta.get("fisheye_relative_yaw_deg_used")
    header = (
        f"yaw={yaw:+g} pitch={pitch:+g} roll={roll:+g} | "
        f"source={source} cam={camera}"
    )
    if fisheye_yaw is not None:
        header += f" fisheye_yaw={fisheye_yaw}"
    draw.rectangle((0, 0, output.width, 23), fill=(0, 0, 0))
    draw.text((6, 5), header, fill=(255, 255, 255))

    # Connect lateral grid rows at the same forward distance.
    by_forward: Dict[float, List[Dict[str, Any]]] = {}
    for point in projected_points:
        by_forward.setdefault(float(point["forward_m"]), []).append(point)
    for points in by_forward.values():
        visible = [p for p in sorted(points, key=lambda item: float(item["right_m"])) if p["valid"]]
        if len(visible) >= 2:
            xy = [(float(p["pixel_uv"][0]), float(p["pixel_uv"][1])) for p in visible]
            draw.line(xy, fill=(255, 255, 255), width=1)

    for point in projected_points:
        u, v = [float(x) for x in point["pixel_uv"]]
        if not point["valid"]:
            continue

        color = point_color(point)
        radius = 5 if abs(float(point["right_m"])) < 1e-6 else 4
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(0, 0, 0), width=1)
        draw.text((u + 7, v - 7), str(point["label"]), fill=color)

    invalid_count = sum(1 for point in projected_points if not point["valid"])
    if invalid_count:
        draw.rectangle((0, output.height - 20, 210, output.height), fill=(0, 0, 0))
        draw.text((6, output.height - 16), f"{invalid_count} projected points off-image", fill=(255, 180, 180))
    return output


def pose_name(pose: Tuple[float, float, float]) -> str:
    def fmt(value: float) -> str:
        prefix = "p" if value >= 0 else "m"
        return f"{prefix}{abs(value):g}".replace(".", "p")

    yaw, pitch, roll = pose
    return f"yaw_{fmt(yaw)}_pitch_{fmt(pitch)}_roll_{fmt(roll)}"


def make_contact_sheet(images: Sequence[Image.Image], *, columns: int = 2, spacing: int = 8) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet")
    columns = max(1, int(columns))
    rows = math.ceil(len(images) / columns)
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    sheet = Image.new(
        "RGB",
        (columns * cell_w + spacing * (columns - 1), rows * cell_h + spacing * (rows - 1)),
        color=(255, 255, 255),
    )
    for idx, image in enumerate(images):
        col = idx % columns
        row = idx // columns
        x = col * (cell_w + spacing)
        y = row * (cell_h + spacing)
        sheet.paste(image, (x, y))
    return sheet


def warp_front_pinhole(
    image: torch.Tensor,
    *,
    K: np.ndarray,
    pose: Tuple[float, float, float],
) -> torch.Tensor:
    """Render a same-center virtual pinhole view from fixed image_00."""
    yaw, pitch, roll = pose
    if abs(yaw) < 1e-8 and abs(pitch) < 1e-8 and abs(roll) < 1e-8:
        return image

    image_np = (
        image.detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    height, width = int(image_np.shape[0]), int(image_np.shape[1])

    R_front_to_virtual = _make_virtual_rectify_rotation(
        float(yaw),
        pitch_deg=float(pitch),
        roll_deg=float(roll),
    )
    K = K.astype(np.float64)
    K_inv = np.linalg.inv(K)

    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=0).reshape(3, -1)

    dirs_virtual = K_inv @ pixels
    dirs_front = R_front_to_virtual.T @ dirs_virtual
    pixels_front = K @ dirs_front
    z = pixels_front[2, :]
    map_x = np.full_like(z, -1.0, dtype=np.float32)
    map_y = np.full_like(z, -1.0, dtype=np.float32)
    valid = z > 1e-8
    map_x[valid] = (pixels_front[0, valid] / z[valid]).astype(np.float32)
    map_y[valid] = (pixels_front[1, valid] / z[valid]).astype(np.float32)
    map_x = map_x.reshape(height, width)
    map_y = map_y.reshape(height, width)

    warped = cv2.remap(
        image_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return torch.from_numpy(warped).to(torch.float32).permute(2, 0, 1) / 255.0


def rotate_front_pose(
    T_front_to_world: np.ndarray,
    pose: Tuple[float, float, float],
) -> np.ndarray:
    """Apply local yaw/pitch/roll delta to image_00 and return virtual cam->world."""
    yaw, pitch, roll = pose
    R_front_to_virtual = _make_virtual_rectify_rotation(
        float(yaw),
        pitch_deg=float(pitch),
        roll_deg=float(roll),
    )
    R_virtual_to_front = R_front_to_virtual.T
    T_virtual_to_world = T_front_to_world.astype(np.float64).copy()
    T_virtual_to_world[:3, :3] = T_front_to_world[:3, :3].astype(np.float64) @ R_virtual_to_front
    return T_virtual_to_world


def build_front_pose_sample(
    *,
    drive_path: Path,
    frame: int,
    pose: Tuple[float, float, float],
    virtual_size: Tuple[int, int],
    seed: int,
) -> Dict[str, Any]:
    dataset = Kitti360dDataset(
        drives=str(drive_path),
        frames=[int(frame)],
        mode="front",
        virtual_size=virtual_size,
        front_resize=virtual_size,
        front_center_crop=None,
        seed=int(seed),
        return_bgr=False,
    )
    sample = dataset[0]
    if bool(sample.get("meta", {}).get("dummy", False)):
        raise RuntimeError(f"Dataset returned a dummy front sample for frame={frame}")

    K = sample["K"].detach().cpu().numpy()
    T_front_to_world = sample["T_cam_to_world"].detach().cpu().numpy()
    T_virtual_to_world = rotate_front_pose(T_front_to_world, pose)

    sample = dict(sample)
    sample["image"] = warp_front_pinhole(sample["image"], K=K, pose=pose)
    sample["T_cam_to_world"] = torch.from_numpy(T_virtual_to_world).to(torch.float32)
    meta = dict(sample.get("meta", {}))
    meta.update(
        {
            "geometry_source": "front",
            "mode": "front",
            "physical_camera": "image_00",
            "virtual_yaw_deg": float(pose[0]),
            "virtual_pitch_deg": float(pose[1]),
            "virtual_roll_deg": float(pose[2]),
        }
    )
    sample["meta"] = meta
    return sample


def build_pose_sample(
    *,
    drive_path: Path,
    frame: int,
    camera: str,
    pose: Tuple[float, float, float],
    virtual_size: Tuple[int, int],
    hfov: float,
    seed: int,
    source: str,
) -> Dict[str, Any]:
    if source == "front":
        return build_front_pose_sample(
            drive_path=drive_path,
            frame=frame,
            pose=pose,
            virtual_size=virtual_size,
            seed=seed,
        )
    if source != "fisheye":
        raise ValueError(f"Unknown source: {source}")

    yaw, pitch, roll = pose
    dataset = Kitti360dDataset(
        drives=str(drive_path),
        frames=[int(frame)],
        mode="fisheye_virtual",
        yaw_mode="vehicle_relative",
        fisheye_camera=camera,
        vehicle_relative_yaw_deg=float(yaw),
        random_vehicle_relative_yaw=False,
        random_fisheye_relative_yaw=False,
        virtual_hfov_deg=float(hfov),
        virtual_size=virtual_size,
        front_resize=virtual_size,
        pitch_deg=float(pitch),
        roll_deg=float(roll),
        seed=int(seed),
        return_bgr=False,
    )
    sample = dataset[0]
    if bool(sample.get("meta", {}).get("dummy", False)):
        raise RuntimeError(f"Dataset returned a dummy sample for frame={frame}, pose={pose}")
    meta = dict(sample.get("meta", {}))
    meta["geometry_source"] = "fisheye"
    sample = dict(sample)
    sample["meta"] = meta
    return sample


def main() -> None:
    args = parse_args()
    poses = parse_pose_triples(args.poses)
    data_dir = Path(args.data_dir)
    drive_path = resolve_drive_path(data_dir, args.drive)
    if not drive_path.is_dir():
        raise FileNotFoundError(f"Drive directory not found: {drive_path}")

    virtual_w, virtual_h = int(args.virtual_size[0]), int(args.virtual_size[1])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {
        "drive": str(drive_path),
        "frame": int(args.frame),
        "source": str(args.source),
        "camera": str(args.camera),
        "virtual_size": [virtual_w, virtual_h],
        "hfov_deg": float(args.hfov),
        "forward_m": [float(x) for x in args.forward_m],
        "lateral_m": [float(x) for x in args.lateral_m],
        "poses": [],
    }
    overlays: List[Image.Image] = []

    print(f"drive={drive_path}")
    print(f"frame={args.frame}, source={args.source}, camera={args.camera}, size={virtual_w}x{virtual_h}")
    print(f"output_dir={output_dir}")

    for pose in poses:
        sample = build_pose_sample(
            drive_path=drive_path,
            frame=int(args.frame),
            camera=str(args.camera),
            pose=pose,
            virtual_size=(virtual_w, virtual_h),
            hfov=float(args.hfov),
            seed=int(args.seed),
            source=str(args.source),
        )

        image = tensor_image_to_pil(sample["image"])
        K = sample["K"].detach().cpu().numpy()
        T_cam_to_world = sample["T_cam_to_world"].detach().cpu().numpy()
        T_imu_to_world = sample["T_imu_to_world"].detach().cpu().numpy()
        camera_height_m = float(sample["camera_height_m"])
        ground_z = float(T_cam_to_world[2, 3] - camera_height_m)

        ground_points = build_vehicle_ground_points(
            T_imu_to_world,
            forward_m=args.forward_m,
            lateral_m=args.lateral_m,
            ground_z=ground_z,
        )
        projected_points = project_world_points(
            ground_points,
            K=K,
            T_cam_to_world=T_cam_to_world,
            image_w=virtual_w,
            image_h=virtual_h,
        )

        overlay = draw_overlay(
            image,
            pose=pose,
            projected_points=projected_points,
            meta=sample.get("meta", {}),
        )
        overlays.append(overlay)

        name = pose_name(pose)
        overlay_path = output_dir / f"{name}.png"
        raw_path = output_dir / f"{name}_raw.png"
        overlay.save(overlay_path)
        image.save(raw_path)

        valid_count = sum(1 for point in projected_points if point["valid"])
        center_points = [
            point for point in projected_points
            if abs(float(point["right_m"])) < 1e-6 and point["valid"]
        ]
        center_uv = ", ".join(
            f"F{point['forward_m']:g}=({point['pixel_uv'][0]:.1f},{point['pixel_uv'][1]:.1f})"
            for point in center_points
        )
        print(
            f"{name}: valid={valid_count}/{len(projected_points)} "
            f"{center_uv} saved={overlay_path}"
        )

        all_results["poses"].append(
            {
                "name": name,
                "pose_yaw_pitch_roll_deg": [float(pose[0]), float(pose[1]), float(pose[2])],
                "overlay_path": str(overlay_path),
                "raw_path": str(raw_path),
                "K": K.tolist(),
                "T_cam_to_world": T_cam_to_world.tolist(),
                "T_imu_to_world": T_imu_to_world.tolist(),
                "camera_height_m": camera_height_m,
                "ground_z": ground_z,
                "meta": to_jsonable(sample.get("meta", {})),
                "projected_points": to_jsonable(projected_points),
            }
        )

    contact_sheet = make_contact_sheet(overlays, columns=2)
    contact_sheet_path = output_dir / "contact_sheet.png"
    contact_sheet.save(contact_sheet_path)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"contact_sheet={contact_sheet_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
