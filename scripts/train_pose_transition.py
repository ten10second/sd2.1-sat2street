#!/usr/bin/env python3
"""Train a camera-action latent transition probe on pose-chain groups."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Subset

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data import Kitti360dDataset
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.pose_transition import PoseTransitionProbe, TransitionHead
from models.sd_model import _resolve_hf_snapshot_path
from scripts.train import (
    DEFAULT_SD21_BASE_REPO,
    _config_get,
    _load_runtime_config,
    _load_split_from_yaml,
    _safe_collate,
    _worker_init_fn,
)


logger = logging.getLogger(__name__)


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train_pose_transition.log"),
        ],
    )


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def _encode_images_to_latents(
    vae: AutoencoderKL,
    images: torch.Tensor,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"pose transition training requires images [B,V,C,H,W], got {tuple(images.shape)}")
    batch_size, num_views = int(images.shape[0]), int(images.shape[1])
    flat_images = images.reshape(batch_size * num_views, *images.shape[2:])
    normalized = flat_images * 2.0 - 1.0
    latent_dist = vae.encode(normalized).latent_dist
    if deterministic and hasattr(latent_dist, "mode"):
        latents = latent_dist.mode()
    else:
        latents = latent_dist.sample()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    return latents * scaling_factor


def _load_vae(
    *,
    base_model: str,
    revision: Optional[str],
    device: torch.device,
) -> AutoencoderKL:
    resolved_base_model = _resolve_hf_snapshot_path(base_model, revision=revision)
    load_source = str(resolved_base_model) if resolved_base_model is not None else base_model
    kwargs: Dict[str, Any] = {}
    if revision is not None and resolved_base_model is None:
        kwargs["revision"] = revision
    if resolved_base_model is not None:
        kwargs["local_files_only"] = True
        logger.info("Using cached base model snapshot: %s", resolved_base_model)
    vae = AutoencoderKL.from_pretrained(load_source, subfolder="vae", **kwargs)
    vae.to(device=device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    return vae


def _dataset_kwargs(config: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    data_cfg = config.get("data") if isinstance(config.get("data"), Mapping) else {}
    front_resize_cfg = data_cfg.get("front_resize", [640, 256])
    front_resize = tuple(int(x) for x in front_resize_cfg)
    return {
        "mode": str(data_cfg.get("mode", "fisheye_virtual")),
        "yaw_mode": str(data_cfg.get("yaw_mode", "vehicle_relative")),
        "view_set": "pose_chain",
        "pose_chains": data_cfg.get(
            "pose_chains",
            [
                {"name": "right", "yaws": ["front", 60.0, 90.0, 120.0]},
                {"name": "left", "yaws": ["front", -60.0, -90.0, -120.0]},
            ],
        ),
        "virtual_size": front_resize,
        "front_resize": front_resize,
        "front_center_crop": None,
        "random_fisheye_relative_yaw": False,
        "random_vehicle_relative_yaw": False,
        "vehicle_yaw_min_deg": float(data_cfg.get("vehicle_yaw_min_deg", 60.0)),
        "vehicle_yaw_max_deg": float(data_cfg.get("vehicle_yaw_max_deg", 120.0)),
        "vehicle_yaw_sampling": "fixed_list",
        "vehicle_yaw_fixed_list": data_cfg.get(
            "vehicle_yaw_fixed_list",
            ["front", -120.0, -90.0, -60.0, 60.0, 90.0, 120.0],
        ),
        "front_sample_prob": 0.0,
        "pitch_deg": float(data_cfg.get("pitch_deg", 0.0)),
        "roll_deg": float(data_cfg.get("roll_deg", 0.0)),
        "seed": int(args.seed),
        "return_bgr": False,
    }


def _limit_dataset(dataset: Kitti360dDataset, max_samples: Optional[int]) -> Kitti360dDataset | Subset:
    if max_samples is None:
        return dataset
    sample_count = min(int(max_samples), len(dataset))
    return Subset(dataset, list(range(sample_count)))


def _build_dataloaders(config: Mapping[str, Any], args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    data_path = Path(args.data_dir)
    split_yaml = Path(args.split_yaml) if args.split_yaml is not None else data_path / "train_test_split_config.yaml"
    train_dirs, train_frames, val_dirs, val_frames = _load_split_from_yaml(data_path, split_yaml)
    train_dataset = Kitti360dDataset(
        drives=train_dirs,
        frames=train_frames,
        **_dataset_kwargs(config, args),
    )
    val_dataset = Kitti360dDataset(
        drives=val_dirs,
        frames=val_frames,
        **_dataset_kwargs(config, args),
    )
    train_dataset = _limit_dataset(train_dataset, args.max_train_samples)
    val_dataset = _limit_dataset(val_dataset, args.max_val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available() and str(args.device).startswith("cuda"),
        drop_last=True,
        collate_fn=_safe_collate,
        worker_init_fn=_worker_init_fn if int(args.num_workers) > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available() and str(args.device).startswith("cuda"),
        drop_last=True,
        collate_fn=_safe_collate,
        worker_init_fn=_worker_init_fn if int(args.num_workers) > 0 else None,
    )
    return train_loader, val_loader


def _build_satellite_encoder(config: Mapping[str, Any], args: argparse.Namespace) -> SatelliteConditionEncoder:
    sat_cfg = dict(_config_get(dict(config), ("model", "satellite_encoder"), {}) or {})
    sat_cfg.pop("name", None)
    if args.satellite_embed_dim is not None:
        sat_cfg["embed_dim"] = int(args.satellite_embed_dim)
    if args.satellite_num_heads is not None:
        sat_cfg["num_heads"] = int(args.satellite_num_heads)
    if args.satellite_num_layers is not None:
        sat_cfg["num_layers"] = int(args.satellite_num_layers)
    return SatelliteConditionEncoder(**sat_cfg)


def _scalar_item(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().item())
    return float(value)


def _append_scalars(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def _outputs_to_record(prefix: str, outputs: Mapping[str, Any]) -> Dict[str, float]:
    return {
        f"{prefix}/loss": _scalar_item(outputs["loss"]),
        f"{prefix}/transition_loss": _scalar_item(outputs["transition_loss"]),
        f"{prefix}/front_to_side_loss": _scalar_item(outputs["front_to_side_loss"]),
        f"{prefix}/side_to_side_loss": _scalar_item(outputs["side_to_side_loss"]),
        f"{prefix}/cycle_loss": _scalar_item(outputs["cycle_loss"]),
        f"{prefix}/composition_loss": _scalar_item(outputs["composition_loss"]),
        f"{prefix}/num_pairs": _scalar_item(outputs["num_pairs"]),
    }


@torch.no_grad()
def _evaluate(
    *,
    model: PoseTransitionProbe,
    vae: AutoencoderKL,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    epoch: int,
    step: int,
    max_batches: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    sums: Dict[str, float] = {}
    num_batches = 0
    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = _move_batch_to_device(batch, device)
            images = batch["image"]
            image_hw = tuple(int(x) for x in images.shape[-2:])
            latents = _encode_images_to_latents(vae, images, deterministic=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    latents=latents.float(),
                    sat_images=batch["sat"].float(),
                    batch=batch,
                    image_hw=image_hw,
                )
            loss = outputs["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite validation transition loss at step {step}: {float(loss.item())}")
            for key, value in _outputs_to_record("val", outputs).items():
                sums[key] = sums.get(key, 0.0) + float(value)
            num_batches += 1
    finally:
        model.train(was_training)
    if num_batches == 0:
        raise RuntimeError("validation dataloader produced zero batches")
    record = {key: value / float(num_batches) for key, value in sums.items()}
    record.update({
        "step": int(step),
        "epoch": int(epoch),
        "val/num_batches": float(num_batches),
    })
    return record


def _camera_action_metadata() -> Dict[str, Any]:
    return {
        "relative_SE3": "rotation6d + translation from inverse(T_A_cam_to_world) @ T_B_cam_to_world",
        "intrinsics": "source and target fx/fy/cx/cy normalized by image width/height",
        "camera_identity": "source/target camera_id embeddings for front(image_00), left fisheye(image_02), right fisheye(image_03), unknown",
        "view_type": "source/target view type embeddings for front vs fisheye-derived virtual perspective",
        "yaw": "sin/cos source yaw, target yaw, delta yaw plus delta/pi; auxiliary only",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    epoch: int,
    model: PoseTransitionProbe,
    optimizer: torch.optim.Optimizer,
    run_config: Mapping[str, Any],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_step": int(step),
        "checkpoint_epoch": int(epoch),
        "view_set": run_config.get("view_set"),
        "transition_probe_enabled": run_config.get("transition_probe_enabled"),
        "satellite_encoder_trainable": run_config.get("satellite_encoder_trainable"),
        "camera_action": run_config.get("camera_action"),
    }
    checkpoint = {
        "step": int(step),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_config": _jsonable(dict(run_config)),
        "metadata": _jsonable(metadata),
        "trainer_metadata": _jsonable(metadata),
    }
    path = checkpoint_dir / f"checkpoint_step_{int(step):06d}.pt"
    torch.save(checkpoint, path)
    torch.save(checkpoint, checkpoint_dir / "checkpoint_latest.pt")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pose-chain camera-action latent transition probe.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split_yaml", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("output/pose_transition_probe_smoke"))
    parser.add_argument("--base_model", type=str, default=DEFAULT_SD21_BASE_REPO)
    parser.add_argument("--base_model_revision", type=str, default=None)
    parser.add_argument("--hf_home", type=Path, default=Path(".hf-home"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=0)
    parser.add_argument("--num_val_batches", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--satellite_embed_dim", type=int, default=None)
    parser.add_argument("--satellite_num_heads", type=int, default=None)
    parser.add_argument("--satellite_num_layers", type=int, default=None)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--cycle_weight", type=float, default=0.1)
    parser.add_argument("--composition_weight", type=float, default=0.05)
    parser.add_argument("--mse_weight", type=float, default=0.1)
    parser.add_argument("--freeze_satellite_encoder", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    config = _load_runtime_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.output_dir / "logs")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    logger.info("Loading pose-chain dataset from %s", args.data_dir)
    train_loader, val_loader = _build_dataloaders(config, args)
    logger.info("Train batches per epoch: %d", len(train_loader))
    logger.info("Validation batches: %d", len(val_loader))

    logger.info("Loading frozen VAE")
    vae = _load_vae(base_model=args.base_model, revision=args.base_model_revision, device=device)

    satellite_encoder = _build_satellite_encoder(config, args).to(device)
    if args.freeze_satellite_encoder:
        for param in satellite_encoder.parameters():
            param.requires_grad = False
    transition_head = TransitionHead(
        latent_channels=int(getattr(vae.config, "latent_channels", 4) or 4),
        sat_token_dim=int(satellite_encoder.embed_dim),
        hidden_channels=int(args.hidden_channels),
    )
    model = PoseTransitionProbe(
        satellite_encoder=satellite_encoder,
        transition_head=transition_head,
        cycle_weight=float(args.cycle_weight),
        composition_weight=float(args.composition_weight),
        mse_weight=float(args.mse_weight),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scalar_log_path = args.output_dir / "logs" / "scalars.jsonl"
    if scalar_log_path.exists() and args.resume_from_checkpoint is None:
        scalar_log_path.unlink()

    data_cfg = config.get("data") if isinstance(config.get("data"), Mapping) else {}
    run_config = _jsonable({
        **vars(args),
        "output_dir": str(args.output_dir),
        "data_dir": str(args.data_dir),
        "split_yaml": str(args.split_yaml) if args.split_yaml is not None else None,
        "view_set": "pose_chain",
        "pose_chains": data_cfg.get("pose_chains"),
        "transition_probe_enabled": True,
        "satellite_encoder_trainable": not bool(args.freeze_satellite_encoder),
        "camera_action": _camera_action_metadata(),
    })
    with (args.output_dir / "run_config.yaml").open("w") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=True)

    global_step = 0
    start_epoch = 1
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        logger.info("Resumed checkpoint %s at step=%d, next_epoch=%d", args.resume_from_checkpoint, global_step, start_epoch)

    latest_checkpoint: Optional[Path] = None
    model.train()
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and args.mixed_precision != "no"

    def validate(epoch_value: int, step_value: int) -> Dict[str, float]:
        record = _evaluate(
            model=model,
            vae=vae,
            dataloader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            epoch=epoch_value,
            step=step_value,
            max_batches=int(args.num_val_batches),
        )
        _append_scalars(scalar_log_path, record)
        logger.info(
            "val step=%d loss=%.6f transition=%.6f front_to_side=%.6f side_to_side=%.6f",
            step_value,
            record["val/loss"],
            record["val/transition_loss"],
            record["val/front_to_side_loss"],
            record["val/side_to_side_loss"],
        )
        return record

    if int(args.val_every) > 0:
        validate(epoch_value=max(0, start_epoch - 1), step_value=global_step)

    epoch = start_epoch - 1
    if global_step >= int(args.max_steps):
        logger.info("Checkpoint step=%d already reached max_steps=%d", global_step, int(args.max_steps))

    for epoch in range(start_epoch, start_epoch + int(args.epochs)):
        if global_step >= int(args.max_steps):
            break
        for batch in train_loader:
            global_step += 1
            batch = _move_batch_to_device(batch, device)
            images = batch["image"]
            image_hw = tuple(int(x) for x in images.shape[-2:])
            with torch.no_grad():
                latents = _encode_images_to_latents(vae, images, deterministic=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    latents=latents.float(),
                    sat_images=batch["sat"].float(),
                    batch=batch,
                    image_hw=image_hw,
                )
                loss = outputs["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite transition loss at step {global_step}: {float(loss.detach().item())}")
            loss.backward()
            optimizer.step()

            record = {
                "step": global_step,
                "epoch": epoch,
                **_outputs_to_record("train", outputs),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
            }
            _append_scalars(scalar_log_path, record)
            if global_step % max(1, int(args.log_every)) == 0:
                logger.info(
                    "step=%d loss=%.6f transition=%.6f front_to_side=%.6f side_to_side=%.6f",
                    global_step,
                    record["train/loss"],
                    record["train/transition_loss"],
                    record["train/front_to_side_loss"],
                    record["train/side_to_side_loss"],
                )
            if args.save_every and global_step % int(args.save_every) == 0:
                latest_checkpoint = _save_checkpoint(
                    output_dir=args.output_dir,
                    step=global_step,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    run_config=run_config,
                )
                logger.info("Saved checkpoint: %s", latest_checkpoint)
            if int(args.val_every) > 0 and global_step % int(args.val_every) == 0:
                validate(epoch_value=epoch, step_value=global_step)
            if global_step >= int(args.max_steps):
                break
        if global_step >= int(args.max_steps):
            break

    if int(args.val_every) <= 0 or global_step % int(args.val_every) != 0:
        validate(epoch_value=epoch, step_value=global_step)

    latest_checkpoint = _save_checkpoint(
        output_dir=args.output_dir,
        step=global_step,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        run_config=run_config,
    )
    logger.info("Saved final checkpoint: %s", latest_checkpoint)
    logger.info("Scalars: %s", scalar_log_path)


if __name__ == "__main__":
    main()
