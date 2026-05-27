#!/usr/bin/env python3
"""Diagnose clean logit-level geometry addressing on a real KITTI-360 batch."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data import Kitti360dDataset
from models.encoders.satellite_condition_encoder import SatelliteConditionEncoder
from models.sd_model import create_sd_model

_train_mod = runpy.run_path(str(_project_root / "scripts" / "train.py"))


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _build_dataset(args: argparse.Namespace, config: Dict[str, Any]) -> Kitti360dDataset:
    data_path = Path(args.data_dir)
    train_dirs, train_frames, val_dirs, val_frames = _train_mod["_load_split_from_yaml"](
        data_path,
        Path(args.split_yaml),
    )
    drives = train_dirs if args.dataset_split == "train" else val_dirs
    frames = train_frames if args.dataset_split == "train" else val_frames

    data_cfg = dict(config.get("data", {}) or {})
    front_resize = tuple(int(x) for x in data_cfg.get("front_resize", [640, 256]))
    fixed_list = data_cfg.get("vehicle_yaw_fixed_list")
    if fixed_list is None:
        fixed_list = ["front", -120.0, -90.0, -60.0, 60.0, 90.0, 120.0]

    return Kitti360dDataset(
        drives=drives,
        frames=frames,
        mode=str(data_cfg.get("mode", "fisheye_virtual")),
        yaw_mode=str(data_cfg.get("yaw_mode", "vehicle_relative")),
        view_set=str(data_cfg.get("view_set", "single")),
        virtual_size=front_resize,
        front_resize=front_resize,
        front_center_crop=None,
        random_fisheye_relative_yaw=False,
        random_vehicle_relative_yaw=False,
        vehicle_yaw_min_deg=float(data_cfg.get("vehicle_yaw_min_deg", 60.0)),
        vehicle_yaw_max_deg=float(data_cfg.get("vehicle_yaw_max_deg", 120.0)),
        vehicle_yaw_sampling=str(data_cfg.get("vehicle_yaw_sampling", "fixed_list")),
        vehicle_yaw_fixed_list=fixed_list,
        front_sample_prob=float(data_cfg.get("front_sample_prob", 0.0)),
        pitch_deg=float(data_cfg.get("pitch_deg", 0.0)),
        roll_deg=float(data_cfg.get("roll_deg", 0.0)),
        seed=int(config.get("seed", args.seed)),
        return_bgr=False,
    )


def _samples_for_first_yaw_group(dataset: Kitti360dDataset) -> List[Dict[str, Any]]:
    if len(dataset) < 7:
        raise ValueError(f"Need at least 7 expanded samples, got {len(dataset)}")
    samples = [dataset[idx] for idx in range(7)]
    frame_ids = {int(sample["frame_id"]) for sample in samples}
    if len(frame_ids) != 1:
        raise ValueError(f"First 7 samples are not one fixed-yaw group: {sorted(frame_ids)}")
    return samples


def _stack_geometry(samples: List[Dict[str, Any]], device: str) -> Dict[str, Any]:
    return {
        "sat": torch.stack([sample["sat"] for sample in samples], dim=0).to(device),
        "K": torch.stack([sample["K"] for sample in samples], dim=0).to(device),
        "T_cam_to_world": torch.stack([sample["T_cam_to_world"] for sample in samples], dim=0).to(device),
        "T_imu_to_world": torch.stack([sample["T_imu_to_world"] for sample in samples], dim=0).to(device),
        "camera_height_m": torch.tensor(
            [float(sample["camera_height_m"]) for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        "image_size": tuple(int(x) for x in samples[0]["image"].shape[-2:]),
    }


def _encoder_config(config: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(config.get("model", {}) or {})
    enc_cfg = dict(model_cfg.get("satellite_encoder", {}) or {})
    enc_cfg.pop("name", None)
    for key in list(enc_cfg):
        if key.startswith("perspective_"):
            enc_cfg.pop(key)
    return enc_cfg


def _run_encoder_geometry_check(
    *,
    enc_cfg: Dict[str, Any],
    batch: Dict[str, Any],
    samples: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoder = SatelliteConditionEncoder(**enc_cfg).to(batch["sat"].device).eval()

    with torch.no_grad():
        out = encoder(
            batch["sat"],
            K=batch["K"],
            T_cam_to_world=batch["T_cam_to_world"],
            T_imu_to_world=batch["T_imu_to_world"],
            camera_height_m=batch["camera_height_m"],
            image_size=batch["image_size"],
        )

    front_tokens = out.tokens[0].float()
    front_uv = out.perspective_uv[0].float()
    base_norm = front_tokens.norm(dim=-1).mean().clamp_min(1e-6)
    uv_norm = front_uv.norm(dim=-1).mean().clamp_min(1e-6)
    rows = []
    token_ratios = []
    uv_ratios = []
    for idx, sample in enumerate(samples):
        token_delta = (out.tokens[idx].float() - front_tokens).norm(dim=-1).mean()
        uv_delta = (out.perspective_uv[idx].float() - front_uv).norm(dim=-1).mean()
        token_ratio = token_delta / base_norm
        uv_ratio = uv_delta / uv_norm
        token_ratios.append(token_ratio)
        uv_ratios.append(uv_ratio)
        valid = out.perspective_valid[idx].detach().bool()
        rows.append(
            {
                "view_name": sample["meta"].get("view_name"),
                "vehicle_yaw_deg": sample["meta"].get("vehicle_yaw_deg_used"),
                "valid_ratio": float(valid.float().mean().cpu()),
                "invalid_ratio": float((~valid).float().mean().cpu()),
                "sat_token_yaw_delta": float(token_delta.cpu()),
                "sat_token_yaw_delta_to_base": float(token_ratio.cpu()),
                "sat_perspective_uv_yaw_delta": float(uv_delta.cpu()),
                "sat_perspective_uv_yaw_delta_to_base": float(uv_ratio.cpu()),
            }
        )

    return {
        "rows": rows,
        "mean_sat_token_yaw_delta_to_base": float(torch.stack(token_ratios).mean().cpu()),
        "max_sat_token_yaw_delta_to_base": float(torch.stack(token_ratios).max().cpu()),
        "mean_sat_perspective_uv_yaw_delta_to_base": float(torch.stack(uv_ratios).mean().cpu()),
        "max_sat_perspective_uv_yaw_delta_to_base": float(torch.stack(uv_ratios).max().cpu()),
        "additive_satellite_parameter_names": [
            name
            for name, _ in encoder.named_parameters()
            if "perspective_pe" in name or "perspective_pos_encoder" in name
        ],
    }


def _single_sample_batch(sample: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    return {
        "sat_images": sample["sat"].unsqueeze(0).to(device),
        "target_images": sample["image"].unsqueeze(0).to(device),
        "K": sample["K"].unsqueeze(0).to(device),
        "T_cam_to_world": sample["T_cam_to_world"].unsqueeze(0).to(device),
        "T_imu_to_world": sample["T_imu_to_world"].unsqueeze(0).to(device),
        "camera_height_m": torch.tensor([float(sample["camera_height_m"])], dtype=torch.float32, device=device),
    }


def _collect_gradients(model: torch.nn.Module) -> Dict[str, Any]:
    buckets = {
        "geometry_score_gate": ".processor.geometry_score_gate",
        "geometry_score_proj": ".processor.geometry_score_proj",
        "removed_additive_satellite_pe": "satellite_encoder.perspective_",
        "removed_additive_query_pe": ".processor.query_uv_",
    }
    result: Dict[str, Any] = {}
    for bucket, pattern in buckets.items():
        examples = []
        count = 0
        nonzero = 0
        for name, param in model.named_parameters():
            if pattern not in name:
                continue
            count += 1
            grad = param.grad
            norm = None if grad is None else float(grad.detach().float().norm().cpu())
            if grad is not None and norm > 0.0:
                nonzero += 1
            if len(examples) < 8:
                examples.append([name, norm])
        result[bucket] = {
            "count": count,
            "nonzero_grad_count": nonzero,
            "examples": examples,
        }
    return result


def _scalar_output(outputs: Dict[str, Any], name: str) -> float | None:
    value = outputs.get(name)
    if not torch.is_tensor(value):
        return None
    return float(value.detach().float().cpu())


def _build_model(config: Dict[str, Any], device: str):
    model_cfg = dict(config.get("model", {}) or {})
    query_geometry_score_config = _train_mod["_resolve_query_geometry_score_config"](config)
    attention_alignment_config = _train_mod["_resolve_attention_alignment_config"](config)
    return create_sd_model(
        base_model=str(model_cfg.get("base_model", "sd2-community/stable-diffusion-2-1-base")),
        freeze_base=bool(model_cfg.get("freeze_base", True)),
        torch_dtype=None,
        cond_drop_prob=0.0,
        query_geometry_score_enabled=query_geometry_score_config["enabled"],
        query_geometry_score_dim=query_geometry_score_config["dim"],
        query_geometry_score_num_freqs=query_geometry_score_config["num_freqs"],
        query_geometry_score_gate_init=query_geometry_score_config["gate_init"],
        query_geometry_score_layers=query_geometry_score_config["layers"],
        query_geometry_score_max_query_tokens=query_geometry_score_config["max_query_tokens"],
        query_geometry_score_mode=query_geometry_score_config["mode"],
        query_geometry_candidate_radius=query_geometry_score_config["candidate_radius"],
        query_geometry_candidate_min_k=query_geometry_score_config["candidate_min_k"],
        query_geometry_candidate_invalid_penalty=query_geometry_score_config["candidate_invalid_penalty"],
        query_semantic_score_dim=query_geometry_score_config["semantic_score_dim"],
        query_semantic_score_alpha=query_geometry_score_config["semantic_alpha_max"],
        attention_alignment_enabled=attention_alignment_config["enabled"],
        attention_alignment_loss_weight=attention_alignment_config["loss_weight"],
        attention_alignment_layers=attention_alignment_config["layers"],
        attention_alignment_max_query_tokens=attention_alignment_config["max_query_tokens"],
        attention_alignment_valid_radius=attention_alignment_config["valid_radius"],
        attention_alignment_invalid_attention_weight=attention_alignment_config["invalid_attention_weight"],
        satellite_encoder_config=dict(model_cfg.get("satellite_encoder", {}) or {}),
    ).to(device)


def _run_backward_check(
    *,
    model: torch.nn.Module,
    sample: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    model.train()
    model.vae.eval()
    batch = _single_sample_batch(sample, device)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outputs = model(**batch)
        outputs["loss"].backward()

    logit_balance_keys = [
        "attention_alignment_content_logits_std",
        "attention_alignment_content_logits_abs_mean",
        "attention_alignment_content_logits_top_gap",
        "attention_alignment_geometry_bias_std",
        "attention_alignment_geometry_bias_abs_mean",
        "attention_alignment_geometry_bias_top_gap",
        "attention_alignment_geometry_to_content_std_ratio",
        "attention_alignment_geometry_to_content_abs_ratio",
        "attention_alignment_geometry_to_content_top_gap_ratio",
        "attention_alignment_attention_geometry_kl",
        "attention_alignment_valid_attention_mass_without_geometry",
        "attention_alignment_target_attention_lift_without_geometry",
        "attention_alignment_target_attention_lift_geometry_delta",
        "attention_alignment_target_logit_gap_without_geometry",
        "attention_alignment_target_logit_gap_geometry_delta",
        "attention_alignment_geometry_score_gate_raw",
        "attention_alignment_geometry_score_runtime_scale",
    ]

    return {
        "sample": {
            "frame_id": int(sample["frame_id"]),
            "view_name": sample["meta"].get("view_name"),
            "vehicle_yaw_deg": sample["meta"].get("vehicle_yaw_deg_used"),
        },
        "loss": float(outputs["loss"].detach().float().cpu()),
        "denoise_loss": float(outputs["denoise_loss"].detach().float().cpu()),
        "attention_alignment_loss": float(outputs["attention_alignment_loss"].detach().float().cpu()),
        "attention_alignment_loss_weight": float(outputs["attention_alignment_loss_weight"].detach().float().cpu()),
        "attention_alignment_loss_is_differentiable": float(
            outputs["attention_alignment_loss_is_differentiable"].detach().float().cpu()
        ),
        "geometry_score_bias_std": (
            float(outputs["attention_alignment_geometry_score_bias_std"].detach().float().cpu())
            if torch.is_tensor(outputs.get("attention_alignment_geometry_score_bias_std"))
            else None
        ),
        "geometry_score_gate": (
            float(outputs["attention_alignment_geometry_score_gate"].detach().float().cpu())
            if torch.is_tensor(outputs.get("attention_alignment_geometry_score_gate"))
            else None
        ),
        "logit_balance": {
            key.removeprefix("attention_alignment_"): _scalar_output(outputs, key)
            for key in logit_balance_keys
            if _scalar_output(outputs, key) is not None
        },
        "perspective_valid_ratio": float(outputs["sat_state"].perspective_valid.detach().float().mean().cpu()),
        "additive_parameter_names": [
            name
            for name, _ in model.named_parameters()
            if "perspective_pe" in name
            or "perspective_pos_encoder" in name
            or "query_uv_gate" in name
            or "query_uv_encoder" in name
        ],
        "gradients": _collect_gradients(model),
    }


def _run_overfit_smoke(
    *,
    model: torch.nn.Module,
    sample: Dict[str, Any],
    device: str,
    steps: int,
    lr: float,
) -> Dict[str, Any]:
    if steps <= 0:
        return {"steps": 0}
    model.train()
    model.vae.eval()
    batch = _single_sample_batch(sample, device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(lr),
    )
    losses = []
    geometry_bias_stds = []
    geometry_gates = []
    geometry_to_content_ratios = []
    attention_geometry_kls = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs["loss"].backward()
        optimizer.step()
        losses.append(float(outputs["loss"].detach().float().cpu()))
        if torch.is_tensor(outputs.get("attention_alignment_geometry_score_bias_std")):
            geometry_bias_stds.append(float(outputs["attention_alignment_geometry_score_bias_std"].detach().float().cpu()))
        if torch.is_tensor(outputs.get("attention_alignment_geometry_score_gate")):
            geometry_gates.append(float(outputs["attention_alignment_geometry_score_gate"].detach().float().cpu()))
        if torch.is_tensor(outputs.get("attention_alignment_geometry_to_content_std_ratio")):
            geometry_to_content_ratios.append(
                float(outputs["attention_alignment_geometry_to_content_std_ratio"].detach().float().cpu())
            )
        if torch.is_tensor(outputs.get("attention_alignment_attention_geometry_kl")):
            attention_geometry_kls.append(
                float(outputs["attention_alignment_attention_geometry_kl"].detach().float().cpu())
            )
    return {
        "steps": int(steps),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "geometry_score_bias_std_last": geometry_bias_stds[-1] if geometry_bias_stds else None,
        "geometry_score_gate_last": geometry_gates[-1] if geometry_gates else None,
        "geometry_to_content_std_ratio_last": geometry_to_content_ratios[-1] if geometry_to_content_ratios else None,
        "attention_geometry_kl_last": attention_geometry_kls[-1] if attention_geometry_kls else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data_dir", type=str, default="/media/shizhm/Lenovo/KITTI-360")
    parser.add_argument("--split_yaml", type=str, default="splits/sync03_sync04_split.yaml")
    parser.add_argument("--dataset_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default="output/geometry_addressing_diagnostic")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_backward", action="store_true")
    parser.add_argument("--overfit_steps", type=int, default=0)
    parser.add_argument("--overfit_lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("HF_HOME", str(_project_root / ".hf-home"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    config = _load_config(Path(args.config))
    dataset = _build_dataset(args, config)
    samples = _samples_for_first_yaw_group(dataset)
    batch = _stack_geometry(samples, args.device)

    result: Dict[str, Any] = {
        "config": str(Path(args.config)),
        "data_dir": str(Path(args.data_dir)),
        "dataset_split": args.dataset_split,
        "frame_id": int(samples[0]["frame_id"]),
        "geometry_path": _run_encoder_geometry_check(
            enc_cfg=_encoder_config(config),
            batch=batch,
            samples=samples,
            seed=args.seed,
        ),
    }
    if not args.skip_backward or int(args.overfit_steps) > 0:
        model = _build_model(config, args.device)
        if not args.skip_backward:
            result["backward_check"] = _run_backward_check(
                model=model,
                sample=samples[3],
                device=args.device,
            )
        if int(args.overfit_steps) > 0:
            result["overfit_smoke"] = _run_overfit_smoke(
                model=model,
                sample=samples[3],
                device=args.device,
                steps=int(args.overfit_steps),
                lr=float(args.overfit_lr),
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "geometry_addressing_summary.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
