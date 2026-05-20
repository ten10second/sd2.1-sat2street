"""Comprehensive checkpoint diagnostic: gates, PE weights, self-attn, valid ratio."""
from __future__ import annotations
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
import argparse
import numpy as np


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_gate_values(state_dict: dict) -> None:
    """Check query_uv_gate values for all attn2 processor layers."""
    _section("1. Query UV Gate values (should be >> 0 if learning)")

    gate_keys = sorted([k for k in state_dict if "query_uv_gate" in k])
    if not gate_keys:
        print("  No query_uv_gate found — Q UV PE was disabled for this run.")
        return

    all_zeros = True
    for k in gate_keys:
        v = state_dict[k]
        val = float(v.detach().cpu())
        print(f"  {k}: {val:.6f}")
        if abs(val) > 0.01:
            all_zeros = False

    if all_zeros:
        print()
        print("  ⚠️  ALL gates ≈ 0! Q UV PE contributed NOTHING to cross-attention.")
    else:
        print()
        print("  ✓ Gates are non-zero, Q UV PE is active.")


def check_perspective_pe(state_dict: dict) -> None:
    """Check perspective_pos_encoder MLP weights."""
    _section("2. Perspective Position Encoder weights")

    pe_keys = sorted([k for k in state_dict if "perspective_pos_encoder" in k])
    if not pe_keys:
        print("  No perspective_pos_encoder found.")
        return

    for k in pe_keys:
        w = state_dict[k]
        tag = ""
        if "mlp.1.weight" in k or "mlp.4.weight" in k:
            d = (w - 1.0).abs().max().item()
            tag = f"← LN weight, max|w-1|={d:.6f} {'⚠️ NOT LEARNING' if d < 1e-6 else '✓'}"
        elif "mlp.1.bias" in k or "mlp.4.bias" in k:
            d = w.abs().max().item()
            tag = f"← LN bias, max|b|={d:.6f} {'⚠️ NOT LEARNING' if d < 1e-6 else '✓'}"
        elif "mlp.0.weight" in k or "mlp.3.weight" in k:
            tag = f"← Linear, mean={w.float().mean():.4f} std={w.float().std():.4f}"
        print(f"  {k}: shape={tuple(w.shape)}  {tag}")


def check_self_attn(state_dict: dict) -> None:
    """Check satellite encoder self-attention out_proj (zero-init)."""
    _section("3. Satellite Encoder Self-Attention out_proj (zero-init → should grow)")

    out_proj_keys = sorted([
        k for k in state_dict
        if "satellite_encoder.self_attn.layers" in k
        and ("out_proj" in k or "linear2" in k)
    ])
    if not out_proj_keys:
        print("  No self_attn out_proj keys found.")
        return

    max_abs_all = 0.0
    for k in out_proj_keys:
        w = state_dict[k]
        ma = float(w.abs().max())
        max_abs_all = max(max_abs_all, ma)
        marker = "⚠️ DEAD (max_abs < 0.01)" if ma < 0.01 else "✓"
        print(f"  {k}: max_abs={ma:.6f}  {marker}")

    if max_abs_all < 0.01:
        print()
        print("  ⚠️  Self-attention is effectively identity — sat patches don't share info.")


def check_grid_pos_embed(state_dict: dict) -> None:
    """Check if learnable grid position embedding has changed from init."""
    _section("4. Grid Position Embedding")

    key = "satellite_encoder.grid_pos_embed"
    if key not in state_dict:
        print(f"  Key '{key}' not found.")
        return

    w = state_dict[key]
    print(f"  shape={tuple(w.shape)}, mean={w.float().mean():.6f}, std={w.float().std():.6f}")
    # init was trunc_normal(std=0.02), so initial std ≈ 0.02
    std_val = float(w.float().std())
    if abs(std_val - 0.02) < 0.005:
        print("  ⚠️  std still ≈ 0.02 — grid PE barely changed from init.")
    else:
        print("  ✓ grid PE has moved from init.")


def check_attn2_kv(state_dict: dict) -> None:
    """Check attn2 to_k / to_v weight statistics."""
    _section("5. attn2 to_k / to_v (trainable projections)")

    kv_keys = sorted([
        k for k in state_dict
        if ".attn2.to_k." in k or ".attn2.to_v." in k
    ])
    if not kv_keys:
        print("  No attn2 to_k/to_v keys found.")
        return

    for k in kv_keys[:8]:  # show first 8
        w = state_dict[k]
        print(f"  {k}: mean={w.float().mean():.6f}  std={w.float().std():.6f}")
    if len(kv_keys) > 8:
        print(f"  ... ({len(kv_keys)} total keys)")


def check_sat_tokens_on_data(checkpoint_path: str, data_dir: str, device: str) -> None:
    """Run forward pass on a few samples and check sat token stats + valid ratio."""
    _section("6. Runtime checks: perspective_valid ratio & sat token stats")

    from models.sd_model import create_sd_model, load_model_checkpoint
    from data.kitti360d_dataset import Kitti360dDataset

    print("  Loading model...")
    model = create_sd_model(
        query_uv_pe_enabled=True,
        query_uv_gate_init=0.0,
    )
    load_model_checkpoint(model, Path(checkpoint_path), device)
    model = model.to(device)
    model.eval()

    data_path = Path(data_dir)
    dataset = Kitti360dDataset(
        drives=str(data_path / "2013_05_28_drive_0003_sync"),
        frames=list(range(0, 200)),
        mode="fisheye_virtual",
        yaw_mode="vehicle_relative",
        vehicle_yaw_sampling="fixed_list",
        vehicle_yaw_fixed_list=["front", 60.0, 90.0, 120.0, -60.0, -90.0, -120.0],
        view_set="single",
        seed=42,
    )

    valid_ratios = []
    num_zero = 0
    for idx in range(min(len(dataset), 200)):
        sample = dataset[idx]
        sat = sample["sat"].unsqueeze(0).to(device)
        target_size = tuple(sample["image"].shape[-2:])

        with torch.no_grad():
            sat_state = model.encode_satellite(
                sat,
                K=sample["K"].unsqueeze(0).to(device),
                T_cam_to_world=sample["T_cam_to_world"].unsqueeze(0).to(device),
                T_imu_to_world=sample["T_imu_to_world"].unsqueeze(0).to(device),
                camera_height_m=torch.tensor([float(sample["camera_height_m"])]).to(device),
                image_size=target_size,
            )

        if sat_state.perspective_valid is not None:
            ratio = sat_state.perspective_valid.float().mean().item()
            valid_ratios.append(ratio)
            if ratio == 0.0:
                num_zero += 1

        if idx < 3:
            tokens = sat_state.tokens
            print(
                f"  sample[{idx}]: tokens mean={tokens.float().mean():.4f} "
                f"std={tokens.float().std():.4f} "
                f"nonzero={((tokens.abs() > 1e-6).float().mean() * 100):.1f}%"
            )
            if sat_state.perspective_valid is not None:
                val_sum = sat_state.perspective_valid.sum().item()
                print(f"           perspective_valid: {val_sum}/{sat_state.perspective_valid.numel()} "
                      f"({sat_state.perspective_valid.float().mean():.4f})")

    if valid_ratios:
        arr = np.array(valid_ratios)
        print(f"\n  Summary over {len(arr)} samples:")
        print(f"    mean valid ratio: {arr.mean():.4f}")
        print(f"    min/max/zero: {arr.min():.4f} / {arr.max():.4f} / {num_zero}")

        if arr.mean() < 0.01:
            print("  ⚠️  perspective_valid ≈ 0%! Z bug still present?")
        elif arr.mean() < 0.1:
            print("  ⚠️  Very low valid ratio (< 10%).")
        else:
            print("  ✓ perspective_valid ratio looks healthy.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/media/shizhm/Lenovo/KITTI-360")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weights_only", action="store_true",
                        help="Only inspect checkpoint weights (no model loading or data)")
    args = parser.parse_args()

    print(f"Diagnosing checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    print(f"State dict keys: {len(state_dict)}")

    # 1-5: Pure weight inspection (no model/data needed)
    check_gate_values(state_dict)
    check_perspective_pe(state_dict)
    check_self_attn(state_dict)
    check_grid_pos_embed(state_dict)
    check_attn2_kv(state_dict)

    # 6: Runtime checks (need model + data)
    if not args.weights_only:
        try:
            check_sat_tokens_on_data(args.checkpoint, args.data_dir, args.device)
        except Exception as e:
            print(f"\n  ⚠️  Runtime checks failed: {e}")
    else:
        _section("6. Skipped (--weights_only)")

    print()
    print("=" * 70)
    print("  Diagnosis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()