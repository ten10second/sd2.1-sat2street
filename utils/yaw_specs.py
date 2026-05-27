"""Canonical vehicle-relative yaw view specs for Stage 1 yaw-only training."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


STAGE1_FIXED_YAWS: Tuple[float, ...] = (-120.0, -90.0, -60.0, 60.0, 90.0, 120.0)
DIAGNOSTIC_YAWS: Tuple[float, ...] = (-120.0, -90.0, -60.0, -30.0, 30.0, 60.0, 90.0, 120.0)


def yaw_view_name(yaw_deg: float) -> str:
    prefix = "p" if yaw_deg > 0 else "m" if yaw_deg < 0 else ""
    return f"yaw_{prefix}{abs(float(yaw_deg)):g}".replace(".", "p")


def yaw_view_specs(yaws: Sequence[float]) -> Tuple[Tuple[str, float], ...]:
    return tuple((yaw_view_name(float(yaw)), float(yaw)) for yaw in yaws)


TRAIN_FIXED_YAW_SWEEP_SPECS: Tuple[Tuple[str, float], ...] = yaw_view_specs(STAGE1_FIXED_YAWS)
DIAGNOSTIC_YAW_SWEEP_SPECS: Tuple[Tuple[str, float], ...] = yaw_view_specs(DIAGNOSTIC_YAWS)

YAW_SWEEP_PRESETS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "train_fixed": TRAIN_FIXED_YAW_SWEEP_SPECS,
    "diagnostic": DIAGNOSTIC_YAW_SWEEP_SPECS,
}


def stage1_fixed_yaw_list() -> List[float]:
    return [float(yaw) for yaw in STAGE1_FIXED_YAWS]


def maybe_front_view_spec(include_front: bool) -> List[Tuple[str, Optional[float]]]:
    return [("front", None)] if include_front else []
