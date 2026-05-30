#!/usr/bin/env python3
"""Initialize a pose-chain manual decision record from formal gate outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.check_pose_chain_gate_outputs import (
    DEFAULT_EXPECTED_SAT_CONDITION_MODE,
    REQUIRED_VALIDATION_SCALAR_KEYS,
    GateScan,
    scan_gate_outputs,
)


def _git_value(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _metadata_checkpoint_epoch(metadata: Mapping) -> Optional[float]:
    checkpoint_gate_metadata = metadata.get("checkpoint_gate_metadata")
    if isinstance(checkpoint_gate_metadata, Mapping):
        trainer_metadata = checkpoint_gate_metadata.get("trainer_metadata")
        if isinstance(trainer_metadata, Mapping):
            checkpoint_epoch = trainer_metadata.get("checkpoint_epoch")
            if isinstance(checkpoint_epoch, (int, float)):
                return float(checkpoint_epoch)
    checkpoint_epoch = metadata.get("checkpoint_epoch")
    if isinstance(checkpoint_epoch, (int, float)):
        return float(checkpoint_epoch)
    return None


def _format_scalar(scalars: Mapping[str, float], key: str) -> str:
    value = scalars.get(key)
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return ""


def _frame_rows(scan: GateScan, limit: int = 20) -> list[str]:
    rows: list[str] = []
    for frame in scan.common_frames[:limit]:
        rows.append(f"| `{frame.label}` |  |  |  |  |")
    if not rows:
        rows.append("|  |  |  |  |  |")
    return rows


def _format_derived_scalar_lines(scalars: Mapping[str, float]) -> list[str]:
    lines: list[str] = []
    mixed = scalars.get("val/attention_alignment_target_attention_lift_mixed")
    geometry_only = scalars.get("val/attention_alignment_target_attention_lift_geometry_only")
    without_geometry = scalars.get("val/attention_alignment_target_attention_lift_without_geometry")
    if isinstance(mixed, (int, float)) and isinstance(geometry_only, (int, float)):
        lines.append(
            "- `val/derived_target_lift_mixed_minus_geometry_only`: "
            f"{float(mixed) - float(geometry_only):.6g}"
        )
    if isinstance(mixed, (int, float)) and isinstance(without_geometry, (int, float)):
        lines.append(
            "- `val/derived_target_lift_mixed_minus_without_geometry`: "
            f"{float(mixed) - float(without_geometry):.6g}"
        )
    return lines


def render_decision_record(
    scan: GateScan,
    *,
    train_fixed_dir: Path,
    heldout_dir: Path,
    scalars_jsonl: Optional[Path],
    checker_report: Optional[Path],
    contact_sheet: Optional[Path] = None,
    branch: Optional[str] = None,
    commit: Optional[str] = None,
) -> str:
    """Render a markdown decision record prefilled from a gate scan."""
    train_metadata = scan.train_fixed.metadata
    checkpoint_epoch = _metadata_checkpoint_epoch(train_metadata)
    branch = branch if branch is not None else _git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = commit if commit is not None else _git_value(["rev-parse", "--short", "HEAD"])

    lines = [
        "# Pose-Chain Gate Decision",
        "",
        "## Experiment",
        "",
        f"- branch: `{branch}`" if branch else "- branch:",
        f"- commit: `{commit}`" if commit else "- commit:",
        f"- checkpoint: `{train_metadata.get('checkpoint', '')}`",
        f"- checkpoint epoch: {checkpoint_epoch:g}" if checkpoint_epoch is not None else "- checkpoint epoch:",
        f"- split yaml: `{train_metadata.get('split_yaml', '')}`",
        f"- dataset split: `{train_metadata.get('dataset_split', '')}`",
        f"- scalars jsonl: `{scalars_jsonl}`" if scalars_jsonl is not None else "- scalars jsonl:",
        f"- train_fixed output dir: `{train_fixed_dir}`",
        f"- heldout output dir: `{heldout_dir}`",
        f"- checker report: `{checker_report}`" if checker_report is not None else "- checker report:",
        f"- contact sheet: `{contact_sheet}`" if contact_sheet is not None else "- contact sheet:",
        "",
        "## Checker Evidence",
        "",
        f"- checker status: `{'READY_FOR_MANUAL_TEST_GATE' if scan.is_output_complete else 'INCOMPLETE_OUTPUTS'}`",
        f"- paired frames: {len(scan.common_frames)}",
        f"- required paired frames: {scan.min_common_frames}",
    ]
    for key in REQUIRED_VALIDATION_SCALAR_KEYS:
        lines.append(f"- `{key}`: {_format_scalar(scan.latest_scalars, key)}")
    lines.extend(_format_derived_scalar_lines(scan.latest_scalars))

    if scan.cross_errors:
        lines.extend(["", "## Checker Errors"])
        lines.extend(f"- {error}" for error in scan.cross_errors)

    lines.extend(
        [
            "",
            "## Visual Review Sample",
            "",
            "Record at least 20 paired validation/test frames. Prefer frames where the",
            "satellite road layout is visible enough that a yaw/layout error is meaningful.",
            "",
            "| drive/frame | satellite ambiguity | train_fixed continuity | heldout interpolation | failure mode |",
            "| --- | --- | --- | --- | --- |",
            *_frame_rows(scan),
            "",
            "## Decision",
            "",
            "Choose one:",
            "",
            "- `PASS_V1_V2`: train_fixed views preserve continuous road/layout geometry and",
            "  heldout yaws do not collapse, flip, or rotate inconsistently on clear-enough",
            "  test frames.",
            "- `MOVE_TO_V3_LATENT_ACTION`: attention/coverage metrics look plausible but",
            "  generated road direction/layout still breaks on clear-enough test frames, or",
            "  heldout yaws fail while trained yaw anchors are acceptable.",
            "- `RERUN_OR_DEBUG_DATA`: outputs are incomplete, the checker fails, satellite",
            "  evidence is too ambiguous on the reviewed frames, or validation diagnostics",
            "  are not trustworthy enough to judge the algorithm.",
            "",
            "Selected decision:",
            "",
            "## Rationale",
            "",
            "Summarize the evidence in a few concrete sentences. Name representative",
            "drive/frame examples for pass/fail cases and distinguish data ambiguity from",
            "algorithm failure.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a pose-chain gate decision record from train_fixed/heldout outputs."
    )
    parser.add_argument("--train_fixed_dir", type=Path, required=True)
    parser.add_argument("--heldout_dir", type=Path, required=True)
    parser.add_argument("--scalars_jsonl", type=Path, default=None)
    parser.add_argument("--checker_report", type=Path, default=None)
    parser.add_argument("--contact_sheet", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_common_frames", type=int, default=20)
    parser.add_argument(
        "--expected_sat_condition_mode",
        type=str,
        default=DEFAULT_EXPECTED_SAT_CONDITION_MODE,
    )
    parser.add_argument("--require_scalars", action="store_true")
    parser.add_argument("--require_checkpoint_state", action="store_true")
    parser.add_argument("--require_metric_sanity", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scan = scan_gate_outputs(
        args.train_fixed_dir,
        args.heldout_dir,
        scalars_jsonl=args.scalars_jsonl,
        min_common_frames=args.min_common_frames,
        expected_sat_condition_mode=args.expected_sat_condition_mode or None,
        require_scalars=bool(args.require_scalars),
        require_checkpoint_state=bool(args.require_checkpoint_state),
        require_metric_sanity=bool(args.require_metric_sanity),
    )
    rendered = render_decision_record(
        scan,
        train_fixed_dir=args.train_fixed_dir,
        heldout_dir=args.heldout_dir,
        scalars_jsonl=args.scalars_jsonl,
        checker_report=args.checker_report,
        contact_sheet=args.contact_sheet,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote decision record to {args.output}")
    if args.strict and not scan.is_output_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
