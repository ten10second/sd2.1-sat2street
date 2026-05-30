#!/usr/bin/env python3
"""Inspect pose-chain training/gate artifacts without mutating a run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
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


CHECKPOINT_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


@dataclass
class RunStatus:
    output_dir: Path
    gate_output_dir: Path
    train_fixed_dir: Path
    heldout_dir: Path
    gate_final_report: Path
    latest_checkpoint: Optional[Path] = None
    latest_checkpoint_epoch: Optional[int] = None
    scalars_jsonl: Optional[Path] = None
    latest_train_scalars: Mapping[str, float] = field(default_factory=dict)
    latest_val_scalars: Mapping[str, float] = field(default_factory=dict)
    final_selected_decision: Optional[str] = None
    gate_scan: Optional[GateScan] = None
    missing_gate_artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _checkpoint_epoch(path: Path) -> Optional[int]:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _latest_checkpoint(output_dir: Path) -> tuple[Optional[Path], Optional[int]]:
    checkpoints_dir = output_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if checkpoints_dir.is_dir():
        for path in checkpoints_dir.glob("checkpoint_epoch_*.pt"):
            epoch = _checkpoint_epoch(path)
            if epoch is not None:
                candidates.append((epoch, path))
    if not candidates:
        return None, None
    epoch, path = max(candidates, key=lambda item: item[0])
    return path, epoch


def _load_latest_scalar_records(scalars_jsonl: Path) -> tuple[Mapping[str, float], Mapping[str, float], list[str]]:
    latest_train: Mapping[str, float] = {}
    latest_val: Mapping[str, float] = {}
    warnings: list[str] = []
    if not scalars_jsonl.is_file():
        return latest_train, latest_val, [f"scalars jsonl not found: {scalars_jsonl}"]

    for line_number, line in enumerate(scalars_jsonl.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            warnings.append(f"invalid JSON in scalars line {line_number}: {exc}")
            continue
        if not isinstance(record, dict):
            continue
        train_keys = [str(key) for key in record if str(key).startswith("train/")]
        val_keys = [str(key) for key in record if str(key).startswith("val/")]
        has_train_metric = any(key != "train/epoch" for key in train_keys)
        if train_keys and (has_train_metric or not val_keys):
            latest_train = record
        if val_keys:
            latest_val = record
    return latest_train, latest_val, warnings


def _required_gate_artifacts(
    *,
    train_fixed_dir: Path,
    heldout_dir: Path,
    gate_report: Path,
    gate_decision: Path,
    gate_contact_sheet: Path,
    gate_final_report: Path,
) -> list[Path]:
    return [
        train_fixed_dir,
        heldout_dir,
        gate_report,
        gate_decision,
        gate_contact_sheet,
        gate_final_report,
    ]


FINAL_DECISION_RE = re.compile(r"selected decision:\s*`([^`]+)`")


def _read_final_selected_decision(final_report: Path) -> Optional[str]:
    if not final_report.is_file():
        return None
    try:
        text = final_report.read_text()
    except OSError:
        return None
    match = FINAL_DECISION_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip() or None


def inspect_run_status(
    *,
    output_dir: Path,
    gate_output_dir: Path,
    train_fixed_dir: Optional[Path] = None,
    heldout_dir: Optional[Path] = None,
    gate_report: Optional[Path] = None,
    gate_decision: Optional[Path] = None,
    gate_contact_sheet: Optional[Path] = None,
    gate_final_report: Optional[Path] = None,
    scalars_jsonl: Optional[Path] = None,
    min_common_frames: int = 20,
    expected_sat_condition_mode: Optional[str] = DEFAULT_EXPECTED_SAT_CONDITION_MODE,
) -> RunStatus:
    scalars_path = scalars_jsonl if scalars_jsonl is not None else output_dir / "logs" / "scalars.jsonl"
    train_fixed_path = train_fixed_dir if train_fixed_dir is not None else gate_output_dir / "pose_chain_gate_train_fixed"
    heldout_path = heldout_dir if heldout_dir is not None else gate_output_dir / "pose_chain_gate_heldout"
    report_path = gate_report if gate_report is not None else gate_output_dir / "pose_chain_gate_report.md"
    decision_path = gate_decision if gate_decision is not None else gate_output_dir / "pose_chain_gate_decision.md"
    contact_sheet_path = (
        gate_contact_sheet
        if gate_contact_sheet is not None
        else gate_output_dir / "pose_chain_gate_contact_sheet.jpg"
    )
    final_report_path = (
        gate_final_report
        if gate_final_report is not None
        else gate_output_dir / "pose_chain_gate_final_report.md"
    )
    latest_checkpoint, latest_epoch = _latest_checkpoint(output_dir)
    latest_train, latest_val, warnings = _load_latest_scalar_records(scalars_path)

    required_artifacts = _required_gate_artifacts(
        train_fixed_dir=train_fixed_path,
        heldout_dir=heldout_path,
        gate_report=report_path,
        gate_decision=decision_path,
        gate_contact_sheet=contact_sheet_path,
        gate_final_report=final_report_path,
    )
    missing_artifacts = [path for path in required_artifacts if not path.exists()]
    final_decision = _read_final_selected_decision(final_report_path)

    gate_scan: Optional[GateScan] = None
    if train_fixed_path.is_dir() and heldout_path.is_dir():
        gate_scan = scan_gate_outputs(
            train_fixed_path,
            heldout_path,
            scalars_jsonl=scalars_path if scalars_path.is_file() else None,
            min_common_frames=min_common_frames,
            expected_sat_condition_mode=expected_sat_condition_mode,
            require_scalars=False,
            require_checkpoint_state=False,
            require_metric_sanity=False,
        )

    return RunStatus(
        output_dir=output_dir,
        gate_output_dir=gate_output_dir,
        train_fixed_dir=train_fixed_path,
        heldout_dir=heldout_path,
        gate_final_report=final_report_path,
        latest_checkpoint=latest_checkpoint,
        latest_checkpoint_epoch=latest_epoch,
        scalars_jsonl=scalars_path,
        latest_train_scalars=latest_train,
        latest_val_scalars=latest_val,
        final_selected_decision=final_decision,
        gate_scan=gate_scan,
        missing_gate_artifacts=missing_artifacts,
        warnings=warnings,
    )


def _format_scalar(record: Mapping[str, float], key: str) -> str:
    value = record.get(key)
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return "missing"


def _format_first_scalar(record: Mapping[str, float], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return f"{float(value):.6g}"
    return "missing"


def _format_epoch(record: Mapping[str, float], primary_key: str) -> str:
    for key in (primary_key, "train/epoch", "epoch"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return f"{float(value):.6g}"
    return "missing"


def render_status_markdown(status: RunStatus) -> str:
    gate_state = "not_started"
    if status.gate_scan is not None:
        gate_state = "ready" if status.gate_scan.is_output_complete else "incomplete"

    lines = [
        "# Pose-Chain Run Status",
        "",
        "## Training",
        "",
        f"- output dir: `{status.output_dir}`",
        f"- latest checkpoint: `{status.latest_checkpoint}`" if status.latest_checkpoint else "- latest checkpoint: missing",
        f"- latest checkpoint epoch: {status.latest_checkpoint_epoch}" if status.latest_checkpoint_epoch is not None else "- latest checkpoint epoch: missing",
        f"- scalars jsonl: `{status.scalars_jsonl}`" if status.scalars_jsonl is not None else "- scalars jsonl: missing",
        f"- latest train epoch: {_format_epoch(status.latest_train_scalars, 'train/epoch')}",
        f"- latest train raw loss: {_format_first_scalar(status.latest_train_scalars, ('train/raw_loss', 'train/epoch_raw_loss'))}",
        f"- latest val epoch: {_format_epoch(status.latest_val_scalars, 'val/epoch')}",
        f"- latest val loss: {_format_scalar(status.latest_val_scalars, 'val/loss')}",
        "",
        "## Validation Gate Metrics",
        "",
    ]
    for key in REQUIRED_VALIDATION_SCALAR_KEYS:
        lines.append(f"- `{key}`: {_format_scalar(status.latest_val_scalars, key)}")

    lines.extend(
        [
            "",
            "## Gate Artifacts",
            "",
            f"- gate output dir: `{status.gate_output_dir}`",
            f"- train_fixed dir: `{status.train_fixed_dir}`",
            f"- heldout dir: `{status.heldout_dir}`",
            f"- final report: `{status.gate_final_report}`",
            f"- final selected decision: `{status.final_selected_decision}`"
            if status.final_selected_decision
            else "- final selected decision: missing",
            f"- gate scan state: `{gate_state}`",
        ]
    )
    if status.gate_scan is not None:
        lines.extend(
            [
                f"- paired frames: {len(status.gate_scan.common_frames)}",
                f"- required paired frames: {status.gate_scan.min_common_frames}",
                f"- train_fixed complete frames: {status.gate_scan.train_fixed.complete_frames}",
                f"- heldout complete frames: {status.gate_scan.heldout.complete_frames}",
            ]
        )
    if status.missing_gate_artifacts:
        lines.append("- missing gate artifacts:")
        lines.extend(f"  - `{path}`" for path in status.missing_gate_artifacts)
    else:
        lines.append("- missing gate artifacts: none")

    if status.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in status.warnings)
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect pose-chain run/checkpoint/gate status.")
    parser.add_argument("--output_dir", type=Path, default=Path("output/wip_pose_chain_group_conditioning"))
    parser.add_argument("--gate_output_dir", type=Path, default=Path("inference_results"))
    parser.add_argument("--train_fixed_dir", type=Path, default=None)
    parser.add_argument("--heldout_dir", type=Path, default=None)
    parser.add_argument("--gate_report", type=Path, default=None)
    parser.add_argument("--gate_decision", type=Path, default=None)
    parser.add_argument("--gate_contact_sheet", type=Path, default=None)
    parser.add_argument("--gate_final_report", type=Path, default=None)
    parser.add_argument("--scalars_jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min_common_frames", type=int, default=20)
    parser.add_argument(
        "--expected_sat_condition_mode",
        type=str,
        default=DEFAULT_EXPECTED_SAT_CONDITION_MODE,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    status = inspect_run_status(
        output_dir=args.output_dir,
        gate_output_dir=args.gate_output_dir,
        train_fixed_dir=args.train_fixed_dir,
        heldout_dir=args.heldout_dir,
        gate_report=args.gate_report,
        gate_decision=args.gate_decision,
        gate_contact_sheet=args.gate_contact_sheet,
        gate_final_report=args.gate_final_report,
        scalars_jsonl=args.scalars_jsonl,
        min_common_frames=args.min_common_frames,
        expected_sat_condition_mode=args.expected_sat_condition_mode or None,
    )
    rendered = render_status_markdown(status)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Wrote pose-chain run status to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
