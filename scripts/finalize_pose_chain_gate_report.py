#!/usr/bin/env python3
"""Create the final pose-chain gate report after manual visual review."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Mapping, Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.check_pose_chain_gate_outputs import (
    DEFAULT_EXPECTED_SAT_CONDITION_MODE,
    REQUIRED_VALIDATION_SCALAR_KEYS,
    GateScan,
    render_markdown_report,
    scan_gate_outputs,
)
from scripts.init_pose_chain_gate_decision import _metadata_checkpoint_epoch


DECISION_CHOICES = (
    "PASS_V1_V2",
    "MOVE_TO_V3_LATENT_ACTION",
    "RERUN_OR_DEBUG_DATA",
)


def _format_scalar(scalars: Mapping[str, float], key: str) -> str:
    value = scalars.get(key)
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return "missing"


def _decision_next_step(selected_decision: str) -> str:
    if selected_decision == "PASS_V1_V2":
        return (
            "Keep the geometry-first semantic-refine pose-chain design as the "
            "current v1/v2 result. Do not start latent-action auxiliary yet; "
            "use the passed validation/test result for comparison against future variants."
        )
    if selected_decision == "MOVE_TO_V3_LATENT_ACTION":
        return (
            "Start v3 latent-action auxiliary. The gate says attention/coverage "
            "is plausible enough to keep the addressing design, but generation "
            "still needs an explicit pose-transition/layout consistency signal."
        )
    return (
        "Do not use this run to judge the algorithm. Rerun the formal gate or "
        "debug data/checkpoint/runtime consistency before deciding on latent-action auxiliary."
    )


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


def _commands_block(
    *,
    train_fixed_dir: Path,
    heldout_dir: Path,
    checkpoint: Optional[object],
    split_yaml: Optional[object],
    dataset_split: Optional[object],
    runtime_config: Optional[Mapping],
    max_frames: Optional[int],
    scalars_jsonl: Optional[Path],
    checker_report: Optional[Path],
    decision_record: Optional[Path],
    contact_sheet: Optional[Path],
    min_common_frames: int,
    expected_sat_condition_mode: Optional[str],
) -> str:
    checkpoint_arg = str(checkpoint) if checkpoint else "output/wip_pose_chain_group_conditioning/checkpoints/checkpoint_epoch_100.pt"
    split_yaml_arg = str(split_yaml) if split_yaml else "/mnt/shizhm/KITTI-360/train_test_split_config.yaml"
    dataset_split_arg = str(dataset_split) if dataset_split in {"val", "test"} else "$POSE_CHAIN_GATE_SPLIT"
    dataset_split_shell = dataset_split_arg if dataset_split_arg == "$POSE_CHAIN_GATE_SPLIT" else shlex.quote(dataset_split_arg)
    runtime = runtime_config if isinstance(runtime_config, Mapping) else {}
    num_inference_steps = int(runtime.get("num_inference_steps", 50) or 50)
    guidance_scale = float(runtime.get("guidance_scale", 7.5) or 7.5)
    mixed_precision = str(runtime.get("mixed_precision", "bf16") or "bf16")
    frame_count = max(1, int(max_frames)) if isinstance(max_frames, int) and max_frames > 0 else 20
    scalars = str(scalars_jsonl) if scalars_jsonl is not None else "output/wip_pose_chain_group_conditioning/logs/scalars.jsonl"
    checker = str(checker_report) if checker_report is not None else "inference_results/pose_chain_gate_report.md"
    decision = str(decision_record) if decision_record is not None else "inference_results/pose_chain_gate_decision.md"
    contact = str(contact_sheet) if contact_sheet is not None else "inference_results/pose_chain_gate_contact_sheet.jpg"
    sat_mode_arg = expected_sat_condition_mode or ""
    return "\n".join(
        [
            "```bash",
            "export HF_HOME=/mnt/shizhm/sd2.1-sat2street/.hf-home",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export DIFFUSERS_OFFLINE=1",
            "export POSE_CHAIN_GATE_SPLIT=\"${POSE_CHAIN_GATE_SPLIT:-test}\"",
            "",
            "# 8-GPU training entry point, after preflight:",
            "torchrun --standalone --nproc_per_node=8 scripts/train.py \\",
            "  --config configs/train.yaml \\",
            "  --data_dir /mnt/shizhm/KITTI-360 \\",
            "  --split_yaml /mnt/shizhm/KITTI-360/train_test_split_config.yaml \\",
            "  --batch_size 2 \\",
            "  --gradient_accumulation 2 \\",
            "  --validate_every 10 \\",
            "  --wandb_run_name pose_chain_group_v1 \\",
            "  --hf_home \"$HF_HOME\"",
            "",
            "# Validation/test yaw sweeps for the evaluated checkpoint:",
            f"CHECKPOINT={shlex.quote(checkpoint_arg)}",
            "python scripts/infer.py \\",
            "  --config configs/inference.yaml \\",
            "  --mode split_yaw_sweep \\",
            "  --checkpoint \"$CHECKPOINT\" \\",
            "  --data_dir /mnt/shizhm/KITTI-360 \\",
            f"  --split_yaml {shlex.quote(split_yaml_arg)} \\",
            f"  --dataset_split {dataset_split_shell} \\",
            "  --yaw_sweep_preset train_fixed \\",
            f"  --max_frames {frame_count} \\",
            f"  --num_inference_steps {num_inference_steps} \\",
            f"  --guidance_scale {guidance_scale:g} \\",
            f"  --mixed_precision {shlex.quote(mixed_precision)} \\",
            f"  --output_dir {shlex.quote(str(train_fixed_dir))} \\",
            "  --hf_home \"$HF_HOME\"",
            "",
            "python scripts/infer.py \\",
            "  --config configs/inference.yaml \\",
            "  --mode split_yaw_sweep \\",
            "  --checkpoint \"$CHECKPOINT\" \\",
            "  --data_dir /mnt/shizhm/KITTI-360 \\",
            f"  --split_yaml {shlex.quote(split_yaml_arg)} \\",
            f"  --dataset_split {dataset_split_shell} \\",
            "  --yaw_sweep_preset heldout \\",
            f"  --max_frames {frame_count} \\",
            f"  --num_inference_steps {num_inference_steps} \\",
            f"  --guidance_scale {guidance_scale:g} \\",
            f"  --mixed_precision {shlex.quote(mixed_precision)} \\",
            f"  --output_dir {shlex.quote(str(heldout_dir))} \\",
            "  --hf_home \"$HF_HOME\"",
            "",
            "# Formal gate bundle for the evaluated checkpoint:",
            "python scripts/run_pose_chain_gate_bundle.py \\",
            f"  --train_fixed_dir {train_fixed_dir} \\",
            f"  --heldout_dir {heldout_dir} \\",
            f"  --scalars_jsonl {scalars} \\",
            f"  --report_output {checker} \\",
            f"  --decision_output {decision} \\",
            f"  --contact_sheet_output {contact} \\",
            f"  --min_common_frames {int(min_common_frames)} \\",
            f"  --expected_sat_condition_mode {sat_mode_arg} \\",
            "  --require_scalars \\",
            "  --require_checkpoint_state \\",
            "  --require_metric_sanity \\",
            "  --strict",
            "```",
        ]
    )


def render_final_report(
    scan: GateScan,
    *,
    selected_decision: str,
    rationale: str,
    train_fixed_dir: Path,
    heldout_dir: Path,
    scalars_jsonl: Optional[Path],
    checker_report: Optional[Path],
    decision_record: Optional[Path],
    contact_sheet: Optional[Path],
    expected_sat_condition_mode: Optional[str],
) -> str:
    """Render the final manual gate report with the scanned evidence attached."""
    train_metadata = scan.train_fixed.metadata
    checkpoint_epoch = _metadata_checkpoint_epoch(train_metadata)
    status = "READY_FOR_MANUAL_TEST_GATE" if scan.is_output_complete else "INCOMPLETE_OUTPUTS"
    lines = [
        "# Pose-Chain Final Gate Report",
        "",
        "## Decision",
        "",
        f"- selected decision: `{selected_decision}`",
        f"- next step: {_decision_next_step(selected_decision)}",
        "",
        "## Rationale",
        "",
        rationale.strip(),
        "",
        "## Evidence",
        "",
        f"- checker status: `{status}`",
        f"- paired frames: {len(scan.common_frames)}",
        f"- required paired frames: {scan.min_common_frames}",
        f"- dataset split: `{train_metadata.get('dataset_split', '')}`",
        f"- checkpoint: `{train_metadata.get('checkpoint', '')}`",
        f"- checkpoint epoch: {checkpoint_epoch:g}" if checkpoint_epoch is not None else "- checkpoint epoch:",
        f"- split yaml: `{train_metadata.get('split_yaml', '')}`",
        f"- scalars jsonl: `{scalars_jsonl}`" if scalars_jsonl is not None else "- scalars jsonl:",
        f"- checker report: `{checker_report}`" if checker_report is not None else "- checker report:",
        f"- decision record: `{decision_record}`" if decision_record is not None else "- decision record:",
        f"- contact sheet: `{contact_sheet}`" if contact_sheet is not None else "- contact sheet:",
        "",
        "## Key Metrics",
        "",
    ]
    for key in REQUIRED_VALIDATION_SCALAR_KEYS:
        lines.append(f"- `{key}`: {_format_scalar(scan.latest_scalars, key)}")
    lines.extend(_format_derived_scalar_lines(scan.latest_scalars))

    if scan.cross_errors:
        lines.extend(["", "## Gate Errors"])
        lines.extend(f"- {error}" for error in scan.cross_errors)

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            _commands_block(
                train_fixed_dir=train_fixed_dir,
                heldout_dir=heldout_dir,
                checkpoint=train_metadata.get("checkpoint"),
                split_yaml=train_metadata.get("split_yaml"),
                dataset_split=train_metadata.get("dataset_split"),
                runtime_config=train_metadata.get("inference_runtime_config"),
                max_frames=train_metadata.get("num_frames") if isinstance(train_metadata.get("num_frames"), int) else None,
                scalars_jsonl=scalars_jsonl,
                checker_report=checker_report,
                decision_record=decision_record,
                contact_sheet=contact_sheet,
                min_common_frames=scan.min_common_frames,
                expected_sat_condition_mode=expected_sat_condition_mode,
            ),
            "",
            "## Checker Snapshot",
            "",
            render_markdown_report(scan),
            "",
        ]
    )
    return "\n".join(lines)


def _read_rationale(args: argparse.Namespace) -> str:
    if args.rationale_file is not None:
        return args.rationale_file.read_text().strip()
    return str(args.rationale or "").strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize the pose-chain gate decision after manual visual review."
    )
    parser.add_argument("--train_fixed_dir", type=Path, required=True)
    parser.add_argument("--heldout_dir", type=Path, required=True)
    parser.add_argument("--scalars_jsonl", type=Path, default=None)
    parser.add_argument("--checker_report", type=Path, default=None)
    parser.add_argument("--decision_record", type=Path, default=None)
    parser.add_argument("--contact_sheet", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected_decision", choices=DECISION_CHOICES, required=True)
    parser.add_argument("--rationale", type=str, default=None)
    parser.add_argument("--rationale_file", type=Path, default=None)
    parser.add_argument("--min_common_frames", type=int, default=20)
    parser.add_argument(
        "--expected_sat_condition_mode",
        type=str,
        default=DEFAULT_EXPECTED_SAT_CONDITION_MODE,
    )
    parser.add_argument("--require_scalars", action="store_true")
    parser.add_argument("--require_checkpoint_state", action="store_true")
    parser.add_argument("--require_metric_sanity", action="store_true")
    parser.add_argument(
        "--allow_incomplete_for_rerun",
        action="store_true",
        help="Allow finalizing RERUN_OR_DEBUG_DATA when the formal checker is incomplete.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rationale = _read_rationale(args)
    if not rationale:
        raise SystemExit("--rationale or --rationale_file is required")

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
    if not scan.is_output_complete:
        if not (args.allow_incomplete_for_rerun and args.selected_decision == "RERUN_OR_DEBUG_DATA"):
            raise SystemExit(
                "formal gate outputs are incomplete; rerun checker or use "
                "--selected_decision RERUN_OR_DEBUG_DATA --allow_incomplete_for_rerun"
            )

    rendered = render_final_report(
        scan,
        selected_decision=args.selected_decision,
        rationale=rationale,
        train_fixed_dir=args.train_fixed_dir,
        heldout_dir=args.heldout_dir,
        scalars_jsonl=args.scalars_jsonl,
        checker_report=args.checker_report,
        decision_record=args.decision_record,
        contact_sheet=args.contact_sheet,
        expected_sat_condition_mode=args.expected_sat_condition_mode or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(f"Wrote final gate report to {args.output}")
    print("selected_decision:", args.selected_decision)


if __name__ == "__main__":
    main()
