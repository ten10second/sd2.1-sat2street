#!/usr/bin/env python3
"""Write the formal pose-chain gate report and decision record together."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.check_pose_chain_gate_outputs import (
    DEFAULT_EXPECTED_SAT_CONDITION_MODE,
    GateScan,
    render_markdown_report,
    scan_gate_outputs,
)
from scripts.init_pose_chain_gate_decision import render_decision_record


def _fit_to_width(image: Image.Image, width: int) -> Image.Image:
    width = max(1, int(width))
    if image.width == width:
        return image.copy()
    height = max(1, round(float(image.height) * float(width) / float(max(1, image.width))))
    return image.resize((width, height), resample=Image.BILINEAR)


def _summary_image(frame_path: Path, width: int) -> Image.Image:
    image = Image.open(frame_path / "summary.png").convert("RGB")
    return _fit_to_width(image, width)


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text((xy[0] + 1, xy[1] + 1), text, fill=(0, 0, 0))
    draw.text(xy, text, fill=(255, 255, 255))


def write_contact_sheet(
    scan: GateScan,
    output: Path,
    *,
    max_frames: int = 20,
    panel_width: int = 960,
) -> int:
    """Write paired train_fixed/heldout summary rows for visual gate review."""
    rows: list[Image.Image] = []
    selected_frames = scan.common_frames[: max(0, int(max_frames))]
    panel_width = max(64, int(panel_width))
    gutter = 8
    header_h = 34

    for frame in selected_frames:
        train_frame = scan.train_fixed.frames.get(frame)
        heldout_frame = scan.heldout.frames.get(frame)
        if train_frame is None or heldout_frame is None:
            continue
        train_image = _summary_image(train_frame.path, panel_width)
        heldout_image = _summary_image(heldout_frame.path, panel_width)
        image_h = max(train_image.height, heldout_image.height)
        row = Image.new("RGB", (panel_width * 2 + gutter, header_h + image_h), color=(18, 18, 18))
        draw = ImageDraw.Draw(row)
        _draw_text(draw, (8, 8), f"{frame.label} | train_fixed")
        _draw_text(draw, (panel_width + gutter + 8, 8), f"{frame.label} | heldout")
        row.paste(train_image, (0, header_h))
        row.paste(heldout_image, (panel_width + gutter, header_h))
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        empty = Image.new("RGB", (panel_width * 2 + gutter, header_h), color=(18, 18, 18))
        draw = ImageDraw.Draw(empty)
        _draw_text(draw, (8, 8), "No paired train_fixed/heldout summaries available")
        empty.save(output)
        return 0

    total_h = sum(row.height for row in rows)
    sheet = Image.new("RGB", (rows[0].width, total_h), color=(18, 18, 18))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output)
    return len(rows)


def write_gate_bundle(
    *,
    train_fixed_dir: Path,
    heldout_dir: Path,
    report_output: Path,
    decision_output: Path,
    contact_sheet_output: Optional[Path] = None,
    contact_sheet_max_frames: int = 20,
    contact_sheet_panel_width: int = 960,
    scalars_jsonl: Optional[Path] = None,
    min_common_frames: int = 20,
    expected_sat_condition_mode: Optional[str] = DEFAULT_EXPECTED_SAT_CONDITION_MODE,
    require_scalars: bool = False,
    require_checkpoint_state: bool = False,
    require_metric_sanity: bool = False,
) -> GateScan:
    """Scan gate outputs once and write both review artifacts."""
    scan = scan_gate_outputs(
        train_fixed_dir,
        heldout_dir,
        scalars_jsonl=scalars_jsonl,
        min_common_frames=min_common_frames,
        expected_sat_condition_mode=expected_sat_condition_mode,
        require_scalars=require_scalars,
        require_checkpoint_state=require_checkpoint_state,
        require_metric_sanity=require_metric_sanity,
    )

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(render_markdown_report(scan) + "\n")

    if contact_sheet_output is not None:
        write_contact_sheet(
            scan,
            contact_sheet_output,
            max_frames=contact_sheet_max_frames,
            panel_width=contact_sheet_panel_width,
        )

    decision_output.parent.mkdir(parents=True, exist_ok=True)
    decision_output.write_text(
        render_decision_record(
            scan,
            train_fixed_dir=train_fixed_dir,
            heldout_dir=heldout_dir,
            scalars_jsonl=scalars_jsonl,
            checker_report=report_output,
            contact_sheet=contact_sheet_output,
        )
        + "\n"
    )
    return scan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the pose-chain formal gate report and manual decision "
            "record from the same validation/test evidence."
        )
    )
    parser.add_argument("--train_fixed_dir", type=Path, required=True)
    parser.add_argument("--heldout_dir", type=Path, required=True)
    parser.add_argument("--scalars_jsonl", type=Path, default=None)
    parser.add_argument("--report_output", type=Path, required=True)
    parser.add_argument("--decision_output", type=Path, required=True)
    parser.add_argument(
        "--contact_sheet_output",
        type=Path,
        default=None,
        help="Optional paired train_fixed/heldout summary image for visual review.",
    )
    parser.add_argument("--contact_sheet_max_frames", type=int, default=20)
    parser.add_argument("--contact_sheet_panel_width", type=int, default=960)
    parser.add_argument(
        "--min_common_frames",
        type=int,
        default=20,
        help="Minimum paired validation/test frames required for formal readiness.",
    )
    parser.add_argument(
        "--expected_sat_condition_mode",
        type=str,
        default=DEFAULT_EXPECTED_SAT_CONDITION_MODE,
        help=(
            "Expected satellite conditioning mode for the formal gate. "
            "Use an empty string to disable this check for ablation-only diagnostics."
        ),
    )
    parser.add_argument("--require_scalars", action="store_true")
    parser.add_argument("--require_checkpoint_state", action="store_true")
    parser.add_argument("--require_metric_sanity", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 when outputs are incomplete.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scan = write_gate_bundle(
        train_fixed_dir=args.train_fixed_dir,
        heldout_dir=args.heldout_dir,
        scalars_jsonl=args.scalars_jsonl,
        report_output=args.report_output,
        decision_output=args.decision_output,
        contact_sheet_output=args.contact_sheet_output,
        contact_sheet_max_frames=args.contact_sheet_max_frames,
        contact_sheet_panel_width=args.contact_sheet_panel_width,
        min_common_frames=args.min_common_frames,
        expected_sat_condition_mode=args.expected_sat_condition_mode or None,
        require_scalars=bool(args.require_scalars),
        require_checkpoint_state=bool(args.require_checkpoint_state),
        require_metric_sanity=bool(args.require_metric_sanity),
    )
    print(f"Wrote gate report to {args.report_output}")
    print(f"Wrote decision record to {args.decision_output}")
    if args.contact_sheet_output is not None:
        print(f"Wrote contact sheet to {args.contact_sheet_output}")
    print("status:", "READY_FOR_MANUAL_TEST_GATE" if scan.is_output_complete else "INCOMPLETE_OUTPUTS")
    if args.strict and not scan.is_output_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
