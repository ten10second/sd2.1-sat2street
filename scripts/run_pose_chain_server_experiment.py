#!/usr/bin/env python3
"""Run the pose-chain server training, inference, and formal gate sequence."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]

    def shell(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class ExperimentPlan:
    env: Mapping[str, str]
    commands: tuple[CommandSpec, ...]
    checkpoint: Path
    scalars_jsonl: Path
    requires_existing_checkpoint: bool
    requires_existing_scalars: bool
    train_fixed_dir: Path
    heldout_dir: Path
    gate_report: Path
    gate_decision: Path
    gate_contact_sheet: Path


def _checkpoint_path(output_dir: Path, checkpoint_epoch: int, checkpoint_path: Optional[Path]) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path
    return output_dir / "checkpoints" / f"checkpoint_epoch_{int(checkpoint_epoch)}.pt"


def _split_yaml_path(data_dir: Path, split_yaml: Optional[Path]) -> Path:
    return split_yaml if split_yaml is not None else data_dir / "train_test_split_config.yaml"


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.expanduser().resolve(strict=False).relative_to(parent.expanduser().resolve(strict=False))
    except ValueError:
        return False
    return True


def build_experiment_plan(args: argparse.Namespace) -> ExperimentPlan:
    """Build the exact command sequence for a pose-chain server run."""
    data_dir = Path(args.data_dir)
    split_yaml = _split_yaml_path(data_dir, args.split_yaml)
    output_dir = Path(args.output_dir)
    checkpoint = _checkpoint_path(output_dir, args.checkpoint_epoch, args.checkpoint_path)
    gate_output_dir = Path(args.gate_output_dir)
    train_fixed_dir = gate_output_dir / "pose_chain_gate_train_fixed"
    heldout_dir = gate_output_dir / "pose_chain_gate_heldout"
    gate_report = gate_output_dir / "pose_chain_gate_report.md"
    gate_decision = gate_output_dir / "pose_chain_gate_decision.md"
    gate_contact_sheet = gate_output_dir / "pose_chain_gate_contact_sheet.jpg"
    scalars_jsonl = Path(args.scalars_jsonl) if args.scalars_jsonl is not None else output_dir / "logs" / "scalars.jsonl"

    if (
        args.skip_training
        and args.checkpoint_path is not None
        and args.scalars_jsonl is None
        and not _path_is_relative_to(Path(args.checkpoint_path), output_dir)
    ):
        raise ValueError(
            "--skip_training with an external --checkpoint_path requires --scalars_jsonl "
            "from the same training run, so the formal gate cannot mix checkpoint "
            "weights with unrelated validation metrics."
        )

    env = {
        "HF_HOME": str(Path(args.hf_home)),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "POSE_CHAIN_GATE_SPLIT": str(args.gate_split),
        "CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices),
    }

    commands: list[CommandSpec] = [
        CommandSpec(
            "preflight",
            (
                str(args.python),
                "scripts/preflight_pose_chain_group.py",
                "--config",
                str(args.config),
                "--data_dir",
                str(data_dir),
                "--split_yaml",
                str(split_yaml),
                "--hf_home",
                str(args.hf_home),
                "--gate_split",
                str(args.gate_split),
                "--expected_num_gpus",
                str(args.num_gpus),
                "--require_offline_env",
            ),
        )
    ]

    if not args.skip_training:
        commands.append(
            CommandSpec(
                "train",
                (
                    str(args.torchrun),
                    "--standalone",
                    f"--nproc_per_node={int(args.num_gpus)}",
                    "scripts/train.py",
                    "--config",
                    str(args.config),
                    "--data_dir",
                    str(data_dir),
                    "--split_yaml",
                    str(split_yaml),
                    "--output_dir",
                    str(output_dir),
                    "--batch_size",
                    str(args.batch_size),
                    "--gradient_accumulation",
                    str(args.gradient_accumulation),
                    "--validate_every",
                    str(args.validate_every),
                    "--wandb_run_name",
                    str(args.wandb_run_name),
                    "--hf_home",
                    str(args.hf_home),
                ),
            )
        )

    for preset, output in (("train_fixed", train_fixed_dir), ("heldout", heldout_dir)):
        commands.append(
            CommandSpec(
                f"infer_{preset}",
                (
                    str(args.python),
                    "scripts/infer.py",
                    "--config",
                    str(args.inference_config),
                    "--mode",
                    "split_yaw_sweep",
                    "--checkpoint",
                    str(checkpoint),
                    "--data_dir",
                    str(data_dir),
                    "--split_yaml",
                    str(split_yaml),
                    "--dataset_split",
                    str(args.gate_split),
                    "--yaw_sweep_preset",
                    preset,
                    "--max_frames",
                    str(args.max_frames),
                    "--num_inference_steps",
                    str(args.num_inference_steps),
                    "--guidance_scale",
                    str(args.guidance_scale),
                    "--mixed_precision",
                    str(args.mixed_precision),
                    "--output_dir",
                    str(output),
                    "--hf_home",
                    str(args.hf_home),
                ),
            )
        )

    commands.append(
        CommandSpec(
            "gate_bundle",
            (
                str(args.python),
                "scripts/run_pose_chain_gate_bundle.py",
                "--train_fixed_dir",
                str(train_fixed_dir),
                "--heldout_dir",
                str(heldout_dir),
                "--scalars_jsonl",
                str(scalars_jsonl),
                "--report_output",
                str(gate_report),
                "--decision_output",
                str(gate_decision),
                "--contact_sheet_output",
                str(gate_contact_sheet),
                "--min_common_frames",
                str(args.min_common_frames),
                "--expected_sat_condition_mode",
                "normal",
                "--require_scalars",
                "--require_checkpoint_state",
                "--require_metric_sanity",
                "--strict",
            ),
        )
    )

    return ExperimentPlan(
        env=env,
        commands=tuple(commands),
        checkpoint=checkpoint,
        scalars_jsonl=scalars_jsonl,
        requires_existing_checkpoint=bool(args.skip_training),
        requires_existing_scalars=bool(args.skip_training),
        train_fixed_dir=train_fixed_dir,
        heldout_dir=heldout_dir,
        gate_report=gate_report,
        gate_decision=gate_decision,
        gate_contact_sheet=gate_contact_sheet,
    )


def _print_plan(plan: ExperimentPlan) -> None:
    print("# Environment")
    for key, value in plan.env.items():
        print(f"export {key}={shlex.quote(str(value))}")
    print("")
    for command in plan.commands:
        print(f"# {command.name}")
        print(command.shell())
        print("")


def run_plan(plan: ExperimentPlan, *, dry_run: bool = False) -> None:
    if dry_run:
        _print_plan(plan)
        return

    if plan.requires_existing_checkpoint and not plan.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found before inference: {plan.checkpoint}")
    if plan.requires_existing_scalars and not plan.scalars_jsonl.is_file():
        raise FileNotFoundError(f"scalars_jsonl not found before inference/gate: {plan.scalars_jsonl}")

    env = os.environ.copy()
    env.update({key: str(value) for key, value in plan.env.items()})
    for command in plan.commands:
        if command.name.startswith("infer_") and not plan.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found before {command.name}: {plan.checkpoint}")
        print(f"[pose-chain] running {command.name}: {command.shell()}", flush=True)
        subprocess.run(command.argv, check=True, env=env)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pose-chain group server training, split yaw sweeps, and the formal gate bundle."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--inference_config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split_yaml", type=Path, default=None)
    parser.add_argument("--hf_home", type=Path, default=Path(".hf-home"))
    parser.add_argument("--gate_split", choices=("val", "test"), default="test")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--cuda_visible_devices", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--torchrun", type=str, default="torchrun")
    parser.add_argument("--output_dir", type=Path, default=Path("output/wip_pose_chain_group_conditioning"))
    parser.add_argument("--gate_output_dir", type=Path, default=Path("inference_results"))
    parser.add_argument("--checkpoint_epoch", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--scalars_jsonl", type=Path, default=None)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument("--wandb_run_name", type=str, default="pose_chain_group_v1")
    parser.add_argument("--max_frames", type=int, default=20)
    parser.add_argument("--min_common_frames", type=int, default=20)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    plan = build_experiment_plan(args)
    run_plan(plan, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
