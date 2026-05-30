# Pose-Chain Group Runbook

This branch trains the geometry-first semantic-refine model with grouped pose chains:

- `right`: `front -> +60 -> +90 -> +120`
- `left`: `front -> -60 -> -90 -> -120`

Each sample shares one satellite content memory across the chain, then applies each view's own camera pose/geometry for cross-attention addressing.
All configured pose chains must have the same number of views so batching,
loss averaging, and chain diagnostics keep a consistent group shape. Chain
names must be unique, and `yaws`/`views` must be YAML lists rather than a
comma-separated string.

For the Codex-facing implementation checklist and completion gate, see
`docs/pose_chain_group_codex_plan.md`.
For the post-checker research decision record, use
`docs/pose_chain_gate_decision_template.md`.

## 8-GPU Training

Use offline Hugging Face cache on the server so startup does not depend on network access:

```bash
cd /mnt/shizhm/sd2.1-sat2street
conda activate "${POSE_CHAIN_CONDA_ENV:-maskgit}"

export HF_HOME=/mnt/shizhm/sd2.1-sat2street/.hf-home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export POSE_CHAIN_GATE_SPLIT="${POSE_CHAIN_GATE_SPLIT:-test}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python scripts/preflight_pose_chain_group.py \
  --config configs/train.yaml \
  --data_dir /mnt/shizhm/KITTI-360 \
  --split_yaml /mnt/shizhm/KITTI-360/train_test_split_config.yaml \
  --hf_home "$HF_HOME" \
  --gate_split "$POSE_CHAIN_GATE_SPLIT" \
  --expected_num_gpus 8 \
  --require_offline_env

torchrun --standalone --nproc_per_node=8 \
  scripts/train.py \
  --config configs/train.yaml \
  --data_dir /mnt/shizhm/KITTI-360 \
  --split_yaml /mnt/shizhm/KITTI-360/train_test_split_config.yaml \
  --batch_size 2 \
  --gradient_accumulation 2 \
  --validate_every 10 \
  --wandb_run_name pose_chain_group_v1 \
  --hf_home "$HF_HOME"
```

Or run the full server sequence with one checked command:

```bash
cd /mnt/shizhm/sd2.1-sat2street
conda activate "${POSE_CHAIN_CONDA_ENV:-maskgit}"

python scripts/run_pose_chain_server_experiment.py \
  --data_dir /mnt/shizhm/KITTI-360 \
  --split_yaml /mnt/shizhm/KITTI-360/train_test_split_config.yaml \
  --hf_home /mnt/shizhm/sd2.1-sat2street/.hf-home \
  --gate_split "${POSE_CHAIN_GATE_SPLIT:-test}" \
  --num_gpus 8 \
  --cuda_visible_devices 0,1,2,3,4,5,6,7
```

Use `--dry_run` first if you want to inspect the exact preflight, training,
inference, and gate bundle commands without running them. Use `--skip_training
--checkpoint_path /path/to/checkpoint.pt` to rerun only inference and the gate
for an existing checkpoint. If that checkpoint does not live under the current
`--output_dir`, the runner requires `--scalars_jsonl /path/to/scalars.jsonl` so
the gate uses the validation metrics from the same training run.

During or after the run, inspect status without mutating outputs:

```bash
python scripts/inspect_pose_chain_run_status.py \
  --output_dir output/wip_pose_chain_group_conditioning \
  --gate_output_dir inference_results \
  --min_common_frames 20
```

For legacy or smoke outputs with custom directory names, pass
`--train_fixed_dir`, `--heldout_dir`, and the gate artifact paths explicitly,
including `--gate_final_report` if the final report is not under
`--gate_output_dir`.

Expected output roots:

- checkpoints: `output/wip_pose_chain_group_conditioning/checkpoints/`
- scalar gate metrics: `output/wip_pose_chain_group_conditioning/logs/scalars.jsonl`
- visualizations: `output/wip_pose_chain_group_conditioning/visualizations/`
- formal gate report/decision/contact sheet/final report:
  `inference_results/pose_chain_gate_*.md` and
  `inference_results/pose_chain_gate_contact_sheet.jpg`

Validation is run on rank 0 over the full validation/test split. It is not distributed-sharded.
The split yaml may name this section either `val` or `test`; use the same
`--dataset_split` value for both yaw-sweep commands because the gate checker
requires matching metadata.

## Local End-to-End Smoke

This smoke only checks that grouped training, checkpoint metadata, split yaw-sweep inference, and the gate checker all run. It is not a visual-quality result.

```bash
cd /home/shizhm/sd2.1-sat2street
conda activate maskgit
export HF_HOME="$(pwd)/.hf-home"

SMOKE_DIR=output/pose_chain_group_smoke
mkdir -p "$SMOKE_DIR"
printf '1\n' > "$SMOKE_DIR/train_frames.txt"
printf '978\n' > "$SMOKE_DIR/test_frames.txt"
cat > "$SMOKE_DIR/split.yaml" <<EOF
train:
  - drive: 2013_05_28_drive_0003_sync
    frames_file: $(pwd)/$SMOKE_DIR/train_frames.txt
test:
  - drive: 2013_05_28_drive_0003_sync
    frames_file: $(pwd)/$SMOKE_DIR/test_frames.txt
EOF

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 \
python scripts/train.py \
  --config configs/train.yaml \
  --data_dir /media/shizhm/Lenovo/KITTI-360 \
  --split_yaml "$SMOKE_DIR/split.yaml" \
  --output_dir "$SMOKE_DIR/run" \
  --epochs 1 \
  --warmup 0 \
  --batch_size 1 \
  --gradient_accumulation 1 \
  --num_workers 0 \
  --validate_every 1 \
  --visualize_every 0 \
  --wandb_mode disabled \
  --no_tensorboard \
  --hf_home "$HF_HOME"

CHECKPOINT="$SMOKE_DIR/run/checkpoints/checkpoint_epoch_1.pt"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 \
python scripts/infer.py \
  --config configs/inference.yaml \
  --mode split_yaw_sweep \
  --checkpoint "$CHECKPOINT" \
  --data_dir /media/shizhm/Lenovo/KITTI-360 \
  --split_yaml "$SMOKE_DIR/split.yaml" \
  --dataset_split test \
  --yaw_sweep_preset train_fixed \
  --max_frames 1 \
  --num_inference_steps 1 \
  --guidance_scale 1.0 \
  --mixed_precision bf16 \
  --output_dir "$SMOKE_DIR/infer_train_fixed" \
  --hf_home "$HF_HOME"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 \
python scripts/infer.py \
  --config configs/inference.yaml \
  --mode split_yaw_sweep \
  --checkpoint "$CHECKPOINT" \
  --data_dir /media/shizhm/Lenovo/KITTI-360 \
  --split_yaml "$SMOKE_DIR/split.yaml" \
  --dataset_split test \
  --yaw_sweep_preset heldout \
  --max_frames 1 \
  --num_inference_steps 1 \
  --guidance_scale 1.0 \
  --mixed_precision bf16 \
  --output_dir "$SMOKE_DIR/infer_heldout" \
  --hf_home "$HF_HOME"

python scripts/run_pose_chain_gate_bundle.py \
  --train_fixed_dir "$SMOKE_DIR/infer_train_fixed" \
  --heldout_dir "$SMOKE_DIR/infer_heldout" \
  --scalars_jsonl "$SMOKE_DIR/run/logs/scalars.jsonl" \
  --report_output "$SMOKE_DIR/gate_report.md" \
  --decision_output "$SMOKE_DIR/gate_decision.md" \
  --contact_sheet_output "$SMOKE_DIR/gate_contact_sheet.jpg" \
  --min_common_frames 1 \
  --expected_sat_condition_mode normal \
  --require_scalars \
  --require_checkpoint_state \
  --require_metric_sanity \
  --strict
```

## Validation/Test-Split Yaw Sweeps

Run both presets on the same checkpoint and same validation/test split. For
the current KITTI-360 `train_test_split_config.yaml`, use
`POSE_CHAIN_GATE_SPLIT=test` so the inference metadata explicitly says the
gate evidence came from the test split. Use `POSE_CHAIN_GATE_SPLIT=val` only
with a split YAML that has an explicit `val` entry, or when you intentionally
want the repository's `val -> test` fallback. Keep the value identical across
both presets. The gate checker rejects train-split output, split-name
mismatches, and checkpoint mismatches.

```bash
cd /mnt/shizhm/sd2.1-sat2street
conda activate "${POSE_CHAIN_CONDA_ENV:-maskgit}"

export HF_HOME=/mnt/shizhm/sd2.1-sat2street/.hf-home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export POSE_CHAIN_GATE_SPLIT="${POSE_CHAIN_GATE_SPLIT:-test}"

CHECKPOINT=/mnt/shizhm/sd2.1-sat2street/output/wip_pose_chain_group_conditioning/checkpoints/checkpoint_epoch_100.pt
test -f "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT"; exit 1; }

python scripts/infer.py \
  --config configs/inference.yaml \
  --mode split_yaw_sweep \
  --checkpoint "$CHECKPOINT" \
  --data_dir /mnt/shizhm/KITTI-360 \
  --split_yaml /mnt/shizhm/KITTI-360/train_test_split_config.yaml \
  --dataset_split "$POSE_CHAIN_GATE_SPLIT" \
  --yaw_sweep_preset train_fixed \
  --max_frames 20 \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --mixed_precision bf16 \
  --output_dir inference_results/pose_chain_gate_train_fixed \
  --hf_home /mnt/shizhm/sd2.1-sat2street/.hf-home

python scripts/infer.py \
  --config configs/inference.yaml \
  --mode split_yaw_sweep \
  --checkpoint "$CHECKPOINT" \
  --data_dir /mnt/shizhm/KITTI-360 \
  --split_yaml /mnt/shizhm/KITTI-360/train_test_split_config.yaml \
  --dataset_split "$POSE_CHAIN_GATE_SPLIT" \
  --yaw_sweep_preset heldout \
  --max_frames 20 \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --mixed_precision bf16 \
  --output_dir inference_results/pose_chain_gate_heldout \
  --hf_home /mnt/shizhm/sd2.1-sat2street/.hf-home
```

`train_fixed` checks the trained yaw anchors:

- `front`
- `-120, -90, -60`
- `+60, +90, +120`

`heldout` checks interpolation/generalization:

- `-105, -75, -45`
- `+45, +75, +105`

## Gate Bundle

```bash
python scripts/run_pose_chain_gate_bundle.py \
  --train_fixed_dir inference_results/pose_chain_gate_train_fixed \
  --heldout_dir inference_results/pose_chain_gate_heldout \
  --scalars_jsonl output/wip_pose_chain_group_conditioning/logs/scalars.jsonl \
  --report_output inference_results/pose_chain_gate_report.md \
  --decision_output inference_results/pose_chain_gate_decision.md \
  --contact_sheet_output inference_results/pose_chain_gate_contact_sheet.jpg \
  --min_common_frames 20 \
  --expected_sat_condition_mode normal \
  --require_scalars \
  --require_checkpoint_state \
  --require_metric_sanity \
  --strict
```

This writes:

- `inference_results/pose_chain_gate_report.md`
- `inference_results/pose_chain_gate_decision.md`
- `inference_results/pose_chain_gate_contact_sheet.jpg`

Use the contact sheet and decision record for the final manual visual/metric
judgment. The standalone scripts remain available if you need to regenerate only
one artifact: `scripts/check_pose_chain_gate_outputs.py` and
`scripts/init_pose_chain_gate_decision.py`.

The checker verifies:

- both presets use `mode=split_yaw_sweep`
- both presets use `dataset_split=val` or `dataset_split=test`
- both presets use the same checkpoint
- both presets use the same split yaml and the same inference runtime config
  (`num_inference_steps`, `guidance_scale`, seed, precision, memory mode, and
  satellite conditioning mode)
- both presets use normal satellite conditioning for the formal gate, not a
  sat-zero ablation
- validation scalars are provided for the formal gate
- validation scalars contain a validation epoch matching the checkpoint epoch
- validation scalars contain the required gate diagnostics:
  denoise loss, target attention lift, geometry-only/semantic-only/without-
  geometry comparisons, semantic-to-geometry ratio, and chain coverage overlap /
  centroid shift / valid pair metrics
- with `--require_metric_sanity`, validation scalars pass conservative sanity
  checks: target lift with geometry is above chance, geometry lift beats the
  without-geometry baseline, and coverage/valid-pair ratios are in `[0, 1]`
- the checkpoint and inference metadata both enable
  `query_geometry_score` with `mode=geometry_first_semantic_refine`
- the checkpoint metadata has `attention_alignment_enabled=true`
- the checkpoint file itself does not contain removed additive PE state keys
  such as satellite `perspective_pe` or query `query_uv` PE parameters
- the checkpoint metadata says `view_set=pose_chain` and includes the expected
  overlapping pose chains:
  - `right`: `front -> +60 -> +90 -> +120`
  - `left`: `front -> -60 -> -90 -> -120`
- inference geometry config matches the checkpoint's saved training config
- each preset's `run_metadata_*.yaml` `num_frames` matches the actual number of
  frame output directories
- both presets cover the same drive/frame set
- both presets cover at least the requested minimum paired frame count
- each frame has summary images and complete per-view outputs
- required PNG outputs are readable images, not empty or corrupt files
- each per-view `metadata.yaml` matches the output drive/frame, preset view
  name/yaw, and nested dataset sample yaw recorded under
  `meta.vehicle_yaw_deg_used`
- when `--scalars_jsonl` is provided, the file exists and contains validation
  records with the required `val/` gate metrics

`READY_FOR_MANUAL_TEST_GATE` means the outputs are complete and comparable. It
does not mean the method passed visually; the pass/fail decision still comes
from the manual test-split criteria below.

## Manual Gate Criteria

Pass v1/v2 only if the test split shows:

- `train_fixed` views preserve continuous road/layout geometry across adjacent yaw steps
- `heldout` views do not collapse, flip, or rotate inconsistently
- validation scalars remain plausible:
  - target attention lift is positive
  - chain attention coverage moves continuously
  - semantic-to-geometry ratio does not dominate early addressing

Move to v3 latent-action auxiliary if:

- attention/coverage metrics look correct, but generated road direction or layout is still wrong on test frames
- heldout yaw views fail while train_fixed anchors look acceptable
- repeated failures occur on frames where satellite road evidence is not fully ambiguous under tree shadow

Record the final decision in `docs/pose_chain_gate_decision_template.md` after
copying it into the experiment output directory.

After filling the decision, write a final gate report with the explicit decision
and rationale:

```bash
python scripts/finalize_pose_chain_gate_report.py \
  --train_fixed_dir inference_results/pose_chain_gate_train_fixed \
  --heldout_dir inference_results/pose_chain_gate_heldout \
  --scalars_jsonl output/wip_pose_chain_group_conditioning/logs/scalars.jsonl \
  --checker_report inference_results/pose_chain_gate_report.md \
  --decision_record inference_results/pose_chain_gate_decision.md \
  --contact_sheet inference_results/pose_chain_gate_contact_sheet.jpg \
  --output inference_results/pose_chain_gate_final_report.md \
  --selected_decision MOVE_TO_V3_LATENT_ACTION \
  --rationale_file inference_results/pose_chain_gate_rationale.txt \
  --min_common_frames 20 \
  --expected_sat_condition_mode normal \
  --require_scalars \
  --require_checkpoint_state \
  --require_metric_sanity
```

Use one of `PASS_V1_V2`, `MOVE_TO_V3_LATENT_ACTION`, or
`RERUN_OR_DEBUG_DATA`. The finalizer rescans the formal outputs before writing
the report; it will not let an incomplete gate be marked as pass.

After finalization, status should show both a ready gate scan and the selected
decision:

```bash
python scripts/inspect_pose_chain_run_status.py \
  --output_dir output/wip_pose_chain_group_conditioning \
  --gate_output_dir inference_results \
  --min_common_frames 20
```
