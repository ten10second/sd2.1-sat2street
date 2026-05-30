# Pose-Chain Group Codex Plan

## Objective

Implement and validate a pose-chain group version of the geometry-first
semantic-refine model on branch `wip-pose-chain-group-conditioning`.

The method must train on overlapping yaw chains while sharing one satellite
content memory per location:

- right chain: `front -> +60 -> +90 -> +120`
- left chain: `front -> -60 -> -90 -> -120`

Each view in the chain must use its own camera pose and projected geometry for
cross-attention addressing. The training loss is averaged across all views in
the chain.

## Execution Plan

1. Dataset grouping
   - Add `view_set=pose_chain`.
   - Expand each frame into configurable named pose chains.
   - Return stacked per-view target images, intrinsics, poses, camera heights,
     BEV query coordinates, masks, view names, and yaw metadata.
   - Keep satellite image/content shared once per chain sample.

2. Model path
   - Encode the satellite image once for each chain sample.
   - Repeat the satellite content memory across views.
   - Project satellite patch BEV coordinates into each view independently using
     that view's `K`, pose, and camera height.
   - Flatten `[batch, view]` into the UNet denoising batch only after per-view
     geometry has been assigned.
   - Keep additive perspective/query PE disabled; use logit-level
     geometry-first semantic-refine addressing.

3. Training and diagnostics
   - Average denoising loss across all views.
   - Keep attention alignment supervision.
   - Log validation metrics for target lift, geometry-only/semantic-only/
     without-geometry comparisons, semantic-to-geometry ratio, and chain
     coverage overlap, centroid shift, and valid pair ratio.
   - Save checkpoint metadata containing `view_set`, `pose_chains`,
     `pose_chain_group_size`, validation interval, and distributed settings.

4. Inference gate
   - Run `split_yaw_sweep` on validation or test split only.
   - Generate both `train_fixed` and `heldout` presets from the same checkpoint,
     split yaml, runtime config, and satellite conditioning mode.
   - Save per-view GT, generated image, satellite coverage, projected satellite
     view, comparison panel, and metadata.

5. Gate checker
   - Reject train-split outputs.
   - Require paired `train_fixed` and `heldout` outputs for the same
     drive/frame set.
   - Require the checkpoint metadata to say `view_set=pose_chain` and include
     the expected right/left overlapping chains.
   - Require checkpoint and inference metadata to use
     `query_geometry_score_enabled=true` and
     `query_geometry_score_mode=geometry_first_semantic_refine`.
   - Require checkpoint metadata to keep `attention_alignment_enabled=true`.
   - Inspect the checkpoint state dict for the formal gate and reject removed
     additive perspective/query PE parameters.
   - Require matching checkpoint, split yaml, inference runtime config, and
     validation epoch.
   - Require validation scalar logs for the formal gate.
   - Require conservative scalar sanity for the formal gate: positive geometry
     target lift, geometry lift above the without-geometry baseline, and valid
     chain ratio ranges.
   - Require readable PNG outputs and per-view metadata matching drive, frame,
     view name, yaw, nested sample yaw, and satellite conditioning mode.

## Completion Gate

This branch is not complete merely because the smoke test passes.

The v1/v2 method is ready for research judgment only after a server run
produces:

- a trained pose-chain checkpoint from the full or intended training split
- validation scalar logs from the same checkpoint epoch
- `train_fixed` yaw-sweep outputs on validation/test split
- `heldout` yaw-sweep outputs on the same validation/test frames
- at least 20 paired validation/test frames passing
  `scripts/run_pose_chain_gate_bundle.py --strict`
- a completed decision record based on
  `docs/pose_chain_gate_decision_template.md`, initialized by the same bundle
  command that wrote the checker report
- a final explicit gate report from
  `scripts/finalize_pose_chain_gate_report.py`, containing one of
  `PASS_V1_V2`, `MOVE_TO_V3_LATENT_ACTION`, or `RERUN_OR_DEBUG_DATA`

Pass v1/v2 if test-split visuals show continuous road/layout geometry across
adjacent trained yaw anchors and stable interpolation on heldout yaws.

Move to v3 latent-action auxiliary if attention/coverage metrics are correct
but test-split generation still flips, rotates, or breaks road layout on frames
where the satellite evidence is not fully ambiguous.

## Current Local Verification

The local machine can verify code path correctness, not final visual quality.

Required local checks before pushing to the multi-GPU server:

```bash
git diff --check
/home/shizhm/anaconda3/envs/maskgit/bin/python -m unittest discover tests
/home/shizhm/anaconda3/envs/maskgit/bin/python scripts/run_pose_chain_gate_bundle.py \
  --train_fixed_dir output/pose_chain_group_smoke/infer_train_fixed \
  --heldout_dir output/pose_chain_group_smoke/infer_heldout \
  --scalars_jsonl output/pose_chain_group_smoke/run/logs/scalars.jsonl \
  --report_output output/pose_chain_group_smoke/gate_report.md \
  --decision_output output/pose_chain_group_smoke/gate_decision.md \
  --contact_sheet_output output/pose_chain_group_smoke/gate_contact_sheet.jpg \
  --min_common_frames 1 \
  --expected_sat_condition_mode normal \
  --require_scalars \
  --require_checkpoint_state \
  --require_metric_sanity \
  --strict
```

The formal server gate must use `--min_common_frames 20` and validation/test
outputs, not the local one-frame smoke output.
