# Pose-Chain Gate Decision

Use this after the formal checker reports `READY_FOR_MANUAL_TEST_GATE` on
validation/test outputs. This file records the research decision; it is not a
replacement for the checker report.

## Experiment

- branch:
- commit:
- checkpoint:
- checkpoint epoch:
- split yaml:
- dataset split: `val` or `test`
- scalars jsonl:
- train_fixed output dir:
- heldout output dir:
- checker report:

## Checker Evidence

- checker status: `READY_FOR_MANUAL_TEST_GATE` / `INCOMPLETE_OUTPUTS`
- paired frames:
- required paired frames:
- `val/attention_alignment_target_attention_lift_mixed`:
- `val/attention_alignment_target_attention_lift_geometry_only`:
- `val/attention_alignment_target_attention_lift_without_geometry`:
- `val/attention_alignment_semantic_to_geometry_ratio`:
- `val/attention_alignment_chain_attention_coverage_overlap`:
- `val/attention_alignment_chain_attention_centroid_shift`:
- `val/chain/coverage_overlap`:
- `val/chain/coverage_centroid_shift`:

## Visual Review Sample

Record at least 20 paired validation/test frames. Prefer frames where the
satellite road layout is visible enough that a yaw/layout error is meaningful.

| drive/frame | satellite ambiguity | train_fixed continuity | heldout interpolation | failure mode |
| --- | --- | --- | --- | --- |
|  | low / medium / high | pass / fail | pass / fail |  |
|  | low / medium / high | pass / fail | pass / fail |  |
|  | low / medium / high | pass / fail | pass / fail |  |

## Decision

Choose one:

- `PASS_V1_V2`: train_fixed views preserve continuous road/layout geometry and
  heldout yaws do not collapse, flip, or rotate inconsistently on clear-enough
  test frames.
- `MOVE_TO_V3_LATENT_ACTION`: attention/coverage metrics look plausible but
  generated road direction/layout still breaks on clear-enough test frames, or
  heldout yaws fail while trained yaw anchors are acceptable.
- `RERUN_OR_DEBUG_DATA`: outputs are incomplete, the checker fails, satellite
  evidence is too ambiguous on the reviewed frames, or validation diagnostics
  are not trustworthy enough to judge the algorithm.

Selected decision:

## Rationale

Summarize the evidence in a few concrete sentences. Name representative
drive/frame examples for pass/fail cases and distinguish data ambiguity from
algorithm failure.
