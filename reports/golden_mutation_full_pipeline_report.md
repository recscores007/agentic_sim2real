# Golden Mutation Full Pipeline Report

- Test status: pass
- Variants: 7
- Pass definition: A mutation test passes when all expected alert codes are observed. Full pipeline status may be pass or fail; fail means a downstream gate blocked weak evidence.
- Mutation report: `./outputs/golden_mutation_pipeline_report/mutation_report.json`

| Mutation | Test | Pipeline Behavior | Readiness | Transfer | Release Gap | Expected Alerts | Observed Alerts | Blockers |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| clean_baseline | pass | pass / smoke_review_only | ready | 0.575 | 0.425 | `none` | `none` | none |
| drop_all_camera_links | pass | pass / smoke_review_only | needs_attention | 0.575 | 0.425 | `heldout_frames_missing, low_frame_link_coverage, episode_frame_gaps` | `heldout_frames_missing, low_frame_link_coverage, episode_frame_gaps` | none |
| heldout_without_frames | pass | pass / smoke_review_only | needs_attention | 0.575 | 0.425 | `heldout_frames_missing, episode_frame_gaps` | `heldout_frames_missing, episode_frame_gaps` | none |
| orphan_sbl_frames | pass | pass / smoke_review_only | needs_attention | 0.575 | 0.425 | `orphan_frames_detected` | `orphan_frames_detected` | none |
| low_rate_telemetry | pass | pass / smoke_review_only | needs_attention | 0.676 | 0.324 | `delay_under_sampled` | `delay_under_sampled` | none |
| auto_positive_labels | pass | pass / smoke_review_only | needs_attention | 0.515 | 0.485 | `auto_positive_success_labels` | `auto_positive_success_labels` | none |
| fk_proxy_pose | pass | blocked | needs_attention | 0.500 | 0.500 | `fk_proxy_pose_validation` | `fk_proxy_pose_validation` | sim_eval_regression: candidate pose p95 error exceeds calibration gate; release_candidate_gate: sim_eval_regression failed: ['candidate pose p95 error exceeds calibration gate'] |

## Interpretation

- `clean_baseline` proves the repaired synthetic baseline is not noisy: no readiness alerts were emitted.
- Camera mutations prove the pipeline catches missing frame links globally and specifically on heldout data.
- `orphan_sbl_frames` proves extracted frames outside canonical records are surfaced to the user.
- `low_rate_telemetry` proves delay confidence is not trusted when sample rate is too low.
- `auto_positive_labels` proves all-positive threshold labels are discounted instead of treated as real success.
- `fk_proxy_pose` proves circular pose validation is caught; the full pipeline also blocked downstream promotion through `sim_eval_regression` and `release_candidate_gate`.

## Artifact Pointers

- Each variant has a full harness folder under `outputs/golden_mutation_pipeline_report/full_harness/<mutation_id>/`.
- Each full harness folder contains `scorecard.json`, `run_record.json`, `real_data_manifest.json`, `scoreboard.json`, and `ui/index.html`.
