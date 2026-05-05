# Golden Dataset Mutation Report

- Status: pass
- Source dataset: `./golden/real_datasets/data_readiness_stress`
- Variants: 7

| Variant | Status | Expected | Observed | Missing |
| --- | --- | --- | --- | --- |
| clean_baseline | pass | `none` | `none` | `none` |
| drop_all_camera_links | pass | `heldout_frames_missing, low_frame_link_coverage, episode_frame_gaps` | `heldout_frames_missing, low_frame_link_coverage, episode_frame_gaps` | `none` |
| heldout_without_frames | pass | `heldout_frames_missing, episode_frame_gaps` | `heldout_frames_missing, episode_frame_gaps` | `none` |
| orphan_sbl_frames | pass | `orphan_frames_detected` | `orphan_frames_detected` | `none` |
| low_rate_telemetry | pass | `delay_under_sampled` | `delay_under_sampled` | `none` |
| auto_positive_labels | pass | `auto_positive_success_labels` | `auto_positive_success_labels` | `none` |
| fk_proxy_pose | pass | `fk_proxy_pose_validation` | `fk_proxy_pose_validation` | `none` |
