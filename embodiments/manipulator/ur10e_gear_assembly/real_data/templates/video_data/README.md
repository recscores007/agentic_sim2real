# Uploaded Video Evidence

Use this folder for customer-uploaded videos that should be watched by the
pipeline before sim parameters are tuned.

Expected files:

- `index.csv`: one row per uploaded video.
- `analysis.json`: output from the configured video analyzer.
- Video files such as `camera_calibration.mp4`, `task_rollout.mp4`, or `contact_friction.mp4`.

The repository ignores actual video files by default. Keep customer videos in
private storage or local run folders, and only commit small template files.

## Video Index

Columns:

- `episode_index`
- `timestamp_start`
- `timestamp_end`
- `video_path`
- `video_type`
- `view`
- `notes`

Recommended `video_type` values:

- `camera_calibration`
- `task_rollout`
- `object_friction`
- `gripper_contact`

## Analysis Contract

`analysis.json` should contain camera and friction metrics produced by a video
analyzer. The built-in harness validates the contract and applies bounded tuning
recommendations; the pixel-level computer-vision implementation can be supplied
through `video_evidence.analysis_command`.
