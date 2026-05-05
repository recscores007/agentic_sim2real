# Real Data Folder

This folder is where you put real robot task data before running the agentic
sim2real pipeline. The checked-in example is UR10e gear assembly, but the folder
layout is meant to stay portable.

The pipeline needs one aligned file:

```text
real_data/<session_name>/aligned/records.jsonl
```

You can create it from raw subfolders with:

```bash
./scripts/prepare_real_data.sh real_data/example_session
```

Then run the skills and evaluator on that session:

```bash
DATASET=real_data/example_session ./scripts/run_skill_harness.sh
DATASET=real_data/example_session ./scripts/run_evaluation_loop.sh
```

`load_records()` automatically looks for:

```text
aligned/records.jsonl
records.jsonl
```

inside a session directory.

## Session Layout

```text
real_data/<session_name>/
  camera_data/
    index.csv
    color/
    depth/
  joint_data/
    joint_states.csv
  pose_data/
    shaft_pose.csv
  contact_data/
    contact.csv
  episode_labels/
    labels.csv
  calibration/
    calibration.json
  aligned/
    records.jsonl
    prepare_summary.json
```

## Required Files

| File | Required | Purpose |
| --- | --- | --- |
| `joint_data/joint_states.csv` | Yes | Timestamped policy actions, joint positions, joint velocities, optional end-effector pose |
| `pose_data/shaft_pose.csv` | Yes | FoundationPose/RealSense shaft pose estimate and optional reference pose |
| `episode_labels/labels.csv` | Strongly recommended | Success/failure labels and failure modes |
| `contact_data/contact.csv` | Recommended | Contact force or force proxy |
| `camera_data/index.csv` | Recommended | Image/depth file provenance |
| `calibration/calibration.json` | Recommended | Robot calibration, camera intrinsics/extrinsics, frame names |

## What The Human Provides

- Camera frames or ROS bag references
- Joint state/action logs
- FoundationPose shaft pose estimates
- Reference/validation shaft poses when available
- Contact force or force-proxy logs
- Episode labels: success, slip, jam, pose miss, camera dropout, calibration issue
- Calibration file paths and frame names

## Output Format

The converter creates JSONL rows like:

```json
{
  "episode_index": 0,
  "timestamp": 0.033,
  "action": [0.01, 0.0, -0.006, 0.004, 0.001, 0.001],
  "joint_state": [0.10, -1.20, 1.31, -1.65, -1.57, 0.02],
  "joint_velocity": [0.01, 0.0, -0.01, 0.0, 0.0, 0.0],
  "shaft_pose_estimate": [0.503, 0.104, 0.222, 0.0, 0.0, 0.010, 0.9999],
  "shaft_pose_reference": [0.500, 0.100, 0.220, 0.0, 0.0, 0.0, 1.0],
  "contact_force": 4.0,
  "success": true,
  "failure_mode": null
}
```

That is the format consumed by:

```bash
ur-gear-agentic analyze --dataset real_data/<session_name> --out outputs/<session_name>
ur-gear-agentic run-harness --dataset real_data/<session_name> --out outputs/<session_name>_harness
ur-gear-agentic run-evaluation-loop --dataset real_data/<session_name> --out outputs/<session_name>_eval
```

## Templates

Use `real_data/templates/` when creating a new collection session. It contains
the expected CSV headers and example values.
