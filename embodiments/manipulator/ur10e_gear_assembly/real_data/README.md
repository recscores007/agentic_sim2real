# Real Data Folder

This folder is where you put real robot task data for the UR10e gear assembly
manipulator example before running the agentic sim2real pipeline.

The same folder pattern should be repeated inside each embodiment:

```text
embodiments/<embodiment_type>/<embodiment_name>/real_data/
```

The pipeline needs one aligned file:

```text
embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name>/aligned/records.jsonl
```

You can inspect a submitted session with:

```bash
agentic-sim2real inspect-real-data \
  --root . \
  --session embodiments/manipulator/ur10e_gear_assembly/real_data/example_session
```

You can create aligned records from CSV subfolders with:

```bash
./scripts/prepare_real_data.sh embodiments/manipulator/ur10e_gear_assembly/real_data/example_session
```

Then run the skills and evaluator on that session:

```bash
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/example_session ./scripts/run_skill_harness.sh
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/example_session ./scripts/run_evaluation_loop.sh
```

`load_records()` automatically looks for:

```text
aligned/records.jsonl
records.jsonl
```

inside a session directory.

## Session Layout

```text
embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name>/
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
| `calibration/calibration.json` | Yes for embodiment sessions | Robot calibration, camera intrinsics/extrinsics, frame names |

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
  "object_pose_estimate": [0.503, 0.104, 0.222, 0.0, 0.0, 0.010, 0.9999],
  "object_pose_reference": [0.500, 0.100, 0.220, 0.0, 0.0, 0.0, 1.0],
  "contact_force": 4.0,
  "success": true,
  "failure_mode": null
}
```

The UR source CSV can still be named `shaft_pose.csv`. The aligned records use
the portable `object_pose_*` fields so the same skills work for other
embodiments.

That is the format consumed by:

```bash
agentic-sim2real analyze --dataset embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name> --out outputs/<session_name>
agentic-sim2real run-harness --dataset embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name> --out outputs/<session_name>_harness
agentic-sim2real run-evaluation-loop --dataset embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name> --out outputs/<session_name>_eval
```

## Templates

Use `embodiments/manipulator/ur10e_gear_assembly/real_data/templates/` when
creating a new collection session. It contains the expected CSV headers and
example values.
