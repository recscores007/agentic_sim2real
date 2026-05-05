# Real Data

Each embodiment keeps its real data locally:

```text
embodiments/<embodiment_type>/<embodiment_name>/real_data/<session_name>/
```

Expected session layout:

```text
camera_data/
  index.csv
  color/
  depth/
joint_data/
  joint_states.csv
pose_data/
  object_pose.csv
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

Use `object_pose.csv` for portable embodiments. If an embodiment has a task-specific pose source name, declare that in its `embodiment.json` manifest.

For physics SysID, add `command_0 ... command_N` or
`joint_command_0 ... joint_command_N` to `joint_data/joint_states.csv` so the
agent can distinguish commanded joint positions from measured joint positions.
