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

Use `object_pose.csv` for portable embodiments. Existing task-specific folders
may keep a legacy object name such as `shaft_pose.csv`; the loader accepts both.
