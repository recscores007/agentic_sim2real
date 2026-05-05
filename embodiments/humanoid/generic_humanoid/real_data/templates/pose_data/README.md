# Object Pose Data

File: `object_pose.csv`

Required columns:

- `episode_index`
- `timestamp`
- `estimate_x`, `estimate_y`, `estimate_z`
- `estimate_qx`, `estimate_qy`, `estimate_qz`, `estimate_qw`

Optional reference columns:

- `reference_x`, `reference_y`, `reference_z`
- `reference_qx`, `reference_qy`, `reference_qz`, `reference_qw`

Use the object, target, base frame, or task feature observed by perception for this embodiment.
