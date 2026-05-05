# Shaft Pose Data

File: `shaft_pose.csv`

Required estimate columns:

- `episode_index`
- `timestamp`
- `estimate_x`, `estimate_y`, `estimate_z`
- `estimate_qx`, `estimate_qy`, `estimate_qz`, `estimate_qw`

Recommended reference columns:

- `reference_x`, `reference_y`, `reference_z`
- `reference_qx`, `reference_qy`, `reference_qz`, `reference_qw`

The estimate is what the policy sees from FoundationPose/RealSense. The
reference is the best available validation pose used to measure perception
error.
