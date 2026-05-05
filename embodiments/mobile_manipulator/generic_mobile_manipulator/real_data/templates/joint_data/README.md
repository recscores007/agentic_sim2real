# Joint Data

File: `joint_states.csv`

Required columns:

- `episode_index`
- `timestamp`
- `action_0 ... action_N`
- `joint_0 ... joint_N`

Optional columns:

- `joint_vel_0 ... joint_vel_N`
- `ee_x`, `ee_y`, `ee_z`, `ee_qx`, `ee_qy`, `ee_qz`, `ee_qw`

The converter detects the action and joint dimensions from the numbered
columns, so the same format works for arms, humanoids, and mobile manipulators.
