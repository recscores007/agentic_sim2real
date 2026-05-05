# Joint Data

File: `joint_states.csv`

Required columns:

- `episode_index`
- `timestamp`
- `action_0` ... `action_5`
- `joint_0` ... `joint_5`

Recommended columns:

- `joint_vel_0` ... `joint_vel_5`
- `command_0` ... `command_5` or `joint_command_0` ... `joint_command_5` when IsaacLab-Newton SysID should fit actuator parameters
- `ee_x`, `ee_y`, `ee_z`, `ee_qx`, `ee_qy`, `ee_qz`, `ee_qw`

Joint order:

```text
0 shoulder_pan_joint
1 shoulder_lift_joint
2 elbow_joint
3 wrist_1_joint
4 wrist_2_joint
5 wrist_3_joint
```
