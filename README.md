# UR10e Gear Assembly Agentic Sim2Real

This repo is a UR arm version of the Isaac Lab gear assembly tutorial, with an
offline agent layer added for SysID, domain-randomization tuning, and
AutoResearch experiment planning.

It targets:

- UR10e arm
- Robotiq 2F-140 gripper by default, with 2F-85 supported by config
- Isaac Lab task `Isaac-Deploy-GearAssembly-UR10e-2F140-v0`
- FoundationPose + RealSense shaft-pose observations
- Isaac ROS Manipulation gear assembly workflow with cuMotion

The repo does not move the robot by itself. Real robot commands are blocked by a
human gate.

## Sources Double Checked

- Isaac Lab gear assembly policy tutorial:
  https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html
- Isaac ROS gear assembly deployment tutorial:
  https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/tutorials/sim_to_real/tutorial_gear_assembly.html

Key facts used here:

- Training task: `Isaac-Deploy-GearAssembly-UR10e-2F140-v0`
- 2F-85 task: `Isaac-Deploy-GearAssembly-UR10e-2F85-v0`
- Policy observations: `joint_pos`, `joint_vel`, `gear_shaft_pos`, `gear_shaft_quat`
- Shaft pose source: FoundationPose + RealSense depth
- Tutorial training noise: `gear_shaft_pos` +/- 0.005 m and `gear_shaft_quat` +/- 0.01 per component
- Tutorial deployment rate: 30 Hz
- Tutorial 2F-140 action scale: 0.0325; 2F-85 default: 0.025
- Full training command uses `--headless --num_envs 256 --video --video_length 800 --video_interval 5000`
- Real robot pose/calibration should be under 1 cm error before insertion

## Current Tutorial Process vs Agentic Process

| Area | Current tutorial method | Agentic version in this repo |
| --- | --- | --- |
| Physics tuning | Human visual comparison of real vs sim videos | Agent estimates delay, stiction proxy, contact spikes, and proposes parameter sweeps |
| Pose noise | Fixed training noise from expected FoundationPose/RealSense error | Agent fits shaft-pose noise from repeatability logs and flags if it exceeds the 1 cm gate |
| Domain randomization | Tutorial ranges are chosen up front | Agent keeps tutorial defaults, then recommends tighten/widen changes from real logs |
| Action scale | Manual gripper-specific value, 0.025 or 0.0325 | Agent checks whether stiction needs a larger scale or contact force requires a smaller one |
| SysID | Human-guided step response and inspection | Agent turns step/log data into stiffness, damping, friction, delay, and action-scale candidates |
| AutoResearch | Not part of the tutorial | Agent ranks experiments E1-E4 and writes an experiment plan before any real run |
| Human role | High throughout | Still required for calibration, labels, approvals, and all real robot motion |
| Time target | Often several weeks if done manually | One-week loop: train, calibrate, collect small logs, run agent, retry best experiment |

## What The Agent Does

The agent is offline and evidence-driven:

1. Reads real logs from JSONL/JSON/CSV.
2. Estimates control rate, action/state delay, stiction/deadband proxy, pose noise, contact force risk, and reset scatter.
3. Produces:
   - `gap_estimates.json`
   - `autoresearch_plan.json`
   - `agentic_params.yaml`
   - `transfer_score.json`
   - `report.md`
4. Suggests which Isaac Lab/Isaac ROS parameters to tune.
5. Keeps real robot deployment behind a human gate.

Human inputs still required:

- UR calibration file path
- hand-eye/camera calibration validation result
- gripper type
- policy checkpoint path plus `agent.yaml` and `env.yaml`
- gear model and asset paths
- pose-estimation repeatability samples
- run labels: success, slip, jam, pose miss, camera dropout, calibration issue
- explicit approval before real robot motion

## Repo Map

```text
configs/ur10e_gear_assembly.example.json
sample_data/real_log_demo.jsonl
scripts/
  _env.sh
  print_tutorial_commands.sh
  isaaclab_train.sh
  isaaclab_play.sh
  ros_preflight.sh
  collect_real_logs.sh
  run_agentic_pipeline.sh
  real_robot_human_gate.sh
  isaac_ros_launch_gear.sh
ur_agentic/
  cli.py
  config.py
  dataset.py
  metrics.py
  sysid.py
  autoresearch.py
  report.py
  safety.py
tests/test_pipeline.py
```

## Step By Step

### 0. Clone and install

```bash
git clone https://github.com/recscores007/so101.git
cd so101
python3 -m pip install -e .
cp configs/ur10e_gear_assembly.example.json configs/ur10e_gear_assembly.local.json
export UR_GEAR_CONFIG=$PWD/configs/ur10e_gear_assembly.local.json
```

Edit `configs/ur10e_gear_assembly.local.json` for your Isaac Lab root, Isaac
ROS workspace, gripper type, `ROS_DOMAIN_ID`, and manipulator config path.

### 1. Print tutorial commands

```bash
./scripts/print_tutorial_commands.sh
```

### 2. Visualize the Isaac Lab environment

```bash
./scripts/isaaclab_train.sh visualize
```

Equivalent tutorial command:

```bash
cd ~/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Deploy-GearAssembly-UR10e-2F140-v0 \
  --num_envs 4
```

For the Robotiq 2F-85 gripper, set `robot.gripper_type` to `robotiq_2f_85`.

### 3. Train the policy

```bash
./scripts/isaaclab_train.sh full
```

Equivalent tutorial command:

```bash
cd ~/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Deploy-GearAssembly-UR10e-2F140-v0 \
  --headless \
  --num_envs 256 \
  --video --video_length 800 --video_interval 5000
```

Training is expected to take about 12-24 hours for a robust insertion policy.

### 4. Configure Isaac ROS for gear assembly

In the Isaac ROS Manipulation config:

- set `workflow_type` to `GEAR_ASSEMBLY`
- set `gear_assembly_model_path`
- set `gear_assembly_model_file_name`
- set `gripper_type`
- set `setup`
- set `moveit_collision_objects_scene_file`
- set `cumotion_urdf_file_path`
- set `cumotion_xrdf_file_path`
- set `ur_calibration_file_path`
- set `gear_assembly_model_frequency` to 30 Hz
- set `gear_assembly_offset_for_place_pose` to 0.34 for 2F-140 or 0.32 for 2F-85
- for 2F-140, set `action_scale_joint_space` values to 0.0325

### 5. Run real-robot preflight, without moving the robot

```bash
./scripts/ros_preflight.sh
```

Run the tutorial's manual-on-robot validation tests before full deployment:

```bash
export ENABLE_MANIPULATOR_TESTING=manual_on_robot
launch_test $(ros2 pkg prefix --share isaac_ros_manipulation_bringup)/test/test_pose_estimation_error_test.py
bash ${ISAAC_ROS_WS}/src/isaac_ros_manipulation/isaac_ros_manipulation_bringup/test/compare_pose_estimation_results.sh
```

Do not proceed if pose error is above 1 cm.

### 6. Collect a real log packet

```bash
./scripts/collect_real_logs.sh rosbags/ur_gear_day1
```

Convert the observations you need into JSONL with this schema:

```json
{
  "episode_index": 0,
  "timestamp": 0.033,
  "action": [0.01, 0.0, -0.006, 0.004, 0.001, 0.001],
  "joint_state": [0.10, -1.20, 1.31, -1.65, -1.57, 0.02],
  "shaft_pose_estimate": [0.503, 0.104, 0.222, 0.0, 0.0, 0.010, 0.9999],
  "shaft_pose_reference": [0.500, 0.100, 0.220, 0.0, 0.0, 0.0, 1.0],
  "contact_force": 4.0,
  "success": true,
  "failure_mode": null
}
```

### 7. Run the offline agent

```bash
./scripts/run_agentic_pipeline.sh \
  --dataset ./sample_data/real_log_demo.jsonl \
  --out ./outputs/demo
```

Installed CLI equivalent:

```bash
ur-gear-agentic --config configs/ur10e_gear_assembly.local.json analyze \
  --dataset ./sample_data/real_log_demo.jsonl \
  --out ./outputs/demo
```

Read `outputs/demo/report.md` and `outputs/demo/autoresearch_plan.json`.

### 8. Use AutoResearch to improve sim2real

Run these in order:

1. `E1_perception_noise_replay`: tune `gear_shaft_pos` and `gear_shaft_quat` noise from pose-repeatability logs.
2. `E2_ur10e_sysid_step_response`: tune stiffness, damping, friction, delay, and action scale from UR response logs.
3. `E3_reset_and_fixture_generalization`: compare real fixture scatter to tutorial base/gear pose randomization.
4. `E4_contact_action_scale_gate`: test whether the action scale overcomes stiction without causing force spikes.

Only promote the best candidate to real robot evaluation.

### 9. Launch the workflow only after the human gate

Checklist:

- UR remote program loaded and paused/stopped
- Robotiq Tool I/O set to User
- workspace clear
- emergency stop tested
- calibration and pose-estimation validation passed
- cuMotion ready
- human standing by

Then:

```bash
export I_ACCEPT_UR_REAL_ROBOT_RISK=yes
./scripts/isaac_ros_launch_gear.sh
```

In a second terminal, after `cuMotion is ready for planning queries!`:

```bash
export I_ACCEPT_UR_REAL_ROBOT_RISK=yes
./scripts/real_robot_human_gate.sh --send-goal
```

The tutorial then prompts the human to click the peg stand and gear in
`rqt_image_view`.

## Domain Randomization In This Tutorial

The tutorial uses three major randomization families:

| Family | What it means | Tutorial values |
| --- | --- | --- |
| Shaft pose observation noise | How perception sees the shaft | `gear_shaft_pos` +/- 0.005 m, `gear_shaft_quat` +/- 0.01/component |
| Base and gear pose randomization | Where the fixture and gear start in simulation | base x +/- 10 cm, y +/- 25 cm, z +/- 10 cm, roll/pitch +/- 2 deg, yaw +/- 30 deg; gear xy +/- 2 cm, z 5.75-7.75 cm, rpy +/- 5 deg |
| Actuator/contact randomization | How the UR arm and contacts respond | stiffness scale 0.75-1.5, damping scale 0.3-3.0, joint friction add 0.3-0.7 Nm, nominal friction 0.75, restitution 0 |

The agent does not replace these. It measures whether the real setup is inside
these ranges and proposes focused changes when it is not.

## Safety Boundary

`check-real-gate` fails unless this exact environment variable is set:

```bash
export I_ACCEPT_UR_REAL_ROBOT_RISK=yes
```

That is intentional. Agentic sim2real should improve the experiments, not hide
the fact that a human still owns hardware safety.
