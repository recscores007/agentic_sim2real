# UR10e Gear Assembly Agentic Skills

This repo turns the Isaac Lab / Isaac ROS UR10e gear assembly tutorial into an
agentic, skill-based sim2real pipeline.

The important change is this:

```text
tutorial step -> atomic skill -> agent runs skill -> harness validates skill -> release gate
```

Every skill has a manifest, input/output contract, validators, quality score,
evidence files, and release-blocking flag. AutoResearch proposes improvements,
but the validation harness decides whether they are good enough to release.

Real robot motion is never automatic. It remains human-gated.

## Skills And Evaluator At A Glance

This repo is organized around atomic skills. An atomic skill is one small,
testable capability in the sim2real pipeline: environment preflight, policy
artifact audit, pose repeatability, SysID, domain randomization update, action
scale sweep, regression testing, release gating, or real-robot approval.

Each skill is treated like a releasable unit:

- it has a `skills/<skill_id>/skill.json` manifest
- it declares inputs, outputs, dependencies, owner agent, and pass/fail gates
- it writes evidence files under `outputs/.../skills/<skill_id>/`
- it returns `status`, `quality_score`, `confidence`, warnings, and blocking failures
- it can be validated alone or as part of the full release harness

The AutoResearch agent is the proposal engine. It reads real logs and skill
metrics, then proposes ranked experiments and candidate parameter updates for
SysID, domain randomization, action scale, perception noise, and reset
generalization. It is intentionally bounded: AutoResearch can propose changes
and promote candidates to human review, but it cannot pass its own work or run
the robot.

The evaluator is the measurement and release system:

| Component | What It Does | Can It Change Parameters? | Can It Approve Hardware? |
| --- | --- | --- | --- |
| Atomic skills | Run one bounded validation task and write evidence | No | No |
| AutoResearch agent | Propose experiments and candidate parameter updates | Proposes only | No |
| Evaluator harness | Measures every skill, scores quality, records evidence | No | No |
| Critic | Challenges weak evidence, warnings, regressions, and low confidence | No | No |
| Release gate | Blocks or promotes candidate to human review | No | No |
| Human hardware gate | Approves supervised real-robot execution | Yes, by explicit approval | Yes |

The full evaluation architecture is:

```text
Agent proposes -> Evaluator measures -> Critic challenges -> Release gate decides -> Human approves hardware
```

That means the repo is not just "agent runs scripts." It is an agentic
sim2real system with explicit skill contracts, self-improvement proposals,
deterministic validation, critic review, and a hard human gate before hardware.

## What This Targets

- UR10e arm
- Robotiq 2F-140 gripper by default, with 2F-85 supported by config
- Isaac Lab task `Isaac-Deploy-GearAssembly-UR10e-2F140-v0`
- Isaac ROS Manipulation gear assembly workflow with cuMotion
- FoundationPose + RealSense shaft pose observations
- Offline SysID, domain randomization tuning, AutoResearch planning, and release validation

Sources double checked:

- Isaac Lab gear assembly tutorial:
  https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html
- Isaac ROS gear assembly deployment tutorial:
  https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/tutorials/sim_to_real/tutorial_gear_assembly.html

Tutorial facts encoded in the skill contracts:

- Policy observations: `joint_pos`, `joint_vel`, `gear_shaft_pos`, `gear_shaft_quat`
- Shaft pose source: FoundationPose + RealSense depth
- Training noise: `gear_shaft_pos` +/- 0.005 m, `gear_shaft_quat` +/- 0.01/component
- Deployment rate: 30 Hz
- 2F-140 action scale: 0.0325
- 2F-85 action scale: 0.025
- Real pose/calibration gate: under 1 cm error before insertion

## Repo Map

```text
skills/                         Atomic skill contracts
  env_preflight/skill.json
  isaaclab_task_check/skill.json
  policy_artifact_audit/skill.json
  ros_preflight/skill.json
  pose_repeatability/skill.json
  sysid_step_response/skill.json
  domain_randomization_update/skill.json
  action_scale_sweep/skill.json
  autoresearch_planner/skill.json
  sim_eval_regression/skill.json
  release_candidate_gate/skill.json
  real_robot_gate/skill.json

ur_agentic/
  skill_harness.py              Skill runner, validators, scoreboard, release gate
  evaluation_loop.py            Agent/Evaluator/Critic/Release/Human trace
  autoresearch.py               Experiment planner
  sysid.py                      Sim-real gap and SysID recommendations
  metrics.py                    Delay, stiction, pose, contact metrics
  cli.py                        Command-line entrypoint

scripts/
  prepare_real_data.sh          Convert raw real_data session to records.jsonl
  run_skill_harness.sh          Default validation harness
  run_evaluation_loop.sh        Five-stage evaluation trace
  run_autoresearch_loop.sh      Harness plus AutoResearch evidence path
  isaaclab_train.sh             Tutorial training wrapper
  ros_preflight.sh              ROS validation helper
  real_robot_human_gate.sh      Human-gated action sender

golden/sample_inputs/           Golden validation fixtures
real_data/                      Real-data templates and example session
sample_data/real_log_demo.jsonl Sample real-log schema
agents/README.md                Agent responsibilities
harness/README.md               Harness design
harness/threshold_policy.json   Safety/spec/statistical/regression thresholds
```

## Atomic Skills

Atomic skills are the smallest validated building blocks in the pipeline. The
agent can run them repeatedly, compare evidence, and improve proposals, but the
release gate decides whether the whole chain is good enough for human review.

| Skill | Agent | Purpose | Release Blocking |
| --- | --- | --- | --- |
| `env_preflight` | `orchestrator_agent` | Check local tools and environment | Yes |
| `isaaclab_task_check` | `sim_agent` | Validate task id, observations, gripper, action scale | Yes |
| `policy_artifact_audit` | `sim_agent` | Check `agent.yaml`, `env.yaml`, checkpoint metadata | Yes |
| `ros_preflight` | `orchestrator_agent` | Check ROS gear workflow config | Yes |
| `pose_repeatability` | `perception_agent` | Validate shaft pose error under 1 cm | Yes |
| `sysid_step_response` | `sysid_agent` | Estimate delay, stiction, contact, SysID targets | Yes |
| `domain_randomization_update` | `dr_agent` | Propose bounded DR updates | Yes |
| `action_scale_sweep` | `sysid_agent` | Propose safe action-scale candidates | Yes |
| `autoresearch_planner` | `autoresearch_agent` | Generate and rank experiments | Yes |
| `sim_eval_regression` | `critic_agent` | Compare candidate vs baseline | Yes |
| `release_candidate_gate` | `safety_agent` | Aggregate evidence and block weak releases | Yes |
| `real_robot_gate` | `safety_agent` | Require human approval before hardware command | Yes, skipped by default |

Each skill writes:

```json
{
  "skill_id": "pose_repeatability",
  "status": "pass",
  "quality_score": 0.95,
  "confidence": 0.75,
  "blocking_failures": [],
  "warnings": [],
  "evidence_files": ["outputs/harness_demo/skills/pose_repeatability/pose_repeatability.json"]
}
```

## How AutoResearch Is Used

AutoResearch is the experiment designer and self-improvement loop. It does not
move the robot.

```text
real logs
  -> skill metrics
  -> AutoResearch hypotheses
  -> candidate parameter changes
  -> validation harness
  -> critic regression
  -> release gate
  -> human review
```

For this task, AutoResearch focuses on:

- perception noise around `gear_shaft_pos` and `gear_shaft_quat`
- UR10e stiffness, damping, friction, stiction, and delay
- base/gear pose randomization coverage
- action scale vs contact force
- sim regression before any hardware run

Promotion rule:

```text
AutoResearch may promote a candidate only to human review.
It never promotes directly to unattended robot execution.
```

## Evaluation Loop

The repo now makes the evaluator architecture explicit:

```text
Agent proposes.
Evaluator measures.
Critic challenges.
Release gate decides.
Human approves hardware.
```

This is the core evaluator feature. It separates proposing from measuring and
approval:

- the agent creates a candidate
- the evaluator measures the candidate against thresholds
- the critic looks for reasons the evidence is weak
- the release gate applies pass/fail policy
- the human decides whether a supervised real-robot run is allowed

Run it:

```bash
./scripts/run_evaluation_loop.sh
```

Equivalent CLI:

```bash
ur-gear-agentic --config "$UR_GEAR_CONFIG" run-evaluation-loop \
  --root . \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/evaluation_demo
```

Outputs:

```text
outputs/evaluation_demo/
  agent_proposal.json
  evaluator_measurements.json
  critic_challenges.json
  release_decision.json
  human_hardware_gate.json
  evaluation_trace.json
  evaluation_trace.md
```

What each role owns:

| Stage | Code | Authority |
| --- | --- | --- |
| Agent proposes | `ur_agentic/autoresearch.py` | Creates hypotheses and candidate parameter families |
| Evaluator measures | `ur_agentic/skill_harness.py` | Runs deterministic skills and writes metrics/evidence |
| Critic challenges | `ur_agentic/evaluation_loop.py` | Flags low confidence, warnings, regressions, and failed skills |
| Release gate decides | `ur_agentic/evaluation_loop.py` | Blocks or promotes to human review; never autoruns robot |
| Human approves hardware | `ur_agentic/safety.py` | Requires explicit supervised hardware approval |

Thresholds live in `harness/threshold_policy.json`:

- hard safety thresholds
- tutorial/spec thresholds
- statistical thresholds
- regression thresholds

Agents may propose new parameters, but the evaluator and release gate own
pass/fail.

## Real Data Folder

Put real robot data under `real_data/<session_name>/`.

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
```

Use the template:

```bash
cp -R real_data/templates real_data/ur10e_day1
```

Fill in:

- `joint_data/joint_states.csv`: policy actions, joint positions, joint velocities, optional end-effector pose
- `pose_data/shaft_pose.csv`: FoundationPose/RealSense shaft pose estimate and reference pose if available
- `camera_data/index.csv`: color/depth image paths or frame provenance
- `contact_data/contact.csv`: force or force-proxy samples
- `episode_labels/labels.csv`: success, failure mode, notes
- `calibration/calibration.json`: UR calibration, camera frames, hand-eye metadata

Convert raw subfolders into the pipeline format:

```bash
./scripts/prepare_real_data.sh real_data/ur10e_day1
```

This writes:

```text
real_data/ur10e_day1/aligned/records.jsonl
real_data/ur10e_day1/aligned/prepare_summary.json
```

Then run the pipeline directly on the session folder:

```bash
DATASET=real_data/ur10e_day1 ./scripts/run_skill_harness.sh
DATASET=real_data/ur10e_day1 ./scripts/run_evaluation_loop.sh
```

The loader automatically uses `aligned/records.jsonl` when a session directory
is passed as `--dataset`.

## Step By Step: Local Skill Harness

### 1. Clone and install

```bash
git clone https://github.com/recscores007/so101.git
cd so101
python3 -m pip install -e .
cp configs/ur10e_gear_assembly.example.json configs/ur10e_gear_assembly.local.json
export UR_GEAR_CONFIG=$PWD/configs/ur10e_gear_assembly.local.json
```

Edit `configs/ur10e_gear_assembly.local.json` for your Isaac Lab root, Isaac
ROS workspace, gripper type, `ROS_DOMAIN_ID`, and manipulator config path.

### 2. List the skills

```bash
ur-gear-agentic --config "$UR_GEAR_CONFIG" list-skills
```

### 3. Validate every skill manifest

```bash
ur-gear-agentic --config "$UR_GEAR_CONFIG" validate-skills --root .
```

This checks that every skill has:

- id
- owner agent
- input contract
- output contract
- validators
- quality gate
- human/release flags

### 4. Run the full offline harness

```bash
./scripts/run_skill_harness.sh
```

Equivalent CLI:

```bash
ur-gear-agentic --config "$UR_GEAR_CONFIG" run-harness \
  --root . \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/harness_demo
```

Expected outputs:

```text
outputs/harness_demo/
  scoreboard.json
  release_candidate.json
  skills/<skill_id>/result.json
```

The default harness skips `real_robot_gate`, because hardware requires human
approval.

### 5. Inspect the scoreboard

```bash
cat outputs/harness_demo/scoreboard.json
```

Release status is pass only if every release-blocking offline skill passes.

### 6. Run only one skill

```bash
ur-gear-agentic --config "$UR_GEAR_CONFIG" run-harness \
  --root . \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/pose_only \
  --skill pose_repeatability
```

Use this while developing or debugging one skill.

### 7. Run AutoResearch loop

```bash
./scripts/run_autoresearch_loop.sh
```

Key output:

```text
outputs/autoresearch_demo/skills/autoresearch_planner/autoresearch_plan.json
```

The plan contains:

- hypothesis
- agent action
- human action
- parameter change
- promotion rule

### 8. Run the full evaluation trace

```bash
./scripts/run_evaluation_loop.sh
cat outputs/evaluation_demo/evaluation_trace.md
```

This is the easiest way to see the full proposal -> measurement -> critique ->
decision -> human gate chain.

### 9. Prepare a real-data session

```bash
cp -R real_data/templates real_data/ur10e_day1
# Fill the CSV files and calibration.json with real session data.
./scripts/prepare_real_data.sh real_data/ur10e_day1
DATASET=real_data/ur10e_day1 ./scripts/run_evaluation_loop.sh
```

## Step By Step: Tutorial Training

### 1. Print tutorial commands

```bash
./scripts/print_tutorial_commands.sh
```

### 2. Visualize the Isaac Lab task

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

For Robotiq 2F-85, set `robot.gripper_type` to `robotiq_2f_85` in the config.

### 4. Replace golden policy artifacts

After training, copy your real Isaac Lab outputs into your release artifact
folder:

```text
agent.yaml
env.yaml
checkpoint metadata or pointer to model checkpoint
```

The harness currently uses `golden/sample_inputs/policy_artifacts/` for offline
validation. For a real release, point the policy audit skill to your real
artifact bundle or replace the golden sample with your release candidate.

## Step By Step: Real Data And SysID

### 1. Run ROS preflight

```bash
./scripts/ros_preflight.sh
```

Before insertion, run the tutorial's manual-on-robot validation:

```bash
export ENABLE_MANIPULATOR_TESTING=manual_on_robot
launch_test $(ros2 pkg prefix --share isaac_ros_manipulation_bringup)/test/test_pose_estimation_error_test.py
bash ${ISAAC_ROS_WS}/src/isaac_ros_manipulation/isaac_ros_manipulation_bringup/test/compare_pose_estimation_results.sh
```

Do not continue if pose error is above 1 cm.

### 2. Record real observations

```bash
./scripts/collect_real_logs.sh rosbags/ur_gear_day1
```

Convert your observations into JSONL like:

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

### 3. Run the harness on real logs

```bash
DATASET=/path/to/your_real_logs.jsonl \
OUT=outputs/real_day1 \
./scripts/run_skill_harness.sh
```

Review:

```text
outputs/real_day1/scoreboard.json
outputs/real_day1/release_candidate.json
outputs/real_day1/skills/sysid_step_response/gap_estimates.json
outputs/real_day1/skills/autoresearch_planner/autoresearch_plan.json
```

## Step By Step: Release Gate

The release gate passes only when these are true:

- all manifests are valid
- every release-blocking offline skill passes
- pose p95 error is under 1 cm
- SysID evidence exists
- DR updates stay bounded
- action-scale candidate stays within contact-force limits
- sim regression evidence is present
- `safe_to_autorun_robot` is false
- human approval is still required

If any blocking validator fails, the release candidate is blocked.

## Step By Step: Human-Gated Robot Run

Checklist:

- UR remote program loaded and paused/stopped
- Robotiq Tool I/O set to User
- workspace clear
- emergency stop tested
- calibration validation passed
- pose repeatability passed
- cuMotion reports ready
- human standing by

Terminal 1:

```bash
export I_ACCEPT_UR_REAL_ROBOT_RISK=yes
./scripts/isaac_ros_launch_gear.sh
```

Terminal 2, after `cuMotion is ready for planning queries!`:

```bash
export I_ACCEPT_UR_REAL_ROBOT_RISK=yes
./scripts/real_robot_human_gate.sh --send-goal
```

The tutorial then prompts the human to click the peg stand and gear in
`rqt_image_view`.

## Domain Randomization Covered

| Family | Meaning | Tutorial Defaults |
| --- | --- | --- |
| Shaft pose observation noise | How perception sees the shaft | `gear_shaft_pos` +/- 0.005 m, `gear_shaft_quat` +/- 0.01/component |
| Base and gear pose randomization | Where fixture and gear start in sim | base x +/- 10 cm, y +/- 25 cm, z +/- 10 cm, roll/pitch +/- 2 deg, yaw +/- 30 deg |
| Gear relative pose | Gear start relative to base | xy +/- 2 cm, z 5.75-7.75 cm, rpy +/- 5 deg |
| Actuator/contact | How the UR and contact respond | stiffness 0.75-1.5, damping 0.3-3.0, joint friction add 0.3-0.7 Nm, friction 0.75 |

The agent measures whether your real setup is inside these assumptions and
proposes bounded updates when it is not.

## Developer Checks

```bash
PYTHONPYCACHEPREFIX=/tmp/ur_pycache python3 -m py_compile ur_agentic/*.py tests/test_pipeline.py
PYTHONPATH=. python3 -m unittest discover -s tests
PYTHONPATH=. python3 -m ur_agentic.cli --config configs/ur10e_gear_assembly.example.json validate-skills --root .
PYTHONPATH=. python3 -m ur_agentic.cli --config configs/ur10e_gear_assembly.example.json run-harness --root . --dataset sample_data/real_log_demo.jsonl --out /tmp/ur_harness
PYTHONPATH=. python3 -m ur_agentic.cli --config configs/ur10e_gear_assembly.example.json run-evaluation-loop --root . --dataset sample_data/real_log_demo.jsonl --out /tmp/ur_eval
bash -n scripts/*.sh
```

CI runs these checks on GitHub.
