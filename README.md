# Agentic Sim2Real Skills

This repo is a portable `agentic_sim2real` framework for LLM-orchestrated,
skill-based sim2real workflows across embodiments such as manipulators,
humanoids, and mobile manipulators. The UR10e gear assembly tutorial is the
first concrete manipulator example, not the framework boundary.

The important change is this:

```text
workflow step -> LLM chooses skill -> harness validates skill -> journal updates -> release gate
```

Every skill has a manifest, input/output contract, validators, quality score,
evidence files, and release-blocking flag. The LLM orchestrator chooses the next
skill from the catalog and writes a journal, but the validation harness decides
whether each result is good enough to release.

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
| LLM orchestrator | Chooses the next valid skill from context, scorecards, and the manifest catalog | Proposes skill calls only | No |
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

## What The LLM Handles

The LLM is used for the non-deterministic reasoning layer: triage, ordering,
hypotheses, experiment planning, critique, and explanation. It is not the
measurement layer and it is not the safety authority.

Every LLM choice must reduce to one of these auditable objects:

- a valid atomic skill call
- a bounded candidate parameter proposal
- a critic challenge tied to evidence
- a report that cites generated evidence

The deterministic harness then validates that object before anything can be
released.

| Non-Deterministic LLM Responsibility | What It Does | Why LLM Handles It | How It Is Atomically Validated |
| --- | --- | --- | --- |
| Gap triage | Decides whether the likely gap is perception, actuator, contact, latency, policy, deployment, or domain randomization | Real sim2real failures are messy and often have multiple plausible causes | User gap hints are recorded, then targeted skills such as `pose_repeatability`, `sysid_step_response`, `action_scale_sweep`, and `sim_eval_regression` produce metrics |
| Skill ordering | Chooses which atomic skill to run next | Multiple valid investigation paths may exist | Guardrails allow only known skills with satisfied dependencies; each decision is written to `llm_orchestrator/journal.jsonl` |
| Hypothesis generation | Proposes explanations such as "stiction is causing low action response" or "pose noise is dominating failures" | Diagnosis requires interpretation and creative comparison across evidence | The hypothesis must become a bounded parameter family and be tested by SysID, perception, DR, action-scale, or regression skills |
| Experiment planning | Suggests Newton/PACE SysID, action-scale sweep, domain-randomization update, more logs, or a replay test | Good experiment order depends on context, cost, and missing evidence | Each experiment maps to a manifest-backed skill with declared inputs, outputs, validators, and pass/fail gates |
| Critic role | Challenges weak evidence, sparse data, low confidence, skipped backends, or unsafe assumptions | Challenge quality is judgment-heavy and benefits from broad context | Critic output cannot approve release; the deterministic release gate treats unresolved challenges as blockers or warnings |
| Report/explanation | Explains what was learned and what should happen next | Natural-language synthesis is useful for humans reviewing the run | Reports must cite scoreboards, evidence files, decisions, and release artifacts; reports have no authority to change gates |

## How To Atomically Validate

Atomic validation means every agentic claim is checked by the smallest
relevant skill before it can affect release.

```text
LLM proposes one skill or hypothesis
  -> guardrails check skill id, dependencies, and safety limits
  -> harness runs the atomic skill
  -> skill writes evidence and a quality score
  -> evaluator records measurements
  -> critic challenges weak evidence
  -> release gate blocks or promotes to human review
  -> human approves any hardware motion
```

Useful commands:

```bash
# Validate skill contracts.
agentic-sim2real --config configs/ur10e_gear_assembly.example.json validate-skills --root .

# Run one atomic validator against a dataset.
agentic-sim2real --config configs/ur10e_gear_assembly.example.json run-harness \
  --root . \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/one_skill_pose_check \
  --skill pose_repeatability

# Let the LLM orchestrator choose skills, seeded by an initial gap hypothesis.
agentic-sim2real --config configs/ur10e_gear_assembly.example.json run-llm-loop \
  --root . \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/llm_contact_triage \
  --gap-hint contact
```

After any run, inspect:

```text
outputs/<run>/llm_orchestrator/journal.jsonl
outputs/<run>/llm_orchestrator/steps/step_###_context.json
outputs/<run>/llm_orchestrator/steps/step_###_decision.json
outputs/<run>/scoreboard.json
outputs/<run>/skills/<skill_id>/
```

The rule is simple: if the LLM cannot point to a skill result, metric,
scoreboard entry, critic finding, or release decision, it is only a suggestion.

## LLM Orchestrator Runtime

The runtime agent is now an LLM-style orchestrator. It receives:

- task/config summary
- dataset path
- user-provided initial gap hints, such as `perception`, `actuator`, `contact`,
  `latency`, `domain_randomization`, `deployment`, or `policy`
- skill manifest catalog
- completed scorecards
- runnable skills whose dependencies are satisfied
- guardrails for release and hardware approval

At every step it must return a decision:

```json
{
  "action": "run_skill",
  "skill_id": "real_data_quality_gate",
  "rationale": "Validate real logs before SysID.",
  "expected_evidence": ["real_data_quality.json", "result.json"],
  "risk_checks": ["records and calibration must be present"],
  "confidence": 0.9
}
```

Allowed actions are `run_skill`, `stop`, and `request_human_review`. The
orchestrator is not trusted blindly. The guardrails reject invalid LLM choices:
unknown skills, missing dependencies, repeated skills, early release gates, and
hardware-facing skills without explicit real-robot approval.

The default provider is `scripted`, a deterministic LLM test double for CI and
goldens. To plug in a real LLM, use the command provider:

```bash
agentic-sim2real --config "$AGENTIC_SIM2REAL_CONFIG" run-llm-loop \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/llm_run \
  --llm-provider command \
  --llm-command-json '["python3", "path/to/your_llm_agent.py", "--input", "{input}", "--output", "{output}"]'
```

The LLM command reads `AGENTIC_SIM2REAL_LLM_INPUT_JSON` and writes the decision
JSON to `AGENTIC_SIM2REAL_LLM_OUTPUT_JSON`. Every decision is journaled in:

```text
outputs/<run>/llm_orchestrator/journal.jsonl
outputs/<run>/llm_orchestrator/steps/step_###_context.json
outputs/<run>/llm_orchestrator/steps/step_###_decision.json
```

If you already suspect the gap family, steer the first-pass triage with one or
more gap hints:

```bash
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1 \
./scripts/run_llm_orchestrator.sh --gap-hint actuator --gap-hint contact
```

Gap hints are **hypotheses**, not permissions. They change skill ordering only
when a hinted skill is already runnable. The guardrails still require data
quality, dependencies, release scoring, and human hardware approval.

## Autorun And SysID Status

If you provide a complete embodiment-scoped real-data session folder, the
pipeline can run the full offline validation chain without human intervention.
For the UR manipulator example, that path is:

```text
embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name>
```

```bash
./scripts/prepare_real_data.sh embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name>
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name> ./scripts/run_skill_harness.sh
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name> ./scripts/run_llm_orchestrator.sh
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name> ./scripts/run_evaluation_loop.sh
```

That offline chain includes adapter-based data inspection, data alignment, a
real-data quality gate, LLM-selected skill calls, local SysID, optional
Newton/PACE SysID, AutoResearch proposals, evaluator scoring, critic review,
and release gate decision.

It does not authorize unattended hardware. The real robot remains behind the
human hardware gate, and `safe_to_autorun_robot` is always false.

Current SysID status:

| Question | Current Answer |
| --- | --- |
| Can the offline pipeline autorun from an embodiment real-data session? | Yes. If `aligned/records.jsonl` is present it is used directly; if CSV subfolders are present the harness auto-prepares aligned records. |
| Can it autorun a physical robot with no human? | No. Hardware is intentionally human-gated. |
| Is SysID currently using IsaacLab-Newton from `es-rl/IsaacLab-Newton`? | Not by default. The local estimator in `agentic_sim2real/sysid.py` runs by default. When `config.sysid.newton_enabled=true`, the portable `newton_sysid` skill uses the built-in Newton bridge or a custom `newton_command`. |
| What if IsaacLab-Newton is not installed? | Configure `pace_sysid` as a backup with `sysid.pace_enabled=true` and `sysid.pace_root=/path/to/pace-sim2real`, or keep using the local log-based fallback. |
| What does current SysID estimate? | Delay, stiction/deadband proxy, contact summary, pose noise, reset scatter, action-scale bounds, and domain-randomization recommendations. |
| Where do physics SysID backends fit? | `newton_sysid` converts canonical records to IsaacLab-Newton SAGE CSVs. `pace_sysid` converts canonical records to PACE `time`/`dof_pos`/`des_dof_pos` data and can run a configured PACE command. Both remain optional unless required in config. |

## Skill Portability Direction

We should keep many skills only when each one is a portable validation unit. The
goal is not "one skill per robot" or "one skill per tutorial." The goal is a
small library of reusable sim2real skill families, with robot/task specifics
provided by config, adapters, datasets, and thresholds.

So yes, the repo currently has many skills on purpose, but they should stay only
if they remain useful and generic. They are split because each skill owns a
different evidence boundary: perception evidence, SysID evidence, policy
artifact evidence, regression evidence, and hardware approval evidence. Merging
them would make failures harder to diagnose and would make self-improvement less
auditable.

Current skills are useful because they map to generic release questions:

| Current Skill | Portable Skill Family | Why Keep It |
| --- | --- | --- |
| `env_preflight` | Environment readiness | Every robot pipeline needs tool/runtime checks. |
| `isaaclab_task_check` | Simulation task contract check | Every sim task needs observation/action/reward/artifact contract validation. |
| `policy_artifact_audit` | Policy artifact audit | Any learned policy needs reproducible metadata and checkpoint evidence. |
| `ros_preflight` | Middleware/deployment preflight | ROS is the current adapter; the skill concept generalizes to other deployment stacks. |
| `real_data_quality_gate` | Data contract validation | Any real-data submission needs alignment, completeness, calibration, and consistency checks before SysID. |
| `pose_repeatability` | Perception repeatability | Any vision or state-estimation pipeline needs measured repeatability. |
| `sysid_step_response` | System identification evidence | Any sim2real transfer needs actuator, latency, friction, and contact evidence. |
| `newton_sysid` | Optional physics SysID adapter | Newton fitting should be a portable skill that consumes canonical records, not robot-specific raw logs. |
| `pace_sysid` | Optional backup physics SysID adapter | PACE can back up Newton when a PACE-compatible task/config or custom command is provided. |
| `domain_randomization_update` | Sim parameter proposal | Any sim2real pipeline needs bounded randomization updates from data. |
| `action_scale_sweep` | Controller/action range validation | Any policy deployment needs action scaling and saturation checks. |
| `autoresearch_planner` | Experiment proposal and ranking | The self-improvement loop should be task-agnostic. |
| `sim_eval_regression` | Candidate-vs-baseline regression | Every candidate needs regression evidence before promotion. |
| `release_candidate_gate` | Release decision | Every candidate needs a final policy gate. |
| `real_robot_gate` | Human hardware approval | Every physical robot pipeline needs explicit supervised approval. |

Portability rule:

```text
Keep the skill if the question is generic.
Move robot/task details into config, adapters, templates, and threshold policy.
Do not create a new skill just because the robot changed.
```

## Replace Any Skill

The pipeline is modular by design. A user can replace any built-in skill with a
custom skill that has the same `id` and emits the same result contract.

Replacement order:

```text
skills/            built-in skill library
custom_skills/     automatic local overrides
--skill-dir        explicit override directories, applied last
```

Example:

```bash
./scripts/run_skill_harness.sh \
  --skill-dir examples/custom_skills \
  --skill env_preflight
```

Custom command-runner manifest excerpt:

```json
{
  "id": "env_preflight",
  "implementation": "external_command",
  "runner": "command",
  "command": ["python3", "examples/custom_skills/env_preflight/run.py"],
  "quality_gate": {"min_score": 0.7},
  "human_required": false,
  "release_blocking": true,
  "real_robot": false
}
```

See `examples/custom_skills/env_preflight/skill.json` for a complete manifest.

The harness gives command skills:

- `AGENTIC_SIM2REAL_SKILL_INPUT_JSON`: input bundle with config, dataset, manifest, output paths, and previous skill results
- `AGENTIC_SIM2REAL_SKILL_OUTPUT_JSON`: path where the skill writes its result JSON
- `AGENTIC_SIM2REAL_SKILL_OUT_DIR`: evidence directory for this skill
- `AGENTIC_SIM2REAL_SKILL_MANIFEST_DIR`: directory containing the selected `skill.json`
- `AGENTIC_SIM2REAL_ROOT`, `AGENTIC_SIM2REAL_CONFIG_PATH`, `AGENTIC_SIM2REAL_DATASET`, `AGENTIC_SIM2REAL_SKILL_ID`

The skill result contract is stable:

```json
{
  "status": "pass",
  "quality_score": 0.95,
  "confidence": 0.8,
  "blocking_failures": [],
  "warnings": [],
  "evidence_files": ["my_evidence.json"],
  "metrics": {"custom": true}
}
```

The evaluator and release gate treat built-in and custom skills the same way.
This is what makes the architecture portable: users can swap skill
implementations while keeping the same validation, critic, and release policy.

If a custom skill can command hardware, mark `"real_robot": true`. The default
harness skips real-robot skills unless `--include-real` is explicitly passed
after human approval.

## Current UR10e Example

- UR10e arm as the current example robot
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

UR10e tutorial facts encoded in the config and skill contracts:

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
  real_data_quality_gate/skill.json
  pose_repeatability/skill.json
  sysid_step_response/skill.json
  newton_sysid/skill.json
  pace_sysid/skill.json
  domain_randomization_update/skill.json
  action_scale_sweep/skill.json
  autoresearch_planner/skill.json
  sim_eval_regression/skill.json
  release_candidate_gate/skill.json
  real_robot_gate/skill.json

custom_skills/                  Drop-in local skill overrides
examples/custom_skills/         Example external command-runner replacement skill

embodiments/
  manipulator/
    ur10e_gear_assembly/
      real_data/                UR example templates and sessions

agentic_sim2real/
  adapters.py                   Embodiment adapter and real-data source contract
  data_quality.py                Pre-SysID quality gate for canonical records
  llm_orchestrator.py            LLM decision loop, guardrails, journal
  skill_harness.py              Skill runner, validators, scoreboard, release gate
  evaluation_loop.py            LLM/Agent/Evaluator/Critic/Release/Human trace
  autoresearch.py               Experiment planner
  sysid.py                      Sim-real gap and SysID recommendations
  metrics.py                    Delay, stiction, pose, contact metrics
  cli.py                        Command-line entrypoint

scripts/
  prepare_real_data.sh          Convert raw real_data session to records.jsonl
  run_skill_harness.sh          Default validation harness
  run_llm_orchestrator.sh       LLM-selected skill loop and journal
  run_evaluation_loop.sh        Five-stage evaluation trace
  run_autoresearch_loop.sh      LLM loop plus AutoResearch evidence path
  isaaclab_train.sh             Tutorial training wrapper
  ros_preflight.sh              ROS validation helper
  real_robot_human_gate.sh      Human-gated action sender

golden/sample_inputs/           Golden validation fixtures
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
| `ros_preflight` | `orchestrator_agent` | Check deployment middleware config | Yes |
| `real_data_quality_gate` | `evaluator_agent` | Validate aligned records, calibration, and data completeness before SysID | Yes |
| `pose_repeatability` | `perception_agent` | Validate object pose error under the configured gate | Yes |
| `sysid_step_response` | `sysid_agent` | Estimate delay, stiction, contact, SysID targets | Yes |
| `newton_sysid` | `sysid_agent` | Optionally run IsaacLab-Newton fitting from canonical records | No, unless required in config |
| `pace_sysid` | `sysid_agent` | Optional PACE backup SysID when Newton is unavailable | No, unless required in config |
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
  -> LLM orchestrator selects next skill
  -> skill metrics and scorecards
  -> AutoResearch hypotheses
  -> candidate parameter changes
  -> validation harness and journal
  -> critic regression
  -> release gate
  -> human review
```

For this task, AutoResearch focuses on:

- perception noise around `gear_shaft_pos` and `gear_shaft_quat`
- robot stiffness, damping, friction, stiction, and delay
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
LLM orchestrator chooses skills.
Evaluator measures.
Critic challenges.
Release gate decides.
Human approves hardware.
```

This is the core evaluator feature. It separates proposing from measuring and
approval:

- the agent creates a candidate and chooses the next skill call
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
agentic-sim2real --config "$AGENTIC_SIM2REAL_CONFIG" run-evaluation-loop \
  --root . \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/evaluation_demo
```

Outputs:

```text
outputs/evaluation_demo/
  agent_proposal.json
  harness/llm_orchestrator/journal.jsonl
  harness/scoreboard.json
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
| Agent proposes | `agentic_sim2real/autoresearch.py` | Creates hypotheses and candidate parameter families |
| LLM orchestrator chooses | `agentic_sim2real/llm_orchestrator.py` | Chooses valid skill calls from manifest/context; cannot bypass guardrails |
| Evaluator measures | `agentic_sim2real/skill_harness.py` | Runs deterministic skills and writes metrics/evidence |
| Critic challenges | `agentic_sim2real/evaluation_loop.py` | Flags low confidence, warnings, regressions, and failed skills |
| Release gate decides | `agentic_sim2real/evaluation_loop.py` | Blocks or promotes to human review; never autoruns robot |
| Human approves hardware | `agentic_sim2real/safety.py` | Requires explicit supervised hardware approval |

Thresholds live in `harness/threshold_policy.json`:

- hard safety thresholds
- tutorial/spec thresholds
- statistical thresholds
- regression thresholds

Agents may propose new parameters, but the evaluator and release gate own
pass/fail.

## Embodiment Folders

Each embodiment uses the same portable scaffold and has an `embodiment.json`
manifest that the repo can validate:

```text
embodiments/<embodiment_type>/<embodiment_name>/
  embodiment.json
  configs/
  artifacts/
  evaluation/
  real_data/
    templates/
    example_session/
```

Current embodiments:

- `embodiments/manipulator/generic_manipulator`
- `embodiments/manipulator/ur10e_gear_assembly`
- `embodiments/humanoid/generic_humanoid`
- `embodiments/mobile_manipulator/generic_mobile_manipulator`

Use the generic folders as copyable starting points. Keep reusable agent skills
at the repo root; put robot-specific data, config references, artifacts, and
threshold notes inside the embodiment.

Validate the scaffold contract:

```bash
./scripts/validate_embodiments.sh
```

## Real Data Folder

Put real robot data inside the relevant embodiment:

```text
embodiments/<embodiment_type>/<embodiment_name>/real_data/<session_name>/
```

Portable session layout:

```text
embodiments/<embodiment_type>/<embodiment_name>/real_data/<session_name>/
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
```

For the UR10e gear assembly manipulator example, use:

```text
embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name>/
```

Use the template:

```bash
cp -R embodiments/manipulator/ur10e_gear_assembly/real_data/templates   embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1
```

Fill in:

- `joint_data/joint_states.csv`: policy actions, measured joint positions, optional commanded joint positions, joint velocities, optional end-effector pose
- `pose_data/object_pose.csv`: perception object/target pose estimate and reference pose if available
- `camera_data/index.csv`: color/depth image paths or frame provenance
- `contact_data/contact.csv`: force or force-proxy samples
- `episode_labels/labels.csv`: success, failure mode, notes
- `calibration/calibration.json`: robot calibration, camera frames, hand-eye metadata

The UR tutorial still uses `pose_data/shaft_pose.csv` because the task object is
the gear shaft. The loader accepts both `object_pose.csv` and `shaft_pose.csv`,
and generated pipeline records use the portable `object_pose_estimate` and
`object_pose_reference` fields.

Convert raw subfolders into the pipeline format:

```bash
./scripts/prepare_real_data.sh embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1
```

This writes:

```text
embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1/aligned/records.jsonl
embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1/aligned/prepare_summary.json
```

Then run the pipeline directly on the session folder:

```bash
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1 ./scripts/run_skill_harness.sh
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1 ./scripts/run_evaluation_loop.sh
```

The loader automatically uses `aligned/records.jsonl` when a session directory
is passed as `--dataset`. If aligned records are missing but CSV subfolders are
complete, `run-harness` and `run-evaluation-loop` auto-create them through the
adapter-based ingestor.

For IsaacLab-Newton SysID, include commanded joint-position columns in the same
joint order as measured state: `command_0 ... command_N` or
`joint_command_0 ... joint_command_N`. The bridge will not silently treat policy
actions as joint-position commands unless `sysid.newton_command_source="action"`
or `sysid.newton_allow_action_as_command=true` is set.

Inspect a submitted session before running the full loop:

```bash
agentic-sim2real inspect-real-data \
  --root . \
  --session embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1
```

Raw `rosbag2` or image-only submissions are detected, but they need an
embodiment `real_data.external_ingestor_command` that converts them into the
same canonical `aligned/records.jsonl` contract. This keeps raw parsing modular
instead of hardcoding UR topics into the core pipeline.

## Step By Step: Local Skill Harness

### 1. Clone and install

```bash
git clone https://github.com/recscores007/agentic_sim2real.git
cd agentic_sim2real
python3 -m pip install -e .
cp configs/ur10e_gear_assembly.example.json configs/ur10e_gear_assembly.local.json
export AGENTIC_SIM2REAL_CONFIG=$PWD/configs/ur10e_gear_assembly.local.json
```

Edit `configs/ur10e_gear_assembly.local.json` for your Isaac Lab root, Isaac
ROS workspace, gripper type, `ROS_DOMAIN_ID`, and manipulator config path.

To enable IsaacLab-Newton later, keep the core pipeline unchanged and configure
the portable Newton skill. The built-in bridge converts aligned records to the
SAGE-style `control.csv`, `state_motor.csv`, and `joint_list.txt` files expected
by `es-rl/IsaacLab-Newton/scripts/sysid/run_sysid.py`:

```json
{
  "sysid": {
    "newton_enabled": true,
    "require_newton": false,
    "newton_root": "/path/to/IsaacLab-Newton",
    "newton_robot_name": "ur10e",
    "newton_joint_names": [
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
    ],
    "newton_joint_types": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
    "newton_command_source": "auto",
    "newton_run_mode": "run",
    "min_newton_records": 5,
    "newton_max_iter": 100,
    "newton_num_envs": 64,
    "newton_control_freq_hz": 500
  }
}
```

If `newton_command` is also configured, it overrides the built-in bridge. That
custom command receives canonical aligned records and writes JSON evidence. If
Newton is not enabled, `newton_sysid` skips and the local SysID skill remains the
active fallback.

The bridge avoids UR-only assumptions: commanded/measured joint vectors are
portable fields, joint names and Newton joint types come from config, and
non-UR embodiments can replace only the config or the `newton_sysid` skill
without changing the evaluator architecture.

To double-check conversion without launching Isaac Lab or Newton, run:

```bash
PYTHONPATH=. python3 scripts/newton/run_newton_sysid_bridge.py \
  --input outputs/<run>/skills/newton_sysid/newton_input.json \
  --output outputs/<run>/skills/newton_sysid/newton_output.json \
  --prepare-only
```

If IsaacLab-Newton is not installed, PACE can be configured as the next SysID
backend. PACE is task-config driven: its public `scripts/pace/fit.py` expects a
PACE-compatible Isaac Lab task whose `env_cfg.sim2real.data_dir` points to the
same data file the bridge writes. For non-legged or custom embodiments, provide a
matching PACE task/config or a custom `pace_command`.

```json
{
  "sysid": {
    "sysid_backend_preference": ["newton", "pace", "local"],
    "pace_enabled": true,
    "require_pace": false,
    "pace_root": "/path/to/pace-sim2real",
    "pace_task": "Isaac-Pace-YourRobot-v0",
    "pace_robot_name": "your_robot",
    "pace_data_dir": "your_robot/chirp_data.pt",
    "pace_run_mode": "run",
    "min_pace_records": 5,
    "pace_num_envs": 4096
  }
}
```

If the public PACE task does not match your embodiment, keep `pace_enabled=false`
or set a custom command. The command receives `{input}` as the prepared PACE
input summary JSON, `{data_file}` as the generated `.pt` file, `{dataset}` as
the original real-data session, and `{output}` as the expected result JSON.

```json
{
  "sysid": {
    "pace_enabled": true,
    "pace_command": [
      "python3",
      "path/to/your_pace_adapter.py",
      "--input",
      "{input}",
      "--data-file",
      "{data_file}",
      "--output",
      "{output}"
    ]
  }
}
```

The preflight report now calls out which SysID backends are available:
Newton, PACE, and the local fallback. For PACE it also checks whether the repo
entrypoint exists, whether a custom command is configured, and whether the
default PACE path has both `pace_task` and `pace_data_dir`.

### 2. List the skills

```bash
agentic-sim2real --config "$AGENTIC_SIM2REAL_CONFIG" list-skills
```

### 3. Validate every skill manifest

```bash
agentic-sim2real --config "$AGENTIC_SIM2REAL_CONFIG" validate-skills --root .
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
agentic-sim2real --config "$AGENTIC_SIM2REAL_CONFIG" run-harness \
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

### 5. Run the LLM-orchestrated loop

```bash
./scripts/run_llm_orchestrator.sh
```

Equivalent CLI:

```bash
agentic-sim2real --config "$AGENTIC_SIM2REAL_CONFIG" run-llm-loop \
  --root . \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/llm_orchestrator_demo \
  --gap-hint perception
```

Expected outputs:

```text
outputs/llm_orchestrator_demo/
  llm_orchestrator/journal.jsonl
  llm_orchestrator/orchestrator_summary.json
  llm_orchestrator/steps/step_###_context.json
  llm_orchestrator/steps/step_###_decision.json
  scoreboard.json
  release_candidate.json
```

The default `scripted` provider is deterministic. Use `--gap-hint actuator`,
`--gap-hint perception`, `--gap-hint contact`, or another supported family to
steer first-pass triage. Use `--llm-provider command` to delegate each decision
to a real LLM adapter.

### 6. Inspect the scoreboard

```bash
cat outputs/harness_demo/scoreboard.json
```

Release status is pass only if every release-blocking offline skill passes.

### 7. Run only one skill

```bash
agentic-sim2real --config "$AGENTIC_SIM2REAL_CONFIG" run-harness \
  --root . \
  --dataset sample_data/real_log_demo.jsonl \
  --out outputs/pose_only \
  --skill pose_repeatability
```

Use this while developing or debugging one skill.

### 8. Run AutoResearch loop

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
cp -R embodiments/manipulator/ur10e_gear_assembly/real_data/templates \
  embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1
# Fill the CSV files and calibration.json with real session data.
./scripts/prepare_real_data.sh embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1 ./scripts/run_evaluation_loop.sh
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

This checks command availability (`python3`, ROS tools, `git`, GPU utility when
present), repo structure, Isaac Lab root/script paths, Isaac ROS workspace and
deployment config, ROS middleware settings, and SysID backend availability
(`newton`, `pace`, `local`).

Before insertion, run the tutorial's manual-on-robot validation:

```bash
export ENABLE_MANIPULATOR_TESTING=manual_on_robot
launch_test $(ros2 pkg prefix --share isaac_ros_manipulation_bringup)/test/test_pose_estimation_error_test.py
bash ${ISAAC_ROS_WS}/src/isaac_ros_manipulation/isaac_ros_manipulation_bringup/test/compare_pose_estimation_results.sh
```

Do not continue if pose error is above 1 cm.

### 2. Record real observations

```bash
./scripts/collect_real_logs.sh rosbags/agentic_sim2real_day1
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
- real-data quality gate passes before SysID
- pose p95 error is under 1 cm
- SysID evidence exists
- Newton SysID either passes, is skipped as optional, or is explicitly required by config
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
export I_ACCEPT_AGENTIC_SIM2REAL_REAL_ROBOT_RISK=yes
./scripts/isaac_ros_launch_gear.sh
```

Terminal 2, after `cuMotion is ready for planning queries!`:

```bash
export I_ACCEPT_AGENTIC_SIM2REAL_REAL_ROBOT_RISK=yes
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
PYTHONPYCACHEPREFIX=/tmp/agentic_sim2real_pycache python3 -m py_compile agentic_sim2real/*.py tests/test_pipeline.py
PYTHONPATH=. python3 -m unittest discover -s tests
PYTHONPATH=. python3 -m agentic_sim2real.cli --config configs/ur10e_gear_assembly.example.json inspect-real-data --session embodiments/manipulator/ur10e_gear_assembly/real_data/example_session --root .
PYTHONPATH=. python3 -m agentic_sim2real.cli --config configs/ur10e_gear_assembly.example.json validate-skills --root .
PYTHONPATH=. python3 -m agentic_sim2real.cli --config configs/ur10e_gear_assembly.example.json run-harness --root . --dataset sample_data/real_log_demo.jsonl --out /tmp/agentic_sim2real_harness
PYTHONPATH=. python3 -m agentic_sim2real.cli --config configs/ur10e_gear_assembly.example.json run-llm-loop --root . --dataset sample_data/real_log_demo.jsonl --out /tmp/agentic_sim2real_llm
PYTHONPATH=. python3 -m agentic_sim2real.cli --config configs/ur10e_gear_assembly.example.json run-evaluation-loop --root . --dataset sample_data/real_log_demo.jsonl --out /tmp/agentic_sim2real_eval
bash -n scripts/*.sh
```

CI runs these checks on GitHub.
