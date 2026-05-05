# Evaluator Architecture

This repo uses an LLM-orchestrated evaluation loop:

```text
Agent proposes.
LLM orchestrator chooses skills.
Evaluator measures.
Critic challenges.
Release gate decides.
Human approves hardware.
```

## 1. Agent Proposes

The agent creates hypotheses and candidate changes. In this repo, the
AutoResearch planner proposes:

- perception-noise experiments
- robot SysID experiments
- fixture/generalization experiments
- action-scale/contact experiments

The proposal is written to `agent_proposal.json`.

The agent cannot pass itself, relax safety gates, or run the real robot.

## 1.5. LLM Orchestrator Chooses Skills

The LLM orchestrator reads the task context, manifest catalog, completed
scorecards, user-provided gap hints, and runnable skills. It proposes one action
at a time:

- `run_skill`
- `stop`
- `request_human_review`

Guardrails reject invalid decisions before any skill runs:

- missing dependencies
- unknown skills
- repeated skills unless retries are explicitly enabled
- release gates before required evidence exists
- hardware-facing skills without explicit approval

The orchestrator writes `llm_orchestrator/journal.jsonl`.

Gap hints such as `perception`, `actuator`, `contact`, `latency`,
`domain_randomization`, `deployment`, and `policy` are treated as starting
hypotheses. They bias skill ordering only after prerequisites are complete.

## 2. Evaluator Measures

The evaluator is deterministic. It executes the LLM-selected skill, runs
validators, computes metrics, and writes evidence.

Examples:

- `real_data_quality_gate` checks canonical record completeness before SysID.
- `pose_repeatability` measures object pose p95 error.
- `sysid_step_response` measures delay and stiction proxy.
- `newton_sysid` optionally runs IsaacLab-Newton fitting from aligned records.
- `action_scale_sweep` measures whether a candidate action scale is inside limits.

The evaluator writes `evaluator_measurements.json`.

## 3. Critic Challenges

The critic looks for weak evidence:

- failed skills
- low confidence
- warnings
- regression against baseline
- unsafe or unsupported promotion claims

The critic writes `critic_challenges.json`.

## 4. Release Gate Decides

The release gate aggregates all release-blocking skills and critic findings. It
can return:

- `promote_to_human_review`
- `blocked`

It always writes `safe_to_autorun_robot: false`.

The release gate writes `release_decision.json`.

## 5. Human Approves Hardware

The hardware gate requires:

```bash
export I_ACCEPT_AGENTIC_SIM2REAL_REAL_ROBOT_RISK=yes
```

That approval means only supervised hardware execution is allowed. It does not
authorize unattended robot motion.

The human gate writes `human_hardware_gate.json`.

## Threshold Policy

Thresholds live in `threshold_policy.json` and are divided into:

- hard safety thresholds
- embodiment/spec thresholds
- data-quality thresholds
- SysID/Newton thresholds
- statistical thresholds
- regression thresholds

Agents may propose parameter changes, but the threshold policy decides whether
the candidate is releasable.

## IsaacLab-Newton Status

The default SysID path is the local log-based estimator in
`agentic_sim2real/sysid.py`.

IsaacLab-Newton is represented by the portable `newton_sysid` skill. Configure
`config.sysid.newton_command` to run a Newton fitting entrypoint. The skill
receives canonical aligned records, writes fitted parameters and residual
metrics, and feeds the same evaluator and release gate. It is optional unless
`release.profile="release_candidate"` or `config.sysid.require_newton=true`.
When Newton/PACE evidence is absent, the harness records `evidence_missing`
rather than a clean pass.
