# Evaluator Architecture

This repo uses a five-role evaluation loop:

```text
Agent proposes.
Evaluator measures.
Critic challenges.
Release gate decides.
Human approves hardware.
```

## 1. Agent Proposes

The agent creates hypotheses and candidate changes. In this repo, the
AutoResearch planner proposes:

- perception-noise experiments
- UR10e SysID experiments
- fixture/generalization experiments
- action-scale/contact experiments

The proposal is written to `agent_proposal.json`.

The agent cannot pass itself, relax safety gates, or run the real robot.

## 2. Evaluator Measures

The evaluator is deterministic. It runs skill validators, computes metrics, and
writes evidence.

Examples:

- `pose_repeatability` measures shaft pose p95 error.
- `sysid_step_response` measures delay and stiction proxy.
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
export I_ACCEPT_UR_REAL_ROBOT_RISK=yes
```

That approval means only supervised hardware execution is allowed. It does not
authorize unattended robot motion.

The human gate writes `human_hardware_gate.json`.

## Threshold Policy

Thresholds live in `threshold_policy.json` and are divided into:

- hard safety thresholds
- tutorial/spec thresholds
- statistical thresholds
- regression thresholds

Agents may propose parameter changes, but the threshold policy decides whether
the candidate is releasable.
