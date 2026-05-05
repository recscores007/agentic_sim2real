# Validation Harness

The harness validates skill quality before release.

Validation layers:

1. Static manifest validation: every skill has an input/output contract.
2. Unit validation: Python tests exercise the harness and offline pipeline.
3. Golden validation: sample logs and baseline metrics must produce a passing scoreboard.
4. Release validation: blocking failures prevent promotion to human review.

The manifest contract is documented in `schemas/skill_manifest.schema.json`.
The threshold policy is documented in `threshold_policy.json`.

Default command:

```bash
./scripts/run_skill_harness.sh
```

Outputs:

```text
outputs/harness_demo/
  scoreboard.json
  release_candidate.json
  skills/<skill_id>/result.json
```

The harness may promote a candidate to human review. It never authorizes
unattended robot motion.

## Modular Skill Replacement

The harness loads skills in layers:

```text
skills/            built-in skills
custom_skills/     automatic local overrides
--skill-dir        explicit override directories
```

If two manifests have the same `id`, the later layer replaces the earlier one.
This lets users replace any skill without changing the evaluator or release
gate.

Run the included example replacement:

```bash
./scripts/run_skill_harness.sh \
  --skill-dir examples/custom_skills \
  --skill env_preflight
```

Command-runner skills receive:

- `UR_SKILL_INPUT_JSON`
- `UR_SKILL_OUTPUT_JSON`
- `UR_SKILL_OUT_DIR`
- `UR_SKILL_MANIFEST_DIR`
- `UR_ROOT`
- `UR_CONFIG`
- `UR_DATASET`
- `UR_SKILL_ID`

They must write the standard skill result JSON. The harness still applies the
manifest quality gate and release-blocking behavior after the custom command
returns.

## Autorun Boundary

A complete `real_data/<session_name>/` folder can drive the offline pipeline
without human intervention after it is converted to `aligned/records.jsonl`.
That includes skill execution, AutoResearch proposal, evaluator scoring, critic
review, and release-gate output.

Physical robot motion is not part of offline autorun. The release gate always
keeps `safe_to_autorun_robot: false`, and the hardware step requires explicit
human approval.

## SysID Boundary

Current SysID is log-based and implemented in `ur_agentic/sysid.py`. It
estimates delay, stiction/deadband, pose noise, reset scatter, contact summary,
and bounded recommendations.

The repo does not yet invoke IsaacLab-Newton. Newton fitting should be added as
a portable `newton_sysid` skill, not embedded directly into a UR-specific skill.

## Five-Stage Evaluation

Use the evaluation loop when you want the repo to explain the decision process,
not just run the skill DAG:

```bash
./scripts/run_evaluation_loop.sh
```

It writes:

```text
outputs/evaluation_demo/
  agent_proposal.json
  evaluator_measurements.json
  critic_challenges.json
  release_decision.json
  human_hardware_gate.json
  evaluation_trace.md
```

Meaning:

| Stage | Output | Purpose |
| --- | --- | --- |
| Agent proposes | `agent_proposal.json` | AutoResearch hypotheses and candidate parameter families |
| Evaluator measures | `evaluator_measurements.json` | Skill metrics, scores, and evidence files |
| Critic challenges | `critic_challenges.json` | Low-confidence evidence, warnings, regressions, and failures |
| Release gate decides | `release_decision.json` | Block or promote to human review |
| Human approves hardware | `human_hardware_gate.json` | Explicit supervised hardware approval state |
