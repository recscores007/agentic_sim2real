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

LLM-orchestrated command:

```bash
./scripts/run_llm_orchestrator.sh
```

The LLM loop asks a provider for the next skill decision, validates that
decision against dependencies and safety guardrails, runs the selected skill
through the same deterministic harness, then appends the decision/result to
`llm_orchestrator/journal.jsonl`.

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

- `AGENTIC_SIM2REAL_SKILL_INPUT_JSON`
- `AGENTIC_SIM2REAL_SKILL_OUTPUT_JSON`
- `AGENTIC_SIM2REAL_SKILL_OUT_DIR`
- `AGENTIC_SIM2REAL_SKILL_MANIFEST_DIR`
- `AGENTIC_SIM2REAL_ROOT`
- `AGENTIC_SIM2REAL_CONFIG_PATH`
- `AGENTIC_SIM2REAL_DATASET`
- `AGENTIC_SIM2REAL_SKILL_ID`

They must write the standard skill result JSON. The harness still applies the
manifest quality gate and release-blocking behavior after the custom command
returns.

## Autorun Boundary

A complete embodiment-scoped real-data session can drive the offline pipeline
without human intervention. If `aligned/records.jsonl` already exists, the
harness uses it directly. If the session contains complete CSV subfolders, the
harness auto-creates aligned records through the embodiment adapter before
running skills.
For the UR manipulator example, use
`embodiments/manipulator/ur10e_gear_assembly/real_data/<session_name>`.
That includes real-data inspection, the `real_data_quality_gate`, SysID,
AutoResearch proposal, evaluator scoring, critic review, and release-gate
output. Raw `rosbag2` or image-only sessions are detected, then routed to an
embodiment `real_data.external_ingestor_command` when one is configured.

Physical robot motion is not part of offline autorun. The release gate always
keeps `safe_to_autorun_robot: false`, and the hardware step requires explicit
human approval.

## SysID Boundary

Current default SysID is log-based and implemented in
`agentic_sim2real/sysid.py`. It estimates delay, stiction/deadband, pose noise,
reset scatter, contact summary, and bounded recommendations.

IsaacLab-Newton is represented by the portable `newton_sysid` skill. It is
skipped by default, runs only when `config.sysid.newton_command` is configured,
and can be made mandatory with `config.sysid.require_newton=true`.

## LLM-Orchestrated Evaluation

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
| LLM orchestrator chooses | `harness/llm_orchestrator/journal.jsonl` | Stepwise skill decisions with guardrail results |
| Evaluator measures | `evaluator_measurements.json` | Skill metrics, scores, and evidence files |
| Critic challenges | `critic_challenges.json` | Low-confidence evidence, warnings, regressions, and failures |
| Release gate decides | `release_decision.json` | Block or promote to human review |
| Human approves hardware | `human_hardware_gate.json` | Explicit supervised hardware approval state |
