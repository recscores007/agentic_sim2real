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
