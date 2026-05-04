# Validation Harness

The harness validates skill quality before release.

Validation layers:

1. Static manifest validation: every skill has an input/output contract.
2. Unit validation: Python tests exercise the harness and offline pipeline.
3. Golden validation: sample logs and baseline metrics must produce a passing scoreboard.
4. Release validation: blocking failures prevent promotion to human review.

The manifest contract is documented in `schemas/skill_manifest.schema.json`.

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
