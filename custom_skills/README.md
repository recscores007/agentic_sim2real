# Custom Skills

Put replacement skills here when you want this repo to load them automatically.

Any folder shaped like this replaces the built-in skill with the same `id`:

```text
custom_skills/<skill_id>/
  skill.json
  run.py
```

You can also keep custom skills outside the repo and pass them explicitly:

```bash
./scripts/run_skill_harness.sh --skill-dir /path/to/my_skills
./scripts/run_evaluation_loop.sh --skill-dir /path/to/my_skills
```

Replacement rule:

```text
skills/            built-ins
custom_skills/     automatic local overrides
--skill-dir        explicit overrides, applied last
```

The replacement skill must emit the same result contract as a built-in skill:

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

The evaluator and release gate do not care whether the skill is built-in or
custom. They only read the result contract and evidence files.

Command-runner skills receive these environment variables:

- `UR_SKILL_INPUT_JSON`
- `UR_SKILL_OUTPUT_JSON`
- `UR_SKILL_OUT_DIR`
- `UR_SKILL_MANIFEST_DIR`
- `UR_ROOT`
- `UR_CONFIG`
- `UR_DATASET`
- `UR_SKILL_ID`
