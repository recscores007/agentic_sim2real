# Example Session

This is a tiny synthetic real-data session that shows the required structure.

Generate the aligned pipeline file:

```bash
./scripts/prepare_real_data.sh real_data/example_session
```

Run the harness directly on the session folder:

```bash
DATASET=real_data/example_session ./scripts/run_skill_harness.sh
```

The harness will load `real_data/example_session/aligned/records.jsonl`.
