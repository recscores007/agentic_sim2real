# Contributing

Thanks for helping improve Agentic Sim2Real. This repository is safety-sensitive because it can describe workflows that lead toward real robot execution, so changes should be small, reviewable, and backed by evidence.

## Development Setup

```bash
git clone https://github.com/recscores007/agentic_sim2real.git
cd agentic_sim2real
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Before Opening A Pull Request

Run the local checks:

```bash
PYTHONPATH=. python3 -m py_compile agentic_sim2real/*.py tests/test_pipeline.py
PYTHONPATH=. python3 -m unittest discover -s tests
PYTHONPATH=. python3 -m agentic_sim2real.cli --config configs/ur10e_gear_assembly.example.json validate-skills --root .
bash -n scripts/*.sh
```

## Pull Request Checklist

- Explain the problem and the behavior change.
- Include tests or a clear reason tests are not applicable.
- Update README or docs when commands, outputs, configs, or safety behavior change.
- Keep generated outputs, logs, rosbags, checkpoints, and videos out of Git.
- Do not commit secrets, customer data, private robot calibration, or access tokens.
- Do not bypass `release_candidate_gate`, `real_robot_gate`, or human approval checks.

## Adding Or Changing Skills

Each built-in skill should have:

- A manifest under `skills/<skill_id>/skill.json`.
- Declared inputs, outputs, dependencies, owner agent, and release-blocking status.
- Deterministic validation behavior in the harness or an explicit external runner.
- Evidence written to the run output directory.
- Tests for manifest validation and expected scoring behavior.

Custom or customer-specific skills should live outside the built-in `skills/` directory and be loaded with `--skill-dir` unless they are intended to become part of the default harness.

## Documentation Changes

Favor practical examples over long design prose. A first-time user should be able to install the package, run the sample loop, and understand the generated artifacts from the README alone.

## Code Of Conduct

All contributors are expected to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
