# Embodiments

Embodiments hold robot/task-specific assets. The core skills and evaluator stay
portable, while embodiment folders carry configs, local artifacts, evaluation
threshold notes, sample data, and real data for a specific robot class.

Expected layout:

```text
embodiments/<embodiment_type>/<embodiment_name>/
  README.md
  configs/
  artifacts/
  evaluation/
  real_data/
    templates/
    <session_name>/
```

Examples of embodiment types:

- `manipulator`
- `humanoid`
- `mobile_manipulator`

Current scaffolded embodiments:

- `manipulator/generic_manipulator`: portable starting point for fixed-base arms
- `manipulator/ur10e_gear_assembly`: concrete Isaac Lab gear assembly example
- `humanoid/generic_humanoid`: portable starting point for humanoid skills/data
- `mobile_manipulator/generic_mobile_manipulator`: portable starting point for mobile arms

Keep robot-specific logic in the embodiment folder and config. Keep reusable
capability checks in the root `skills/` library so the agent/evaluator loop can
reuse them across embodiments.
