# Embodiments

Embodiments hold robot/task-specific assets. The core skills and evaluator stay
portable, while embodiment folders carry configs, scripts, sample data, and real
data for a specific robot class.

Expected layout:

```text
embodiments/<embodiment_type>/<embodiment_name>/
  real_data/
    templates/
    <session_name>/
```

Examples of embodiment types:

- `manipulator`
- `humanoid`
- `mobile_manipulator`

