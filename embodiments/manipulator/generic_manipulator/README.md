# Generic Embodiment Scaffold

This scaffold is a portable starting point for a new embodiment. Copy or rename
the folder, then replace the templates with robot-specific configs, artifacts,
threshold notes, and real data.

The structure is intentionally the same across embodiment classes:

```text
configs/
artifacts/
evaluation/
real_data/
  templates/
  example_session/
```

Reusable skills should stay in the root `skills/` library. Embodiment folders
should only contain robot/task-specific data, thresholds, config references, and
artifacts.
