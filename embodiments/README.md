# Embodiments

Embodiments hold robot/task-specific assets. The root `skills/` library and the
evaluator stay portable; embodiment folders carry only robot/task-specific
manifests, config references, artifacts, evaluation notes, and real data.

Every embodiment must follow this structure:

```text
embodiments/<embodiment_type>/<embodiment_name>/
  embodiment.json
  README.md
  configs/
  artifacts/
  evaluation/
  real_data/
    README.md
    templates/
      camera_data/
      joint_data/
      pose_data/
      contact_data/
      episode_labels/
      calibration/
      aligned/
    example_session/
      camera_data/
      joint_data/
      pose_data/
      contact_data/
      episode_labels/
      calibration/
      aligned/
```

Current embodiments:

- `manipulator/generic_manipulator`: portable manipulator scaffold
- `manipulator/ur10e_gear_assembly`: concrete Isaac Lab gear assembly example
- `humanoid/generic_humanoid`: portable humanoid scaffold
- `mobile_manipulator/generic_mobile_manipulator`: portable mobile manipulator scaffold

Validate the contract from the repo root:

```bash
python3 -m ur_agentic.cli validate-embodiments --root .
```

Use `object_pose.csv` as the portable pose file name. Task-specific embodiments
may keep an object-specific source file such as `shaft_pose.csv`; the generated
pipeline records still use canonical `object_pose_estimate` and
`object_pose_reference` fields.
