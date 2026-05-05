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
python3 -m agentic_sim2real.cli validate-embodiments --root .
```

Use `object_pose.csv` as the portable pose file name. Task-specific embodiments
may keep an object-specific source file such as `shaft_pose.csv`; the generated
pipeline records still use canonical `object_pose_estimate` and
`object_pose_reference` fields.

Use `command_0 ... command_N` or `joint_command_0 ... joint_command_N` in
`joint_data/joint_states.csv` when a SysID skill needs commanded joint
positions. This stays portable across manipulators, humanoids, and mobile
manipulators; robot-specific joint names belong in config.

`embodiment.json` is also the adapter contract. Keep raw parsing choices here,
not in core skills:

```json
{
  "real_data": {
    "canonical_pose_file": "object_pose.csv",
    "accepted_pose_files": ["object_pose.csv"],
    "accepted_raw_sources": ["aligned_records", "csv_session", "rosbag2", "image_sequence"],
    "external_ingestor_command": []
  }
}
```

If a robot needs `rosbag2` or image-only ingestion, set
`external_ingestor_command` to a command that writes the same canonical
`aligned/records.jsonl` schema. The evaluator and skills should not need to
know the robot's raw topic names.
