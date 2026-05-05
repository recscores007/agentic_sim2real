# Golden Real Dataset: UR10e Video Tuning

This fixture demonstrates how uploaded UR10e task videos feed camera tuning and
environmental friction domain-randomization decisions.

It is a small synthetic golden dataset for pipeline validation. The videos are
not photorealistic robot evidence; they are deterministic clips that make the
video-evidence contract easy to inspect in GitHub and easy to exercise in CI.
Real customer uploads should replace these clips with recorded RGB/depth videos
and should either provide `video_data/analysis.json` or configure
`video_evidence.analysis_command`.

## What The Pipeline Should Decide

- Camera tuning should pass from `video_data/ur10_camera_calibration.m4v`.
- Contact/friction tuning should pass from
  `video_data/ur10_contact_friction.m4v`.
- Domain randomization should use the video-derived object and gripper friction
  values from `video_data/analysis.json`.

Expected tuning values:

| Parameter | Expected value |
| --- | ---: |
| Camera reprojection error | `0.82 px` |
| Camera latency | `0.018 s` |
| Object static friction | `0.88` |
| Object dynamic friction | `0.71` |
| Gripper pad static friction | `1.18` |
| Gripper pad dynamic friction | `0.98` |
| Friction sweep | `[0.73, 1.03]` |

## Run The Fixture

```bash
PYTHONPATH=. python3 -m agentic_sim2real.cli \
  --config configs/ur10e_gear_assembly.example.json \
  run-harness \
  --root . \
  --dataset golden/real_datasets/ur10e_video_tuning \
  --out outputs/ur10e_video_tuning_harness
```

The relevant outputs are:

- `outputs/ur10e_video_tuning_harness/subchecks/video_camera_tuning/video_camera_tuning.json`
- `outputs/ur10e_video_tuning_harness/subchecks/video_contact_friction/video_contact_friction.json`
- `outputs/ur10e_video_tuning_harness/subchecks/domain_randomization_update/domain_randomization_candidate.json`
- `outputs/ur10e_video_tuning_harness/skills/agentic_tuning_plan/sim_params_patch.yaml`
- `outputs/ur10e_video_tuning_harness/skills/agentic_tuning_plan/sim_params_patch.json`
