# UR10e Gear Assembly Manipulator Example

This embodiment folder contains UR10e gear-assembly-specific assets. The
portable skills and evaluator live at the repo root; this folder carries data
and examples for one manipulator embodiment.

This folder follows the shared embodiment structure:

```text
configs/      embodiment-specific config overrides or references
artifacts/    local checkpoints, exported policies, and generated assets
evaluation/   threshold notes and release evidence for this embodiment
real_data/    raw and aligned real robot/session data
```

Real data for this embodiment lives in:

```text
embodiments/manipulator/ur10e_gear_assembly/real_data/
```

Run the example session from the repo root:

```bash
./scripts/prepare_real_data.sh embodiments/manipulator/ur10e_gear_assembly/real_data/example_session
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/example_session ./scripts/run_evaluation_loop.sh
```

The real-data loader also accepts the generic `pose_data/object_pose.csv`
filename. This UR tutorial keeps `shaft_pose.csv` because that is the tutorial
object name.
