# UR10e Gear Assembly Manipulator Example

This embodiment folder contains UR10e gear-assembly-specific assets. The
portable skills and evaluator live at the repo root; this folder carries data
and examples for one manipulator embodiment.

Real data for this embodiment lives in:

```text
embodiments/manipulator/ur10e_gear_assembly/real_data/
```

Run the example session from the repo root:

```bash
./scripts/prepare_real_data.sh embodiments/manipulator/ur10e_gear_assembly/real_data/example_session
DATASET=embodiments/manipulator/ur10e_gear_assembly/real_data/example_session ./scripts/run_evaluation_loop.sh
```

