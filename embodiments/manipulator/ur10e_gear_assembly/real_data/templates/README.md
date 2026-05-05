# Real Data Templates

Copy this folder for a new real-data session:

```bash
cp -R embodiments/manipulator/ur10e_gear_assembly/real_data/templates \
  embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1
```

Then replace the template rows with your real rows and run:

```bash
./scripts/prepare_real_data.sh embodiments/manipulator/ur10e_gear_assembly/real_data/ur10e_day1
```

The converter aligns rows by `episode_index` and nearest `timestamp`.
Default tolerance is 0.05 seconds.
