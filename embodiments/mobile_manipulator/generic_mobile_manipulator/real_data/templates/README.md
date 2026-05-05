# Real Data Templates

Copy this folder to create a new real-data session:

```bash
cp -R embodiments/<embodiment_type>/<embodiment_name>/real_data/templates \
  embodiments/<embodiment_type>/<embodiment_name>/real_data/<session_name>
```

Then replace the example rows with real logs and run:

```bash
./scripts/prepare_real_data.sh embodiments/<embodiment_type>/<embodiment_name>/real_data/<session_name>
```

The converter aligns rows by `episode_index` and nearest `timestamp`.
