# Camera Data

File: `index.csv`

This is image provenance for each sample. The current evaluator does not parse
image pixels directly; it uses this index to keep camera data tied to the pose
and joint logs.

Columns:

- `episode_index`
- `timestamp`
- `camera_name`
- `color_image`
- `depth_image`

Paths should be relative to the session folder, for example:

```text
camera_data/color/episode_000_t000000.png
camera_data/depth/episode_000_t000000.png
```
