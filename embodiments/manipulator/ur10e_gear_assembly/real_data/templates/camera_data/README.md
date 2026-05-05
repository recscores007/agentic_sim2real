# Camera Data

File: `index.csv`

This is image provenance for each sample. The evaluator uses this index to keep
camera data tied to the pose, video, and joint logs.

Columns:

- `episode_index`
- `timestamp`
- `camera_name`
- `color_image`
- `depth_image`
- `video_path` optional path to the source video that produced the frame
- `frame_index` optional frame number inside the source video

Paths should be relative to the session folder, for example:

```text
camera_data/color/episode_000_t000000.png
camera_data/depth/episode_000_t000000.png
```

For customer uploads, put full task or calibration videos under
`video_data/` and list them in `video_data/index.csv`. The video analyzer can
then tune camera intrinsics/extrinsics and latency from `video_data/analysis.json`.
