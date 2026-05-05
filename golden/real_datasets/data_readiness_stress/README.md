# Golden Real Dataset: Data Readiness Stress

This synthetic dataset is intentionally imperfect. It is meant to test whether
the pipeline alerts the user early when real data cannot support strong
sim2real conclusions.

Expected issues:

- Low telemetry rate: records are at 10 Hz while delay observability requires 50 Hz.
- Sparse visual coverage: only BKPIECE has one linked camera frame.
- Heldout visual gap: RRTstar, TRRT, and PRMstar are heldout but have zero frame links.
- Orphan visual data: `frames_full/SBL/` contains extracted frames that are not in `aligned/records.jsonl`.
- Auto-only success labels: all success labels use `success_label_source=tracking_error_threshold`.
- Circular pose validation: object pose estimate and reference are identical FK proxies.
- Missing optional streams: some records omit `joint_velocity` and `contact_force`.

The fixture is embodiment-neutral: planner names are just scenarios, and the
canonical record fields can be consumed by any manipulator, humanoid, or mobile
manipulator adapter that accepts joint/action trajectories plus object pose.
