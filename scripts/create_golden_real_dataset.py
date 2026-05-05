from __future__ import annotations

import base64
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "golden" / "real_datasets" / "data_readiness_stress"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def main() -> None:
    random.seed(21)
    _mkdirs()
    records = _records()
    _write_records(records)
    _write_camera_index(records)
    _write_orphan_sbl_sources()
    _write_calibration()
    _write_readme()
    _write_expected_alerts()


def _mkdirs() -> None:
    for rel in [
        "aligned",
        "camera_data/color/BKPIECE",
        "camera_data/depth/BKPIECE",
        "frames_full/SBL",
        "raw_sources",
        "calibration",
    ]:
        (DATASET / rel).mkdir(parents=True, exist_ok=True)


def _records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        {"episode": 0, "planner": "BKPIECE", "split": "train", "records": 8, "frame_every": 8, "video": True},
        {"episode": 1, "planner": "PRM", "split": "train", "records": 5, "frame_every": 0, "screencast_only": True},
        {"episode": 2, "planner": "RRTstar", "split": "heldout", "records": 4, "frame_every": 0, "video": True},
        {"episode": 3, "planner": "TRRT", "split": "heldout", "records": 4, "frame_every": 0, "video": True},
        {"episode": 4, "planner": "PRMstar", "split": "heldout", "records": 4, "frame_every": 0, "screencast_only": True},
    ]
    for spec in specs:
        rows.extend(_episode_records(**spec))
    return rows


def _episode_records(
    *,
    episode: int,
    planner: str,
    split: str,
    records: int,
    frame_every: int,
    video: bool = False,
    screencast_only: bool = False,
) -> list[dict[str, Any]]:
    out = []
    base_t = episode * 10.0
    base_pose = [0.55 + 0.01 * episode, -0.12 + 0.004 * episode, 0.17, 0.0, 0.0, 0.0, 1.0]
    for i in range(records):
        t = round(base_t + 0.1 * i, 3)
        action = [_jitter(0.008 * math.sin(i + j)) for j in range(6)]
        joint_state = [_jitter(0.18 * episode + 0.004 * i + 0.02 * j) for j in range(6)]
        record: dict[str, Any] = {
            "episode_index": episode,
            "timestamp": t,
            "session_id": "golden_data_readiness_stress",
            "scenario": planner,
            "planner": planner,
            "split": split,
            "action": action,
            "joint_command": action,
            "joint_state": joint_state,
            "ee_pose": [_jitter(0.4 + 0.002 * i), _jitter(-0.1), _jitter(0.2), 0.0, 0.0, 0.0, 1.0],
            "object_pose_estimate": base_pose,
            "object_pose_reference": base_pose,
            "object_pose_estimate_source": "fk_proxy",
            "object_pose_reference_source": "fk_proxy",
            "pose_validation_source": "fk_proxy_only",
            "contact_force": _jitter(12.0 + 2.0 * math.sin(i)) if i % 3 != 0 else None,
            "success": True,
            "success_label_source": "tracking_error_threshold",
            "tracking_error_rad": round(0.0004 + 0.00005 * random.random(), 6),
            "action_scale_used": 0.0325,
            "actuator_deadband_proxy": round(0.006 + 0.001 * random.random(), 6),
            "friction_proxy": round(0.75 + random.uniform(-0.18, 0.18), 4),
            "observed_latency_ms": round(40 + random.uniform(-15, 20), 3),
            "video_file": f"raw_sources/camera_record_{planner}.avi" if video else "",
            "screencast_only": screencast_only,
            "notes": "Golden stress fixture: intentionally low frequency and weak perception provenance.",
        }
        if i % 5 != 0:
            record["joint_velocity"] = [_jitter(0.04 * math.cos(i + j)) for j in range(6)]
        if frame_every and i % frame_every == 0:
            color = f"camera_data/color/{planner}/frame_{i:06d}.png"
            depth = f"camera_data/depth/{planner}/frame_{i:06d}.png"
            (DATASET / color).write_bytes(PNG_1X1)
            (DATASET / depth).write_bytes(PNG_1X1)
            record["camera"] = {
                "camera_name": "golden_fixture_camera",
                "color_image": color,
                "depth_image": depth,
                "timestamp": t + 0.56,
                "timestamp_source": "retimed_video_offset_needed",
            }
        out.append(record)
    return out


def _jitter(value: float) -> float:
    return round(value + random.uniform(-0.0005, 0.0005), 6)


def _write_records(records: list[dict[str, Any]]) -> None:
    path = DATASET / "aligned" / "records.jsonl"
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")


def _write_camera_index(records: list[dict[str, Any]]) -> None:
    with (DATASET / "camera_data" / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "camera_name", "color_image", "depth_image"])
        writer.writeheader()
        for record in records:
            camera = record.get("camera")
            if isinstance(camera, dict):
                writer.writerow(
                    {
                        "timestamp": camera["timestamp"],
                        "camera_name": camera["camera_name"],
                        "color_image": camera["color_image"],
                        "depth_image": camera["depth_image"],
                    }
                )


def _write_orphan_sbl_sources() -> None:
    for idx in range(6):
        (DATASET / "frames_full" / "SBL" / f"frame_{idx:06d}.png").write_bytes(PNG_1X1)
    with (DATASET / "raw_sources" / "SBL_rtde.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "q0", "q1", "q2", "q3", "q4", "q5", "note"])
        for idx in range(6):
            writer.writerow([round(idx * 0.1, 3), *[round(0.01 * idx + 0.02 * j, 5) for j in range(6)], "orphan_frame_source"])
    for planner in ("BKPIECE", "RRTstar", "TRRT"):
        (DATASET / "raw_sources" / f"camera_record_{planner}.avi").write_text(
            "placeholder video marker for golden data-readiness fixture\n"
        )
    (DATASET / "raw_sources" / "PRM_screencast.txt").write_text(
        "placeholder screencast marker; not a robot camera feed\n"
    )


def _write_calibration() -> None:
    payload = {
        "schema": "agentic_sim2real.golden_calibration.v1",
        "camera_name": "golden_fixture_camera",
        "robot_base_frame": "base",
        "camera_frame": "camera_color_optical_frame",
        "T_base_camera": [0.42, -0.31, 0.72, 0.0, 0.0, 0.7071068, 0.7071068],
        "quality": "synthetic_fixture",
    }
    (DATASET / "calibration" / "calibration.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_readme() -> None:
    (DATASET / "README.md").write_text(
        """# Golden Real Dataset: Data Readiness Stress

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
""",
        )


def _write_expected_alerts() -> None:
    payload = {
        "must_detect": [
            "heldout_frames_missing",
            "low_frame_link_coverage",
            "orphan_frames_detected",
            "auto_positive_success_labels",
            "delay_under_sampled",
            "fk_proxy_pose_validation",
            "episode_frame_gaps",
        ],
        "expected_policy": {
            "delay_excluded_from_transfer_score": True,
            "pose_score_cap": 0.5,
            "success_component_trusted": False,
        },
        "notes": "This file is a test oracle for generic data-readiness checks, not a real robot benchmark.",
    }
    (DATASET / "expected_alerts.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
