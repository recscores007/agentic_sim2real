from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


POSE_FILENAMES = ("object_pose.csv", "shaft_pose.csv")


def prepare_real_session(
    session_dir: str | Path,
    out_path: str | Path | None = None,
    tolerance_s: float = 0.05,
) -> dict[str, Any]:
    session = Path(session_dir).expanduser()
    if out_path is None:
        out = session / "aligned" / "records.jsonl"
    else:
        out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    joints = _read_csv_required(session / "joint_data" / "joint_states.csv")
    poses, pose_path = _read_csv_required_any(session / "pose_data", POSE_FILENAMES)
    contacts = _read_csv_optional(session / "contact_data" / "contact.csv")
    cameras = _read_csv_optional(session / "camera_data" / "index.csv")
    labels = _labels_by_episode(_read_csv_optional(session / "episode_labels" / "labels.csv"))

    warnings: list[str] = []
    records = []
    missing_pose_matches = 0
    missing_contact_matches = 0
    missing_camera_matches = 0

    for joint_row in joints:
        episode = int(_required(joint_row, "episode_index"))
        timestamp = float(_required(joint_row, "timestamp"))
        pose_row = _nearest_row(poses, episode, timestamp, tolerance_s)
        contact_row = _nearest_row(contacts, episode, timestamp, tolerance_s)
        camera_row = _nearest_row(cameras, episode, timestamp, tolerance_s)
        label = labels.get(episode, {})

        if pose_row is None:
            missing_pose_matches += 1
            warnings.append(f"episode {episode} t={timestamp}: no object pose match within {tolerance_s}s")
            object_estimate: list[float] = []
            object_reference: list[float] = []
        else:
            object_estimate = _pose_from_prefix(pose_row, "estimate")
            object_reference = _pose_from_prefix(pose_row, "reference", optional=True)

        if contact_row is None:
            missing_contact_matches += 1
            contact_force = None
        else:
            contact_force = _optional_float(contact_row.get("contact_force_n"))

        camera_fields: dict[str, Any] = {}
        if camera_row is None:
            missing_camera_matches += 1
        else:
            camera_fields = {
                "camera_name": camera_row.get("camera_name", ""),
                "color_image": camera_row.get("color_image", ""),
                "depth_image": camera_row.get("depth_image", ""),
            }

        record = {
            "episode_index": episode,
            "timestamp": timestamp,
            "action": _vector_from_prefix(joint_row, "action"),
            "joint_state": _vector_from_prefix(joint_row, "joint"),
            "joint_velocity": _vector_from_prefix(joint_row, "joint_vel", optional=True),
            "ee_pose": _pose_from_prefix(joint_row, "ee", optional=True),
            "object_pose_estimate": object_estimate,
            "object_pose_reference": object_reference,
            "contact_force": contact_force,
            "success": label.get("success"),
            "failure_mode": _clean_failure_mode(label.get("failure_mode")),
            "notes": label.get("notes"),
            "camera": camera_fields,
        }
        records.append(record)

    with out.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "session_dir": _display_path(session),
        "records": len(records),
        "episodes": sorted({record["episode_index"] for record in records}),
        "out": _display_path(out),
        "tolerance_s": tolerance_s,
        "missing_pose_matches": missing_pose_matches,
        "missing_contact_matches": missing_contact_matches,
        "missing_camera_matches": missing_camera_matches,
        "warnings": warnings,
        "required_pipeline_file": _display_path(out),
        "pose_file": _display_path(pose_path),
        "record_pose_contract": "object_pose_estimate/object_pose_reference",
    }
    summary_path = out.parent / "prepare_summary.json"
    summary["summary_path"] = _display_path(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _read_csv_required(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required real-data file missing: {path}")
    return _read_csv(path)


def _read_csv_required_any(parent: Path, filenames: tuple[str, ...]) -> tuple[list[dict[str, str]], Path]:
    for filename in filenames:
        path = parent / filename
        if path.exists():
            return _read_csv(path), path
    options = ", ".join(str(parent / filename) for filename in filenames)
    raise FileNotFoundError(f"Required real-data file missing. Expected one of: {options}")


def _read_csv_optional(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_csv(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _labels_by_episode(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    labels = {}
    for row in rows:
        if row.get("episode_index") in (None, ""):
            continue
        labels[int(row["episode_index"])] = row
    return labels


def _nearest_row(rows: list[dict[str, str]], episode: int, timestamp: float, tolerance_s: float) -> dict[str, str] | None:
    best_row = None
    best_dt = None
    for row in rows:
        if row.get("episode_index") in (None, "") or row.get("timestamp") in (None, ""):
            continue
        if int(row["episode_index"]) != episode:
            continue
        dt = abs(float(row["timestamp"]) - timestamp)
        if dt <= tolerance_s and (best_dt is None or dt < best_dt):
            best_row = row
            best_dt = dt
    return best_row


def _vector_from_prefix(row: dict[str, str], prefix: str, optional: bool = False) -> list[float]:
    values = []
    indices = _prefix_indices(row, prefix)
    if not indices:
        if optional:
            return []
        raise ValueError(f"Missing required columns {prefix}_0 ... {prefix}_N")
    if indices != list(range(indices[-1] + 1)):
        raise ValueError(f"Columns for {prefix} must be contiguous from {prefix}_0")
    for idx in indices:
        value = row.get(f"{prefix}_{idx}")
        if value in (None, ""):
            if optional:
                return []
            raise ValueError(f"Missing required column {prefix}_{idx}")
        values.append(float(value))
    return values


def _prefix_indices(row: dict[str, str], prefix: str) -> list[int]:
    indices = []
    marker = f"{prefix}_"
    for key in row:
        if not key.startswith(marker):
            continue
        suffix = key[len(marker) :]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _pose_from_prefix(row: dict[str, str], prefix: str, optional: bool = False) -> list[float]:
    keys = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z", f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw"]
    values = []
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            if optional:
                return []
            raise ValueError(f"Missing required column {key}")
        values.append(float(value))
    return values


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required column {key}")
    return value


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _clean_failure_mode(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.lower() in {"none", "null", "success", "succeeded"}:
        return None
    return text


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)
