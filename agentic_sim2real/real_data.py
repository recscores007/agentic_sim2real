from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .adapters import load_embodiment_adapter


def inspect_real_session(
    session_dir: str | Path,
    root: str | Path = ".",
    embodiment_id: str | None = None,
) -> dict[str, Any]:
    session = _resolve_path(session_dir, root)
    adapter = load_embodiment_adapter(root, embodiment_id=embodiment_id, session_dir=session)
    pose_files = [session / "pose_data" / filename for filename in adapter.accepted_pose_files]
    selected_pose = next((path for path in pose_files if path.exists()), None)
    aligned_records = session / "aligned" / "records.jsonl"
    files = {
        "aligned_records": aligned_records.exists(),
        "joint_states": (session / "joint_data" / "joint_states.csv").exists(),
        "pose": selected_pose is not None,
        "contact": (session / "contact_data" / "contact.csv").exists(),
        "camera_index": (session / "camera_data" / "index.csv").exists(),
        "calibration": (session / "calibration" / "calibration.json").exists(),
    }
    raw_sources = {
        "csv_session": bool(files["joint_states"] and files["pose"]),
        "aligned_records": bool(files["aligned_records"]),
        "rosbag2": _has_rosbag2_source(session),
        "image_sequence": _has_image_sequence(session),
    }
    blockers = []
    if not raw_sources["aligned_records"] and not raw_sources["csv_session"]:
        blockers.append("session needs aligned/records.jsonl or CSV joint_data plus pose_data")
    if (raw_sources["rosbag2"] or raw_sources["image_sequence"]) and not raw_sources["csv_session"]:
        if adapter.external_ingestor_command:
            blockers.append("external ingestor command is configured but has not produced aligned records yet")
        else:
            blockers.append("raw rosbag/image sources need an embodiment external_ingestor_command")

    if raw_sources["aligned_records"]:
        status = "aligned_records_ready"
    elif raw_sources["csv_session"]:
        status = "csv_session_ready"
    elif raw_sources["rosbag2"] or raw_sources["image_sequence"]:
        status = "needs_external_ingestor"
    else:
        status = "incomplete"

    return {
        "status": status,
        "session_dir": _display_path(session),
        "adapter": adapter.to_dict(),
        "raw_sources": raw_sources,
        "files": {key: bool(value) for key, value in files.items()},
        "selected_pose_file": _display_path(selected_pose) if selected_pose else None,
        "accepted_pose_files": adapter.accepted_pose_files,
        "quality_inputs": {
            "calibration_present": bool(files["calibration"]),
            "camera_index_present": bool(files["camera_index"]),
            "contact_present": bool(files["contact"]),
        },
        "blockers": blockers,
    }


def ensure_aligned_dataset(
    dataset_path: str | Path,
    root: str | Path = ".",
    embodiment_id: str | None = None,
    tolerance_s: float = 0.05,
) -> Path:
    source = _resolve_path(dataset_path, root)
    if not source.is_dir():
        return source
    report = inspect_real_session(source, root=root, embodiment_id=embodiment_id)
    if report["status"] == "aligned_records_ready":
        return source
    if report["status"] == "csv_session_ready":
        prepare_real_session(source, root=root, embodiment_id=embodiment_id, tolerance_s=tolerance_s)
        return source
    blockers = "; ".join(report.get("blockers", [])) or "no supported real-data source found"
    raise ValueError(f"Cannot align dataset session {source}: {blockers}")


def prepare_real_session(
    session_dir: str | Path,
    out_path: str | Path | None = None,
    tolerance_s: float = 0.05,
    root: str | Path = ".",
    embodiment_id: str | None = None,
) -> dict[str, Any]:
    session = _resolve_path(session_dir, root)
    adapter = load_embodiment_adapter(root, embodiment_id=embodiment_id, session_dir=session)
    if out_path is None:
        out = session / "aligned" / "records.jsonl"
    else:
        out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    inspection = inspect_real_session(session, root=root, embodiment_id=embodiment_id)
    if inspection["status"] == "aligned_records_ready" and not inspection["raw_sources"]["csv_session"]:
        if out != session / "aligned" / "records.jsonl":
            shutil.copyfile(session / "aligned" / "records.jsonl", out)
        records = sum(1 for line in out.read_text().splitlines() if line.strip())
        summary = {
            "session_dir": _display_path(session),
            "records": records,
            "episodes": [],
            "out": _display_path(out),
            "tolerance_s": tolerance_s,
            "warnings": ["used existing aligned/records.jsonl; raw CSV source was not present"],
            "required_pipeline_file": _display_path(out),
            "pose_file": None,
            "record_pose_contract": "object_pose_estimate/object_pose_reference",
            "adapter": adapter.to_dict(),
            "inspection": inspection,
        }
        summary_path = out.parent / "prepare_summary.json"
        summary["summary_path"] = _display_path(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary
    if inspection["status"] not in {"csv_session_ready", "aligned_records_ready"}:
        blockers = "; ".join(inspection.get("blockers", [])) or "missing required CSV inputs"
        raise ValueError(f"Cannot prepare real-data session: {blockers}")

    joints = _read_csv_required(session / "joint_data" / "joint_states.csv")
    poses, pose_path = _read_csv_required_any(session / "pose_data", tuple(adapter.accepted_pose_files))
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

        joint_command = _first_optional_vector_from_prefix(
            joint_row,
            "command",
            "joint_command",
            "commanded_joint",
            "commanded_joint_position",
            "target_joint",
            "target_joint_position",
            "control",
            "control_position",
            "motor_command",
        )
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
        if joint_command:
            record["joint_command"] = joint_command
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
        "adapter": adapter.to_dict(),
        "inspection": inspection,
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


def _first_optional_vector_from_prefix(row: dict[str, str], *prefixes: str) -> list[float]:
    for prefix in prefixes:
        values = _vector_from_prefix(row, prefix, optional=True)
        if values:
            return values
    return []


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


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _resolve_path(path: str | Path, root: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = Path(root).expanduser() / candidate
    if root_candidate.exists():
        return root_candidate
    return candidate


def _has_rosbag2_source(session: Path) -> bool:
    return any(
        path.name == "metadata.yaml" or path.suffix in {".db3", ".mcap"}
        for path in session.rglob("*")
        if path.is_file()
    )


def _has_image_sequence(session: Path) -> bool:
    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}
    camera_dir = session / "camera_data"
    if not camera_dir.exists():
        return False
    return any(
        path.suffix.lower() in image_suffixes
        for path in camera_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
