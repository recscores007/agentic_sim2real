from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Record:
    episode_index: int
    timestamp: float
    action: list[float]
    joint_state: list[float]
    joint_velocity: list[float]
    ee_pose: list[float]
    shaft_pose_estimate: list[float]
    shaft_pose_reference: list[float]
    contact_force: float | None = None
    success: bool | None = None
    failure_mode: str | None = None
    raw: dict[str, Any] | None = None


def load_records(path: str | Path) -> list[Record]:
    source = Path(path).expanduser()
    if source.is_dir():
        preferred = [
            source / "aligned" / "records.jsonl",
            source / "records.jsonl",
        ]
        existing = [candidate for candidate in preferred if candidate.exists()]
        if existing:
            rows = _load_jsonl(existing[0])
        else:
            rows: list[dict[str, Any]] = []
            for child in sorted(source.rglob("*")):
                if child.suffix == ".jsonl":
                    rows.extend(_load_jsonl(child))
                elif child.suffix == ".json":
                    rows.extend(_load_json(child))
                elif child.suffix == ".csv":
                    rows.extend(_load_csv(child))
    elif source.suffix == ".jsonl":
        rows = _load_jsonl(source)
    elif source.suffix == ".json":
        rows = _load_json(source)
    elif source.suffix == ".csv":
        rows = _load_csv(source)
    else:
        raise ValueError(f"Unsupported dataset path: {source}")

    records = [_normalize(row) for row in rows]
    if not records:
        raise ValueError(f"No records found in {source}")
    return sorted(records, key=lambda r: (r.episode_index, r.timestamp))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
    return rows


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return list(payload["records"])
    raise ValueError(f"{path}: expected list or object with records")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _normalize(row: dict[str, Any]) -> Record:
    episode = _first(row, "episode_index", "episode", "episode_id", default=0)
    timestamp = _first(row, "timestamp", "time", "t", default=0.0)
    action = _vector(_first(row, "action", "joint_action", "policy_action", default=[]))
    joint_state = _vector(_first(row, "joint_state", "joint_pos", "joint_positions", "state", default=[]))
    joint_velocity = _vector(_first(row, "joint_velocity", "joint_vel", "joint_velocities", default=[]))
    ee_pose = _vector(_first(row, "ee_pose", "tcp_pose", "wrist_pose", default=[]))
    shaft_est = _vector(
        _first(
            row,
            "shaft_pose_estimate",
            "gear_shaft_pose_estimate",
            "gear_shaft_pose",
            "shaft_pose",
            "object_pose_estimate",
            "object_pose",
            "target_pose_estimate",
            default=[],
        )
    )
    shaft_ref = _vector(
        _first(
            row,
            "shaft_pose_reference",
            "shaft_pose_ground_truth",
            "shaft_pose_measured",
            "gear_shaft_pose_reference",
            "object_pose_reference",
            "object_pose_ground_truth",
            "target_pose_reference",
            default=[],
        )
    )
    contact_force = _optional_float(_first(row, "contact_force", "contact_force_n", "force_n", default=None))
    success = _optional_bool(_first(row, "success", "episode_success", default=None))
    failure_mode = _first(row, "failure_mode", "failure", default=None)

    if not action:
        raise ValueError("Every record needs an action vector")
    if not joint_state:
        raise ValueError("Every record needs joint_state or joint_pos")

    return Record(
        episode_index=int(episode),
        timestamp=float(timestamp),
        action=action,
        joint_state=joint_state,
        joint_velocity=joint_velocity,
        ee_pose=ee_pose,
        shaft_pose_estimate=shaft_est,
        shaft_pose_reference=shaft_ref,
        contact_force=contact_force,
        success=success,
        failure_mode=str(failure_mode) if failure_mode not in (None, "") else None,
        raw=dict(row),
    )


def _first(row: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _vector(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            value = [part for part in stripped.replace(";", ",").split(",") if part.strip()]
    if not isinstance(value, Iterable):
        raise ValueError(f"Expected vector, got {value!r}")
    return [float(v) for v in value]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "success", "succeeded"}:
        return True
    if text in {"false", "no", "0", "fail", "failed"}:
        return False
    return None
