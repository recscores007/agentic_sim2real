from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .adapters import load_embodiment_adapter
from .config import PipelineConfig
from .dataset import Record, load_records
from .real_data import inspect_real_session


def evaluate_real_data_quality(
    dataset_path: str | Path,
    config: PipelineConfig,
    root: str | Path = ".",
    embodiment_id: str | None = None,
) -> dict[str, Any]:
    source = Path(dataset_path).expanduser()
    records = load_records(source)
    blockers: list[str] = []
    warnings: list[str] = []
    metrics = _record_metrics(records)

    min_episodes = int(config.agent.get("min_real_episodes_for_gate", 3))
    if metrics["records"] == 0:
        blockers.append("no aligned records found")
    if metrics["episodes"] < min_episodes:
        blockers.append(f"only {metrics['episodes']} episodes found; gate requires at least {min_episodes}")
    if len(metrics["action_dims"]) != 1:
        blockers.append(f"action dimensions are inconsistent: {metrics['action_dims']}")
    if len(metrics["joint_dims"]) != 1:
        blockers.append(f"joint dimensions are inconsistent: {metrics['joint_dims']}")
    if metrics["missing_object_pose_estimate"] > 0:
        blockers.append(f"{metrics['missing_object_pose_estimate']} records are missing object_pose_estimate")
    if metrics["missing_object_pose_reference"] > 0:
        blockers.append(f"{metrics['missing_object_pose_reference']} records are missing object_pose_reference")
    if metrics["timestamp_regressions"] > 0:
        blockers.append(f"{metrics['timestamp_regressions']} per-episode timestamp regressions found")

    if metrics["contact_coverage"] < 0.5:
        warnings.append("contact coverage is below 50 percent; SysID contact/friction confidence will be limited")
    if metrics["success_label_coverage"] < 0.5:
        warnings.append("episode success labels are sparse; regression confidence will be limited")
    if metrics["joint_velocity_coverage"] < 0.5:
        warnings.append("joint velocity coverage is sparse; delay/stiction estimates may be less confident")
    if metrics["heldout_episodes"] == 0:
        warnings.append("no held-out episodes are marked; release-candidate promotion should use a held-out session")

    session_report: dict[str, Any] | None = None
    if source.is_dir():
        session_report = inspect_real_session(source, root=root, embodiment_id=embodiment_id)
        if not session_report["quality_inputs"]["calibration_present"]:
            blockers.append("calibration/calibration.json is required for embodiment-scoped real-data sessions")
        if not session_report["quality_inputs"]["camera_index_present"]:
            warnings.append("camera_data/index.csv is missing; perception provenance will be limited")
    else:
        adapter = load_embodiment_adapter(root, embodiment_id=embodiment_id)
        session_report = {
            "status": "file_dataset",
            "dataset": str(source),
            "adapter": adapter.to_dict(),
            "quality_inputs": {
                "calibration_present": None,
                "camera_index_present": None,
            },
        }

    score = _quality_score(blockers, warnings, metrics)
    return {
        "status": "fail" if blockers else "pass",
        "quality_score": score,
        "confidence": min(1.0, metrics["episodes"] / max(float(min_episodes), 1.0)),
        "blocking_failures": blockers,
        "warnings": warnings,
        "metrics": metrics,
        "session_report": session_report,
    }


def _record_metrics(records: list[Record]) -> dict[str, Any]:
    by_episode: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        by_episode[record.episode_index].append(record)

    timestamp_regressions = 0
    for rows in by_episode.values():
        ordered = sorted(rows, key=lambda item: item.timestamp)
        if [row.timestamp for row in rows] != [row.timestamp for row in ordered]:
            timestamp_regressions += 1

    total = len(records)
    contact_count = sum(1 for record in records if record.contact_force is not None)
    success_count = sum(1 for record in records if record.success is not None)
    joint_velocity_count = sum(1 for record in records if record.joint_velocity)
    return {
        "records": total,
        "episodes": len(by_episode),
        "episode_ids": sorted(by_episode),
        "split_counts": _split_counts(records),
        "session_ids": _session_ids(records),
        "heldout_episodes": _heldout_episodes(records),
        "heldout_session_count": _heldout_session_count(records),
        "action_dims": sorted({len(record.action) for record in records}),
        "joint_dims": sorted({len(record.joint_state) for record in records}),
        "joint_velocity_coverage": _ratio(joint_velocity_count, total),
        "contact_coverage": _ratio(contact_count, total),
        "success_label_coverage": _ratio(success_count, total),
        "missing_object_pose_estimate": sum(1 for record in records if not record.object_pose_estimate),
        "missing_object_pose_reference": sum(1 for record in records if not record.object_pose_reference),
        "timestamp_regressions": timestamp_regressions,
    }


def _quality_score(blockers: list[str], warnings: list[str], metrics: dict[str, Any]) -> float:
    score = 1.0
    score -= min(0.75, 0.25 * len(blockers))
    score -= min(0.2, 0.05 * len(warnings))
    if metrics["contact_coverage"] < 0.5:
        score -= 0.05
    if metrics["success_label_coverage"] < 0.5:
        score -= 0.05
    return round(max(0.0, score), 3)


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 3)


def _split_counts(records: list[Record]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        split = _split_name(record)
        counts[split] = counts.get(split, 0) + 1
    return counts


def _session_ids(records: list[Record]) -> list[str]:
    sessions = {_session_id(record) for record in records}
    return sorted(session for session in sessions if session)


def _heldout_episodes(records: list[Record]) -> int:
    return len({record.episode_index for record in records if _is_heldout(record)})


def _heldout_session_count(records: list[Record]) -> int:
    return len({_session_id(record) for record in records if _is_heldout(record) and _session_id(record)})


def _split_name(record: Record) -> str:
    raw = record.raw or {}
    value = raw.get("split") or raw.get("dataset_split") or raw.get("eval_split") or raw.get("partition")
    return str(value).strip().lower() if value not in (None, "") else "unmarked"


def _session_id(record: Record) -> str:
    raw = record.raw or {}
    value = raw.get("session_id") or raw.get("session") or raw.get("session_name") or raw.get("run_id")
    return str(value).strip() if value not in (None, "") else ""


def _is_heldout(record: Record) -> bool:
    split = _split_name(record)
    raw = record.raw or {}
    heldout_flag = raw.get("heldout") or raw.get("is_holdout") or raw.get("holdout")
    if isinstance(heldout_flag, bool):
        return heldout_flag
    if heldout_flag not in (None, ""):
        return str(heldout_flag).strip().lower() in {"1", "true", "yes", "heldout", "holdout"}
    return split in {"heldout", "holdout", "test", "validation", "val"}
