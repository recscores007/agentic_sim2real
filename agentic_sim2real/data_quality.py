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
    readiness = evaluate_data_readiness(records, config, dataset_path=source)

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
    warnings.extend(item["message"] for item in readiness["action_items"] if item["severity"] == "warning")
    blockers.extend(item["message"] for item in readiness["action_items"] if item["severity"] == "blocker")

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
        "data_readiness": readiness,
        "session_report": session_report,
    }


def evaluate_data_readiness(
    records: list[Record],
    config: PipelineConfig,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize whether submitted real data can support sim2real claims.

    These checks are intentionally generic. They inspect canonical fields and
    provenance hints, so the same gate works for manipulators, humanoids, and
    mobile manipulators.
    """

    source = Path(dataset_path).expanduser() if dataset_path else None
    total = len(records)
    frame_link_records = [record for record in records if _record_frame_paths(record)]
    heldout_records = [record for record in records if _is_heldout(record)]
    heldout_frame_records = [record for record in heldout_records if _record_frame_paths(record)]
    train_records = [record for record in records if not _is_heldout(record)]
    train_frame_records = [record for record in train_records if _record_frame_paths(record)]
    referenced_frames = _referenced_frame_paths(records, source)
    discovered_frames = _discover_frame_files(source)
    orphan_frames = sorted(str(path) for path in discovered_frames - referenced_frames)
    existing_frame_record_count = (
        sum(1 for record in frame_link_records if _record_has_existing_frame(record, source))
        if source and source.is_dir()
        else None
    )
    label_report = _success_label_report(records)
    pose_report = _pose_validation_report(records)
    rate_hz = _estimated_rate_hz(records)
    min_delay_sample_hz = float(config.agent.get("min_delay_sample_hz", 50.0))
    delay_observability_status = (
        "unknown"
        if rate_hz is None
        else "adequate"
        if rate_hz >= min_delay_sample_hz
        else "under_sampled"
    )
    by_episode = _episode_readiness(records, source)
    action_items = _readiness_action_items(
        total=total,
        frame_link_count=len(frame_link_records),
        heldout_count=len(heldout_records),
        heldout_frame_count=len(heldout_frame_records),
        train_count=len(train_records),
        train_frame_count=len(train_frame_records),
        orphan_count=len(orphan_frames),
        label_report=label_report,
        pose_report=pose_report,
        rate_hz=rate_hz,
        min_delay_sample_hz=min_delay_sample_hz,
        by_episode=by_episode,
    )
    status = "needs_attention" if action_items else "ready"
    return {
        "status": status,
        "records": total,
        "frame_link_coverage": _ratio(len(frame_link_records), total),
        "train_frame_link_coverage": _ratio(len(train_frame_records), len(train_records)),
        "heldout_frame_link_coverage": _ratio(len(heldout_frame_records), len(heldout_records)),
        "existing_frame_coverage": None if existing_frame_record_count is None else _ratio(existing_frame_record_count, len(frame_link_records)),
        "referenced_frame_count": len(referenced_frames),
        "discovered_frame_count": len(discovered_frames),
        "orphan_frame_count": len(orphan_frames),
        "orphan_frame_examples": orphan_frames[:10],
        "estimated_rate_hz": rate_hz,
        "min_delay_sample_hz": round(min_delay_sample_hz, 3),
        "delay_observability_status": delay_observability_status,
        "success_labels": label_report,
        "pose_validation": pose_report,
        "episodes": by_episode,
        "action_items": action_items,
        "score_impacts": {
            "delay_excluded_from_transfer_score": delay_observability_status == "under_sampled",
            "success_component_trusted": label_report["trust_level"] == "human_or_mixed",
            "pose_score_cap": 0.5 if pose_report["validation_source"] != "vision_validated" else 1.0,
        },
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
        "frame_link_coverage": _ratio(sum(1 for record in records if _record_frame_paths(record)), total),
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


def _record_frame_paths(record: Record) -> list[str]:
    raw = record.raw or {}
    paths: list[str] = []
    camera = raw.get("camera")
    if isinstance(camera, dict):
        for key in ("color_image", "depth_image", "image", "frame_path", "rgb_path", "depth_path"):
            value = camera.get(key)
            if value not in (None, ""):
                paths.append(str(value))
    for key in (
        "frame_path",
        "image_path",
        "rgb_path",
        "color_image",
        "depth_image",
        "camera_frame",
        "camera_frame_path",
        "visual_frame",
    ):
        value = raw.get(key)
        if value not in (None, ""):
            paths.append(str(value))
    return sorted(set(paths))


def _referenced_frame_paths(records: list[Record], source: Path | None) -> set[Path]:
    if source is None:
        return set()
    root = _dataset_root(source)
    paths = set()
    for record in records:
        for value in _record_frame_paths(record):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = root / path
            paths.add(path.resolve())
    return paths


def _record_has_existing_frame(record: Record, source: Path) -> bool:
    root = _dataset_root(source)
    for value in _record_frame_paths(record):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return True
    return False


def _discover_frame_files(source: Path | None) -> set[Path]:
    if source is None or not source.is_dir():
        return set()
    root = _dataset_root(source)
    frame_roots = [root / "frames_full", root / "camera_data" / "color", root / "camera_data" / "depth"]
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    paths: set[Path] = set()
    for frame_root in frame_roots:
        if not frame_root.exists():
            continue
        for path in frame_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                paths.add(path.resolve())
    return paths


def _dataset_root(source: Path) -> Path:
    if source.is_dir():
        return source
    if source.parent.name == "aligned":
        return source.parent.parent
    return source.parent


def _success_label_report(records: list[Record]) -> dict[str, Any]:
    labeled = [record for record in records if record.success is not None]
    source_counts: dict[str, int] = {}
    for record in labeled:
        source = _raw_first(record, "success_label_source", "label_source", "outcome_source", default="unknown")
        source_counts[str(source)] = source_counts.get(str(source), 0) + 1
    positive = sum(1 for record in labeled if record.success is True)
    auto_sources = {
        "auto",
        "automatic",
        "tracking_error_threshold",
        "threshold",
        "heuristic",
        "generated",
        "script",
        "unknown",
    }
    human_sources = {"human", "manual", "operator", "reviewed", "human_reviewed"}
    normalized_sources = {source.strip().lower() for source in source_counts}
    has_human = bool(normalized_sources & human_sources)
    all_auto = bool(source_counts) and all(source in auto_sources for source in normalized_sources)
    all_positive = bool(labeled) and positive == len(labeled)
    return {
        "labeled_records": len(labeled),
        "positive_records": positive,
        "coverage": _ratio(len(labeled), len(records)),
        "source_counts": dict(sorted(source_counts.items())),
        "all_positive": all_positive,
        "all_auto_generated": all_auto,
        "trust_level": "human_or_mixed" if has_human else "auto_only" if all_auto else "unknown",
    }


def _pose_validation_report(records: list[Record]) -> dict[str, Any]:
    samples = 0
    identical = 0
    source_counts: dict[str, int] = {}
    frame_linked = 0
    vision_validated = 0
    for record in records:
        if _record_frame_paths(record):
            frame_linked += 1
        est = record.object_pose_estimate
        ref = record.object_pose_reference
        if len(est) >= 3 and len(ref) >= 3:
            samples += 1
            if _vectors_close(est, ref):
                identical += 1
        source = _raw_first(
            record,
            "pose_validation_source",
            "object_pose_estimate_source",
            "shaft_pose_estimate_source",
            "pose_source",
            default="unknown",
        )
        source_text = str(source).strip().lower()
        source_counts[source_text] = source_counts.get(source_text, 0) + 1
        if any(token in source_text for token in ("vision", "camera", "foundationpose", "aruco", "apriltag", "vslam")):
            vision_validated += 1

    identical_ratio = _ratio(identical, samples)
    normalized_sources = set(source_counts)
    if samples == 0:
        validation_source = "missing"
    elif vision_validated > 0 and identical_ratio < 0.95:
        validation_source = "vision_validated"
    elif identical_ratio >= 0.95 or any("fk" in source or "forward_kinematics" in source for source in normalized_sources):
        validation_source = "fk_proxy_only"
    elif frame_linked > 0:
        validation_source = "frame_linked_unverified"
    else:
        validation_source = "unproven"
    return {
        "samples": samples,
        "validation_source": validation_source,
        "identical_estimate_reference_ratio": identical_ratio,
        "frame_linked_records": frame_linked,
        "source_counts": dict(sorted(source_counts.items())),
    }


def _vectors_close(a: list[float], b: list[float], eps: float = 1e-9) -> bool:
    n = min(len(a), len(b))
    return n > 0 and all(abs(float(a[i]) - float(b[i])) <= eps for i in range(n))


def _estimated_rate_hz(records: list[Record]) -> float | None:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_episode[record.episode_index].append(record.timestamp)
    dts: list[float] = []
    for values in by_episode.values():
        ordered = sorted(values)
        dts.extend(b - a for a, b in zip(ordered, ordered[1:]) if b > a)
    if not dts:
        return None
    dts = sorted(dts)
    mid = len(dts) // 2
    median_dt = dts[mid] if len(dts) % 2 else (dts[mid - 1] + dts[mid]) / 2.0
    return round(1.0 / median_dt, 3) if median_dt > 0 else None


def _episode_readiness(records: list[Record], source: Path | None) -> list[dict[str, Any]]:
    by_episode: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        by_episode[record.episode_index].append(record)
    rows = []
    for episode, episode_records in sorted(by_episode.items()):
        frame_links = sum(1 for record in episode_records if _record_frame_paths(record))
        split = _split_name(episode_records[0])
        scenario = _raw_first(episode_records[0], "scenario", "algorithm", "planner", "algo", default=f"episode_{episode}")
        row = {
            "episode_index": episode,
            "scenario": str(scenario),
            "split": split,
            "records": len(episode_records),
            "frame_link_coverage": _ratio(frame_links, len(episode_records)),
            "success_label_sources": _success_label_report(episode_records)["source_counts"],
            "pose_validation_source": _pose_validation_report(episode_records)["validation_source"],
        }
        if source and source.is_dir():
            existing = sum(1 for record in episode_records if _record_has_existing_frame(record, source))
            row["existing_frame_coverage"] = _ratio(existing, frame_links)
        rows.append(row)
    return rows


def _readiness_action_items(
    *,
    total: int,
    frame_link_count: int,
    heldout_count: int,
    heldout_frame_count: int,
    train_count: int,
    train_frame_count: int,
    orphan_count: int,
    label_report: dict[str, Any],
    pose_report: dict[str, Any],
    rate_hz: float | None,
    min_delay_sample_hz: float,
    by_episode: list[dict[str, Any]],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    frame_coverage = _ratio(frame_link_count, total)
    train_frame_coverage = _ratio(train_frame_count, train_count)
    heldout_frame_coverage = _ratio(heldout_frame_count, heldout_count)
    if heldout_count > 0 and heldout_frame_count == 0:
        items.append(
            _action_item(
                "warning",
                "user",
                "heldout_frames_missing",
                "held-out records have no linked camera frames",
                "Pose/perception validation cannot be trusted on held-out data.",
                "Extract or link camera frames for held-out episodes that have video; mark screencast-only episodes as a limitation.",
            )
        )
    if total > 0 and frame_coverage < 0.2:
        items.append(
            _action_item(
                "warning",
                "user",
                "low_frame_link_coverage",
                f"camera frame link coverage is {frame_coverage:.3f}",
                "Pose-based conclusions will be sparse and the critic should down-rank perception claims.",
                "Link frames to records or explicitly run a no-vision characterization profile.",
            )
        )
    elif train_count > 0 and train_frame_coverage < 0.5:
        items.append(
            _action_item(
                "warning",
                "user",
                "sparse_training_frames",
                f"training frame link coverage is {train_frame_coverage:.3f}",
                "The agent can tune trajectory/contact parameters, but camera-pose noise evidence is weak.",
                "Improve frame extraction/linking before drawing strong perception-noise conclusions.",
            )
        )
    if orphan_count > 0:
        items.append(
            _action_item(
                "warning",
                "pipeline",
                "orphan_frames_detected",
                f"{orphan_count} image files are present but not referenced by records",
                "Useful video evidence may be sitting outside the canonical pipeline input.",
                "Run ingestion/linking to convert orphan frames and matching telemetry into aligned records.",
            )
        )
    if label_report["all_positive"] and label_report["all_auto_generated"]:
        items.append(
            _action_item(
                "warning",
                "user",
                "auto_positive_success_labels",
                "success labels are auto-generated and all positive",
                "A perfect success component would be a labeling artifact, not real policy performance.",
                "Keep success_label_source per record, add human-reviewed labels, and include failure cases before policy-release claims.",
            )
        )
    if rate_hz is not None and rate_hz < min_delay_sample_hz:
        items.append(
            _action_item(
                "warning",
                "pipeline",
                "delay_under_sampled",
                f"trajectory rate is {rate_hz:.3f} Hz; delay estimation requires at least {min_delay_sample_hz:.3f} Hz",
                "Sub-sample actuation delay cannot be estimated reliably from this log.",
                "Exclude delay confidence from transfer scoring, or collect higher-rate telemetry for latency SysID.",
            )
        )
    if pose_report["validation_source"] == "fk_proxy_only":
        items.append(
            _action_item(
                "warning",
                "pipeline",
                "fk_proxy_pose_validation",
                "pose estimate/reference appear to be the same FK proxy",
                "A pose score of 1.0 would be circular unless real vision validation is present.",
                "Cap the pose component, flag pose_validation_source=fk_proxy_only, and add real visual pose estimates when needed.",
            )
        )
    elif pose_report["validation_source"] in {"missing", "unproven", "frame_linked_unverified"}:
        items.append(
            _action_item(
                "warning",
                "user",
                "pose_validation_unproven",
                f"pose validation source is {pose_report['validation_source']}",
                "The pipeline can tune non-vision parameters, but pose-noise confidence is limited.",
                "Add vision-derived object pose estimates/references or mark this run as trajectory-only characterization.",
            )
        )
    zero_frame_episodes = [
        row for row in by_episode if row["records"] > 0 and row["frame_link_coverage"] == 0.0
    ]
    if zero_frame_episodes:
        examples = ", ".join(str(row["scenario"]) for row in zero_frame_episodes[:5])
        items.append(
            _action_item(
                "warning",
                "user",
                "episode_frame_gaps",
                f"{len(zero_frame_episodes)} episode(s) have zero frame links",
                "Some planners/scenarios cannot support visual holdout checks.",
                f"Prioritize frame extraction/linking for: {examples}.",
            )
        )
    return items


def _action_item(
    severity: str,
    owner: str,
    code: str,
    message: str,
    impact: str,
    recommended_fix: str,
) -> dict[str, str]:
    return {
        "severity": severity,
        "owner": owner,
        "code": code,
        "message": message,
        "impact": impact,
        "recommended_fix": recommended_fix,
    }


def _raw_first(record: Record, *keys: str, default: Any = None) -> Any:
    raw = record.raw or {}
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return default
