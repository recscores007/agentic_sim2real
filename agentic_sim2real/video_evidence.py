from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import PipelineConfig


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VIDEO_INDEX_CANDIDATES = (
    "video_data/index.csv",
    "camera_data/video_index.csv",
    "contact_data/video_index.csv",
)
VIDEO_ANALYSIS_CANDIDATES = (
    "video_data/analysis.json",
    "video_data/video_analysis.json",
    "camera_data/video_analysis.json",
    "contact_data/video_analysis.json",
)


def collect_video_evidence(
    dataset_path: str | Path,
    root: str | Path,
    config: PipelineConfig,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    session = resolve_video_session(dataset_path, root)
    summary: dict[str, Any] = {
        "status": "no_session",
        "session_dir": None,
        "video_count": 0,
        "indexed_video_count": 0,
        "analysis_files": [],
        "analysis": {},
        "warnings": [],
    }
    if session is None or not session.exists():
        summary["warnings"].append("no real-data session directory was available for uploaded video evidence")
        return summary

    entries = _read_video_index(session)
    discovered = _discover_videos(session)
    indexed_paths = {_resolve_video_path(session, entry.get("video_path", "")) for entry in entries if entry.get("video_path")}
    indexed_paths = {path for path in indexed_paths if path is not None}
    missing_index_paths = sorted(str(path.relative_to(session)) for path in indexed_paths if not path.exists())
    unindexed = sorted(path for path in discovered if path not in indexed_paths)
    analysis_files, analysis = _read_analysis_files(session)

    summary.update(
        {
            "status": "ready" if discovered or entries or analysis else "no_video_evidence",
            "session_dir": str(session),
            "video_count": len(discovered),
            "indexed_video_count": len(entries),
            "videos": [_video_file_payload(path, session) for path in discovered],
            "index_entries": entries,
            "missing_index_paths": missing_index_paths,
            "unindexed_videos": [str(path.relative_to(session)) for path in unindexed],
            "analysis_files": [str(path) for path in analysis_files],
            "analysis": analysis,
            "warnings": [],
        }
    )
    if missing_index_paths:
        summary["warnings"].append(f"{len(missing_index_paths)} indexed video path(s) do not exist")
    if unindexed:
        summary["warnings"].append(f"{len(unindexed)} uploaded video file(s) are present but not listed in a video index")

    command_payload = _run_video_analysis_command(summary, config, root, work_dir)
    if command_payload:
        command_analysis = dict(command_payload.get("analysis", command_payload))
        summary["analysis"] = _deep_merge_dicts(summary["analysis"], command_analysis)
        summary["analysis_command"] = command_payload.get("analysis_command", {})
    return summary


def camera_tuning_from_video(summary: dict[str, Any], config: PipelineConfig) -> dict[str, Any]:
    camera_videos = _videos_for(summary, {"camera", "calibration", "perception", "rgb", "depth"})
    analysis = _first_analysis(summary, "camera_calibration", "camera_tuning", "camera")
    gate = float(config.video_evidence.get("reprojection_error_gate_px", 1.5))
    warnings = list(summary.get("warnings", []))
    failures: list[str] = []

    if not camera_videos and not analysis:
        return {
            "status": "evidence_missing",
            "quality_score": 0.0,
            "confidence": 1.0,
            "blocking_failures": [],
            "warnings": ["no uploaded camera/calibration video evidence was found"],
            "metrics": {
                "camera_video_count": 0,
                "analysis_available": False,
                "suggested_camera_parameters": {},
            },
        }

    reprojection_error = _optional_float(analysis.get("reprojection_error_px"))
    if reprojection_error is None:
        warnings.append("camera video exists but no reprojection_error_px was reported by video analysis")
        status = "evidence_missing"
        quality = 0.45 if camera_videos else 0.0
    elif reprojection_error > gate:
        failures.append(f"camera reprojection_error_px {reprojection_error:.3f} exceeds gate {gate:.3f}")
        status = "fail"
        quality = 0.25
    else:
        status = "pass"
        quality = 0.9

    observed_targets = int(analysis.get("observed_target_count", analysis.get("target_observations", 0)) or 0)
    confidence = _bounded_float(analysis.get("confidence"), default=min(1.0, max(0.5, observed_targets / 20.0 if observed_targets else 0.65)))
    suggested = {
        "intrinsics": analysis.get("suggested_intrinsics", {}),
        "extrinsic_delta": analysis.get("suggested_extrinsic_delta", analysis.get("extrinsic_delta", {})),
        "latency_seconds": analysis.get("camera_latency_s", analysis.get("latency_seconds")),
        "reprojection_error_px": reprojection_error,
    }
    return {
        "status": status,
        "quality_score": quality,
        "confidence": confidence,
        "blocking_failures": failures,
        "warnings": warnings,
        "metrics": {
            "camera_video_count": len(camera_videos),
            "analysis_available": bool(analysis),
            "reprojection_error_px": reprojection_error,
            "reprojection_error_gate_px": gate,
            "observed_target_count": observed_targets,
            "suggested_camera_parameters": suggested,
            "videos": camera_videos,
        },
    }


def friction_tuning_from_video(summary: dict[str, Any], config: PipelineConfig) -> dict[str, Any]:
    friction_videos = _videos_for(summary, {"friction", "contact", "gripper", "slip", "rollout", "task"})
    analysis = _first_analysis(summary, "contact_friction", "friction", "material_friction", "gripper_friction")
    warnings = list(summary.get("warnings", []))
    failures: list[str] = []
    min_confidence = float(config.video_evidence.get("min_friction_confidence", 0.55))

    if not friction_videos and not analysis:
        return {
            "status": "evidence_missing",
            "quality_score": 0.0,
            "confidence": 1.0,
            "blocking_failures": [],
            "warnings": ["no uploaded contact/friction video evidence was found"],
            "metrics": {
                "friction_video_count": 0,
                "analysis_available": False,
                "suggested_sim_params": {},
            },
        }

    static_mu = _first_float(analysis, "object_static_friction", "static_friction", "mu_static")
    dynamic_mu = _first_float(analysis, "object_dynamic_friction", "dynamic_friction", "mu_dynamic")
    gripper_static = _first_float(analysis, "gripper_pad_static_friction", "gripper_static_friction")
    gripper_dynamic = _first_float(analysis, "gripper_pad_dynamic_friction", "gripper_dynamic_friction")
    slip_ratio = _optional_float(analysis.get("slip_ratio"))
    stick_slip_events = int(analysis.get("stick_slip_events", 0) or 0)
    confidence = _bounded_float(analysis.get("confidence"), default=0.65 if analysis else 0.35)

    if static_mu is None and dynamic_mu is None and slip_ratio is None:
        warnings.append("friction video exists but no friction coefficient or slip metric was reported by video analysis")
        status = "evidence_missing"
        quality = 0.45 if friction_videos else 0.0
    elif confidence < min_confidence:
        warnings.append(f"video friction confidence {confidence:.3f} is below recommended {min_confidence:.3f}")
        status = "evidence_missing"
        quality = 0.45
    else:
        for label, value in [
            ("object_static_friction", static_mu),
            ("object_dynamic_friction", dynamic_mu),
            ("gripper_pad_static_friction", gripper_static),
            ("gripper_pad_dynamic_friction", gripper_dynamic),
        ]:
            if value is not None and (value <= 0.0 or value > 2.5):
                failures.append(f"{label}={value} is outside conservative sim tuning bounds")
        status = "fail" if failures else "pass"
        quality = 0.3 if failures else 0.85

    suggested = _friction_sim_params(
        static_mu=static_mu,
        dynamic_mu=dynamic_mu,
        gripper_static=gripper_static,
        gripper_dynamic=gripper_dynamic,
        slip_ratio=slip_ratio,
        spread=float(config.video_evidence.get("default_friction_spread", 0.15)),
    )
    return {
        "status": status,
        "quality_score": quality,
        "confidence": confidence,
        "blocking_failures": failures,
        "warnings": warnings,
        "metrics": {
            "friction_video_count": len(friction_videos),
            "analysis_available": bool(analysis),
            "object_static_friction": static_mu,
            "object_dynamic_friction": dynamic_mu,
            "gripper_pad_static_friction": gripper_static,
            "gripper_pad_dynamic_friction": gripper_dynamic,
            "slip_ratio": slip_ratio,
            "stick_slip_events": stick_slip_events,
            "suggested_sim_params": suggested,
            "videos": friction_videos,
        },
    }


def apply_video_friction_to_domain_randomization(dr: dict[str, Any], video_metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not video_metrics:
        return dr
    suggested = video_metrics.get("suggested_sim_params")
    if not isinstance(suggested, dict) or not suggested:
        return dr
    updated = json.loads(json.dumps(dr))
    contact = updated.setdefault("actuator_and_contact_randomization", {})
    object_material = suggested.get("object_material", {})
    gripper_pad = suggested.get("gripper_pad", {})
    if object_material.get("static_friction") is not None:
        contact["object_material_static_friction"] = object_material["static_friction"]
    if object_material.get("dynamic_friction") is not None:
        contact["object_material_dynamic_friction"] = object_material["dynamic_friction"]
    if gripper_pad.get("static_friction") is not None:
        contact["gripper_pad_static_friction"] = gripper_pad["static_friction"]
    if gripper_pad.get("dynamic_friction") is not None:
        contact["gripper_pad_dynamic_friction"] = gripper_pad["dynamic_friction"]
    if suggested.get("friction_sweep_for_agent_experiments"):
        contact["friction_sweep_for_agent_experiments"] = suggested["friction_sweep_for_agent_experiments"]
    if object_material.get("static_friction") is not None and object_material.get("dynamic_friction") is not None:
        contact["material_nominal_static_dynamic_friction"] = round(
            (float(object_material["static_friction"]) + float(object_material["dynamic_friction"])) / 2.0,
            3,
        )
    contact["video_friction_source"] = "video_contact_friction"
    return updated


def resolve_video_session(dataset_path: str | Path, root: str | Path = ".") -> Path | None:
    root_path = Path(root).expanduser().resolve()
    source = Path(dataset_path).expanduser()
    if not source.is_absolute():
        source = root_path / source
    source = source.resolve()
    if source.is_dir():
        return source

    summary_path = source.parent / "prepare_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            session_dir = summary.get("session_dir")
            if session_dir:
                session = Path(str(session_dir)).expanduser()
                return session if session.is_absolute() else (root_path / session).resolve()
        except (json.JSONDecodeError, OSError):
            pass

    try:
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            session_dir = row.get("source_session_dir") or row.get("video_session_dir")
            if session_dir:
                session = Path(str(session_dir)).expanduser()
                return session if session.is_absolute() else (root_path / session).resolve()
            break
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return None


def _run_video_analysis_command(
    summary: dict[str, Any],
    config: PipelineConfig,
    root: str | Path,
    work_dir: str | Path | None,
) -> dict[str, Any] | None:
    command = [str(item) for item in config.video_evidence.get("analysis_command", [])]
    if not command or work_dir is None:
        return None
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "video_analysis_input.json"
    output_path = out_dir / "video_analysis_output.json"
    input_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    session = summary.get("session_dir") or ""
    formatted = [
        item.format(input=input_path, output=output_path, session=session, root=Path(root).expanduser().resolve())
        for item in command
    ]
    env = os.environ.copy()
    env.update(
        {
            "AGENTIC_SIM2REAL_VIDEO_INPUT_JSON": str(input_path),
            "AGENTIC_SIM2REAL_VIDEO_OUTPUT_JSON": str(output_path),
            "AGENTIC_SIM2REAL_VIDEO_SESSION": str(session),
        }
    )
    timeout_s = float(config.video_evidence.get("analysis_timeout_s", 600))
    try:
        completed = subprocess.run(
            formatted,
            cwd=Path(root).expanduser().resolve(),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "analysis_command": {
                "command": formatted,
                "error": str(exc),
                "status": "fail",
            }
        }
    payload: dict[str, Any]
    try:
        payload = json.loads(output_path.read_text()) if output_path.exists() else json.loads(completed.stdout)
    except Exception as exc:
        payload = {
            "analysis_command": {
                "command": formatted,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "error": f"could not parse output: {exc}",
                "status": "fail",
            }
        }
    payload.setdefault("analysis_command", {})
    payload["analysis_command"].update(
        {
            "command": formatted,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "input": str(input_path),
            "output": str(output_path),
        }
    )
    return payload


def _read_video_index(session: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for rel in VIDEO_INDEX_CANDIDATES:
        path = session / rel
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                cleaned = {key: value for key, value in row.items() if key is not None}
                cleaned["index_file"] = rel
                entries.append(cleaned)
    return entries


def _read_analysis_files(session: Path) -> tuple[list[Path], dict[str, Any]]:
    files: list[Path] = []
    merged: dict[str, Any] = {}
    for rel in VIDEO_ANALYSIS_CANDIDATES:
        path = session / rel
        if not path.exists():
            continue
        files.append(path)
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            merged.setdefault("invalid_analysis_files", []).append(rel)
            continue
        merged = _deep_merge_dicts(merged, payload)
    return files, merged


def _discover_videos(session: Path) -> list[Path]:
    candidates = []
    for dirname in ("video_data", "camera_data", "contact_data"):
        base = session / dirname
        if base.exists():
            candidates.extend(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    return sorted(set(candidates))


def _video_file_payload(path: Path, session: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(session)),
        "bytes": stat.st_size,
        "suffix": path.suffix.lower(),
    }


def _resolve_video_path(session: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else session / path


def _videos_for(summary: dict[str, Any], labels: set[str]) -> list[dict[str, Any]]:
    videos = list(summary.get("videos", []))
    entries = list(summary.get("index_entries", []))
    selected_paths: set[str] = set()
    for entry in entries:
        haystack = " ".join(str(entry.get(key, "")) for key in ("video_type", "role", "view", "notes", "video_path")).lower()
        if any(label in haystack for label in labels):
            selected_paths.add(str(entry.get("video_path", "")))
    selected = []
    for item in videos:
        path = str(item.get("path", ""))
        if path in selected_paths or any(label in path.lower() for label in labels):
            selected.append(item)
    return selected


def _first_analysis(summary: dict[str, Any], *keys: str) -> dict[str, Any]:
    analysis = summary.get("analysis", {})
    if not isinstance(analysis, dict):
        return {}
    for key in keys:
        value = analysis.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _friction_sim_params(
    *,
    static_mu: float | None,
    dynamic_mu: float | None,
    gripper_static: float | None,
    gripper_dynamic: float | None,
    slip_ratio: float | None,
    spread: float,
) -> dict[str, Any]:
    nominal = static_mu if static_mu is not None else dynamic_mu if dynamic_mu is not None else 0.75
    if slip_ratio is not None and slip_ratio > 0.1:
        spread = max(spread, min(0.35, 0.15 + slip_ratio))
    lower = round(max(0.1, float(nominal) - spread), 3)
    upper = round(min(2.0, float(nominal) + spread), 3)
    return {
        "object_material": {
            "static_friction": None if static_mu is None else round(static_mu, 3),
            "dynamic_friction": None if dynamic_mu is None else round(dynamic_mu, 3),
        },
        "gripper_pad": {
            "static_friction": None if gripper_static is None else round(gripper_static, 3),
            "dynamic_friction": None if gripper_dynamic is None else round(gripper_dynamic, 3),
        },
        "friction_sweep_for_agent_experiments": [lower, upper],
    }


def _first_float(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _optional_float(data.get(key))
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_float(value: Any, *, default: float) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        return round(max(0.0, min(1.0, default)), 3)
    return round(max(0.0, min(1.0, parsed)), 3)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
