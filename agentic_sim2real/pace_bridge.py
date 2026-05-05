from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .dataset import Record, load_records
from .newton_bridge import _commanded_positions, _resolve_joint_names, _sysid_cfg


def run_pace_bridge(
    input_payload: dict[str, Any],
    work_dir: str | Path,
    output_path: str | Path | None = None,
    force_prepare_only: bool = False,
) -> dict[str, Any]:
    config = input_payload.get("config", {}) if isinstance(input_payload.get("config", {}), dict) else {}
    sysid_cfg = _sysid_cfg(config)
    required = bool(sysid_cfg.get("require_pace", False))
    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    try:
        prep = prepare_pace_input(input_payload["dataset"], config, work)
    except Exception as exc:
        payload = _payload(
            status="fail" if required else "evidence_missing",
            reason=f"PACE input preparation failed: {exc}",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={"pace_input_prepared": False, "pace_available": False},
            warnings=[] if required else [str(exc)],
            blocking_failures=[str(exc)] if required else [],
        )
        return _write_optional_output(payload, output_path)

    run_mode = str(sysid_cfg.get("pace_run_mode", "run")).lower()
    if force_prepare_only or run_mode in {"prepare_only", "convert_only", "dry_run"}:
        payload = _payload(
            status="not_applicable",
            reason="PACE input prepared; run mode is prepare_only.",
            confidence=1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "pace_input_prepared": True, "pace_ran": False, "pace_run_mode": run_mode},
            warnings=list(prep["warnings"]),
            evidence_files=list(prep["evidence_files"]),
        )
        return _write_optional_output(payload, output_path)

    command = [str(item) for item in sysid_cfg.get("pace_command", [])]
    pace_root = _pace_root(sysid_cfg)
    if not command:
        if not pace_root:
            payload = _payload(
                status="fail" if required else "evidence_missing",
                reason="PACE root is not configured. Set sysid.pace_root or PACE_SIM2REAL_ROOT.",
                confidence=0.0 if required else 1.0,
                quality_score=0.0,
                metrics={**prep["metrics"], "pace_input_prepared": True, "pace_available": False},
                warnings=[] if required else ["PACE input was prepared, but pace-sim2real root is missing."],
                blocking_failures=["PACE root is missing"] if required else [],
                evidence_files=list(prep["evidence_files"]),
            )
            return _write_optional_output(payload, output_path)
        try:
            command = _default_pace_command(pace_root, sysid_cfg)
        except ValueError as exc:
            payload = _payload(
                status="fail" if required else "evidence_missing",
                reason=str(exc),
                confidence=0.0 if required else 1.0,
                quality_score=0.0,
                metrics={**prep["metrics"], "pace_input_prepared": True, "pace_available": bool(pace_root)},
                warnings=[] if required else [str(exc)],
                blocking_failures=[str(exc)] if required else [],
                evidence_files=list(prep["evidence_files"]),
            )
            return _write_optional_output(payload, output_path)

    formatted_values = {
        "input": str(prep["input_summary"]),
        "summary": str(prep["input_summary"]),
        "dataset": str(input_payload.get("dataset", "")),
        "output": str(output_path or work / "pace_output.json"),
        "root": str(input_payload.get("root", "")),
        "pace_root": str(pace_root or ""),
        "data_file": str(prep["pace_data_file"]),
    }
    command = [item.format(**formatted_values) for item in command]
    command_log = work / "pace_command_log.json"
    env = os.environ.copy()
    if pace_root:
        env["PACE_SIM2REAL_ROOT"] = str(pace_root)
    env.update(
        {
            "AGENTIC_SIM2REAL_PACE_INPUT_JSON": str(prep["input_summary"]),
            "AGENTIC_SIM2REAL_PACE_OUTPUT_JSON": str(output_path or work / "pace_output.json"),
            "AGENTIC_SIM2REAL_PACE_DATA_FILE": str(prep["pace_data_file"]),
            "AGENTIC_SIM2REAL_SKILL_OUTPUT_JSON": str(output_path or work / "pace_output.json"),
        }
    )
    try:
        completed = subprocess.run(
            command,
            cwd=pace_root or work,
            env=env,
            text=True,
            capture_output=True,
            timeout=float(sysid_cfg.get("pace_timeout_s", 1800)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        command_log.write_text(json.dumps({"command": command, "error": str(exc)}, indent=2, sort_keys=True) + "\n")
        payload = _payload(
            status="fail" if required else "evidence_missing",
            reason=f"PACE command could not complete: {exc}",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "pace_input_prepared": True, "pace_available": False},
            warnings=[] if required else [str(exc)],
            blocking_failures=[str(exc)] if required else [],
            evidence_files=[*prep["evidence_files"], str(command_log)],
        )
        return _write_optional_output(payload, output_path)

    command_log.write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if completed.returncode != 0:
        payload = _payload(
            status="fail" if required else "evidence_missing",
            reason=f"PACE command exited {completed.returncode}",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "pace_input_prepared": True, "pace_available": False, "pace_ran": True},
            warnings=[] if required else [f"PACE command exited {completed.returncode}; see command log."],
            blocking_failures=[f"PACE command exited {completed.returncode}"] if required else [],
            evidence_files=[*prep["evidence_files"], str(command_log)],
        )
        return _write_optional_output(payload, output_path)

    if output_path and Path(output_path).expanduser().exists():
        try:
            payload = json.loads(Path(output_path).expanduser().read_text())
            payload.setdefault("status", "pass")
            payload.setdefault("confidence", payload.get("metrics", {}).get("confidence", 0.0))
            payload.setdefault("quality_score", payload["confidence"])
            payload.setdefault("warnings", [])
            payload.setdefault("blocking_failures", [])
            payload.setdefault("evidence_files", [])
            payload.setdefault("metrics", {})
            payload["metrics"] = {**prep["metrics"], **payload["metrics"]}
            payload["metrics"].setdefault("pace_input_prepared", True)
            payload["metrics"].setdefault("pace_ran", True)
            payload["evidence_files"] = [*prep["evidence_files"], str(command_log), *payload["evidence_files"]]
            return payload
        except Exception:
            pass

    parsed = parse_pace_outputs(config, prep)
    confidence = parsed["confidence"]
    failures = []
    min_confidence = float(sysid_cfg.get("min_pace_confidence", 0.6))
    if confidence < min_confidence:
        failures.append(f"PACE SysID confidence {confidence:.3f} below required {min_confidence:.3f}")
    payload = _payload(
        status="fail" if failures else "pass",
        reason="PACE SysID completed.",
        confidence=confidence,
        quality_score=confidence,
        metrics={**prep["metrics"], **parsed["metrics"], "pace_input_prepared": True, "pace_ran": True},
        warnings=[*prep["warnings"], *parsed["warnings"]],
        blocking_failures=failures,
        evidence_files=[*prep["evidence_files"], str(command_log), *parsed["evidence_files"]],
        fitted_parameters=parsed["fitted_parameters"],
    )
    return _write_optional_output(payload, output_path)


def prepare_pace_input(dataset_path: str | Path, config: dict[str, Any], work_dir: str | Path) -> dict[str, Any]:
    records = load_records(dataset_path)
    sysid_cfg = _sysid_cfg(config)
    selected_records, selected_episode, warnings = _select_episode(records, int(sysid_cfg.get("min_pace_records", sysid_cfg.get("min_newton_records", 5))))
    joint_dim = len(selected_records[0].joint_state)
    joint_names = _resolve_joint_names(config, joint_dim)

    first_timestamp = selected_records[0].timestamp
    time_data = []
    measured = []
    commanded = []
    command_source_counts: dict[str, int] = {}
    for record in selected_records:
        command, source = _commanded_positions(record, sysid_cfg, joint_dim)
        if command is None:
            raise ValueError(
                "PACE SysID needs commanded joint positions. Add joint_command/command_* fields "
                "or explicitly set sysid.newton_command_source='action' when actions are already joint-position commands."
            )
        time_data.append(float(record.timestamp - first_timestamp))
        measured.append([float(item) for item in record.joint_state])
        commanded.append([float(item) for item in command])
        command_source_counts[source] = command_source_counts.get(source, 0) + 1

    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    pace_root = _pace_root(sysid_cfg)
    data_dir = str(sysid_cfg.get("pace_data_dir", "")).strip()
    if pace_root and data_dir:
        pace_data_file = pace_root / "data" / data_dir
    else:
        pace_data_file = work / "pace_data" / "chirp_data.pt"
    pace_data_file.parent.mkdir(parents=True, exist_ok=True)

    evidence_files = []
    torch_warning = ""
    try:
        import torch  # type: ignore

        torch.save(
            {
                "time": torch.tensor(time_data, dtype=torch.float32),
                "dof_pos": torch.tensor(measured, dtype=torch.float32),
                "des_dof_pos": torch.tensor(commanded, dtype=torch.float32),
            },
            pace_data_file,
        )
        evidence_files.append(str(pace_data_file))
    except Exception as exc:
        torch_warning = f"torch is required to write PACE .pt input; wrote JSON preview only: {exc}"
        preview_path = work / "pace_data_preview.json"
        preview_path.write_text(
            json.dumps(
                {"time": time_data, "dof_pos": measured, "des_dof_pos": commanded, "joint_order": joint_names},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        evidence_files.append(str(preview_path))

    summary_path = work / "pace_input_summary.json"
    summary = {
        "dataset": str(Path(dataset_path).expanduser()),
        "selected_episode": selected_episode,
        "records": len(selected_records),
        "joint_order": joint_names,
        "pace_data_file": str(pace_data_file),
        "pace_data_dir": data_dir,
        "command_source_counts": command_source_counts,
        "warnings": [*warnings, *([torch_warning] if torch_warning else [])],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    evidence_files.append(str(summary_path))
    return {
        "pace_data_file": str(pace_data_file),
        "input_summary": str(summary_path),
        "metrics": {
            "pace_input_records": len(selected_records),
            "pace_joint_count": joint_dim,
            "pace_selected_episode": selected_episode,
            "pace_command_source_counts": command_source_counts,
            "pace_data_file": str(pace_data_file),
        },
        "warnings": summary["warnings"],
        "evidence_files": evidence_files,
    }


def parse_pace_outputs(config: dict[str, Any], prep: dict[str, Any]) -> dict[str, Any]:
    sysid_cfg = _sysid_cfg(config)
    output_root = _pace_output_root(sysid_cfg)
    latest = _latest_child(output_root) if output_root and output_root.exists() else None
    evidence_files = []
    warnings = []
    fitted_parameters: dict[str, Any] = {}
    metrics: dict[str, Any] = {"pace_output_root": str(output_root) if output_root else "", "pace_output_dir": str(latest) if latest else ""}
    if latest is None:
        warnings.append("PACE output directory was not found; command may have written custom evidence only.")
        return {"confidence": 0.55, "fitted_parameters": fitted_parameters, "metrics": metrics, "warnings": warnings, "evidence_files": evidence_files}

    mean_files = sorted(latest.glob("mean_*.pt"))
    evidence_files.extend(str(path) for path in mean_files[-1:])
    for name in ["best_trajectory.pt", "progress.pt", "config.pt"]:
        path = latest / name
        if path.exists():
            evidence_files.append(str(path))
    if not mean_files:
        warnings.append("PACE output directory exists but contains no mean_*.pt fitted-parameter checkpoint.")
        return {"confidence": 0.55, "fitted_parameters": fitted_parameters, "metrics": metrics, "warnings": warnings, "evidence_files": evidence_files}

    try:
        import torch  # type: ignore

        mean = torch.load(mean_files[-1], map_location="cpu")
        values = mean.detach().cpu().tolist() if hasattr(mean, "detach") else list(mean)
        joint_names = _resolve_joint_names(config, int(prep["metrics"]["pace_joint_count"]))
        fitted_parameters = _pace_params_from_vector(values, joint_names)
        metrics["pace_parameter_count"] = len(values)
        progress_path = latest / "progress.pt"
        if progress_path.exists():
            progress = torch.load(progress_path, map_location="cpu")
            scores = progress.get("scores_buffer") if isinstance(progress, dict) else None
            if scores is not None:
                metrics["pace_best_score"] = float(torch.min(scores).item())
    except Exception as exc:
        warnings.append(f"Could not parse PACE torch outputs: {exc}")
        return {"confidence": 0.6, "fitted_parameters": fitted_parameters, "metrics": metrics, "warnings": warnings, "evidence_files": evidence_files}

    return {"confidence": 0.82, "fitted_parameters": fitted_parameters, "metrics": metrics, "warnings": warnings, "evidence_files": evidence_files}


def _pace_params_from_vector(values: list[float], joint_names: list[str]) -> dict[str, Any]:
    n = len(joint_names)
    if len(values) < 4 * n + 1:
        return {"raw_mean": values}
    return {
        "armature": dict(zip(joint_names, values[0:n])),
        "viscous_friction": dict(zip(joint_names, values[n : 2 * n])),
        "static_dynamic_friction": dict(zip(joint_names, values[2 * n : 3 * n])),
        "encoder_bias": dict(zip(joint_names, values[3 * n : 4 * n])),
        "delay_steps": values[4 * n],
    }


def _default_pace_command(pace_root: Path, sysid_cfg: dict[str, Any]) -> list[str]:
    fit = pace_root / "scripts" / "pace" / "fit.py"
    source = pace_root / "source" / "pace_sim2real"
    if not fit.exists() or not source.exists():
        raise ValueError(f"PACE fit entrypoint or source package is missing under {pace_root}")
    task = str(sysid_cfg.get("pace_task", "")).strip()
    data_dir = str(sysid_cfg.get("pace_data_dir", "")).strip()
    if not task or not data_dir:
        raise ValueError(
            "PACE fit.py is task-config driven. Set sysid.pace_task and sysid.pace_data_dir "
            "to match env_cfg.sim2real.data_dir, or provide sysid.pace_command."
        )
    return [
        str(sysid_cfg.get("pace_python") or sys.executable),
        str(fit),
        "--task",
        task,
        "--num_envs",
        str(int(sysid_cfg.get("pace_num_envs", 4096))),
        "--headless",
    ]


def _select_episode(records: list[Record], min_records: int) -> tuple[list[Record], int, list[str]]:
    by_episode: dict[int, list[Record]] = {}
    for record in records:
        by_episode.setdefault(record.episode_index, []).append(record)
    episode = max(by_episode, key=lambda key: len(by_episode[key]))
    warnings = []
    if len(by_episode) > 1:
        warnings.append("PACE expects one continuous excitation trajectory; selected the longest episode.")
    selected = sorted(by_episode[episode], key=lambda item: item.timestamp)
    if len(selected) < min_records:
        raise ValueError("selected PACE episode has too few records for SysID")
    return selected, episode, warnings


def _pace_root(sysid_cfg: dict[str, Any]) -> Path | None:
    root = str(sysid_cfg.get("pace_root") or os.environ.get("PACE_SIM2REAL_ROOT", "")).strip()
    return Path(root).expanduser().resolve() if root else None


def _pace_output_root(sysid_cfg: dict[str, Any]) -> Path | None:
    explicit = str(sysid_cfg.get("pace_output_dir", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = _pace_root(sysid_cfg)
    robot = str(sysid_cfg.get("pace_robot_name") or "").strip()
    if root and robot:
        return root / "logs" / "pace" / robot
    return None


def _latest_child(path: Path) -> Path | None:
    children = [child for child in path.iterdir() if child.is_dir()]
    if not children:
        return None
    return max(children, key=lambda child: child.stat().st_mtime)


def _payload(
    status: str,
    reason: str,
    confidence: float,
    quality_score: float,
    metrics: dict[str, Any],
    warnings: list[str] | None = None,
    blocking_failures: list[str] | None = None,
    evidence_files: list[str] | None = None,
    fitted_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "reason": reason,
        "confidence": confidence,
        "quality_score": quality_score,
        "metrics": metrics,
        "warnings": warnings or [],
        "blocking_failures": blocking_failures or [],
        "evidence_files": evidence_files or [],
    }
    if fitted_parameters is not None:
        payload["fitted_parameters"] = fitted_parameters
    return payload


def _write_optional_output(payload: dict[str, Any], output_path: str | Path | None) -> dict[str, Any]:
    if output_path:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
