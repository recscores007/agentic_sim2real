from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from .dataset import Record, load_records


KNOWN_NEWTON_JOINT_NAMES: dict[str, list[str]] = {
    "ur10e": [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
}

KNOWN_NEWTON_JOINT_TYPES: dict[str, list[str]] = {
    "ur10e": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
}

NEWTON_SUPPORTED_ROBOTS = {"h1", "h1_right_arm", "ur10e", "so101"}
COMMAND_VECTOR_KEYS = (
    "joint_command",
    "joint_commands",
    "commanded_joint_state",
    "commanded_joint_states",
    "commanded_joint_position",
    "commanded_joint_positions",
    "target_joint_state",
    "target_joint_states",
    "target_joint_position",
    "target_joint_positions",
    "control_position",
    "control_positions",
    "motor_command",
    "motor_commands",
)
COMMAND_PREFIXES = (
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


def run_newton_bridge_from_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    force_prepare_only: bool = False,
) -> dict[str, Any]:
    payload = json.loads(Path(input_path).expanduser().read_text())
    if output_path is None:
        output_path = payload.get("expected_output_json")
    if work_dir is None:
        work_dir = Path(output_path).expanduser().parent if output_path else Path.cwd() / "newton_bridge"
    return run_newton_bridge(payload, work_dir=work_dir, output_path=output_path, force_prepare_only=force_prepare_only)


def run_newton_bridge(
    input_payload: dict[str, Any],
    work_dir: str | Path,
    output_path: str | Path | None = None,
    force_prepare_only: bool = False,
) -> dict[str, Any]:
    config = _config(input_payload)
    sysid_cfg = _sysid_cfg(config)
    required = bool(sysid_cfg.get("require_newton", False))
    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    bridge_input_dir = Path(sysid_cfg.get("newton_bridge_input_dir") or work / "newton_real_data").expanduser()
    if not bridge_input_dir.is_absolute():
        bridge_input_dir = (work / bridge_input_dir).resolve()

    try:
        prep = prepare_newton_input(
            dataset_path=input_payload["dataset"],
            config=config,
            out_dir=bridge_input_dir,
        )
    except Exception as exc:
        payload = _status_payload(
            status="fail" if required else "evidence_missing",
            reason=f"Newton input preparation failed: {exc}",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={"newton_input_prepared": False, "newton_available": False},
            warnings=[] if required else [str(exc)],
            blocking_failures=[str(exc)] if required else [],
        )
        return _write_optional_output(payload, output_path)

    run_mode = str(sysid_cfg.get("newton_run_mode", "run")).lower()
    prepare_only = force_prepare_only or run_mode in {"prepare_only", "convert_only", "dry_run"}
    if prepare_only:
        payload = _status_payload(
            status="not_applicable",
            reason="Newton input prepared; run mode is prepare_only.",
            confidence=1.0,
            quality_score=0.0,
            metrics={
                **prep["metrics"],
                "newton_input_prepared": True,
                "newton_ran": False,
                "newton_run_mode": run_mode,
            },
            warnings=list(prep["warnings"]),
            evidence_files=list(prep["evidence_files"]),
        )
        return _write_optional_output(payload, output_path)

    newton_root = _resolve_newton_root(input_payload, sysid_cfg)
    if not newton_root:
        payload = _status_payload(
            status="fail" if required else "evidence_missing",
            reason="IsaacLab-Newton root is not configured. Set sysid.newton_root or ISAACLAB_NEWTON_ROOT.",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "newton_input_prepared": True, "newton_available": False},
            warnings=[] if required else ["Newton input was prepared, but IsaacLab-Newton root is missing."],
            blocking_failures=["IsaacLab-Newton root is missing"] if required else [],
            evidence_files=list(prep["evidence_files"]),
        )
        return _write_optional_output(payload, output_path)

    script_path = newton_root / "scripts" / "sysid" / "run_sysid.py"
    if not script_path.exists():
        payload = _status_payload(
            status="fail" if required else "evidence_missing",
            reason=f"IsaacLab-Newton SysID entrypoint not found: {script_path}",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "newton_input_prepared": True, "newton_available": False},
            warnings=[] if required else [f"Missing Newton entrypoint: {script_path}"],
            blocking_failures=[f"Missing Newton entrypoint: {script_path}"] if required else [],
            evidence_files=list(prep["evidence_files"]),
        )
        return _write_optional_output(payload, output_path)

    fit_output_dir = Path(sysid_cfg.get("newton_output_dir") or work / "newton_fit").expanduser()
    if not fit_output_dir.is_absolute():
        fit_output_dir = (work / fit_output_dir).resolve()
    fit_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        command = _newton_command(script_path, newton_root, bridge_input_dir, fit_output_dir, config)
    except ValueError as exc:
        payload = _status_payload(
            status="fail" if required else "evidence_missing",
            reason=str(exc),
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "newton_input_prepared": True, "newton_available": True, "newton_ran": False},
            warnings=[] if required else [str(exc)],
            blocking_failures=[str(exc)] if required else [],
            evidence_files=list(prep["evidence_files"]),
        )
        return _write_optional_output(payload, output_path)
    command_log = work / "newton_bridge_command_log.json"
    env = os.environ.copy()
    env["ISAACLAB_NEWTON_ROOT"] = str(newton_root)
    try:
        completed = subprocess.run(
            command,
            cwd=newton_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=float(sysid_cfg.get("newton_timeout_s", 900)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        command_log.write_text(json.dumps({"command": command, "error": str(exc)}, indent=2, sort_keys=True) + "\n")
        payload = _status_payload(
            status="fail" if required else "evidence_missing",
            reason=f"IsaacLab-Newton run could not complete: {exc}",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "newton_input_prepared": True, "newton_available": False},
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
                "fit_output_dir": str(fit_output_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if completed.returncode != 0:
        payload = _status_payload(
            status="fail" if required else "evidence_missing",
            reason=f"IsaacLab-Newton exited {completed.returncode}",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "newton_input_prepared": True, "newton_available": False},
            warnings=[] if required else [f"Newton exited {completed.returncode}; see command log."],
            blocking_failures=[f"Newton exited {completed.returncode}"] if required else [],
            evidence_files=[*prep["evidence_files"], str(command_log)],
        )
        return _write_optional_output(payload, output_path)

    parsed = parse_newton_outputs(fit_output_dir)
    best_params = parsed.get("best_params", {})
    if not best_params:
        payload = _status_payload(
            status="fail" if required else "evidence_missing",
            reason=f"IsaacLab-Newton did not write parseable best_params.yaml in {fit_output_dir}",
            confidence=0.0 if required else 1.0,
            quality_score=0.0,
            metrics={**prep["metrics"], "newton_input_prepared": True, "newton_available": True, "newton_ran": True},
            warnings=[] if required else ["Newton ran but produced no parseable fitted parameters."],
            blocking_failures=["Newton produced no parseable fitted parameters"] if required else [],
            evidence_files=[*prep["evidence_files"], str(command_log), *parsed.get("evidence_files", [])],
        )
        return _write_optional_output(payload, output_path)

    failures = []
    best_mse = parsed["metrics"].get("best_mse")
    max_best_mse = sysid_cfg.get("max_newton_best_mse")
    if max_best_mse is not None and best_mse is not None and float(best_mse) > float(max_best_mse):
        failures.append(f"Newton best_mse {float(best_mse):.6g} exceeds configured max {float(max_best_mse):.6g}")

    confidence = _newton_confidence(parsed)
    payload = _status_payload(
        status="fail" if failures else "pass",
        reason="IsaacLab-Newton SysID completed.",
        confidence=confidence,
        quality_score=confidence,
        metrics={
            **prep["metrics"],
            **parsed["metrics"],
            "newton_input_prepared": True,
            "newton_available": True,
            "newton_ran": True,
        },
        warnings=[*prep["warnings"], *parsed.get("warnings", [])],
        blocking_failures=failures,
        evidence_files=[*prep["evidence_files"], str(command_log), *parsed.get("evidence_files", [])],
        fitted_parameters=parsed.get("fitted_parameters", {}),
    )
    return _write_optional_output(payload, output_path)


def prepare_newton_input(dataset_path: str | Path, config: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    records = load_records(dataset_path)
    if not records:
        raise ValueError("dataset has no records")

    sysid_cfg = _sysid_cfg(config)
    selected_records, selected_episode, selection_warnings = _select_episode(records, sysid_cfg)
    joint_dim = len(selected_records[0].joint_state)
    joint_names = _resolve_joint_names(config, joint_dim)
    if len(joint_names) != joint_dim:
        raise ValueError(f"Newton joint_names length {len(joint_names)} does not match joint_state length {joint_dim}")

    rows = []
    command_source_counts: dict[str, int] = {}
    warnings = list(selection_warnings)
    first_timestamp = selected_records[0].timestamp
    previous_timestamp: float | None = None
    timestamp_diffs = []

    for record in selected_records:
        if len(record.joint_state) != joint_dim:
            raise ValueError("all selected records must have the same joint_state dimension")
        if previous_timestamp is not None:
            dt = record.timestamp - previous_timestamp
            if dt <= 0:
                raise ValueError("selected Newton episode has non-monotonic timestamps")
            timestamp_diffs.append(dt)
        previous_timestamp = record.timestamp

        command, command_source = _commanded_positions(record, sysid_cfg, joint_dim)
        if command is None:
            raise ValueError(
                "Newton SysID needs commanded joint positions. Add joint_command/command_* fields "
                "or explicitly set sysid.newton_command_source='action' when actions are already joint-position commands."
            )
        if command_source == "action":
            warnings.append(
                "Using action as Newton commanded joint positions; only trust this if the action vector is already "
                "a joint-position command in the same order as joint_state."
            )
        command_source_counts[command_source] = command_source_counts.get(command_source, 0) + 1
        rows.append(
            {
                "timestamp_us": int(round((record.timestamp - first_timestamp) * 1_000_000.0)),
                "command": _coerce_vector(command, joint_dim, "command"),
                "measured": _coerce_vector(record.joint_state, joint_dim, "joint_state"),
            }
        )

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    joint_list_path = out / "joint_list.txt"
    control_path = out / "control.csv"
    state_path = out / "state_motor.csv"
    summary_path = out / "agentic_newton_input_summary.json"

    joint_list_path.write_text("\n".join(joint_names) + "\n")
    _write_newton_positions_csv(control_path, rows, "command")
    _write_newton_positions_csv(state_path, rows, "measured")

    dt_median = statistics.median(timestamp_diffs) if timestamp_diffs else 0.0
    summary = {
        "dataset": str(Path(dataset_path).expanduser()),
        "newton_input_dir": str(out),
        "selected_episode": selected_episode,
        "records": len(rows),
        "joint_names": joint_names,
        "command_source_counts": command_source_counts,
        "dt_median_s": dt_median,
        "control_csv": str(control_path),
        "state_motor_csv": str(state_path),
        "joint_list": str(joint_list_path),
        "warnings": sorted(set(warnings)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return {
        "newton_input_dir": str(out),
        "metrics": {
            "newton_input_dir": str(out),
            "selected_episode": selected_episode,
            "newton_input_records": len(rows),
            "newton_joint_count": len(joint_names),
            "newton_dt_median_s": dt_median,
            "newton_command_source_counts": command_source_counts,
        },
        "warnings": sorted(set(warnings)),
        "evidence_files": [str(joint_list_path), str(control_path), str(state_path), str(summary_path)],
    }


def parse_newton_outputs(output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    best_params_path = out / "best_params.yaml"
    run_summary_path = out / "run_summary.yaml"
    optimization_log_path = out / "optimization_log.csv"

    best_params = _load_yaml(best_params_path) if best_params_path.exists() else {}
    run_summary = _load_yaml(run_summary_path) if run_summary_path.exists() else {}
    optimization = _load_optimization_log(optimization_log_path) if optimization_log_path.exists() else {}

    best_mse = _first_float(
        best_params.get("best_mse") if isinstance(best_params, dict) else None,
        optimization.get("best_mse"),
    )
    metrics = {
        "newton_output_dir": str(out),
        "best_mse": best_mse,
        "optimization_generations": optimization.get("generations"),
        "robot_name": best_params.get("robot_name") if isinstance(best_params, dict) else None,
        "mode": best_params.get("mode") if isinstance(best_params, dict) else None,
    }
    if isinstance(run_summary, dict):
        sysid_summary = run_summary.get("sysid", {})
        if isinstance(sysid_summary, dict):
            metrics["newton_num_envs"] = sysid_summary.get("num_envs")
            metrics["newton_max_iter"] = sysid_summary.get("max_iter")
            metrics["newton_control_freq"] = sysid_summary.get("control_freq")

    evidence = [str(path) for path in (best_params_path, run_summary_path, optimization_log_path) if path.exists()]
    warnings = []
    if not best_params:
        warnings.append("best_params.yaml was missing or empty")
    if not optimization:
        warnings.append("optimization_log.csv was missing or empty")

    return {
        "best_params": best_params,
        "run_summary": run_summary,
        "optimization": optimization,
        "fitted_parameters": _fitted_parameters(best_params),
        "metrics": metrics,
        "warnings": warnings,
        "evidence_files": evidence,
    }


def _config(input_payload: dict[str, Any]) -> dict[str, Any]:
    config = input_payload.get("config", {})
    return config if isinstance(config, dict) else {}


def _sysid_cfg(config: dict[str, Any]) -> dict[str, Any]:
    sysid = config.get("sysid", {})
    return sysid if isinstance(sysid, dict) else {}


def _select_episode(records: list[Record], sysid_cfg: dict[str, Any]) -> tuple[list[Record], int, list[str]]:
    by_episode: dict[int, list[Record]] = {}
    for record in records:
        by_episode.setdefault(record.episode_index, []).append(record)
    requested = sysid_cfg.get("newton_episode_index")
    warnings = []
    if requested is not None:
        episode = int(requested)
        if episode not in by_episode:
            raise ValueError(f"newton_episode_index {episode} not found in dataset")
    else:
        episode = max(by_episode, key=lambda key: len(by_episode[key]))
        if len(by_episode) > 1:
            warnings.append(
                "Newton all-joints SysID expects one continuous trajectory; selected the longest episode. "
                "Set sysid.newton_episode_index to choose a different one."
            )
    selected = sorted(by_episode[episode], key=lambda item: item.timestamp)
    if len(selected) < int(sysid_cfg.get("min_newton_records", 5)):
        raise ValueError("selected Newton episode has too few records for SysID")
    return selected, episode, warnings


def _resolve_joint_names(config: dict[str, Any], joint_dim: int) -> list[str]:
    sysid_cfg = _sysid_cfg(config)
    explicit = sysid_cfg.get("newton_joint_names")
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit]
    robot_cfg = config.get("robot", {}) if isinstance(config.get("robot", {}), dict) else {}
    robot_joint_names = robot_cfg.get("joint_names")
    if isinstance(robot_joint_names, list) and robot_joint_names:
        return [str(item) for item in robot_joint_names]
    robot_name = _newton_robot_name(config)
    if robot_name in KNOWN_NEWTON_JOINT_NAMES:
        return list(KNOWN_NEWTON_JOINT_NAMES[robot_name])
    return [f"joint_{idx}" for idx in range(joint_dim)]


def _newton_robot_name(config: dict[str, Any]) -> str:
    sysid_cfg = _sysid_cfg(config)
    if sysid_cfg.get("newton_robot_name"):
        return str(sysid_cfg["newton_robot_name"])
    robot_cfg = config.get("robot", {}) if isinstance(config.get("robot", {}), dict) else {}
    return str(robot_cfg.get("arm", "ur10e"))


def _newton_joint_types(config: dict[str, Any]) -> list[str]:
    sysid_cfg = _sysid_cfg(config)
    explicit = sysid_cfg.get("newton_joint_types")
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit]
    robot_name = _newton_robot_name(config)
    return list(KNOWN_NEWTON_JOINT_TYPES.get(robot_name, []))


def _commanded_positions(record: Record, sysid_cfg: dict[str, Any], joint_dim: int) -> tuple[list[float] | None, str]:
    source = str(sysid_cfg.get("newton_command_source", "auto")).lower()
    if source == "action":
        return _coerce_vector(record.action, joint_dim, "action"), "action"
    if source in {"joint_state", "measured"}:
        return _coerce_vector(record.joint_state, joint_dim, "joint_state"), "joint_state"

    raw = record.raw or {}
    if source not in {"auto", "command", "joint_command", "commanded_joint_position"}:
        value = _raw_vector(raw, (source,), (), joint_dim)
        if value is not None:
            return value, source
        raise ValueError(f"unknown or unavailable newton_command_source: {source}")

    value, key = _raw_command_vector(raw, joint_dim)
    if value is not None:
        return value, key

    if bool(sysid_cfg.get("newton_allow_action_as_command", False)):
        return _coerce_vector(record.action, joint_dim, "action"), "action"
    return None, ""


def _raw_command_vector(raw: dict[str, Any], joint_dim: int) -> tuple[list[float] | None, str]:
    for key in COMMAND_VECTOR_KEYS:
        value = _vector_value(raw.get(key))
        if value is not None:
            return _coerce_vector(value, joint_dim, key), key
    for prefix in COMMAND_PREFIXES:
        value = _prefix_vector(raw, prefix)
        if value is not None:
            return _coerce_vector(value, joint_dim, prefix), prefix
    return None, ""


def _raw_vector(raw: dict[str, Any], keys: tuple[str, ...], prefixes: tuple[str, ...], joint_dim: int) -> list[float] | None:
    for key in keys:
        value = _vector_value(raw.get(key))
        if value is not None:
            return _coerce_vector(value, joint_dim, key)
    for prefix in prefixes:
        value = _prefix_vector(raw, prefix)
        if value is not None:
            return _coerce_vector(value, joint_dim, prefix)
    return None


def _vector_value(value: Any) -> list[float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            value = json.loads(text)
        else:
            value = [part for part in text.replace(";", ",").split(",") if part.strip()]
    if not isinstance(value, (list, tuple)):
        return None
    return [float(item) for item in value]


def _prefix_vector(raw: dict[str, Any], prefix: str) -> list[float] | None:
    marker = f"{prefix}_"
    indices = []
    for key in raw:
        if key.startswith(marker) and key[len(marker) :].isdigit():
            indices.append(int(key[len(marker) :]))
    if not indices:
        return None
    indices = sorted(indices)
    if indices != list(range(indices[-1] + 1)):
        raise ValueError(f"{prefix}_* columns must be contiguous from {prefix}_0")
    return [float(raw[f"{prefix}_{idx}"]) for idx in indices]


def _coerce_vector(value: list[float], length: int, name: str) -> list[float]:
    if len(value) != length:
        raise ValueError(f"{name} length {len(value)} does not match expected joint dimension {length}")
    return [float(item) for item in value]


def _write_newton_positions_csv(path: Path, rows: list[dict[str, Any]], vector_key: str) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "positions"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"timestamp": row["timestamp_us"], "positions": json.dumps(row[vector_key])})


def _resolve_newton_root(input_payload: dict[str, Any], sysid_cfg: dict[str, Any]) -> Path | None:
    root = sysid_cfg.get("newton_root") or input_payload.get("newton_root") or os.environ.get("ISAACLAB_NEWTON_ROOT")
    if not root:
        return None
    return Path(str(root)).expanduser().resolve()


def _newton_command(
    script_path: Path,
    newton_root: Path,
    real_data_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    sysid_cfg = _sysid_cfg(config)
    python_exe = str(sysid_cfg.get("newton_python") or sys.executable)
    robot_name = _newton_robot_name(config)
    if robot_name not in NEWTON_SUPPORTED_ROBOTS:
        raise ValueError(
            f"IsaacLab-Newton run_sysid.py supports {sorted(NEWTON_SUPPORTED_ROBOTS)}; "
            f"got {robot_name!r}. Add a custom newton_command or set a supported newton_robot_name."
        )
    command = [
        python_exe,
        str(script_path),
        "--robot-name",
        robot_name,
        "--real-data-dir",
        str(real_data_dir),
        "--output-dir",
        str(output_dir),
        "--max-iter",
        str(int(sysid_cfg.get("newton_max_iter", 100))),
        "--num-envs",
        str(int(sysid_cfg.get("newton_num_envs", 64))),
        "--control-freq",
        str(int(sysid_cfg.get("newton_control_freq_hz", 500))),
        "--physics-freq",
        str(int(sysid_cfg.get("newton_physics_freq_hz", sysid_cfg.get("newton_control_freq_hz", 500)))),
        "--headless",
    ]
    joint_types = _newton_joint_types(config)
    if joint_types:
        command.extend(["--joints", ",".join(joint_types)])
    bounds = sysid_cfg.get("newton_bounds_config")
    if bounds:
        bounds_path = Path(str(bounds)).expanduser()
        if not bounds_path.is_absolute():
            bounds_path = (newton_root / bounds_path).resolve()
        command.extend(["--config", str(bounds_path)])
    return command


def _load_optimization_log(path: Path) -> dict[str, Any]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    best_row = min(rows, key=lambda row: _float_or_inf(row.get("best_mse")))
    return {
        "generations": len(rows),
        "best_mse": _first_float(best_row.get("best_mse")),
        "mean_mse": _first_float(best_row.get("mean_mse")),
        "min_mse": _first_float(best_row.get("min_mse")),
        "best_generation": _first_float(best_row.get("generation")),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return _parse_basic_yaml(text)


def _parse_basic_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _fitted_parameters(best_params: Any) -> dict[str, Any]:
    if not isinstance(best_params, dict):
        return {}
    metadata_keys = {"robot_name", "mode", "joint_name", "data_source", "num_files", "best_mse"}
    return {key: value for key, value in best_params.items() if key not in metadata_keys}


def _newton_confidence(parsed: dict[str, Any]) -> float:
    confidence = 0.55
    if parsed.get("best_params"):
        confidence += 0.2
    metrics = parsed.get("metrics", {})
    if metrics.get("best_mse") is not None:
        confidence += 0.1
    if metrics.get("optimization_generations"):
        confidence += 0.05
    return min(confidence, 0.9)


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            result = float(value)
            if math.isfinite(result):
                return result
        except (TypeError, ValueError):
            continue
    return None


def _float_or_inf(value: Any) -> float:
    result = _first_float(value)
    return result if result is not None else float("inf")


def _status_payload(
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
