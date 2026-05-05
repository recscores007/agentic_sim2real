from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .config import PipelineConfig, choose_task


REQUIRED_COMMANDS = ["python3"]
OPTIONAL_COMMANDS = ["git", "ros2", "launch_test", "rqt_image_view", "nvidia-smi"]


def run_preflight(config: PipelineConfig, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    command_checks = _check_commands(checks)
    repo_checks = _check_repo(root_path, checks)
    isaac_lab_checks = _check_isaac_lab(config, checks)
    isaac_ros_checks = _check_isaac_ros(config, checks)
    sysid_checks = _check_sysid_backends(config, checks)

    failures = [check["message"] for check in checks if check["status"] == "fail"]
    warnings = [check["message"] for check in checks if check["status"] == "warn"]
    recommendations = _recommendations(sysid_checks)
    return {
        "status": "fail" if failures else "pass",
        "root": str(root_path),
        "selected_isaac_lab_task": choose_task(config),
        "commands": command_checks,
        "repo": repo_checks,
        "isaac_lab": isaac_lab_checks,
        "isaac_ros": isaac_ros_checks,
        "sysid_backends": sysid_checks,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def _check_commands(checks: list[dict[str, Any]]) -> dict[str, str | None]:
    commands = {}
    for command in [*REQUIRED_COMMANDS, *OPTIONAL_COMMANDS]:
        found = shutil.which(command)
        commands[command] = found
        if command in REQUIRED_COMMANDS and not found:
            _add(checks, "command", command, "fail", f"{command} is required but was not found on PATH")
        elif not found:
            _add(checks, "command", command, "warn", f"{command} not found; related runtime or hardware checks may need a sourced environment")
        else:
            _add(checks, "command", command, "pass", f"{command} found", path=found)
    return commands


def _check_repo(root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    required_dirs = ["agentic_sim2real", "skills", "configs", "embodiments", "scripts"]
    present = {}
    for dirname in required_dirs:
        path = root / dirname
        present[dirname] = path.exists()
        if path.exists():
            _add(checks, "repo", dirname, "pass", f"repo directory present: {dirname}", path=str(path))
        else:
            _add(checks, "repo", dirname, "fail", f"repo directory missing: {dirname}", path=str(path))
    return {"required_dirs": present}


def _check_isaac_lab(config: PipelineConfig, checks: list[dict[str, Any]]) -> dict[str, Any]:
    root = Path(str(config.isaac_lab.get("root", ""))).expanduser()
    task = choose_task(config)
    train_script = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    result = {
        "root": str(root),
        "root_exists": root.exists(),
        "selected_task": task,
        "train_script": str(train_script),
        "train_script_exists": train_script.exists(),
    }
    if root.exists():
        _add(checks, "isaac_lab", "root", "pass", "Isaac Lab root exists", path=str(root))
        if train_script.exists():
            _add(checks, "isaac_lab", "train_script", "pass", "Isaac Lab RSL-RL train script exists", path=str(train_script))
        else:
            _add(checks, "isaac_lab", "train_script", "warn", "Isaac Lab train script not found at configured root", path=str(train_script))
    else:
        _add(checks, "isaac_lab", "root", "warn", "Isaac Lab root does not exist; sim training/evaluation cannot run until configured", path=str(root))
    if task:
        _add(checks, "isaac_lab", "task", "pass", f"selected Isaac Lab task: {task}")
    else:
        _add(checks, "isaac_lab", "task", "fail", "selected Isaac Lab task is empty")
    return result


def _check_isaac_ros(config: PipelineConfig, checks: list[dict[str, Any]]) -> dict[str, Any]:
    workspace = Path(str(config.isaac_ros.get("workspace", ""))).expanduser()
    manipulator_config = str(config.isaac_ros.get("manipulator_config", ""))
    ros_domain_id = str(config.isaac_ros.get("ros_domain_id", ""))
    rmw = str(config.isaac_ros.get("rmw_implementation", ""))
    workflow = str(config.isaac_ros.get("workflow_type", ""))
    result = {
        "workspace": str(workspace),
        "workspace_exists": workspace.exists(),
        "manipulator_config": manipulator_config,
        "ros_domain_id": ros_domain_id,
        "rmw_implementation": rmw,
        "workflow_type": workflow,
    }
    if workspace.exists():
        _add(checks, "isaac_ros", "workspace", "pass", "Isaac ROS workspace exists", path=str(workspace))
    else:
        _add(checks, "isaac_ros", "workspace", "warn", "Isaac ROS workspace does not exist; deployment commands need a sourced Isaac ROS workspace", path=str(workspace))
    if ros_domain_id:
        _add(checks, "isaac_ros", "ros_domain_id", "pass", f"ROS_DOMAIN_ID configured: {ros_domain_id}")
    else:
        _add(checks, "isaac_ros", "ros_domain_id", "fail", "ROS_DOMAIN_ID is empty")
    if rmw == "rmw_cyclonedds_cpp":
        _add(checks, "isaac_ros", "rmw_implementation", "pass", "RMW_IMPLEMENTATION is rmw_cyclonedds_cpp")
    else:
        _add(checks, "isaac_ros", "rmw_implementation", "warn", "RMW_IMPLEMENTATION is not rmw_cyclonedds_cpp")
    if workflow == "GEAR_ASSEMBLY":
        _add(checks, "isaac_ros", "workflow_type", "pass", "workflow_type is GEAR_ASSEMBLY")
    elif workflow:
        _add(checks, "isaac_ros", "workflow_type", "warn", f"workflow_type is {workflow}; verify deployment adapter expects this")
    else:
        _add(checks, "isaac_ros", "workflow_type", "fail", "workflow_type is empty")
    if "$(" in manipulator_config or "$" in manipulator_config:
        _add(checks, "isaac_ros", "manipulator_config", "warn", "manipulator_config contains shell substitution; source ROS and verify it resolves before hardware use")
    elif manipulator_config and Path(manipulator_config).expanduser().exists():
        _add(checks, "isaac_ros", "manipulator_config", "pass", "manipulator_config file exists", path=manipulator_config)
    elif manipulator_config:
        _add(checks, "isaac_ros", "manipulator_config", "warn", "manipulator_config path does not exist yet", path=manipulator_config)
    else:
        _add(checks, "isaac_ros", "manipulator_config", "fail", "manipulator_config is empty")
    return result


def _check_sysid_backends(config: PipelineConfig, checks: list[dict[str, Any]]) -> dict[str, Any]:
    sysid = config.sysid
    newton_root = _configured_root(sysid.get("newton_root"), "ISAACLAB_NEWTON_ROOT")
    pace_root = _configured_root(sysid.get("pace_root"), "PACE_SIM2REAL_ROOT")

    newton_script = newton_root / "scripts" / "sysid" / "run_sysid.py" if newton_root else None
    pace_fit = pace_root / "scripts" / "pace" / "fit.py" if pace_root else None
    pace_source = pace_root / "source" / "pace_sim2real" if pace_root else None

    newton_entrypoint_available = bool(newton_script and newton_script.exists())
    pace_entrypoint_available = bool(pace_fit and pace_fit.exists() and pace_source and pace_source.exists())
    newton_command = [str(item) for item in sysid.get("newton_command", [])]
    pace_command = [str(item) for item in sysid.get("pace_command", [])]
    newton_available = bool(newton_entrypoint_available or newton_command)
    pace_task = str(sysid.get("pace_task", "")).strip()
    pace_data_dir = str(sysid.get("pace_data_dir", "")).strip()
    pace_default_runnable = bool(pace_entrypoint_available and pace_task and pace_data_dir)
    pace_available = bool(pace_command or pace_default_runnable)
    newton_enabled = bool(sysid.get("newton_enabled", False))
    pace_enabled = bool(sysid.get("pace_enabled", False))

    if newton_command:
        _add(checks, "sysid", "newton", "pass", "custom Newton SysID command is configured")
    elif newton_entrypoint_available:
        _add(checks, "sysid", "newton", "pass", "IsaacLab-Newton SysID entrypoint found", path=str(newton_script))
    elif newton_enabled:
        _add(checks, "sysid", "newton", "fail", "Newton SysID is enabled but IsaacLab-Newton is missing or incomplete")
    else:
        _add(checks, "sysid", "newton", "warn", "Newton SysID is not available; configure sysid.newton_root or ISAACLAB_NEWTON_ROOT to use it")

    if pace_command:
        _add(checks, "sysid", "pace", "pass", "custom PACE backup SysID command is configured")
    elif pace_default_runnable:
        _add(checks, "sysid", "pace", "pass", "PACE backup SysID entrypoint found", path=str(pace_fit))
    elif pace_entrypoint_available and pace_enabled:
        _add(checks, "sysid", "pace", "fail", "PACE is enabled but pace_task and pace_data_dir are required for the default PACE fit.py path")
    elif pace_entrypoint_available:
        _add(checks, "sysid", "pace", "warn", "PACE repo found; set sysid.pace_task and sysid.pace_data_dir, or provide sysid.pace_command, before enabling it")
    elif pace_enabled:
        _add(checks, "sysid", "pace", "fail", "PACE SysID is enabled but pace-sim2real is missing or incomplete")
    else:
        _add(checks, "sysid", "pace", "warn", "PACE backup SysID is not configured; set sysid.pace_enabled and sysid.pace_root to use it if Newton is unavailable")

    _add(checks, "sysid", "local", "pass", "local log-based SysID fallback is available")
    return {
        "preference": list(sysid.get("sysid_backend_preference", ["newton", "pace", "local"])),
        "newton": {
            "enabled": newton_enabled,
            "root": str(newton_root) if newton_root else "",
            "entrypoint": str(newton_script) if newton_script else "",
            "entrypoint_available": newton_entrypoint_available,
            "command_configured": bool(newton_command),
            "available": newton_available,
        },
        "pace": {
            "enabled": pace_enabled,
            "root": str(pace_root) if pace_root else "",
            "entrypoint": str(pace_fit) if pace_fit else "",
            "entrypoint_available": pace_entrypoint_available,
            "command_configured": bool(pace_command),
            "available": pace_available,
            "task": pace_task,
            "task_configured": bool(pace_task),
            "data_dir": pace_data_dir,
            "data_dir_configured": bool(pace_data_dir),
        },
        "local": {"enabled": True, "available": True},
    }


def _configured_root(value: Any, env_name: str) -> Path | None:
    root = str(value or os.environ.get(env_name, "")).strip()
    if not root:
        return None
    return Path(root).expanduser().resolve()


def _recommendations(sysid_checks: dict[str, Any]) -> list[str]:
    if sysid_checks["newton"]["available"]:
        return ["Use Newton SysID as the primary physics fitting backend."]
    if sysid_checks["pace"]["available"]:
        if sysid_checks["pace"]["command_configured"]:
            return ["Newton is unavailable; use the configured custom PACE backup SysID command."]
        return ["Newton is unavailable; PACE backup SysID can be used if the configured PACE task matches this embodiment."]
    if sysid_checks["pace"]["entrypoint_available"]:
        return [
            "PACE repo was found, but the default PACE path still needs sysid.pace_task and sysid.pace_data_dir.",
            "For custom embodiments, provide sysid.pace_command instead of overfitting the generic pipeline to one PACE task.",
        ]
    return [
        "Newton and PACE are unavailable; the pipeline will use local log-based SysID only.",
        "To add PACE backup: clone https://github.com/leggedrobotics/pace-sim2real and set sysid.pace_root.",
    ]


def _add(
    checks: list[dict[str, Any]],
    category: str,
    name: str,
    status: str,
    message: str,
    path: str | None = None,
) -> None:
    item: dict[str, Any] = {"category": category, "name": name, "status": status, "message": message}
    if path is not None:
        item["path"] = path
    checks.append(item)
