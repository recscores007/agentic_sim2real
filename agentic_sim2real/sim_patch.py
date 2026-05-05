from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig, choose_task


def build_sim_params_patch(
    *,
    config: PipelineConfig,
    dataset_path: str | Path,
    domain_randomization: dict[str, Any],
    action_scale: dict[str, Any],
    camera_metrics: dict[str, Any] | None = None,
    friction_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a concrete review artifact for Isaac Lab / Isaac Sim parameters."""
    camera_metrics = camera_metrics or {}
    friction_metrics = friction_metrics or {}
    suggested_camera = camera_metrics.get("suggested_camera_parameters", {})
    if not isinstance(suggested_camera, dict):
        suggested_camera = {}

    contact = dict(domain_randomization.get("actuator_and_contact_randomization", {}))
    latency = dict(domain_randomization.get("latency_randomization", {}))
    object_pose_noise = dict(domain_randomization.get("object_pose_observation_noise", {}))
    shaft_pose_noise = dict(domain_randomization.get("shaft_pose_observation_noise", {}))
    placement = dict(
        domain_randomization.get("base_and_gear_pose_randomization")
        or domain_randomization.get("object_and_base_pose_randomization", {})
    )

    task_spec = config.task_spec
    sim_config = task_spec.get("sim_config", {}) if isinstance(task_spec.get("sim_config"), dict) else {}
    object_material = _material_block(contact, friction_metrics, "object_material", "object_material")
    gripper_pad = _material_block(contact, friction_metrics, "gripper_pad", "gripper_pad")
    camera_patch = {
        "intrinsics": suggested_camera.get("intrinsics", {}),
        "extrinsic_delta": suggested_camera.get("extrinsic_delta", {}),
        "latency_seconds": suggested_camera.get("latency_seconds"),
        "reprojection_error_px": suggested_camera.get("reprojection_error_px"),
    }

    patch = {
        "schema": "agentic_sim2real.sim_params_patch.v1",
        "status": "proposed_review_only",
        "generated_by": "agentic_tuning_plan",
        "target": {
            "simulators": ["Isaac Lab", "Isaac Sim"],
            "task": choose_task(config),
            "sim_engine": sim_config.get("engine", "isaac"),
            "sim_config_hash": sim_config.get("hash", ""),
            "dataset": str(dataset_path),
        },
        "review_policy": {
            "apply_automatically": False,
            "real_robot_autorun": False,
            "requires_human_review": True,
        },
        "source_evidence": {
            "camera_video": {
                "analysis_available": bool(camera_metrics.get("analysis_available")),
                "camera_video_count": camera_metrics.get("camera_video_count", 0),
                "reprojection_error_px": camera_metrics.get("reprojection_error_px"),
            },
            "contact_friction_video": {
                "analysis_available": bool(friction_metrics.get("analysis_available")),
                "friction_video_count": friction_metrics.get("friction_video_count", 0),
                "object_static_friction": friction_metrics.get("object_static_friction"),
                "object_dynamic_friction": friction_metrics.get("object_dynamic_friction"),
                "gripper_pad_static_friction": friction_metrics.get("gripper_pad_static_friction"),
                "gripper_pad_dynamic_friction": friction_metrics.get("gripper_pad_dynamic_friction"),
                "slip_ratio": friction_metrics.get("slip_ratio"),
            },
        },
        "patch": {
            "isaac_lab": {
                "env_cfg": {
                    "actions": {
                        "action_scale": action_scale.get("suggested"),
                        "candidate_action_scales": action_scale.get("candidates", []),
                    },
                    "domain_randomization": {
                        "object_pose_observation_noise": object_pose_noise,
                        "shaft_pose_observation_noise": shaft_pose_noise,
                        "base_and_gear_pose_randomization": placement,
                        "actuator_and_contact_randomization": contact,
                        "latency_randomization": latency,
                    },
                    "camera": camera_patch,
                }
            },
            "isaac_sim": {
                "physics_materials": {
                    "task_object": object_material,
                    "gripper_pad": gripper_pad,
                    "material_nominal_static_dynamic_friction": contact.get(
                        "material_nominal_static_dynamic_friction"
                    ),
                    "restitution": contact.get("restitution"),
                },
                "articulation": {
                    "stiffness_scale_log_uniform": contact.get("stiffness_scale_log_uniform"),
                    "damping_scale_log_uniform": contact.get("damping_scale_log_uniform"),
                    "joint_friction_additive_nm": contact.get("joint_friction_additive_nm"),
                },
                "latency": latency,
                "camera": camera_patch,
            },
        },
    }
    patch["operations"] = _operations_from_patch(patch)
    return patch


def write_sim_params_patch(path: str | Path, patch: dict[str, Any]) -> str:
    out = Path(path)
    out.write_text(_to_yaml(patch), encoding="utf-8")
    return str(out)


def write_sim_params_patch_json(path: str | Path, patch: dict[str, Any]) -> str:
    out = Path(path)
    out.write_text(json.dumps(patch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(out)


def _material_block(
    contact: dict[str, Any],
    friction_metrics: dict[str, Any],
    contact_prefix: str,
    suggested_key: str,
) -> dict[str, Any]:
    suggested = friction_metrics.get("suggested_sim_params", {})
    suggested_material = suggested.get(suggested_key, {}) if isinstance(suggested, dict) else {}
    return {
        "static_friction": contact.get(f"{contact_prefix}_static_friction")
        if contact.get(f"{contact_prefix}_static_friction") is not None
        else suggested_material.get("static_friction"),
        "dynamic_friction": contact.get(f"{contact_prefix}_dynamic_friction")
        if contact.get(f"{contact_prefix}_dynamic_friction") is not None
        else suggested_material.get("dynamic_friction"),
    }


def _operations_from_patch(patch: dict[str, Any]) -> list[dict[str, Any]]:
    env_cfg = patch["patch"]["isaac_lab"]["env_cfg"]
    domain = env_cfg["domain_randomization"]
    contact = domain["actuator_and_contact_randomization"]
    camera = env_cfg["camera"]
    operations: list[dict[str, Any]] = []

    _append_operation(operations, "isaac_lab.env_cfg.actions.action_scale", env_cfg["actions"].get("action_scale"), "action_scale_sweep")
    _append_operation(operations, "isaac_lab.env_cfg.actions.candidate_action_scales", env_cfg["actions"].get("candidate_action_scales"), "action_scale_sweep")
    for key, value in domain["object_pose_observation_noise"].items():
        _append_operation(operations, f"isaac_lab.env_cfg.domain_randomization.object_pose_observation_noise.{key}", value, "domain_randomization_update")
    for key, value in domain["shaft_pose_observation_noise"].items():
        _append_operation(operations, f"isaac_lab.env_cfg.domain_randomization.shaft_pose_observation_noise.{key}", value, "domain_randomization_update")
    for key, value in domain["base_and_gear_pose_randomization"].items():
        _append_operation(operations, f"isaac_lab.env_cfg.domain_randomization.base_and_gear_pose_randomization.{key}", value, "domain_randomization_update")
    for key, value in contact.items():
        _append_operation(operations, f"isaac_lab.env_cfg.domain_randomization.actuator_and_contact_randomization.{key}", value, "domain_randomization_update")
    for key, value in domain["latency_randomization"].items():
        _append_operation(operations, f"isaac_lab.env_cfg.domain_randomization.latency_randomization.{key}", value, "domain_randomization_update")
    for key, value in camera.items():
        _append_operation(operations, f"isaac_lab.env_cfg.camera.{key}", value, "video_camera_tuning")
    for material, values in patch["patch"]["isaac_sim"]["physics_materials"].items():
        if isinstance(values, dict):
            for key, value in values.items():
                _append_operation(operations, f"isaac_sim.physics_materials.{material}.{key}", value, "video_contact_friction")
        else:
            _append_operation(operations, f"isaac_sim.physics_materials.{material}", values, "domain_randomization_update")
    return operations


def _append_operation(operations: list[dict[str, Any]], path: str, value: Any, source: str) -> None:
    if value in (None, {}, []):
        return
    operations.append(
        {
            "path": path,
            "value": value,
            "source": source,
            "review_gate": "human_review_required",
        }
    )


def _to_yaml(value: Any, indent: int = 0) -> str:
    text = _yaml_value(value, indent)
    return text if text.endswith("\n") else text + "\n"


def _yaml_value(value: Any, indent: int) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict) and item:
                lines.append(f"{prefix}{_yaml_key(key)}:")
                lines.append(_yaml_value(item, indent + 2))
            elif isinstance(item, list) and item and any(isinstance(child, (dict, list)) for child in item):
                lines.append(f"{prefix}{_yaml_key(key)}:")
                lines.append(_yaml_value(item, indent + 2))
            else:
                lines.append(f"{prefix}{_yaml_key(key)}: {_yaml_inline(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_yaml_value(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}- {_yaml_value(item, indent + 2).strip()}")
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return _yaml_scalar(value)


def _yaml_key(value: Any) -> str:
    return str(value)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _yaml_inline(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
    if isinstance(value, dict) and not value:
        return "{}"
    return _yaml_scalar(value)
