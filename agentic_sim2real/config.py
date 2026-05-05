from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


UI_AUDIENCES = {"customer", "developer"}

DEFAULTS: dict[str, Any] = {
    "isaac_lab": {
        "root": "~/IsaacLab",
        "task_2f140": "Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
        "task_2f85": "Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
        "train_num_envs": 256,
        "visualize_num_envs": 4,
        "video_length": 800,
        "video_interval": 5000,
        "observations": ["joint_pos", "joint_vel", "gear_shaft_pos", "gear_shaft_quat"],
        "rollout_command": [],
        "rollout_metrics_path": "",
        "rollout_timeout_s": 1800,
        "rollout_min_episodes": 20,
        "rollout_success_rate_floor": 0.7,
    },
    "robot": {
        "arm": "ur10e",
        "gripper_type": "robotiq_2f_140",
        "control_rate_hz": 30.0,
        "nominal_action_scale_2f85": 0.025,
        "nominal_action_scale_2f140": 0.0325,
        "max_action_scale": 0.05,
    },
    "perception": {
        "depth_type": "REALSENSE",
        "camera_topic": "/camera_1/color/image_raw",
        "debug_points_topic": "/input_points_debug",
        "pose_error_gate_m": 0.01,
        "training_shaft_pos_noise_m": 0.005,
        "training_shaft_quat_noise_component": 0.01,
    },
    "isaac_ros": {
        "workspace": "~/workspaces/isaac_ros-dev",
        "ros_domain_id": "77",
        "rmw_implementation": "rmw_cyclonedds_cpp",
        "manipulator_config": "$(ros2 pkg prefix --share isaac_ros_manipulation_bringup)/params/ur10e_robotiq_2f_140_gear_assembly.yaml",
        "workflow_type": "GEAR_ASSEMBLY",
        "gear_action": "/gear_assembly",
    },
    "agent": {
        "target_real_episodes": 10,
        "min_real_episodes_for_gate": 3,
        "max_recommended_delay_steps": 8,
        "min_delay_sample_hz": 50.0,
        "precision_task": True,
        "target_real_success": 0.8,
        "gap_target": 0.1,
    },
    "task_spec": {
        "mode": "characterization",
        "task": "",
        "goal": {},
        "scenarios": [],
        "policy_ckpt": "",
        "sim_config": {"engine": "isaac", "hash": ""},
        "real_data": {"source": "", "rollouts": None},
        "skills_allowed": [],
        "budget": {"gpu_hr": 20.0, "wall_hr": 6.0},
        "kill_criteria": {"max_iters": 5, "min_delta": 0.01},
        "owner": "",
        "submitted": "",
        "baseline_gap": None,
        "baseline_real_success": None,
    },
    "policy": {
        "artifact_dir": "golden/sample_inputs/policy_artifacts",
        "allow_sample_artifacts_for_smoke": True,
        "allow_policy_artifact_waiver": False,
        "policy_artifact_waiver_reason": "",
    },
    "release": {
        "profile": "smoke",
        "require_physics_sysid_for_human_review": None,
        "allow_sysid_waiver": False,
        "sysid_waiver_reason": "",
        "require_heldout_session_for_human_review": None,
        "heldout_min_episodes": 1,
        "allow_heldout_waiver": False,
        "heldout_waiver_reason": "",
        "require_camera_video_for_human_review": False,
        "allow_camera_video_waiver": False,
        "camera_video_waiver_reason": "",
        "require_contact_video_for_human_review": False,
        "allow_contact_video_waiver": False,
        "contact_video_waiver_reason": "",
        "require_isaaclab_rollout_for_human_review": None,
        "allow_isaaclab_rollout_waiver": False,
        "isaaclab_rollout_waiver_reason": "",
        "require_user_policy_artifacts_for_human_review": None,
    },
    "llm_orchestrator": {
        "provider": "scripted",
        "command": [],
        "model": "",
        "gap_hints": [],
        "max_steps": 32,
        "max_invalid_decisions": 3,
        "allow_retries": False,
        "budget_skill_calls": 24,
    },
    "ui": {
        "audience": "customer",
    },
    "video_evidence": {
        "analysis_command": [],
        "analysis_timeout_s": 600,
        "reprojection_error_gate_px": 1.5,
        "min_friction_confidence": 0.55,
        "default_friction_spread": 0.15,
    },
    "sysid": {
        "sysid_backend_preference": ["newton", "pace", "local"],
        "newton_enabled": False,
        "require_newton": False,
        "newton_root": "",
        "newton_command": [],
        "newton_robot_name": "",
        "newton_joint_names": [],
        "newton_joint_types": [],
        "newton_command_source": "auto",
        "newton_allow_action_as_command": False,
        "newton_run_mode": "run",
        "min_newton_records": 5,
        "newton_max_iter": 100,
        "newton_num_envs": 64,
        "newton_control_freq_hz": 500,
        "newton_physics_freq_hz": 500,
        "newton_timeout_s": 900,
        "min_newton_confidence": 0.6,
        "pace_enabled": False,
        "require_pace": False,
        "pace_root": "",
        "pace_command": [],
        "pace_task": "",
        "pace_robot_name": "",
        "pace_data_dir": "",
        "pace_run_mode": "run",
        "min_pace_records": 5,
        "pace_num_envs": 4096,
        "pace_timeout_s": 1800,
        "min_pace_confidence": 0.6,
    },
    "safety": {
        "require_human_gate": True,
        "real_robot_gate_env": "I_ACCEPT_AGENTIC_SIM2REAL_REAL_ROBOT_RISK",
        "real_robot_gate_value": "yes",
        "max_contact_force_n": 80.0,
    },
}


@dataclass(frozen=True)
class PipelineConfig:
    isaac_lab: dict[str, Any] = field(default_factory=dict)
    robot: dict[str, Any] = field(default_factory=dict)
    perception: dict[str, Any] = field(default_factory=dict)
    isaac_ros: dict[str, Any] = field(default_factory=dict)
    agent: dict[str, Any] = field(default_factory=dict)
    task_spec: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    release: dict[str, Any] = field(default_factory=dict)
    llm_orchestrator: dict[str, Any] = field(default_factory=dict)
    ui: dict[str, Any] = field(default_factory=dict)
    video_evidence: dict[str, Any] = field(default_factory=dict)
    sysid: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)

    def merged(self) -> dict[str, Any]:
        return {
            "isaac_lab": self.isaac_lab,
            "robot": self.robot,
            "perception": self.perception,
            "isaac_ros": self.isaac_ros,
            "agent": self.agent,
            "task_spec": self.task_spec,
            "policy": self.policy,
            "release": self.release,
            "llm_orchestrator": self.llm_orchestrator,
            "ui": self.ui,
            "video_evidence": self.video_evidence,
            "sysid": self.sysid,
            "safety": self.safety,
        }


def load_config(path: str | Path) -> PipelineConfig:
    data = json.loads(Path(path).expanduser().read_text())
    merged = _deep_merge(DEFAULTS, data)
    return _config_from_merged(merged)


def config_with_ui_audience(config: PipelineConfig, audience: str | None) -> PipelineConfig:
    if audience is None:
        normalize_ui_audience(config.ui.get("audience"))
        return config
    merged = config.merged()
    merged["ui"] = dict(merged.get("ui", {}))
    merged["ui"]["audience"] = normalize_ui_audience(audience)
    return _config_from_merged(merged)


def normalize_ui_audience(audience: Any) -> str:
    value = str(audience or "customer").strip().lower()
    if value not in UI_AUDIENCES:
        allowed = ", ".join(sorted(UI_AUDIENCES))
        raise ValueError(f"ui.audience must be one of: {allowed}")
    return value


def _config_from_merged(merged: dict[str, Any]) -> PipelineConfig:
    merged["ui"] = dict(merged.get("ui", {}))
    merged["ui"]["audience"] = normalize_ui_audience(merged.get("ui", {}).get("audience"))
    return PipelineConfig(
        isaac_lab=dict(merged["isaac_lab"]),
        robot=dict(merged["robot"]),
        perception=dict(merged["perception"]),
        isaac_ros=dict(merged["isaac_ros"]),
        agent=dict(merged["agent"]),
        task_spec=dict(merged["task_spec"]),
        policy=dict(merged["policy"]),
        release=dict(merged["release"]),
        llm_orchestrator=dict(merged["llm_orchestrator"]),
        ui=dict(merged["ui"]),
        video_evidence=dict(merged["video_evidence"]),
        sysid=dict(merged["sysid"]),
        safety=dict(merged["safety"]),
    )


def choose_task(config: PipelineConfig) -> str:
    gripper = str(config.robot.get("gripper_type", "robotiq_2f_140")).lower()
    if "85" in gripper:
        return str(config.isaac_lab["task_2f85"])
    return str(config.isaac_lab["task_2f140"])


def nominal_action_scale(config: PipelineConfig) -> float:
    gripper = str(config.robot.get("gripper_type", "robotiq_2f_140")).lower()
    key = "nominal_action_scale_2f85" if "85" in gripper else "nominal_action_scale_2f140"
    return float(config.robot[key])


def command_env(config: PipelineConfig) -> dict[str, str]:
    return {
        "ISAAC_LAB_ROOT": str(config.isaac_lab["root"]),
        "ISAAC_ROS_WS": str(config.isaac_ros["workspace"]),
        "AGENTIC_SIM2REAL_TASK": choose_task(config),
        "ROS_DOMAIN_ID": str(config.isaac_ros["ros_domain_id"]),
        "RMW_IMPLEMENTATION": str(config.isaac_ros["rmw_implementation"]),
        "AGENTIC_SIM2REAL_DEPLOYMENT_CONFIG": str(config.isaac_ros["manipulator_config"]),
        "AGENTIC_SIM2REAL_ACTION": str(config.isaac_ros["gear_action"]),
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = _deep_merge(value, override.get(key, {}))
        else:
            result[key] = override.get(key, value)
    for key, value in override.items():
        if key not in result:
            result[key] = value
    return result
