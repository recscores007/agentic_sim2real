from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
        "precision_task": True,
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
    sysid: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)

    def merged(self) -> dict[str, Any]:
        return {
            "isaac_lab": self.isaac_lab,
            "robot": self.robot,
            "perception": self.perception,
            "isaac_ros": self.isaac_ros,
            "agent": self.agent,
            "sysid": self.sysid,
            "safety": self.safety,
        }


def load_config(path: str | Path) -> PipelineConfig:
    data = json.loads(Path(path).expanduser().read_text())
    merged = _deep_merge(DEFAULTS, data)
    return PipelineConfig(
        isaac_lab=dict(merged["isaac_lab"]),
        robot=dict(merged["robot"]),
        perception=dict(merged["perception"]),
        isaac_ros=dict(merged["isaac_ros"]),
        agent=dict(merged["agent"]),
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
