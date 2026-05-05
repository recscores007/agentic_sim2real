from __future__ import annotations

from .config import PipelineConfig, nominal_action_scale
from .dataset import Record
from .metrics import (
    estimate_contact,
    estimate_deadband,
    estimate_delay_steps,
    estimate_pose_noise,
    estimate_reset_scatter,
    summarize_records,
)


def estimate_gap(records: list[Record], config: PipelineConfig) -> dict:
    summary = summarize_records(records)
    max_lag = int(config.agent.get("max_recommended_delay_steps", 8))
    delay = estimate_delay_steps(records, max_lag=max_lag)
    deadband = estimate_deadband(records)
    pose_noise = estimate_pose_noise(records)
    contact = estimate_contact(records, float(config.safety.get("max_contact_force_n", 80.0)))
    reset_scatter = estimate_reset_scatter(records)

    rate_hz = summary.get("estimated_rate_hz") or float(config.robot.get("control_rate_hz", 30.0))
    min_delay_sample_hz = float(config.agent.get("min_delay_sample_hz", 50.0))
    delay_observability_status = "adequate" if float(rate_hz) >= min_delay_sample_hz else "under_sampled"
    delay_steps = int(delay["delay_steps"])
    delay_s = delay_steps / rate_hz if rate_hz else 0.0
    action_scale = recommend_action_scale(
        nominal_action_scale(config),
        float(config.robot.get("max_action_scale", 0.05)),
        deadband,
        bool(config.agent.get("precision_task", True)),
        contact,
    )

    domain_randomization = recommend_domain_randomization(
        config,
        delay_steps,
        delay_s,
        deadband,
        pose_noise,
        contact,
        reset_scatter,
    )

    return {
        "summary": summary,
        "delay": {
            **delay,
            "delay_seconds": round(delay_s, 4),
            "sample_rate_hz": round(float(rate_hz), 3),
            "min_sample_rate_hz": round(min_delay_sample_hz, 3),
            "observability_status": delay_observability_status,
            "score_policy": "excluded_when_under_sampled" if delay_observability_status == "under_sampled" else "included",
        },
        "deadband_stiction_proxy": deadband,
        "object_pose_noise": pose_noise,
        "shaft_pose_noise": pose_noise,
        "contact": contact,
        "reset_scatter": reset_scatter,
        "recommendations": {
            "action_scale": action_scale,
            "domain_randomization": domain_randomization,
            "sysid_targets": recommend_sysid_targets(delay_steps, deadband, pose_noise, contact),
            "human_inputs_needed": human_inputs_needed(config),
        },
    }


def recommend_action_scale(
    nominal: float,
    max_scale: float,
    deadband: dict,
    precision_task: bool,
    contact: dict,
) -> dict:
    deadband_norm = float(deadband.get("deadband_command_norm", 0.0))
    swallowed = float(deadband.get("swallowed_command_ratio", 0.0))
    contact_over = float(contact.get("over_limit_ratio", 0.0))
    lower_bound = min(max_scale, max(0.005, deadband_norm * 1.5))
    upper_multiplier = 1.15 if precision_task else 1.25
    upper_bound = min(max_scale, max(lower_bound, nominal * upper_multiplier))

    suggested = nominal
    if swallowed > 0.35:
        suggested = max(suggested, nominal * 1.15, lower_bound)
    if contact_over > 0.05:
        suggested = min(suggested, max(lower_bound, nominal * 0.9))
    suggested = max(lower_bound, min(upper_bound, suggested))

    return {
        "suggested": round(suggested, 5),
        "nominal_from_tutorial_or_config": round(nominal, 5),
        "lower_bound_from_stiction_proxy": round(lower_bound, 5),
        "upper_bound_from_contact_precision": round(upper_bound, 5),
        "rationale": "stiction/deadband sets the lower bound; contact-rich motion and force spikes set the upper bound",
    }


def recommend_domain_randomization(
    config: PipelineConfig,
    delay_steps: int,
    delay_s: float,
    deadband: dict,
    pose_noise: dict,
    contact: dict,
    reset_scatter: dict,
) -> dict:
    measured_pos_p95 = pose_noise.get("position_error_p95_m")
    train_pos_noise = float(config.perception.get("training_shaft_pos_noise_m", 0.005))
    pos_noise = train_pos_noise
    if measured_pos_p95 is not None:
        pos_noise = max(train_pos_noise, min(0.01, float(measured_pos_p95) * 1.2))

    quat_noise = float(config.perception.get("training_shaft_quat_noise_component", 0.01))
    swallowed = float(deadband.get("swallowed_command_ratio", 0.0))
    contact_over = float(contact.get("over_limit_ratio", 0.0))
    friction_spread = 0.1 + min(0.35, swallowed * 0.5 + contact_over * 0.3)

    object_pose_noise = {
        "object_position_uniform_m": [-round(pos_noise, 6), round(pos_noise, 6)],
        "object_quat_uniform_component": [-quat_noise, quat_noise],
        "agent_note": "fit to perception repeatability and calibration tests",
    }
    object_and_base_pose_randomization = {
        "base_pose_x_m": [-0.1, 0.1],
        "base_pose_y_m": [-0.25, 0.25],
        "base_pose_z_m": [-0.1, 0.1],
        "base_roll_pitch_deg": [-2.0, 2.0],
        "base_yaw_deg": [-30.0, 30.0],
        "object_relative_xy_m": [-0.02, 0.02],
        "object_relative_z_m": [0.0575, 0.0775],
        "object_relative_rpy_deg": [-5.0, 5.0],
        "measured_reset_scatter": reset_scatter,
    }

    return {
        "object_pose_observation_noise": object_pose_noise,
        "shaft_pose_observation_noise": {
            "gear_shaft_pos_uniform_m": object_pose_noise["object_position_uniform_m"],
            "gear_shaft_quat_uniform_component": object_pose_noise["object_quat_uniform_component"],
            "agent_note": object_pose_noise["agent_note"],
        },
        "object_and_base_pose_randomization": object_and_base_pose_randomization,
        "base_and_gear_pose_randomization": {
            "base_pose_x_m": object_and_base_pose_randomization["base_pose_x_m"],
            "base_pose_y_m": object_and_base_pose_randomization["base_pose_y_m"],
            "base_pose_z_m": object_and_base_pose_randomization["base_pose_z_m"],
            "base_roll_pitch_deg": object_and_base_pose_randomization["base_roll_pitch_deg"],
            "base_yaw_deg": object_and_base_pose_randomization["base_yaw_deg"],
            "gear_relative_xy_m": object_and_base_pose_randomization["object_relative_xy_m"],
            "gear_relative_z_m": object_and_base_pose_randomization["object_relative_z_m"],
            "gear_relative_rpy_deg": object_and_base_pose_randomization["object_relative_rpy_deg"],
            "measured_reset_scatter": reset_scatter,
        },
        "actuator_and_contact_randomization": {
            "stiffness_scale_log_uniform": [0.75, 1.5],
            "damping_scale_log_uniform": [0.3, 3.0],
            "joint_friction_additive_nm": [0.3, 0.7],
            "friction_sweep_for_agent_experiments": [round(0.75 - friction_spread, 3), round(0.75 + friction_spread, 3)],
            "material_nominal_static_dynamic_friction": 0.75,
            "restitution": 0.0,
        },
        "latency_randomization": {
            "actuation_delay_steps": [max(0, delay_steps - 1), delay_steps + 1],
            "actuation_delay_seconds_center": round(delay_s, 4),
            "camera_latency_seconds": "measure from sensor/controller timestamps before widening this range",
        },
    }


def recommend_sysid_targets(delay_steps: int, deadband: dict, pose_noise: dict, contact: dict) -> list[str]:
    targets = [
        "robot impedance step response",
        "joint stiffness and damping scale",
        "joint friction and stiction",
        "task-object and end-effector contact friction",
        "perception object-pose error",
        "sensor-to-robot calibration error",
    ]
    if delay_steps > 0:
        targets.append("policy-to-controller command delay")
    if float(deadband.get("swallowed_command_ratio", 0.0)) > 0.25:
        targets.append("action scale lower bound from swallowed small commands")
    if pose_noise.get("position_error_p95_m") is None:
        targets.append("pose estimation repeatability packet")
    if float(contact.get("over_limit_ratio", 0.0)) > 0.0:
        targets.append("contact force ceiling and jam or instability modes")
    return targets


def human_inputs_needed(config: PipelineConfig) -> list[str]:
    return [
        "robot calibration file path for the physical robot",
        "sensor-to-robot calibration validation result, target under 1 cm pose error",
        "embodiment and end-effector configuration",
        "policy checkpoint path plus agent.yaml and env.yaml from training",
        "task object or asset paths used by the simulator/deployment stack",
        "10 pose-estimation repeatability samples for the task object",
        "real run labels: success, slip, jam, pose miss, camera dropout, or calibration issue",
        "human approval before each real-robot run",
    ]
