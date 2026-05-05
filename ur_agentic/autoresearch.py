from __future__ import annotations

from .config import PipelineConfig


def build_plan(gap: dict, config: PipelineConfig) -> dict:
    summary = gap["summary"]
    recs = gap["recommendations"]

    experiments = [
        {
            "id": "E1_perception_noise_replay",
            "question": "Does measured object-pose noise explain task failures?",
            "agent_does": "Build replay/sweep configs around measured object position and orientation noise.",
            "human_does": "Run or approve the real calibration/pose-repeatability test and label obvious camera failures.",
            "parameter_change": recs["domain_randomization"]["object_pose_observation_noise"],
            "promote_if": "held-out sim rollouts remain stable and predicted object-pose error is inside the configured real gate",
        },
        {
            "id": "E2_robot_sysid_step_response",
            "question": "Do simulated joint gains/friction produce the same response as the real robot controller?",
            "agent_does": "Estimate delay, stiction proxy, stiffness/damping ranges, and a small action-scale sweep from logs.",
            "human_does": "Approve a low-speed supervised step-response packet or provide existing logs.",
            "parameter_change": recs["domain_randomization"]["actuator_and_contact_randomization"],
            "promote_if": "sim joint tracking error and real tracking error match within the configured tolerance",
        },
        {
            "id": "E3_reset_and_fixture_generalization",
            "question": "Is pose randomization enough for real reset and fixture scatter?",
            "agent_does": "Compare measured reset scatter to base/object pose randomization and propose range tightening or widening.",
            "human_does": "Measure or log where the task object, fixture, or robot reset state lands between trials.",
            "parameter_change": recs["domain_randomization"]["object_and_base_pose_randomization"],
            "promote_if": "real scatter is covered without making training unnecessarily broad",
        },
        {
            "id": "E4_contact_action_scale_gate",
            "question": "Is the action scale high enough to overcome stiction but low enough for contact-rich insertion?",
            "agent_does": "Recommend an action-scale candidate and flag force spikes/jam modes before a robot run.",
            "human_does": "Check workspace, pendant, gripper Tool I/O, emergency stop, and approve the supervised run.",
            "parameter_change": recs["action_scale"],
            "promote_if": "success improves without exceeding contact-force limits or causing overshoot",
        },
    ]

    gate = {
        "min_episodes_required": int(config.agent.get("min_real_episodes_for_gate", 3)),
        "target_episodes": int(config.agent.get("target_real_episodes", 10)),
        "current_episodes": summary["episodes"],
        "human_gate_required": bool(config.safety.get("require_human_gate", True)),
        "safe_to_autorun_robot": False,
        "reason": "This repo intentionally keeps all real-robot motion behind a human gate.",
    }

    transfer_score = compute_transfer_score(gap, config)
    return {
        "gate": gate,
        "transfer_score": transfer_score,
        "experiments": experiments,
        "notes": [
            "AutoResearch is used to select and rank experiments, not to move the robot autonomously.",
            "Keep one parameter family per experiment so the sim-real gap attribution stays readable.",
            "Promote only the best candidate to supervised real evaluation.",
        ],
    }


def compute_transfer_score(gap: dict, config: PipelineConfig) -> dict:
    summary = gap["summary"]
    episode_target = max(1, int(config.agent.get("target_real_episodes", 10)))
    episode_score = min(1.0, summary["episodes"] / episode_target)
    success_rate = summary.get("success_rate")
    success_component = success_rate if success_rate is not None else 0.5
    delay_conf = float(gap["delay"].get("confidence", 0.0))
    deadband_conf = float(gap["deadband_stiction_proxy"].get("confidence", 0.0))
    pose_samples = int(gap["object_pose_noise"].get("samples", 0) or 0)
    pose_score = min(1.0, pose_samples / 20.0)
    contact_over = float(gap["contact"].get("over_limit_ratio", 0.0))
    contact_score = max(0.0, 1.0 - contact_over)

    score = (
        0.2 * episode_score
        + 0.2 * success_component
        + 0.15 * delay_conf
        + 0.15 * deadband_conf
        + 0.15 * pose_score
        + 0.15 * contact_score
    )
    return {
        "score_0_to_1": round(score, 3),
        "episode_score": round(episode_score, 3),
        "success_component": round(success_component, 3),
        "delay_confidence": delay_conf,
        "deadband_confidence": deadband_conf,
        "pose_score": round(pose_score, 3),
        "contact_score": round(contact_score, 3),
        "interpretation": interpret_score(score),
    }


def interpret_score(score: float) -> str:
    if score >= 0.75:
        return "ready for supervised real-robot candidate gate"
    if score >= 0.5:
        return "promising, but run the highest-risk AutoResearch experiment first"
    return "not ready; collect better calibration, pose, and contact logs"
