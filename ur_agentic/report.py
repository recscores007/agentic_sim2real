from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_outputs(out_dir: str | Path, gap: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "gap_estimates": out / "gap_estimates.json",
        "autoresearch_plan": out / "autoresearch_plan.json",
        "transfer_score": out / "transfer_score.json",
        "agentic_params": out / "agentic_params.yaml",
        "report": out / "report.md",
    }

    files["gap_estimates"].write_text(json.dumps(gap, indent=2, sort_keys=True) + "\n")
    files["autoresearch_plan"].write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    files["transfer_score"].write_text(json.dumps(plan["transfer_score"], indent=2, sort_keys=True) + "\n")
    files["agentic_params"].write_text(_write_params_yaml(gap) + "\n")
    files["report"].write_text(_write_report(gap, plan) + "\n")
    return {name: str(path) for name, path in files.items()}


def _write_params_yaml(gap: dict[str, Any]) -> str:
    dr = gap["recommendations"]["domain_randomization"]
    action = gap["recommendations"]["action_scale"]
    lines = [
        "# Candidate parameters generated from real UR gear-assembly logs.",
        "# Review before copying into Isaac Lab or Isaac ROS configs.",
        "action_scale_joint_space:",
    ]
    suggested = action["suggested"]
    for _ in range(6):
        lines.append(f"  - {suggested}")
    lines.extend(
        [
            "shaft_pose_observation_noise:",
            f"  gear_shaft_pos_uniform_m: {dr['shaft_pose_observation_noise']['gear_shaft_pos_uniform_m']}",
            f"  gear_shaft_quat_uniform_component: {dr['shaft_pose_observation_noise']['gear_shaft_quat_uniform_component']}",
            "latency_randomization:",
            f"  actuation_delay_steps: {dr['latency_randomization']['actuation_delay_steps']}",
            "actuator_randomization:",
            f"  stiffness_scale_log_uniform: {dr['actuator_and_contact_randomization']['stiffness_scale_log_uniform']}",
            f"  damping_scale_log_uniform: {dr['actuator_and_contact_randomization']['damping_scale_log_uniform']}",
            f"  joint_friction_additive_nm: {dr['actuator_and_contact_randomization']['joint_friction_additive_nm']}",
        ]
    )
    return "\n".join(lines)


def _write_report(gap: dict[str, Any], plan: dict[str, Any]) -> str:
    score = plan["transfer_score"]
    summary = gap["summary"]
    action = gap["recommendations"]["action_scale"]
    lines = [
        "# UR10e Gear Assembly Agentic Sim2Real Report",
        "",
        "## Readiness",
        "",
        f"- Transfer score: {score['score_0_to_1']} ({score['interpretation']})",
        f"- Records: {summary['records']}",
        f"- Episodes: {summary['episodes']}",
        f"- Success rate: {summary['success_rate']}",
        f"- Estimated control rate: {summary['estimated_rate_hz']} Hz",
        "",
        "## Agent Findings",
        "",
        f"- Delay: {gap['delay']['delay_steps']} steps, {gap['delay']['delay_seconds']} s",
        f"- Stiction proxy: {gap['deadband_stiction_proxy']}",
        f"- Shaft pose noise: {gap['shaft_pose_noise']}",
        f"- Contact: {gap['contact']}",
        f"- Suggested action scale: {action['suggested']} (nominal {action['nominal_from_tutorial_or_config']})",
        "",
        "## Human Inputs Still Needed",
        "",
    ]
    for item in gap["recommendations"]["human_inputs_needed"]:
        lines.append(f"- {item}")
    lines.extend(["", "## AutoResearch Experiments", ""])
    for exp in plan["experiments"]:
        lines.append(f"- {exp['id']}: {exp['question']}")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "This report is offline analysis only. Real UR motion must stay behind the human gate.",
        ]
    )
    return "\n".join(lines)
