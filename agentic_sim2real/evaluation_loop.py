from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .autoresearch import build_plan
from .config import load_config
from .dataset import load_records
from .skill_harness import run_harness
from .sysid import estimate_gap


DEFAULT_THRESHOLD_POLICY = Path("harness/threshold_policy.json")


def run_evaluation_loop(
    root: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path,
    threshold_policy_path: str | Path | None = None,
    include_real: bool = False,
    skill_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    records = load_records(dataset_path)
    gap = estimate_gap(records, config)
    plan = build_plan(gap, config)
    threshold_policy = load_threshold_policy(root_path, threshold_policy_path)

    proposal = agent_proposes(plan, gap, out)
    scoreboard = run_harness(
        root=root_path,
        config_path=config_path,
        dataset_path=dataset_path,
        out_dir=out / "harness",
        include_real=include_real,
        skill_dirs=skill_dirs,
    )
    measurements = evaluator_measures(scoreboard, threshold_policy, out)
    critique = critic_challenges(scoreboard, threshold_policy, out)
    decision = release_gate_decides(scoreboard, critique, out)
    human_gate = human_approves_hardware(config, include_real, out)

    trace = {
        "threshold_policy": threshold_policy,
        "agent_proposes": proposal,
        "evaluator_measures": measurements,
        "critic_challenges": critique,
        "release_gate_decides": decision,
        "human_approves_hardware": human_gate,
    }
    (out / "evaluation_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    (out / "evaluation_trace.md").write_text(write_trace_markdown(trace) + "\n")
    return trace


def load_threshold_policy(root: Path, threshold_policy_path: str | Path | None) -> dict[str, Any]:
    path = Path(threshold_policy_path) if threshold_policy_path else root / DEFAULT_THRESHOLD_POLICY
    if not path.is_absolute():
        path = root / path
    return json.loads(path.read_text())


def agent_proposes(plan: dict[str, Any], gap: dict[str, Any], out: Path) -> dict[str, Any]:
    proposal = {
        "role": "Agent",
        "job": "Generate hypotheses, candidate parameter changes, and experiment order.",
        "source": "AutoResearch planner plus SysID/domain-randomization recommendations",
        "transfer_score": plan["transfer_score"],
        "experiments": plan["experiments"],
        "candidate_parameter_families": {
            "action_scale": gap["recommendations"]["action_scale"],
            "domain_randomization": gap["recommendations"]["domain_randomization"],
            "sysid_targets": gap["recommendations"]["sysid_targets"],
        },
        "authority_limit": "Proposal only. The agent cannot pass itself, change hard safety thresholds, or run the robot.",
    }
    proposal["evidence_file"] = _write_json(out / "agent_proposal.json", proposal)
    return proposal


def evaluator_measures(scoreboard: dict[str, Any], threshold_policy: dict[str, Any], out: Path) -> dict[str, Any]:
    skills = scoreboard["skills"]
    measurements = {
        "role": "Evaluator",
        "job": "Run deterministic skill validators, compute metrics, and write evidence.",
        "status": scoreboard["status"],
        "quality_score": scoreboard["quality_score"],
        "threshold_policy_version": threshold_policy["policy_version"],
        "skill_measurements": {
            skill_id: {
                "status": result["status"],
                "quality_score": result["quality_score"],
                "confidence": result["confidence"],
                "metrics": result["metrics"],
                "evidence_files": result["evidence_files"],
            }
            for skill_id, result in skills.items()
        },
        "authority_limit": "Measures and scores. It does not invent new candidate parameters.",
    }
    measurements["evidence_file"] = _write_json(out / "evaluator_measurements.json", measurements)
    return measurements


def critic_challenges(scoreboard: dict[str, Any], threshold_policy: dict[str, Any], out: Path) -> dict[str, Any]:
    challenges = []
    observations = []
    skills = scoreboard["skills"]
    confidence_floor = float(threshold_policy["statistical_thresholds"]["min_skill_confidence_for_auto_promotion"])

    for skill_id, result in skills.items():
        if result["status"] == "fail":
            challenges.append(f"{skill_id} failed: {result['blocking_failures']}")
        if result["status"] == "skip":
            observations.append(f"{skill_id} skipped: {result.get('warnings', [])}")
        if result["status"] == "pass" and float(result["confidence"]) < confidence_floor:
            challenges.append(
                f"{skill_id} confidence {result['confidence']} is below auto-promotion floor {confidence_floor}"
            )
        for warning in result.get("warnings", []):
            observations.append(f"{skill_id} warning: {warning}")

    sim_eval = skills.get("sim_eval_regression", {})
    success_delta = float(sim_eval.get("metrics", {}).get("success_delta", 0.0))
    if success_delta < float(threshold_policy["regression_thresholds"]["min_success_delta"]):
        challenges.append(
            f"candidate success delta {success_delta} is below required "
            f"{threshold_policy['regression_thresholds']['min_success_delta']}"
        )

    critique = {
        "role": "Critic",
        "job": "Challenge weak evidence, low confidence, regressions, and unsafe assumptions.",
        "status": "pass" if not challenges else "needs_review",
        "challenges": challenges,
        "observations": observations,
        "authority_limit": "Can block or request more evidence. It does not approve hardware.",
    }
    critique["evidence_file"] = _write_json(out / "critic_challenges.json", critique)
    return critique


def release_gate_decides(scoreboard: dict[str, Any], critique: dict[str, Any], out: Path) -> dict[str, Any]:
    blocking = list(scoreboard.get("blocking_failures", []))
    critic_needs_review = critique["status"] != "pass"
    if critic_needs_review:
        blocking.append({"skill_id": "critic_agent", "failures": critique["challenges"]})

    status = "promote_to_human_review" if not blocking else "blocked"
    decision = {
        "role": "Release Gate",
        "job": "Apply pass/fail policy across all release-blocking skills and critic findings.",
        "status": status,
        "blocking_failures": blocking,
        "safe_to_autorun_robot": False,
        "required_human_approvals": [
            skill_id
            for skill_id, result in scoreboard["skills"].items()
            if result.get("human_required")
        ],
        "authority_limit": "Can promote to human review only. It cannot authorize unattended real robot motion.",
    }
    decision["evidence_file"] = _write_json(out / "release_decision.json", decision)
    return decision


def human_approves_hardware(config: Any, include_real: bool, out: Path) -> dict[str, Any]:
    key = str(config.safety.get("real_robot_gate_env", "I_ACCEPT_AGENTIC_SIM2REAL_REAL_ROBOT_RISK"))
    expected = str(config.safety.get("real_robot_gate_value", "yes"))
    actual = os.environ.get(key)
    approved = bool(include_real and actual == expected)
    gate = {
        "role": "Human",
        "job": "Confirm workspace, calibration, pendant state, gripper Tool I/O, emergency stop, and supervision.",
        "status": "approved_for_supervised_run" if approved else "not_approved",
        "required_env": f"{key}={expected}",
        "env_present": actual == expected,
        "include_real_requested": include_real,
        "safe_to_autorun_robot": False,
        "authority_limit": "Human can approve supervised hardware execution; this still does not make the robot autonomous.",
    }
    gate["evidence_file"] = _write_json(out / "human_hardware_gate.json", gate)
    return gate


def write_trace_markdown(trace: dict[str, Any]) -> str:
    sections = [
        "# Evaluation Trace",
        "",
        "## Agent Proposes",
        "",
        f"- Transfer score: {trace['agent_proposes']['transfer_score']['score_0_to_1']}",
        f"- Experiments proposed: {len(trace['agent_proposes']['experiments'])}",
        f"- Authority: {trace['agent_proposes']['authority_limit']}",
        "",
        "## Evaluator Measures",
        "",
        f"- Harness status: {trace['evaluator_measures']['status']}",
        f"- Quality score: {trace['evaluator_measures']['quality_score']}",
        f"- Threshold policy: {trace['evaluator_measures']['threshold_policy_version']}",
        "",
        "## Critic Challenges",
        "",
        f"- Critic status: {trace['critic_challenges']['status']}",
    ]
    challenges = trace["critic_challenges"]["challenges"]
    if challenges:
        sections.extend([f"- {item}" for item in challenges])
    else:
        sections.append("- No critic challenges.")
    observations = trace["critic_challenges"].get("observations", [])
    if observations:
        sections.append("- Observations:")
        sections.extend([f"  - {item}" for item in observations])
    sections.extend(
        [
            "",
            "## Release Gate Decides",
            "",
            f"- Decision: {trace['release_gate_decides']['status']}",
            f"- Safe to autorun robot: {trace['release_gate_decides']['safe_to_autorun_robot']}",
            "",
            "## Human Approves Hardware",
            "",
            f"- Human gate: {trace['human_approves_hardware']['status']}",
            f"- Required env: `{trace['human_approves_hardware']['required_env']}`",
            f"- Safe to autorun robot: {trace['human_approves_hardware']['safe_to_autorun_robot']}",
        ]
    )
    return "\n".join(sections)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return str(path)
