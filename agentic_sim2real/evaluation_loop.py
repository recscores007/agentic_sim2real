from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .artifacts import write_slide_contract_bundle
from .autoresearch import build_plan
from .config import load_config
from .dataset import load_records
from .llm_orchestrator import run_llm_orchestrated_loop
from .real_data import ensure_aligned_dataset
from .release_policy import release_profile, release_requires, release_waiver
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
    llm_provider_name: str | None = None,
    llm_command: list[str] | None = None,
    max_steps: int | None = None,
    gap_hints: list[str] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    resolved_dataset_path = ensure_aligned_dataset(dataset_path, root=root_path)
    records = load_records(resolved_dataset_path)
    gap = estimate_gap(records, config)
    plan = build_plan(gap, config)
    threshold_policy = load_threshold_policy(root_path, threshold_policy_path)

    proposal = agent_proposes(plan, gap, out)
    orchestrator = run_llm_orchestrated_loop(
        root=root_path,
        config_path=config_path,
        dataset_path=resolved_dataset_path,
        out_dir=out / "harness",
        include_real=include_real,
        skill_dirs=skill_dirs,
        provider_name=llm_provider_name,
        provider_command=llm_command,
        max_steps=max_steps,
        gap_hints=gap_hints,
    )
    scoreboard = orchestrator["scoreboard"]
    measurements = evaluator_measures(scoreboard, threshold_policy, out)
    critique = critic_challenges(scoreboard, threshold_policy, config, out)
    decision = release_gate_decides(scoreboard, critique, config, out)
    human_gate = human_approves_hardware(config, include_real, out)

    trace = {
        "threshold_policy": threshold_policy,
        "release_profile": release_profile(config),
        "offline_validation_status": scoreboard.get("offline_validation_status", scoreboard["status"]),
        "human_review_readiness": scoreboard.get("human_review_readiness", "not_ready"),
        "agent_proposes": proposal,
        "llm_orchestrator": {
            key: value
            for key, value in orchestrator.items()
            if key != "scoreboard"
        },
        "evaluator_measures": measurements,
        "critic_challenges": critique,
        "release_gate_decides": decision,
        "human_approves_hardware": human_gate,
    }
    trace["slide_contract_artifacts"] = write_slide_contract_bundle(
        resolved_dataset_path,
        config,
        out,
        scoreboard.get("skills", {}),
        scoreboard,
        config_path=config_path,
        skill_ids=sorted(scoreboard.get("skills", {})),
        trace=trace,
    )
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


def critic_challenges(
    scoreboard: dict[str, Any],
    threshold_policy: dict[str, Any],
    config: Any,
    out: Path,
) -> dict[str, Any]:
    challenges = []
    observations = []
    skills = scoreboard["skills"]
    confidence_floor = float(threshold_policy["statistical_thresholds"]["min_skill_confidence_for_auto_promotion"])

    for skill_id, result in skills.items():
        if result["status"] == "fail":
            challenges.append(f"{skill_id} failed: {result['blocking_failures']}")
        if result["status"] in {"skip", "not_applicable", "not_approved", "evidence_missing"}:
            observations.append(f"{skill_id} status {result['status']}: {result.get('warnings', [])}")
        if result["status"] == "evidence_missing" and result.get("release_blocking"):
            challenges.append(f"{skill_id} is missing release-blocking evidence")
        if result["status"] == "pass" and float(result["confidence"]) < confidence_floor:
            challenges.append(
                f"{skill_id} confidence {result['confidence']} is below auto-promotion floor {confidence_floor}"
            )
        for warning in result.get("warnings", []):
            observations.append(f"{skill_id} warning: {warning}")

    data_quality = skills.get("real_data_quality_gate", {})
    data_metrics = data_quality.get("metrics", {})
    success_coverage = float(data_metrics.get("success_label_coverage", 0.0) or 0.0)
    contact_coverage = float(data_metrics.get("contact_coverage", 0.0) or 0.0)
    joint_velocity_coverage = float(data_metrics.get("joint_velocity_coverage", 0.0) or 0.0)
    strict_release = release_profile(config) != "smoke"
    if success_coverage < float(threshold_policy["data_quality_thresholds"]["min_success_label_coverage"]):
        if strict_release:
            challenges.append("success labels are too sparse for confident release scoring")
        else:
            observations.append("success labels are sparse; smoke profile keeps this as an observation")
    if release_requires(config, "require_heldout_session_for_human_review"):
        heldout_episodes = int(data_metrics.get("heldout_episodes", 0) or 0)
        heldout_min = int(config.release.get("heldout_min_episodes", 1))
        waiver = release_waiver(config, "allow_heldout_waiver", "heldout_waiver_reason")
        if heldout_episodes < heldout_min and not waiver["allowed"]:
            challenges.append("no sufficient held-out session evidence is present")
    if contact_coverage < float(threshold_policy["data_quality_thresholds"]["min_contact_coverage"]):
        observations.append("contact coverage is sparse; contact/action-scale conclusions are weaker")
    if joint_velocity_coverage < 0.5:
        observations.append("joint velocity coverage is sparse; delay/stiction conclusions are weaker")

    newton = skills.get("newton_sysid", {})
    pace = skills.get("pace_sysid", {})
    physics_passed = newton.get("status") == "pass" or pace.get("status") == "pass"
    if release_requires(config, "require_physics_sysid_for_human_review"):
        waiver = release_waiver(config, "allow_sysid_waiver", "sysid_waiver_reason")
        if not physics_passed and not waiver["allowed"]:
            challenges.append("release-candidate profile requires Newton or PACE SysID evidence")
    elif not physics_passed:
        observations.append("no Newton/PACE physics SysID backend produced fitted parameters")

    policy = skills.get("policy_artifact_audit", {})
    if policy.get("metrics", {}).get("using_sample_artifacts"):
        if release_requires(config, "require_user_policy_artifacts_for_human_review"):
            challenges.append("policy artifact audit is still using sample fixtures")
        else:
            observations.append("policy artifact audit is using sample fixtures")

    rollout = skills.get("isaaclab_rollout_regression", {})
    if release_requires(config, "require_isaaclab_rollout_for_human_review"):
        waiver = release_waiver(config, "allow_isaaclab_rollout_waiver", "isaaclab_rollout_waiver_reason")
        if rollout.get("status") != "pass" and not waiver["allowed"]:
            challenges.append("true Isaac Lab rollout regression evidence is missing")

    sim_eval = skills.get("sim_eval_regression", {})
    success_delta = float(sim_eval.get("metrics", {}).get("success_delta", 0.0))
    if success_delta < float(threshold_policy["regression_thresholds"]["min_success_delta"]):
        challenges.append(
            f"candidate success delta {success_delta} is below required "
            f"{threshold_policy['regression_thresholds']['min_success_delta']}"
        )
    if strict_release and success_delta > 0 and success_coverage < float(threshold_policy["data_quality_thresholds"]["min_success_label_coverage"]):
        challenges.append("candidate success improvement is contradicted by sparse success labels")

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


def release_gate_decides(scoreboard: dict[str, Any], critique: dict[str, Any], config: Any, out: Path) -> dict[str, Any]:
    blocking = list(scoreboard.get("blocking_failures", []))
    critic_needs_review = critique["status"] != "pass"
    if critic_needs_review:
        blocking.append({"skill_id": "critic_agent", "failures": critique["challenges"]})

    status = "promote_to_human_review" if not blocking else "blocked"
    profile = release_profile(config)
    strict_profile = profile != "smoke"
    readiness = (
        "ready"
        if status == "promote_to_human_review" and strict_profile
        else "smoke_review_only"
        if status == "promote_to_human_review"
        else "not_ready"
    )
    decision = {
        "role": "Release Gate",
        "job": "Apply pass/fail policy across all release-blocking skills and critic findings.",
        "status": status,
        "release_profile": profile,
        "review_scope": "release_candidate_review" if strict_profile else "smoke_offline_review",
        "offline_validation_status": scoreboard.get("offline_validation_status", scoreboard["status"]),
        "human_review_readiness": readiness,
        "ready_for_human_review": status == "promote_to_human_review",
        "release_candidate_ready": bool(status == "promote_to_human_review" and strict_profile),
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
        f"- Release profile: {trace['release_profile']}",
        f"- Offline validation: {trace['offline_validation_status']}",
        f"- Human review readiness: {trace['human_review_readiness']}",
        "",
        "## Agent Proposes",
        "",
        f"- Transfer score: {trace['agent_proposes']['transfer_score']['score_0_to_1']}",
        f"- Experiments proposed: {len(trace['agent_proposes']['experiments'])}",
        f"- Authority: {trace['agent_proposes']['authority_limit']}",
        "",
        "## LLM Orchestrator",
        "",
        f"- Provider: {trace['llm_orchestrator']['provider']}",
        f"- Status: {trace['llm_orchestrator']['orchestrator_status']}",
        f"- Skill calls: {trace['llm_orchestrator']['skill_calls']}",
        f"- Journal: `{trace['llm_orchestrator']['journal']}`",
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
