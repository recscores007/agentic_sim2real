from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


UI_SCHEMA_VERSION = "agentic_sim2real.pipeline_ui.v1"

AGENTIC_PROPOSAL_SKILLS = {
    "autoresearch_planner": {
        "role": "Hypothesis and experiment planning",
        "why_agentic": "Chooses what to investigate next from messy evidence.",
    },
    "domain_randomization_update": {
        "role": "Bounded sim-parameter proposal",
        "why_agentic": "Turns measured gaps into candidate DR ranges for validation.",
    },
    "action_scale_sweep": {
        "role": "Bounded control-parameter proposal",
        "why_agentic": "Suggests action-scale candidates from stiction/contact evidence.",
    },
}

NONDETERMINISTIC_COVERAGE = [
    {
        "responsibility": "Gap triage",
        "covered_by": "LLM orchestrator",
        "is_atomic_skill": False,
        "skill_ids": ["real_data_quality_gate", "pose_repeatability", "sysid_step_response"],
        "validation": "Guardrails require known skills and complete dependencies; selected skills produce metrics.",
    },
    {
        "responsibility": "Skill ordering",
        "covered_by": "LLM orchestrator",
        "is_atomic_skill": False,
        "skill_ids": ["llm_orchestrator/journal.jsonl"],
        "validation": "Every decision is journaled and rejected if the skill is unknown, repeated, unsafe, or missing dependencies.",
    },
    {
        "responsibility": "Hypothesis generation",
        "covered_by": "autoresearch_planner",
        "is_atomic_skill": True,
        "skill_ids": ["autoresearch_planner"],
        "validation": "Plan must emit enough experiments and cite measured gap evidence.",
    },
    {
        "responsibility": "Experiment planning",
        "covered_by": "autoresearch_planner",
        "is_atomic_skill": True,
        "skill_ids": ["autoresearch_planner", "newton_sysid", "pace_sysid", "sim_eval_regression"],
        "validation": "Proposed experiments must map to manifest-backed skills or explicit human review.",
    },
    {
        "responsibility": "Candidate parameter proposals",
        "covered_by": "domain_randomization_update, action_scale_sweep",
        "is_atomic_skill": True,
        "skill_ids": ["domain_randomization_update", "action_scale_sweep"],
        "validation": "Candidate ranges are bounded by safety caps, pose gates, friction bounds, and contact limits.",
    },
    {
        "responsibility": "Critic challenge",
        "covered_by": "evaluation_loop critic + regression skills",
        "is_atomic_skill": False,
        "skill_ids": ["sim_eval_regression", "isaaclab_rollout_regression"],
        "validation": "Weak evidence, low confidence, sparse labels, skipped backends, and regressions become warnings or blockers.",
    },
    {
        "responsibility": "Report and explanation",
        "covered_by": "scorecard, pipeline_output, run_record",
        "is_atomic_skill": False,
        "skill_ids": ["scorecard.json", "pipeline_output.json", "run_record.json"],
        "validation": "Narrative output has no release authority; it must point back to generated evidence files.",
    },
]


def write_pipeline_ui(
    run_dir: str | Path,
    *,
    run_status: str = "complete",
    journal_path: str | Path | None = None,
) -> dict[str, str]:
    """Write a static, intuitive UI for one pipeline run.

    The UI is intentionally generated from existing artifacts only. It does not
    recompute scores or make release decisions.
    """

    run = Path(run_dir).expanduser().resolve()
    ui_dir = run / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)

    pipeline_input = _read_json(run / "pipeline_input.json")
    rollout_data = _read_json(run / "rollout_data.json")
    scorecard = _read_json(run / "scorecard.json")
    pipeline_output = _read_json(run / "pipeline_output.json")
    run_record = _read_json(run / "run_record.json")
    scoreboard = _read_json(run / "scoreboard.json")
    trace = _read_json(run / "evaluation_trace.json")
    journal = _read_journal(run, journal_path)

    mode = str(pipeline_input.get("mode") or scorecard.get("mode") or "characterization")
    state = {
        "schema_version": UI_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "dir": str(run),
            "status": run_status,
            "version": run_record.get("run", {}).get("run_version") or scorecard.get("run_version"),
            "safe_to_autorun_robot": False,
        },
        "mode": mode,
        "task": pipeline_input.get("task") or scorecard.get("task") or "Agentic Sim2Real",
        "score_labels": {
            "transfer_readiness_score": "Higher is better. Normalized evidence/readiness score from AutoResearch components.",
            "release_gap_score": "Lower is better. Normalized remaining readiness gap, not a physical sim2real distance.",
            "sim2real_gap": "Backward-compatible alias for release_gap_score.",
        },
        "workflow": _workflow(mode, scoreboard, scorecard),
        "pipeline_input": pipeline_input,
        "rollout_summary": _rollout_summary(rollout_data),
        "scorecard": scorecard,
        "data_readiness": scorecard.get("data_readiness", {}),
        "split_scores": scorecard.get("split_scores", {}),
        "pipeline_output": pipeline_output,
        "run_record": run_record,
        "scoreboard": _scoreboard_summary(scoreboard),
        "characterization": scorecard.get("characterization", {}),
        "policy_release": scorecard.get("policy_release", {}),
        "journal": journal,
        "evaluation_trace": _trace_summary(trace),
    }

    state_path = ui_dir / "state.json"
    index_path = ui_dir / "index.html"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    index_path.write_text(_html(state) + "\n")
    return {"ui": str(index_path), "state": str(state_path)}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_journal(run: Path, journal_path: str | Path | None) -> list[dict[str, Any]]:
    candidates = []
    if journal_path:
        candidates.append(Path(journal_path))
    candidates.extend(
        [
            run / "llm_orchestrator" / "journal.jsonl",
            run / "harness" / "llm_orchestrator" / "journal.jsonl",
        ]
    )
    for candidate in candidates:
        path = candidate.expanduser()
        if not path.is_absolute():
            path = run / path
        if not path.exists():
            continue
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(
                {
                    "step": row.get("step"),
                    "status": row.get("status"),
                    "action": row.get("decision", {}).get("action"),
                    "skill_id": row.get("decision", {}).get("skill_id"),
                    "rationale": row.get("decision", {}).get("rationale"),
                    "guardrail": row.get("guardrail", {}),
                }
            )
        return rows
    return []


def _rollout_summary(rollout_data: dict[str, Any]) -> dict[str, Any]:
    rollouts = rollout_data.get("rollouts", [])
    labeled = [item for item in rollouts if item.get("outcome", {}).get("success") is not None]
    successes = [item for item in labeled if item.get("outcome", {}).get("success") is True]
    streams = sorted({name for item in rollouts for name in (item.get("streams") or {})})
    failures: dict[str, int] = {}
    for item in rollouts:
        mode = item.get("outcome", {}).get("failure_mode")
        if mode:
            failures[str(mode)] = failures.get(str(mode), 0) + 1
    return {
        "dataset": rollout_data.get("dataset"),
        "rollout_count": rollout_data.get("rollout_count", len(rollouts)),
        "streams": streams,
        "success_rate": None if not labeled else round(len(successes) / len(labeled), 3),
        "success_labels": len(labeled),
        "failure_modes": failures,
    }


def _scoreboard_summary(scoreboard: dict[str, Any]) -> dict[str, Any]:
    skills = scoreboard.get("skills", {})
    skill_rows = [
        _skill_row(skill_id, result)
        for skill_id, result in sorted(skills.items())
    ]
    agentic_rows = [
        {
            **row,
            "agentic_role": AGENTIC_PROPOSAL_SKILLS[row["skill_id"]]["role"],
            "why_agentic": AGENTIC_PROPOSAL_SKILLS[row["skill_id"]]["why_agentic"],
        }
        for row in skill_rows
        if row["skill_id"] in AGENTIC_PROPOSAL_SKILLS
    ]
    deterministic_rows = [
        row
        for row in skill_rows
        if row["skill_id"] not in AGENTIC_PROPOSAL_SKILLS
    ]
    recommended_actions = _recommended_actions(scoreboard, skill_rows)
    return {
        "status": scoreboard.get("status"),
        "release_profile": scoreboard.get("release_profile"),
        "review_scope": scoreboard.get("review_scope"),
        "quality_score": scoreboard.get("quality_score"),
        "offline_validation_status": scoreboard.get("offline_validation_status"),
        "human_review_readiness": scoreboard.get("human_review_readiness"),
        "release_candidate_ready": scoreboard.get("release_candidate_ready"),
        "hardware_approval_status": scoreboard.get("hardware_approval_status"),
        "safe_to_autorun_robot": False,
        "blocking_failures": scoreboard.get("blocking_failures", []),
        "recommended_actions": recommended_actions,
        "nondeterministic_coverage": NONDETERMINISTIC_COVERAGE,
        "agentic_proposal_skills": agentic_rows,
        "deterministic_validation_skills": deterministic_rows,
        "skills": skill_rows,
    }


def _skill_row(skill_id: str, result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "skill_id": skill_id,
        "status": result.get("status"),
        "quality_score": result.get("quality_score"),
        "confidence": result.get("confidence"),
        "release_blocking": result.get("release_blocking"),
        "human_required": result.get("human_required"),
        "warnings": result.get("warnings", []),
        "blocking_failures": result.get("blocking_failures", []),
    }
    row.update(_skill_action_guidance(skill_id, result))
    return row


def _skill_action_guidance(skill_id: str, result: dict[str, Any]) -> dict[str, str]:
    status = str(result.get("status") or "pending")
    quality = _float_or_none(result.get("quality_score"))
    confidence = _float_or_none(result.get("confidence"))
    release_blocking = bool(result.get("release_blocking", False))
    warnings = [str(item) for item in result.get("warnings", [])]
    failures = [str(item) for item in result.get("blocking_failures", [])]

    guidance = {
        "action_level": "review",
        "quality_meaning": _quality_meaning(status, quality),
        "confidence_meaning": _confidence_meaning(status, confidence),
        "quality_rationale": _quality_rationale(skill_id, result, status, quality, warnings, failures),
        "confidence_rationale": _confidence_rationale(skill_id, result, status, confidence),
        "pipeline_action": "Hold this result for review.",
        "user_action": "Inspect the skill evidence if this affects your release decision.",
    }

    if status == "fail":
        guidance.update(
            {
                "action_level": "blocked",
                "pipeline_action": "Block release promotion until this skill passes.",
                "user_action": failures[0] if failures else "Fix the failing validator and rerun this skill.",
            }
        )
    elif status == "evidence_missing":
        if skill_id == "newton_sysid":
            guidance.update(
                {
                    "action_level": "configure_backend",
                    "pipeline_action": "Use local SysID fallback in smoke runs; require Newton only when the release profile demands physics SysID.",
                    "user_action": "Set sysid.newton_enabled plus sysid.newton_root or sysid.newton_command to get fitted Newton parameters.",
                }
            )
        elif skill_id == "pace_sysid":
            guidance.update(
                {
                    "action_level": "configure_backend",
                    "pipeline_action": "Keep PACE as backup only; continue with Newton or local SysID when allowed.",
                    "user_action": "Set sysid.pace_enabled plus sysid.pace_root or sysid.pace_command if Newton is unavailable.",
                }
            )
        else:
            guidance.update(
                {
                    "action_level": "missing_evidence",
                    "pipeline_action": "Block release if this skill is release-blocking; otherwise continue with a warning.",
                    "user_action": warnings[0] if warnings else "Provide the missing evidence and rerun.",
                }
            )
    elif status == "not_applicable":
        guidance.update(
            {
                "action_level": "not_applicable",
                "pipeline_action": "Ignore for this run profile; do not count it as positive release evidence.",
                "user_action": "Configure this skill only if you are moving from smoke review to release-candidate review.",
            }
        )
        if skill_id == "isaaclab_rollout_regression":
            guidance["user_action"] = "Provide isaac_lab.rollout_command or isaac_lab.rollout_metrics_path for true policy-release validation."
    elif status == "not_approved":
        guidance.update(
            {
                "action_level": "human_approval_required",
                "pipeline_action": "Stop before any real robot command; safe_to_autorun_robot remains false.",
                "user_action": "Human must review evidence and explicitly approve supervised hardware execution.",
            }
        )
    elif status == "pass":
        if confidence is not None and confidence < 0.5:
            guidance.update(
                {
                    "action_level": "collect_more_data",
                    "pipeline_action": "Treat as weak evidence; do not auto-promote beyond smoke review.",
                    "user_action": "Collect more real records, labels, or repeatability samples, then rerun.",
                }
            )
        elif confidence is not None and confidence <= 0.5:
            guidance.update(
                {
                    "action_level": "minimum_evidence",
                    "pipeline_action": "Proceed through smoke validation, but keep this as a critic observation.",
                    "user_action": "Collect stronger data before release-candidate review.",
                }
            )
        elif quality is not None and quality < 0.75:
            guidance.update(
                {
                    "action_level": "improve_quality",
                    "pipeline_action": "Proceed only if downstream gates accept the evidence.",
                    "user_action": "Improve the evidence quality and rerun if this is intended for release.",
                }
            )
        elif skill_id in AGENTIC_PROPOSAL_SKILLS:
            guidance.update(
                {
                    "action_level": "validate_candidate",
                    "pipeline_action": "Use this as a candidate proposal, then validate with regression and gates.",
                    "user_action": "Review the proposed parameters before using them for training or release.",
                }
            )
        else:
            guidance.update(
                {
                    "action_level": "proceed",
                    "pipeline_action": "Feed this passing evidence to dependent skills and release gates.",
                    "user_action": "No immediate action unless warnings or human review apply.",
                }
            )

    if status == "pass" and warnings:
        guidance["user_action"] = _warning_action(skill_id, warnings, guidance["user_action"])
    if release_blocking and guidance["action_level"] in {"blocked", "missing_evidence", "collect_more_data"}:
        guidance["pipeline_action"] += " This is release-blocking."
    return guidance


def _recommended_actions(scoreboard: dict[str, Any], skill_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if scoreboard.get("status") == "fail":
        actions.append(
            {
                "owner": "Pipeline",
                "priority": "blocked",
                "action": "Do not promote this run.",
                "reason": "One or more release-blocking skills failed or are missing required evidence.",
            }
        )
    elif scoreboard.get("human_review_readiness") == "smoke_review_only":
        actions.append(
            {
                "owner": "Pipeline",
                "priority": "review",
                "action": "Treat this as smoke/offline review only.",
                "reason": "The run is not a release-candidate approval package yet.",
            }
        )

    for row in skill_rows:
        skill_id = str(row["skill_id"])
        status = str(row.get("status"))
        confidence = _float_or_none(row.get("confidence"))
        warnings = [str(item) for item in row.get("warnings", [])]
        if status == "not_approved" and skill_id == "real_robot_gate":
            actions.append(
                {
                    "owner": "Human",
                    "priority": "approval_required",
                    "action": "Approve or reject supervised hardware execution.",
                    "reason": "The pipeline never authorizes unattended robot motion.",
                }
            )
        elif status == "evidence_missing" and skill_id in {"newton_sysid", "pace_sysid"}:
            backend = "Newton" if skill_id == "newton_sysid" else "PACE"
            actions.append(
                {
                    "owner": "User",
                    "priority": "configure_backend",
                    "action": f"Configure {backend} SysID if you need fitted physics parameters.",
                    "reason": f"{skill_id} produced no fitted-parameter evidence in this run.",
                }
            )
        elif status == "not_applicable" and skill_id == "isaaclab_rollout_regression":
            actions.append(
                {
                    "owner": "User",
                    "priority": "provide_rollout",
                    "action": "Provide Isaac Lab rollout metrics or a rollout command for release-candidate validation.",
                    "reason": "Smoke review can skip true rollout regression, but release review should not.",
                }
            )
        elif status == "pass" and confidence is not None and confidence <= 0.5:
            actions.append(
                {
                    "owner": "User",
                    "priority": "collect_more_data",
                    "action": f"Collect stronger evidence for {skill_id}.",
                    "reason": f"Confidence is {confidence:.3f}, which is only at or below the auto-promotion floor.",
                }
            )
        elif skill_id == "real_data_quality_gate" and warnings:
            actions.append(
                {
                    "owner": "User",
                    "priority": "improve_data",
                    "action": "Add missing labels, held-out split, or richer joint/contact data before release review.",
                    "reason": "The real-data gate passed but warned that some evidence is sparse.",
                }
            )
        elif skill_id == "policy_artifact_audit" and warnings:
            actions.append(
                {
                    "owner": "User",
                    "priority": "replace_sample_policy",
                    "action": "Replace sample policy artifacts with the real trained policy bundle.",
                    "reason": "Sample artifacts are acceptable for smoke review, not for a real release package.",
                }
            )
    return actions[:8]


def _quality_meaning(status: str, quality: float | None) -> str:
    if status in {"evidence_missing", "not_applicable", "not_approved"}:
        return "No positive validation evidence was produced for this skill."
    if status == "fail":
        return "The validator failed; quality does not support release."
    if quality is None:
        return "No quality score was reported."
    if quality >= 0.9:
        return "Strong validator result."
    if quality >= 0.75:
        return "Usable validator result with some limitations."
    if quality >= 0.5:
        return "Weak validator result; improve before release."
    return "Poor validator result; do not promote from this evidence."


def _confidence_meaning(status: str, confidence: float | None) -> str:
    if confidence is None:
        return "No evidence-confidence score was reported."
    if status in {"evidence_missing", "not_applicable", "not_approved"}:
        return "This can mean the harness is confident about the skip/block, not that the skill succeeded."
    if confidence >= 0.8:
        return "Strong evidence coverage."
    if confidence >= 0.6:
        return "Moderate evidence coverage."
    if confidence >= 0.5:
        return "Minimum evidence coverage; collect more for release."
    return "Low evidence coverage; treat as exploratory."


def _quality_rationale(
    skill_id: str,
    result: dict[str, Any],
    status: str,
    quality: float | None,
    warnings: list[str],
    failures: list[str],
) -> str:
    metrics = dict(result.get("metrics", {}))
    if status == "fail":
        reason = failures[0] if failures else "the skill reported a failing status"
        return f"Quality is low because the validator failed: {reason}."
    if status == "evidence_missing":
        if skill_id == "newton_sysid":
            return "Quality is 0.0 because Newton SysID did not run; sysid.newton_enabled/root/command is not configured."
        if skill_id == "pace_sysid":
            return "Quality is 0.0 because PACE SysID did not run; sysid.pace_enabled/root/command is not configured."
        return "Quality is 0.0 because required evidence was not produced."
    if status == "not_applicable":
        return "Quality is 0.0 because this skill was intentionally skipped for the current run profile."
    if status == "not_approved":
        return "Quality is 0.0 because this is a human approval gate and hardware was not approved."

    if skill_id == "env_preflight":
        return "Quality is 1.0 when required local preflight checks have no blocking failures; warnings remain informational."
    if skill_id == "isaaclab_task_check":
        return "Quality is 1.0 because the configured Isaac Lab task, observation contract, gripper, and action scale passed validation."
    if skill_id == "policy_artifact_audit":
        completeness = metrics.get("artifact_completeness", quality)
        return f"Quality equals policy artifact completeness; this run found completeness={completeness}."
    if skill_id == "ros_preflight":
        return "Quality is 1.0 because the ROS workflow contract matched the expected gear-assembly deployment settings."
    if skill_id == "pose_repeatability":
        samples = metrics.get("samples")
        p95 = metrics.get("position_error_p95_m")
        return f"Quality is 1.0 because pose repeatability passed: samples={samples}, position_error_p95_m={p95}."
    if skill_id == "real_data_quality_gate":
        return _real_data_quality_rationale(metrics, warnings)
    if skill_id == "sysid_step_response":
        return "Quality is 0.75 because the local log-based SysID gate passed, but it is a lightweight estimator rather than fitted Newton/PACE parameters."
    if skill_id == "domain_randomization_update":
        pos_noise = metrics.get("object_pos_noise")
        friction = metrics.get("friction_sweep")
        return f"Quality is 0.9 because proposed DR ranges stayed inside safety bounds: object_pos_noise={pos_noise}, friction_sweep={friction}."
    if skill_id == "action_scale_sweep":
        suggested = metrics.get("suggested")
        candidates = metrics.get("candidates")
        return f"Quality is 0.85 because the suggested action scale stayed within configured limits: suggested={suggested}, candidates={candidates}."
    if skill_id == "autoresearch_planner":
        count = metrics.get("experiment_count")
        return f"Quality is 0.9 because AutoResearch produced the required experiment plan size: experiment_count={count}."
    if skill_id == "sim_eval_regression":
        delta = _float_or_none(metrics.get("success_delta"))
        if delta is not None:
            return f"Quality is 0.8 plus positive success delta, capped at 1.0; success_delta={delta:.3f} gives quality={quality}."
        return "Quality is based on candidate-vs-baseline regression checks and safety gates."
    if skill_id == "isaaclab_rollout_regression":
        return "Quality is 0.0 because no Isaac Lab rollout metrics or rollout command were configured for this smoke run."
    if skill_id == "release_candidate_gate":
        return "Quality is 0.95 because the aggregate release gate found no blocking failures, while hardware approval remains separate."
    if skill_id == "real_robot_gate":
        return "Quality is 1.0 only after explicit human hardware approval; this run stayed blocked by design."
    if quality is None:
        return "No quality rationale is available because no quality score was reported."
    return f"Quality={quality} was assigned by the skill implementation from its validator result."


def _confidence_rationale(
    skill_id: str,
    result: dict[str, Any],
    status: str,
    confidence: float | None,
) -> str:
    metrics = dict(result.get("metrics", {}))
    if confidence is None:
        return "No confidence rationale is available because no confidence score was reported."
    if status in {"evidence_missing", "not_applicable", "not_approved"}:
        return "Confidence here means the harness is confident about the skip/block decision, not that the skill succeeded."
    if skill_id == "real_data_quality_gate":
        episodes = metrics.get("episodes")
        return f"Confidence is min(1.0, episodes / configured minimum episodes); episodes={episodes}, confidence={confidence}."
    if skill_id == "pose_repeatability":
        samples = metrics.get("samples")
        return f"Confidence is min(1.0, pose_samples / 20); samples={samples}, confidence={confidence}."
    if skill_id == "sysid_step_response":
        return "Confidence is the local SysID evidence confidence floor/max from delay and deadband estimates; this sample run sits at the minimum useful floor."
    if skill_id == "policy_artifact_audit":
        return "Confidence is a fixed 0.8 because file presence is deterministic, but deployability still needs human/pipeline review."
    if skill_id == "ros_preflight":
        return "Confidence is a fixed 0.9 because config checks are deterministic but runtime ROS sourcing can still differ."
    if skill_id in {"action_scale_sweep", "domain_randomization_update", "autoresearch_planner", "sim_eval_regression"}:
        return f"Confidence is a conservative heuristic assigned by this proposal/regression skill after bounded checks passed: confidence={confidence}."
    if skill_id == "isaaclab_rollout_regression":
        episodes = metrics.get("episodes") or metrics.get("num_episodes") or metrics.get("rollout_episodes")
        return f"When rollout metrics exist, confidence is rollout episodes divided by the configured minimum; this run reported episodes={episodes}."
    if skill_id in {"env_preflight", "isaaclab_task_check", "release_candidate_gate", "real_robot_gate"}:
        return "Confidence is 1.0 because this is a deterministic gate over explicit config/evidence."
    return f"Confidence={confidence} was assigned by the skill implementation from available evidence coverage."


def _real_data_quality_rationale(metrics: dict[str, Any], warnings: list[str]) -> str:
    readiness = dict(metrics.get("data_readiness", {}))
    parts = [
        "Quality starts at 1.0.",
        f"records={metrics.get('records')}",
        f"episodes={metrics.get('episodes')}",
        f"contact_coverage={metrics.get('contact_coverage')}",
        f"success_label_coverage={metrics.get('success_label_coverage')}",
        f"joint_velocity_coverage={metrics.get('joint_velocity_coverage')}",
    ]
    if readiness:
        parts.extend(
            [
                f"frame_link_coverage={readiness.get('frame_link_coverage')}",
                f"delay_observability={readiness.get('delay_observability_status')}",
                f"pose_validation={readiness.get('pose_validation', {}).get('validation_source')}",
            ]
        )
    if warnings:
        parts.append(f"warnings={len(warnings)} reduce the score.")
    return " ".join(parts)


def _warning_action(skill_id: str, warnings: list[str], fallback: str) -> str:
    if skill_id == "real_data_quality_gate":
        return "Resolve data warnings: add success labels, held-out episodes, joint velocities, calibration, or camera provenance as applicable."
    if skill_id == "policy_artifact_audit":
        return "Replace sample policy artifacts with the real trained policy bundle before release-candidate review."
    return warnings[0] if warnings else fallback


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    if not trace:
        return {}
    return {
        "agent_proposes": trace.get("agent_proposes", {}).get("status", "written"),
        "evaluator_measures": trace.get("evaluator_measures", {}).get("status"),
        "critic_challenges": trace.get("critic_challenges", {}).get("status"),
        "release_gate_decides": trace.get("release_gate_decides", {}).get("status"),
        "human_approves_hardware": trace.get("human_approves_hardware", {}).get("status"),
    }


def _workflow(mode: str, scoreboard: dict[str, Any], scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    validation_status = scoreboard.get("offline_validation_status") or scoreboard.get("status") or "pending"
    human_status = scoreboard.get("hardware_approval_status") or "not_requested"
    release_ready = bool(scoreboard.get("release_candidate_ready"))
    characterization_status = "active" if mode == "characterization" else "complete"
    return [
        {
            "stage": "Real data submitted",
            "owner": "Human",
            "status": "complete" if scorecard else "pending",
            "artifact": "rollout_data.json",
        },
        {
            "stage": "Characterize gaps",
            "owner": "Agent + Evaluator",
            "status": characterization_status,
            "artifact": "scorecard.characterization",
        },
        {
            "stage": "Tune sim parameters",
            "owner": "Agent proposes, Evaluator validates",
            "status": validation_status,
            "artifact": "pipeline_output.changes",
        },
        {
            "stage": "Validate policy release",
            "owner": "Release gate",
            "status": "ready" if release_ready else "not_ready",
            "artifact": "scorecard.policy_release",
        },
        {
            "stage": "Approve hardware",
            "owner": "Human",
            "status": human_status,
            "artifact": "human hardware gate",
        },
    ]


def _html(state: dict[str, Any]) -> str:
    title = escape(str(state.get("task") or "Agentic Sim2Real"))
    state_json = json.dumps(state, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} Pipeline UI</title>
<style>
:root {{
  --ink:#172026;
  --muted:#5a6573;
  --line:#d9e0e8;
  --paper:#ffffff;
  --bg:#f5f7fa;
  --green:#6bae2e;
  --blue:#2c7be5;
  --amber:#f0a429;
  --coral:#e75f51;
  --violet:#7457d5;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Arial, Helvetica, sans-serif; font-size:14px; letter-spacing:0; }}
header {{ background:var(--paper); border-bottom:1px solid var(--line); padding:18px 22px; }}
h1 {{ margin:0; font-size:24px; line-height:1.2; }}
h2 {{ margin:0 0 12px; font-size:16px; }}
h3 {{ margin:0 0 8px; font-size:14px; color:var(--muted); }}
main {{ padding:18px; display:grid; grid-template-columns:1.15fr .85fr; gap:14px; max-width:1440px; margin:0 auto; }}
.topline {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; align-items:center; }}
.pill {{ display:inline-flex; align-items:center; min-height:24px; padding:4px 9px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid transparent; background:#eef2f7; color:#25384d; }}
.agent {{ background:#e8f1ff; color:#174a8b; border-color:#b7d5ff; }}
.human {{ background:#fff2d6; color:#764b00; border-color:#ffd37b; }}
.evaluator {{ background:#ecf8df; color:#325d00; border-color:#bfe59e; }}
.gate {{ background:#f4eefe; color:#4b2b92; border-color:#d8c7ff; }}
.blocked,.fail,.not_ready,.not_approved,.evidence_missing {{ background:#fdebe9; color:#8a261d; border-color:#f5bbb5; }}
.pass,.complete,.ready,.active,.promote_to_human_review {{ background:#e9f7df; color:#2c6100; border-color:#bee89a; }}
.pending,.smoke_review_only,.not_requested {{ background:#fff4dc; color:#765000; border-color:#ffd88a; }}
.panel {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }}
.wide {{ grid-column:1 / -1; }}
.metrics {{ display:grid; grid-template-columns:repeat(4, minmax(150px, 1fr)); gap:10px; }}
.metric {{ border:1px solid var(--line); border-top:4px solid var(--blue); border-radius:8px; padding:12px; min-height:88px; background:#fff; }}
.metric:nth-child(2) {{ border-top-color:var(--green); }}
.metric:nth-child(3) {{ border-top-color:var(--amber); }}
.metric:nth-child(4) {{ border-top-color:var(--coral); }}
.metric span {{ color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; }}
.metric b {{ display:block; margin-top:8px; font-size:24px; line-height:1.1; }}
.workflow {{ display:grid; grid-template-columns:repeat(5, minmax(145px, 1fr)); gap:10px; }}
.stage {{ border:1px solid var(--line); border-radius:8px; padding:12px; min-height:120px; background:#fff; position:relative; overflow:hidden; }}
.stage::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--blue); }}
.stage:nth-child(2)::before {{ background:var(--green); }}
.stage:nth-child(3)::before {{ background:var(--amber); }}
.stage:nth-child(4)::before {{ background:var(--violet); }}
.stage:nth-child(5)::before {{ background:var(--coral); }}
.stage-title {{ font-weight:700; margin-bottom:8px; padding-left:4px; }}
.lane-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.subgrid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }}
.kv {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; min-height:72px; }}
.kv label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }}
.kv strong {{ font-size:18px; overflow-wrap:anywhere; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px; vertical-align:top; }}
th {{ color:var(--muted); font-size:12px; }}
code {{ font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }}
.journal {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:8px; }}
.journal-item {{ border:1px solid var(--line); border-radius:8px; padding:9px; background:#fff; min-height:74px; }}
.action-grid {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px; }}
.action-card {{ border:1px solid var(--line); border-radius:8px; padding:11px; background:#fff; min-height:118px; }}
.action-card b {{ display:block; margin:8px 0 6px; }}
.table-wrap {{ overflow-x:auto; }}
.note {{ color:var(--muted); line-height:1.45; }}
@media (max-width:1050px) {{
  main,.lane-grid {{ grid-template-columns:1fr; }}
  .metrics,.workflow,.journal,.action-grid {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width:640px) {{
  header {{ padding:14px; }}
  main {{ padding:12px; }}
  .metrics,.workflow,.journal,.subgrid,.action-grid {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="topline" id="header-pills"></div>
</header>
<main id="app"></main>
<script>
const state = {state_json};
const fmt = (v) => v === null || v === undefined || v === "" ? "n/a" : String(v);
const pct = (v) => v === null || v === undefined ? "n/a" : `${{Math.round(Number(v) * 100)}}%`;
const num = (v) => v === null || v === undefined ? "n/a" : Number(v).toFixed(3).replace(/\\.000$/, ".0");
const cls = (v) => String(v || "pending").replace(/[^a-zA-Z0-9_-]/g, "_");
const pill = (text, role="") => `<span class="pill ${{cls(text)}} ${{role}}">${{fmt(text)}}</span>`;
const rolePill = (owner) => {{
  const lower = String(owner || "").toLowerCase();
  if (lower.includes("human")) return pill(owner, "human");
  if (lower.includes("gate")) return pill(owner, "gate");
  if (lower.includes("evaluator")) return pill(owner, "evaluator");
  return pill(owner || "Agent", "agent");
}};
const score = state.scorecard || {{}};
const runRecord = state.run_record || {{}};
const lineage = runRecord.lineage || {{}};
const input = state.pipeline_input || {{}};
const out = state.pipeline_output || {{}};
const rollout = state.rollout_summary || {{}};
const board = state.scoreboard || {{}};
const char = state.characterization || {{}};
const policy = state.policy_release || {{}};
const readiness = state.data_readiness || {{}};
const splitScores = state.split_scores || {{}};
const gap = score.release_gap_score ?? score.sim2real_gap;
document.getElementById("header-pills").innerHTML = [
  pill(state.mode),
  pill(state.run.version || "unversioned"),
  pill(board.release_profile || "profile n/a"),
  pill(`offline ${{board.offline_validation_status || board.status || "pending"}}`),
  pill(`hardware ${{board.hardware_approval_status || "not_requested"}}`),
  pill("safe_to_autorun_robot=false", "human")
].join("");

const metricHtml = `
<section class="panel wide">
  <div class="metrics">
    <div class="metric"><span>Transfer readiness</span><b>${{num(score.transfer_readiness_score)}}</b><div class="note">higher is better</div></div>
    <div class="metric"><span>Release gap score</span><b>${{num(gap)}}</b><div class="note">target ${{fmt(score.release_gap_target ?? input.goal?.release_gap_target)}} from config</div></div>
    <div class="metric"><span>Real data</span><b>${{fmt(rollout.rollout_count)}} rollouts</b><div class="note">frames ${{pct(readiness.frame_link_coverage)}}; heldout ${{pct(readiness.heldout_frame_link_coverage)}}</div></div>
    <div class="metric"><span>Run version</span><b>${{fmt(state.run.version)}}</b><div class="note">versioned audit record</div></div>
  </div>
</section>`;

const actionCards = (board.recommended_actions || []).map(item => `<div class="action-card">
  <div>${{pill(item.owner || "Pipeline")}} ${{pill(item.priority || "review")}}</div>
  <b>${{fmt(item.action)}}</b>
  <div class="note">${{fmt(item.reason)}}</div>
</div>`).join("");

const actionHtml = `
<section class="panel wide">
  <h2>Recommended Actions</h2>
  <div class="action-grid">${{actionCards || "<span class='note'>No immediate action. Keep evidence with the run record.</span>"}}</div>
</section>`;

const readinessCards = (readiness.action_items || []).map(item => `<div class="action-card">
  <div>${{pill(item.owner || "Pipeline")}} ${{pill(item.severity || "warning")}}</div>
  <b>${{fmt(item.message)}}</b>
  <div class="note"><b>Impact:</b> ${{fmt(item.impact)}}<br><b>Fix:</b> ${{fmt(item.recommended_fix)}}</div>
</div>`).join("");

const readinessHtml = `
<section class="panel wide">
  <h2>Data Readiness</h2>
  <div class="subgrid">
    <div class="kv"><label>Status</label><strong>${{fmt(readiness.status)}}</strong></div>
    <div class="kv"><label>Frame links</label><strong>${{pct(readiness.frame_link_coverage)}}</strong></div>
    <div class="kv"><label>Heldout frame links</label><strong>${{pct(readiness.heldout_frame_link_coverage)}}</strong></div>
    <div class="kv"><label>Orphan frames</label><strong>${{fmt(readiness.orphan_frame_count)}}</strong></div>
    <div class="kv"><label>Delay observability</label><strong>${{fmt(readiness.delay_observability_status)}}</strong></div>
    <div class="kv"><label>Pose source</label><strong>${{fmt(readiness.pose_validation?.validation_source)}}</strong></div>
  </div>
  <h3 style="margin-top:14px">Early Data Alerts</h3>
  <div class="action-grid">${{readinessCards || "<span class='note'>No data-readiness alerts for this run.</span>"}}</div>
</section>`;

const workflowHtml = `
<section class="panel wide">
  <h2>Pipeline Flow</h2>
  <div class="workflow">
    ${{(state.workflow || []).map(s => `<div class="stage">
      <div class="stage-title">${{fmt(s.stage)}}</div>
      <div>${{rolePill(s.owner)}}</div>
      <div style="margin-top:8px">${{pill(s.status)}}</div>
      <div class="note" style="margin-top:8px"><code>${{fmt(s.artifact)}}</code></div>
    </div>`).join("")}}
  </div>
</section>`;

const charHtml = `
<section class="panel">
  <h2>Characterization Lane</h2>
  <div class="subgrid">
    <div class="kv"><label>Trajectory records</label><strong>${{fmt(char.trajectory_data?.records)}}</strong></div>
    <div class="kv"><label>Estimated rate</label><strong>${{fmt(char.trajectory_data?.estimated_rate_hz)}} Hz</strong></div>
    <div class="kv"><label>Delay</label><strong>${{fmt(char.actuator_latency?.delay_steps)}} steps</strong><div class="note">${{fmt(char.actuator_latency?.observability_status)}}</div></div>
    <div class="kv"><label>Deadband proxy</label><strong>${{num(char.actuator_latency?.deadband_command_norm)}}</strong></div>
    <div class="kv"><label>Pose p95 error</label><strong>${{fmt(char.camera_pose_noise?.position_error_p95_m)}} m</strong><div class="note">${{fmt(char.camera_pose_noise?.validation_source)}}</div></div>
    <div class="kv"><label>Contact over limit</label><strong>${{pct(char.contact?.over_limit_ratio)}}</strong></div>
  </div>
  <p class="note">Used before policy training to fit actuator/friction/latency, camera pose noise, contact limits, and domain-randomization ranges.</p>
</section>`;

const releaseHtml = `
<section class="panel">
  <h2>Policy Release Lane</h2>
  <div class="subgrid">
    <div class="kv"><label>Sim success</label><strong>${{pct(policy.success?.sim)}}</strong></div>
    <div class="kv"><label>Real success</label><strong>${{pct(policy.success?.real)}}</strong></div>
    <div class="kv"><label>Target success</label><strong>${{pct(policy.success?.real_target)}}</strong></div>
    <div class="kv"><label>Release gate</label><strong>${{fmt(policy.release_gate_status || board.status)}}</strong></div>
    <div class="kv"><label>Human review</label><strong>${{fmt(policy.human_review_readiness || board.human_review_readiness)}}</strong></div>
    <div class="kv"><label>Hardware approval</label><strong>${{fmt(policy.hardware_approval_status || board.hardware_approval_status)}}</strong></div>
  </div>
  <p class="note">Used after a candidate policy exists. Human approval is still required for supervised hardware motion.</p>
</section>`;

const recordHtml = `
<section class="panel wide">
  <h2>Versioned Run Record</h2>
  <div class="subgrid">
    <div class="kv"><label>Real-data source</label><strong>${{fmt(lineage.real_data_fed?.source_path)}}</strong></div>
    <div class="kv"><label>Real-data SHA256</label><strong>${{fmt(lineage.real_data_fed?.source_sha256)}}</strong></div>
    <div class="kv"><label>Files recorded</label><strong>${{fmt(lineage.real_data_fed?.file_count)}}</strong></div>
    <div class="kv"><label>Policy checkpoint</label><strong>${{fmt(lineage.policy_checkpoint?.configured)}}</strong></div>
    <div class="kv"><label>Policy SHA256</label><strong>${{fmt(lineage.policy_checkpoint?.sha256)}}</strong></div>
    <div class="kv"><label>Retraining requested</label><strong>${{fmt(lineage.retraining?.requested)}}</strong></div>
  </div>
  <p class="note"><code>run_record.json</code> links this run version to the exact real-data manifest, policy checkpoint, score breakdowns, skill evidence, and release gates.</p>
</section>`;

const splitHtml = `
<section class="panel wide">
  <h2>Train vs Heldout Score</h2>
  <div class="subgrid">
    <div class="kv"><label>Train transfer readiness</label><strong>${{num(splitScores.train?.transfer_readiness_score)}}</strong><div class="note">${{fmt(splitScores.train?.records)}} records</div></div>
    <div class="kv"><label>Heldout transfer readiness</label><strong>${{num(splitScores.heldout?.transfer_readiness_score)}}</strong><div class="note">${{fmt(splitScores.heldout?.records)}} records</div></div>
    <div class="kv"><label>Heldout - train</label><strong>${{num(splitScores.heldout_vs_train_delta)}}</strong></div>
    <div class="kv"><label>Score policy</label><strong>${{fmt(score.transfer_readiness_breakdown?.formula)}}</strong></div>
  </div>
</section>`;

const coverageRows = (board.nondeterministic_coverage || []).map(row => `<tr>
  <td>${{fmt(row.responsibility)}}</td>
  <td>${{fmt(row.covered_by)}}</td>
  <td>${{row.is_atomic_skill ? pill("atomic skill", "agent") : pill("agent role", "gate")}}</td>
  <td><code>${{fmt((row.skill_ids || []).join(", "))}}</code></td>
  <td>${{fmt(row.validation)}}</td>
</tr>`).join("");

const proposalSkills = (board.agentic_proposal_skills || []).map(row => `<tr>
  <td><code>${{row.skill_id}}</code></td>
  <td>${{pill(row.status)}}</td>
  <td>${{pill(row.action_level)}}<div class="note">${{fmt(row.agentic_role)}}</div></td>
  <td>${{num(row.quality_score)}}<div class="note">${{fmt(row.quality_meaning)}}</div></td>
  <td>${{num(row.confidence)}}<div class="note">${{fmt(row.confidence_meaning)}}</div></td>
  <td><b>Q:</b> ${{fmt(row.quality_rationale)}}<br><b>C:</b> ${{fmt(row.confidence_rationale)}}</td>
  <td>${{fmt(row.pipeline_action)}}</td>
  <td>${{fmt(row.user_action)}}</td>
</tr>`).join("");

const deterministicSkills = (board.deterministic_validation_skills || []).map(row => `<tr>
  <td><code>${{row.skill_id}}</code></td>
  <td>${{pill(row.status)}}</td>
  <td>${{pill(row.action_level)}}<div class="note">${{row.release_blocking ? "release-blocking" : "non-blocking"}}; ${{row.human_required ? "human review" : "pipeline-owned"}}</div></td>
  <td>${{num(row.quality_score)}}<div class="note">${{fmt(row.quality_meaning)}}</div></td>
  <td>${{num(row.confidence)}}<div class="note">${{fmt(row.confidence_meaning)}}</div></td>
  <td><b>Q:</b> ${{fmt(row.quality_rationale)}}<br><b>C:</b> ${{fmt(row.confidence_rationale)}}</td>
  <td>${{fmt(row.pipeline_action)}}</td>
  <td>${{fmt(row.user_action)}}</td>
</tr>`).join("");

const nondetHtml = `
<section class="panel wide">
  <h2>Non-Deterministic Coverage</h2>
  <p class="note">The LLM-style part proposes, orders, critiques, or explains. The harness still validates through atomic evidence.</p>
  <div class="table-wrap"><table><thead><tr><th>Responsibility</th><th>Covered by</th><th>Type</th><th>Related skill/artifact</th><th>Validation boundary</th></tr></thead><tbody>${{coverageRows}}</tbody></table></div>
  <h3 style="margin-top:14px">Proposal-Producing Atomic Skills</h3>
  <div class="table-wrap"><table><thead><tr><th>Skill</th><th>Status</th><th>Action level</th><th>Quality</th><th>Confidence</th><th>Why assigned</th><th>Pipeline action</th><th>User action</th></tr></thead><tbody>${{proposalSkills || "<tr><td colspan='8' class='note'>No proposal-producing skills ran in this run.</td></tr>"}}</tbody></table></div>
</section>`;

const deterministicHtml = `
<section class="panel wide">
  <h2>Deterministic Validation Skills</h2>
  <div class="table-wrap"><table><thead><tr><th>Skill</th><th>Status</th><th>Action level</th><th>Quality</th><th>Confidence</th><th>Why assigned</th><th>Pipeline action</th><th>User action</th></tr></thead><tbody>${{deterministicSkills}}</tbody></table></div>
</section>`;

const journal = (state.journal || []).slice(-9).map(row => `<div class="journal-item">
  <b>#${{fmt(row.step)}} ${{fmt(row.skill_id || row.action)}}</b>
  <div style="margin-top:6px">${{pill(row.status)}}</div>
  <div class="note" style="margin-top:6px">${{fmt(row.rationale).slice(0, 120)}}</div>
</div>`).join("");

document.getElementById("app").innerHTML = [
  metricHtml,
  actionHtml,
  readinessHtml,
  workflowHtml,
  `<div class="lane-grid">${{charHtml}}${{releaseHtml}}</div>`,
  splitHtml,
  recordHtml,
  nondetHtml,
  deterministicHtml,
  `<section class="panel wide"><h2>LLM Orchestrator Journal</h2><div class="journal">${{journal || "<span class='note'>No LLM journal found for this run.</span>"}}</div></section>`,
  `<section class="panel wide"><h2>Score Meaning</h2><p class="note">${{fmt(score.score_meaning?.release_gap_score)}} ${{fmt(score.score_meaning?.target_source)}} Formula: ${{fmt(score.score_meaning?.formula)}}</p></section>`
].join("");
</script>
</body>
</html>"""
