from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import write_scorecard_artifacts
from .config import PipelineConfig
from .run_ui import write_pipeline_ui
from .skill_harness import (
    HarnessContext,
    SkillManifest,
    SkillResult,
    ordered_skill_ids,
    prepare_harness,
    run_skill_step,
    write_harness_artifacts,
)


VALID_ACTIONS = {"run_skill", "stop", "request_human_review"}
FOUNDATION_SKILLS = ["project_preflight", "real_data_evidence_gate"]
GAP_HINT_PRIORITY: dict[str, list[str]] = {
    "perception": [
        "project_preflight",
        "real_data_evidence_gate",
        "agentic_tuning_plan",
        "regression_evaluation",
    ],
    "actuator": [
        "project_preflight",
        "real_data_evidence_gate",
        "physics_sysid",
        "agentic_tuning_plan",
        "regression_evaluation",
    ],
    "contact": [
        "project_preflight",
        "real_data_evidence_gate",
        "physics_sysid",
        "agentic_tuning_plan",
        "regression_evaluation",
    ],
    "latency": [
        "project_preflight",
        "real_data_evidence_gate",
        "physics_sysid",
        "agentic_tuning_plan",
        "regression_evaluation",
    ],
    "domain_randomization": [
        "project_preflight",
        "real_data_evidence_gate",
        "physics_sysid",
        "agentic_tuning_plan",
        "regression_evaluation",
    ],
    "deployment": [
        "project_preflight",
        "regression_evaluation",
    ],
    "policy": [
        "project_preflight",
        "real_data_evidence_gate",
        "regression_evaluation",
        "agentic_tuning_plan",
    ],
}
GAP_HINT_ALIASES = {
    "vision": "perception",
    "camera": "perception",
    "pose": "perception",
    "shaft_pose": "perception",
    "foundationpose": "perception",
    "video": "perception",
    "reprojection": "perception",
    "intrinsics": "perception",
    "sysid": "actuator",
    "dynamics": "actuator",
    "stiction": "actuator",
    "joint": "actuator",
    "friction": "contact",
    "gripper": "contact",
    "slip": "contact",
    "force": "contact",
    "insertion": "contact",
    "jam": "contact",
    "jamming": "contact",
    "delay": "latency",
    "dr": "domain_randomization",
    "randomization": "domain_randomization",
    "sim_params": "domain_randomization",
    "ros": "deployment",
    "middleware": "deployment",
    "deploy": "deployment",
    "checkpoint": "policy",
    "regression": "policy",
    "success": "policy",
}


@dataclass(frozen=True)
class OrchestratorDecision:
    action: str
    skill_id: str | None = None
    rationale: str = ""
    expected_evidence: list[str] = field(default_factory=list)
    risk_checks: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "skill_id": self.skill_id,
            "rationale": self.rationale,
            "expected_evidence": self.expected_evidence,
            "risk_checks": self.risk_checks,
            "confidence": round(float(self.confidence), 3),
            "raw": self.raw,
        }


class LLMProvider:
    """Decision provider for the orchestration loop.

    Production deployments should use a provider backed by an LLM tool or
    service. Tests and goldens use ScriptedLLMProvider so the evaluator stays
    deterministic.
    """

    name = "base"

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class ScriptedLLMProvider(LLMProvider):
    """Deterministic LLM-shaped provider used for CI and offline goldens."""

    name = "scripted"

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        runnable = context.get("runnable_skills", [])
        completed = set(context.get("completed_skill_ids", []))
        if runnable:
            skill_id = _prioritized_runnable_skill([str(item) for item in runnable], context)
            skill = context["skills"][skill_id]
            hint_text = _hint_rationale(context)
            return {
                "action": "run_skill",
                "skill_id": skill_id,
                "rationale": f"Run next valid skill from the LLM-visible catalog: {skill['name']}.{hint_text}",
                "expected_evidence": skill.get("outputs", []),
                "risk_checks": skill.get("validators", []),
                "confidence": 0.95,
            }
        if "release_candidate_gate" in completed:
            return {
                "action": "request_human_review",
                "rationale": "All runnable validation skills have completed and release evidence is available.",
                "expected_evidence": ["release_candidate.json", "scoreboard.json"],
                "risk_checks": ["safe_to_autorun_robot must remain false"],
                "confidence": 0.95,
            }
        return {
            "action": "stop",
            "rationale": "No runnable skill remains.",
            "expected_evidence": ["orchestrator_summary.json"],
            "risk_checks": ["release-blocking gaps will be caught by final scoring"],
            "confidence": 0.7,
        }


class CommandLLMProvider(LLMProvider):
    """External LLM adapter.

    The command receives the current orchestration context as JSON and must
    return a decision JSON with action, skill_id, rationale, and confidence. The
    command can use any LLM stack: local model, hosted API, notebook service, or
    an internal agent runner.
    """

    name = "command"

    def __init__(self, command: list[str], work_dir: str | Path, timeout_s: float = 120.0) -> None:
        if not command:
            raise ValueError("command LLM provider requires a non-empty command list")
        self.command = [str(item) for item in command]
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.timeout_s = float(timeout_s)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        step = int(context["step"])
        input_path = self.work_dir / f"step_{step:03d}_llm_input.json"
        output_path = self.work_dir / f"step_{step:03d}_llm_decision.json"
        input_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
        values = {"input": str(input_path), "output": str(output_path), "step": str(step)}
        command = [item.format(**values) for item in self.command]
        env = os.environ.copy()
        env.update(
            {
                "AGENTIC_SIM2REAL_LLM_INPUT_JSON": str(input_path),
                "AGENTIC_SIM2REAL_LLM_OUTPUT_JSON": str(output_path),
                "AGENTIC_SIM2REAL_LLM_STEP": str(step),
            }
        )
        completed = subprocess.run(
            command,
            cwd=self.work_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        log_path = self.work_dir / f"step_{step:03d}_llm_command_log.json"
        log_path.write_text(
            json.dumps(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "input": str(input_path),
                    "output": str(output_path),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if completed.returncode != 0:
            raise RuntimeError(f"LLM command exited {completed.returncode}; see {log_path}")
        if output_path.exists():
            return json.loads(output_path.read_text())
        if completed.stdout.strip():
            return json.loads(completed.stdout)
        raise RuntimeError("LLM command did not write a decision JSON or JSON stdout")


def provider_from_config(
    config: PipelineConfig,
    work_dir: str | Path,
    provider_name: str | None = None,
    command: list[str] | None = None,
) -> LLMProvider:
    llm_cfg = config.llm_orchestrator
    selected = str(provider_name or llm_cfg.get("provider", "scripted")).strip().lower()
    if selected in {"scripted", "mock", "deterministic"}:
        return ScriptedLLMProvider()
    if selected == "command":
        configured_command = command or [str(item) for item in llm_cfg.get("command", [])]
        return CommandLLMProvider(
            configured_command,
            work_dir=work_dir,
            timeout_s=float(llm_cfg.get("timeout_s", 120.0)),
        )
    raise ValueError(f"Unknown LLM orchestrator provider: {selected}")


def run_llm_orchestrated_loop(
    root: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path,
    include_real: bool = False,
    skill_dirs: list[str | Path] | None = None,
    provider: LLMProvider | None = None,
    provider_name: str | None = None,
    provider_command: list[str] | None = None,
    max_steps: int | None = None,
    gap_hints: list[str] | None = None,
    audience: str | None = None,
) -> dict[str, Any]:
    ctx, manifests = prepare_harness(
        root=root,
        config_path=config_path,
        dataset_path=dataset_path,
        out_dir=out_dir,
        include_real=include_real,
        skill_dirs=skill_dirs,
        audience=audience,
    )
    orchestrator_dir = ctx.out_dir / "llm_orchestrator"
    steps_dir = orchestrator_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    llm = provider or provider_from_config(ctx.config, work_dir=steps_dir, provider_name=provider_name, command=provider_command)

    llm_cfg = ctx.config.llm_orchestrator
    initial_gap_hints = normalize_gap_hints([*_coerce_gap_hints(llm_cfg.get("gap_hints", [])), *(gap_hints or [])])
    step_limit = int(max_steps or llm_cfg.get("max_steps", 32))
    invalid_limit = int(llm_cfg.get("max_invalid_decisions", 3))
    budget_skill_calls = int(llm_cfg.get("budget_skill_calls", step_limit))
    allow_retries = bool(llm_cfg.get("allow_retries", False))
    ordered = ordered_skill_ids(manifests)

    results: dict[str, SkillResult] = {}
    journal: list[dict[str, Any]] = []
    invalid_decisions = 0
    stop_reason = "max_steps_exhausted"
    previous_scorecard: dict[str, Any] | None = None

    for step in range(1, step_limit + 1):
        if len(results) >= budget_skill_calls:
            stop_reason = "budget_skill_calls_exhausted"
            break

        context = _orchestration_context(ctx, manifests, ordered, results, step, llm.name, initial_gap_hints)
        context_path = steps_dir / f"step_{step:03d}_context.json"
        context_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")

        try:
            raw_decision = llm.decide(context)
            decision = normalize_decision(raw_decision)
            provider_error = ""
        except Exception as exc:
            raw_decision = {"action": "stop", "rationale": f"LLM provider failed: {exc}", "confidence": 0.0}
            decision = normalize_decision(raw_decision)
            provider_error = str(exc)

        guardrail = validate_decision(decision, manifests, results, ctx.include_real, allow_retries)
        entry: dict[str, Any] = {
            "step": step,
            "provider": llm.name,
            "context_file": str(context_path),
            "decision": decision.to_dict(),
            "guardrail": guardrail,
        }
        if provider_error:
            entry["provider_error"] = provider_error

        if not guardrail["accepted"]:
            invalid_decisions += 1
            entry["status"] = "rejected"
            journal.append(entry)
            _write_step_files(steps_dir, step, entry)
            if invalid_decisions > invalid_limit:
                stop_reason = "too_many_invalid_llm_decisions"
                break
            continue

        invalid_decisions = 0
        if decision.action == "run_skill" and decision.skill_id:
            result = run_skill_step(manifests, ctx, decision.skill_id, results)
            results[decision.skill_id] = result
            entry["status"] = "skill_completed"
            entry["skill_result"] = result.to_dict()
            step_scorecard_dir = orchestrator_dir / "scorecards" / f"step_{step:03d}_{decision.skill_id}"
            step_scoreboard = _iteration_scoreboard(results)
            scorecard_paths = write_scorecard_artifacts(
                ctx.dataset,
                ctx.config,
                step_scorecard_dir,
                results,
                step_scoreboard,
                config_path=ctx.config_path,
                run_id=f"step_{step:03d}_{decision.skill_id}",
                previous_scorecard=previous_scorecard,
            )
            entry["scorecard"] = scorecard_paths
            previous_scorecard_path = Path(scorecard_paths["scorecard"])
            if previous_scorecard_path.exists():
                previous_scorecard = json.loads(previous_scorecard_path.read_text())
        elif decision.action == "request_human_review":
            entry["status"] = "human_review_requested"
            stop_reason = "human_review_requested"
            journal.append(entry)
            _write_step_files(steps_dir, step, entry)
            break
        else:
            entry["status"] = "stopped"
            stop_reason = "llm_stopped"
            journal.append(entry)
            _write_step_files(steps_dir, step, entry)
            break

        journal.append(entry)
        _write_step_files(steps_dir, step, entry)

    journal_path = orchestrator_dir / "journal.jsonl"
    journal_path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in journal) + ("\n" if journal else ""))
    scoreboard = write_harness_artifacts(
        ctx,
        manifests,
        results,
        require_all_release_blocking=True,
    )
    ui_artifacts = write_pipeline_ui(
        ctx.out_dir,
        run_status="complete",
        journal_path=journal_path,
        audience=ctx.config.ui.get("audience"),
    )
    summary = {
        "status": "pass" if scoreboard["status"] == "pass" and stop_reason in {"human_review_requested", "llm_stopped"} else "fail",
        "orchestrator_status": stop_reason,
        "provider": llm.name,
        "steps": len(journal),
        "skill_calls": len(results),
        "completed_skill_ids": list(results),
        "gap_hints": initial_gap_hints,
        "journal": str(journal_path),
        "scoreboard": scoreboard,
        "ui_artifacts": ui_artifacts,
    }
    (orchestrator_dir / "orchestrator_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def normalize_decision(payload: dict[str, Any]) -> OrchestratorDecision:
    raw = dict(payload)
    action = str(payload.get("action", "stop")).strip().lower()
    expected = payload.get("expected_evidence", [])
    risks = payload.get("risk_checks", [])
    if isinstance(expected, str):
        expected = [expected]
    if isinstance(risks, str):
        risks = [risks]
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return OrchestratorDecision(
        action=action,
        skill_id=str(payload["skill_id"]) if payload.get("skill_id") else None,
        rationale=str(payload.get("rationale", "")),
        expected_evidence=[str(item) for item in expected],
        risk_checks=[str(item) for item in risks],
        confidence=max(0.0, min(1.0, confidence)),
        raw=raw,
    )


def validate_decision(
    decision: OrchestratorDecision,
    manifests: dict[str, SkillManifest],
    results: dict[str, SkillResult],
    include_real: bool,
    allow_retries: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if decision.action not in VALID_ACTIONS:
        reasons.append(f"unsupported action: {decision.action}")
    if decision.action == "run_skill":
        if not decision.skill_id:
            reasons.append("run_skill decision must include skill_id")
        elif decision.skill_id not in manifests:
            reasons.append(f"unknown skill_id: {decision.skill_id}")
        else:
            manifest = manifests[decision.skill_id]
            missing_deps = [dep for dep in manifest.depends_on if dep not in results]
            if missing_deps:
                reasons.append(f"dependencies not complete: {missing_deps}")
            if decision.skill_id in results and not allow_retries:
                reasons.append(f"skill already completed: {decision.skill_id}")
            if decision.skill_id == "release_candidate_gate" and not _release_gate_ready(manifests, results):
                reasons.append("release_candidate_gate cannot run until all non-hardware release-blocking skills have results")
            if decision.skill_id == "real_robot_gate" and "release_candidate_gate" not in results:
                reasons.append("real_robot_gate cannot run before release_candidate_gate")
            if manifest.real_robot and decision.skill_id != "real_robot_gate" and not include_real:
                reasons.append(f"real robot skill cannot run without include_real: {decision.skill_id}")
    if decision.action == "request_human_review":
        release = results.get("release_candidate_gate")
        if not release or release.status != "pass":
            reasons.append("human review can be requested only after release_candidate_gate passes")
        if "real_robot_gate" in manifests and "real_robot_gate" not in results:
            reasons.append("human review request must include a real_robot_gate result")
    return {"accepted": not reasons, "reasons": reasons}


def _orchestration_context(
    ctx: HarnessContext,
    manifests: dict[str, SkillManifest],
    ordered: list[str],
    results: dict[str, SkillResult],
    step: int,
    provider_name: str,
    gap_hints: list[dict[str, Any]],
) -> dict[str, Any]:
    priority = _priority_skill_ids(gap_hints)
    return {
        "step": step,
        "provider": provider_name,
        "task": {
            "config_path": str(ctx.config_path),
            "dataset": str(ctx.dataset),
            "include_real": ctx.include_real,
            "safe_to_autorun_robot": False,
            "user_gap_hints": gap_hints,
            "gap_hint_priority_skill_ids": priority,
        },
        "llm_contract": {
            "allowed_actions": sorted(VALID_ACTIONS),
            "decision_schema": {
                "action": "run_skill | stop | request_human_review",
                "skill_id": "required only for run_skill",
                "rationale": "why this step is next",
                "expected_evidence": "list of expected output files",
                "risk_checks": "list of validators or safety checks considered",
                "confidence": "0.0 to 1.0",
            },
            "guardrails": [
                "skills must exist in the manifest catalog",
                "dependencies must be complete before a skill runs",
                "release_candidate_gate runs after non-hardware release-blocking skills",
                "real_robot_gate runs only after release_candidate_gate",
                "safe_to_autorun_robot must remain false",
            ],
        },
        "ordered_skills": ordered,
        "runnable_skills": _runnable_skills(manifests, ordered, results, ctx.include_real),
        "priority_runnable_skills": [
            skill_id
            for skill_id in priority
            if skill_id in _runnable_skills(manifests, ordered, results, ctx.include_real)
        ],
        "completed_skill_ids": list(results),
        "skills": {skill_id: _skill_card(manifest, results.get(skill_id)) for skill_id, manifest in manifests.items()},
        "release_gate_ready": _release_gate_ready(manifests, results),
        "human_gate_ready": "release_candidate_gate" in results,
        "previous_results": {skill_id: result.to_dict() for skill_id, result in results.items()},
    }


def normalize_gap_hints(hints: list[str]) -> list[dict[str, Any]]:
    normalized = []
    seen: set[str] = set()
    for raw_hint in hints:
        raw = str(raw_hint).strip()
        if not raw:
            continue
        key = raw.lower().replace("-", "_").replace(" ", "_")
        canonical = GAP_HINT_ALIASES.get(key, key)
        priority = GAP_HINT_PRIORITY.get(canonical, [])
        signature = f"{canonical}:{raw}"
        if signature in seen:
            continue
        seen.add(signature)
        normalized.append(
            {
                "raw": raw,
                "normalized": canonical,
                "recognized": bool(priority),
                "priority_skill_ids": priority,
                "instruction": (
                    "Use this as an initial hypothesis for skill ordering only. "
                    "Do not skip measurement, dependencies, release gates, or human hardware approval."
                ),
            }
        )
    return normalized


def _coerce_gap_hints(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _priority_skill_ids(gap_hints: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for skill_id in FOUNDATION_SKILLS:
        if skill_id not in ordered:
            ordered.append(skill_id)
    for hint in gap_hints:
        for skill_id in hint.get("priority_skill_ids", []):
            if skill_id not in ordered:
                ordered.append(str(skill_id))
    return ordered


def _prioritized_runnable_skill(runnable: list[str], context: dict[str, Any]) -> str:
    priority = [str(item) for item in context.get("task", {}).get("gap_hint_priority_skill_ids", [])]
    completed = set(context.get("completed_skill_ids", []))
    for foundation in FOUNDATION_SKILLS:
        if foundation in runnable and foundation not in completed:
            return foundation
    for skill_id in priority:
        if skill_id in runnable:
            return skill_id
    return str(runnable[0])


def _hint_rationale(context: dict[str, Any]) -> str:
    hints = context.get("task", {}).get("user_gap_hints", [])
    recognized = [str(item.get("normalized")) for item in hints if item.get("recognized")]
    if not recognized:
        return ""
    return f" User gap hint priority: {', '.join(recognized)}."


def _runnable_skills(
    manifests: dict[str, SkillManifest],
    ordered: list[str],
    results: dict[str, SkillResult],
    include_real: bool,
) -> list[str]:
    runnable = []
    for skill_id in ordered:
        if skill_id in results:
            continue
        manifest = manifests[skill_id]
        if any(dep not in results for dep in manifest.depends_on):
            continue
        if skill_id == "release_candidate_gate" and not _release_gate_ready(manifests, results):
            continue
        if skill_id == "real_robot_gate" and "release_candidate_gate" not in results:
            continue
        if manifest.real_robot and skill_id != "real_robot_gate" and not include_real:
            continue
        runnable.append(skill_id)
    return runnable


def _release_gate_ready(manifests: dict[str, SkillManifest], results: dict[str, SkillResult]) -> bool:
    required = [
        skill_id
        for skill_id, manifest in manifests.items()
        if manifest.release_blocking
        and skill_id not in {"release_candidate_gate", "real_robot_gate"}
        and not manifest.real_robot
    ]
    return all(skill_id in results for skill_id in required)


def _skill_card(manifest: SkillManifest, result: SkillResult | None) -> dict[str, Any]:
    return {
        "id": manifest.skill_id,
        "name": str(manifest.data.get("name", manifest.skill_id)),
        "owner_agent": str(manifest.data.get("owner_agent", "")),
        "description": str(manifest.data.get("description", "")),
        "depends_on": manifest.depends_on,
        "inputs": list(manifest.data.get("inputs", [])),
        "outputs": list(manifest.data.get("outputs", [])),
        "validators": list(manifest.data.get("validators", [])),
        "release_blocking": manifest.release_blocking,
        "human_required": manifest.human_required,
        "real_robot": manifest.real_robot,
        "runner": manifest.runner,
        "status": result.status if result else "not_run",
        "quality_score": result.quality_score if result else None,
        "confidence": result.confidence if result else None,
    }


def _write_step_files(steps_dir: Path, step: int, entry: dict[str, Any]) -> None:
    (steps_dir / f"step_{step:03d}_decision.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")


def _iteration_scoreboard(results: dict[str, SkillResult]) -> dict[str, Any]:
    blocking = [
        {"skill_id": skill_id, "failures": result.blocking_failures}
        for skill_id, result in results.items()
        if result.status == "fail"
    ]
    return {
        "status": "fail" if blocking else "pass",
        "offline_validation_status": "fail" if blocking else "pass",
        "human_review_readiness": "not_ready",
        "release_candidate_ready": False,
        "hardware_approval_status": results.get("real_robot_gate").status if "real_robot_gate" in results else "not_requested",
        "safe_to_autorun_robot": False,
        "blocking_failures": blocking,
        "skills": {skill_id: result.to_dict() for skill_id, result in results.items()},
    }


def load_provider_command_json(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--llm-command-json must be a JSON list of strings")
    return parsed
