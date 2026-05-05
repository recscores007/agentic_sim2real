from __future__ import annotations

import json
import os
import subprocess
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import write_slide_contract_bundle
from .autoresearch import build_plan
from .config import PipelineConfig, choose_task, config_with_ui_audience, load_config, nominal_action_scale
from .data_quality import evaluate_real_data_quality
from .dataset import load_records
from .newton_bridge import run_newton_bridge
from .pace_bridge import run_pace_bridge
from .preflight import run_preflight
from .real_data import ensure_aligned_dataset
from .release_policy import is_sample_policy_artifact_path, release_profile, release_requires, release_waiver
from .run_ui import write_pipeline_ui
from .safety import require_real_robot_gate
from .sim_patch import build_sim_params_patch, write_sim_params_patch, write_sim_params_patch_json
from .sysid import estimate_gap
from .video_evidence import (
    apply_video_friction_to_domain_randomization,
    camera_tuning_from_video,
    collect_video_evidence,
    friction_tuning_from_video,
)


ALLOWED_SKILL_STATUSES = {"pass", "fail", "skip", "not_applicable", "not_approved", "evidence_missing"}
BLOCKING_SKILL_STATUSES = {"fail", "evidence_missing"}
NON_SCORING_STATUSES = {"skip", "not_applicable", "not_approved", "evidence_missing"}
RUN_PROGRESS_SCHEMA = "agentic_sim2real.run_progress.v1"
REQUIRED_MANIFEST_FIELDS = {
    "id",
    "name",
    "owner_agent",
    "description",
    "implementation",
    "inputs",
    "outputs",
    "validators",
    "quality_gate",
    "human_required",
    "release_blocking",
    "real_robot",
}

CONSOLIDATED_SKILL_ALIASES = {
    "env_preflight": "project_preflight",
    "isaaclab_task_check": "project_preflight",
    "policy_artifact_audit": "project_preflight",
    "ros_preflight": "project_preflight",
    "real_data_quality_gate": "real_data_evidence_gate",
    "pose_repeatability": "real_data_evidence_gate",
    "sysid_step_response": "physics_sysid",
    "newton_sysid": "physics_sysid",
    "pace_sysid": "physics_sysid",
    "domain_randomization_update": "agentic_tuning_plan",
    "action_scale_sweep": "agentic_tuning_plan",
    "autoresearch_planner": "agentic_tuning_plan",
    "sim_eval_regression": "regression_evaluation",
    "isaaclab_rollout_regression": "regression_evaluation",
}

SUBCHECK_MIN_SCORES = {
    "env_preflight": 0.7,
    "isaaclab_task_check": 0.9,
    "policy_artifact_audit": 0.7,
    "ros_preflight": 0.8,
    "real_data_quality_gate": 0.7,
    "pose_repeatability": 0.7,
    "video_camera_tuning": 0.6,
    "sysid_step_response": 0.6,
    "newton_sysid": 0.6,
    "pace_sysid": 0.6,
    "video_contact_friction": 0.55,
    "domain_randomization_update": 0.8,
    "action_scale_sweep": 0.7,
    "autoresearch_planner": 0.8,
    "sim_eval_regression": 0.7,
    "isaaclab_rollout_regression": 0.7,
}


@dataclass(frozen=True)
class SkillManifest:
    path: Path
    data: dict[str, Any]

    @property
    def skill_id(self) -> str:
        return str(self.data["id"])

    @property
    def depends_on(self) -> list[str]:
        return [str(item) for item in self.data.get("depends_on", [])]

    @property
    def implementation(self) -> str:
        return str(self.data["implementation"])

    @property
    def runner(self) -> str:
        if self.data.get("runner"):
            return str(self.data["runner"])
        if self.implementation == "external_command":
            return "command"
        return "builtin"

    @property
    def release_blocking(self) -> bool:
        return bool(self.data.get("release_blocking", False))

    @property
    def human_required(self) -> bool:
        return bool(self.data.get("human_required", False))

    @property
    def real_robot(self) -> bool:
        return bool(self.data.get("real_robot", False))

    @property
    def min_score(self) -> float:
        return float(self.data.get("quality_gate", {}).get("min_score", 0.0))


@dataclass
class SkillResult:
    skill_id: str
    status: str
    quality_score: float
    confidence: float
    blocking_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence_files: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    human_required: bool = False
    release_blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "status": self.status,
            "quality_score": round(float(self.quality_score), 3),
            "confidence": round(float(self.confidence), 3),
            "blocking_failures": self.blocking_failures,
            "warnings": self.warnings,
            "evidence_files": self.evidence_files,
            "metrics": self.metrics,
            "human_required": self.human_required,
            "release_blocking": self.release_blocking,
        }


@dataclass
class HarnessContext:
    root: Path
    config_path: Path
    config: PipelineConfig
    dataset: Path
    out_dir: Path
    include_real: bool = False

    @property
    def skill_dir(self) -> Path:
        return self.out_dir / "skills"


def load_manifests(root: str | Path, skill_dirs: list[str | Path] | None = None) -> list[SkillManifest]:
    base = Path(root)
    manifest_by_id: dict[str, SkillManifest] = {}
    manifest_dirs = [base / "skills"]
    default_custom_dir = base / "custom_skills"
    if default_custom_dir.exists():
        manifest_dirs.append(default_custom_dir)
    for skill_dir in skill_dirs or []:
        path = Path(skill_dir)
        manifest_dirs.append(path if path.is_absolute() else base / path)

    for manifest_dir in manifest_dirs:
        if not manifest_dir.exists():
            continue
        for path in sorted(manifest_dir.glob("*/skill.json")):
            data = json.loads(path.read_text())
            manifest_by_id[str(data.get("id", path.parent.name))] = SkillManifest(path=path, data=data)
    manifests = list(manifest_by_id.values())
    if not manifests:
        raise ValueError(f"No skill manifests found under {base / 'skills'} or overlays")
    return manifests


def validate_manifest(manifest: SkillManifest) -> list[str]:
    errors = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest.data))
    if missing:
        errors.append(f"missing required fields: {missing}")
    if manifest.data.get("id") != manifest.path.parent.name:
        errors.append("manifest id must match containing directory name")
    if not isinstance(manifest.data.get("validators"), list) or not manifest.data.get("validators"):
        errors.append("validators must be a non-empty list")
    if not isinstance(manifest.data.get("inputs"), list):
        errors.append("inputs must be a list")
    if not isinstance(manifest.data.get("outputs"), list):
        errors.append("outputs must be a list")
    if not isinstance(manifest.data.get("quality_gate"), dict):
        errors.append("quality_gate must be an object")
    runner = str(manifest.data.get("runner", "command" if manifest.data.get("implementation") == "external_command" else "builtin"))
    if runner not in {"builtin", "command"}:
        errors.append("runner must be 'builtin' or 'command'")
    if runner == "command":
        command = manifest.data.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            errors.append("command-runner skills must provide a non-empty string list in command")
    return errors


def validate_all_manifests(root: str | Path, skill_dirs: list[str | Path] | None = None) -> dict[str, Any]:
    manifests = load_manifests(root, skill_dirs=skill_dirs)
    results = {}
    for manifest in manifests:
        errors = validate_manifest(manifest)
        results[manifest.skill_id] = {
            "status": "pass" if not errors else "fail",
            "errors": errors,
            "path": str(manifest.path),
        }
    return {
        "status": "pass" if all(item["status"] == "pass" for item in results.values()) else "fail",
        "skills": results,
    }


def run_harness(
    root: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path,
    include_real: bool = False,
    only_skill: str | None = None,
    skill_dirs: list[str | Path] | None = None,
    audience: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
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
    if only_skill and only_skill not in manifests and only_skill in CONSOLIDATED_SKILL_ALIASES:
        only_skill = CONSOLIDATED_SKILL_ALIASES[only_skill]
    ordered = ordered_skill_ids(manifests)
    if only_skill:
        ordered = [skill_id for skill_id in ordered if skill_id == only_skill]
        if not ordered:
            raise ValueError(f"Unknown skill: {only_skill}")

    results: dict[str, SkillResult] = {}
    ctx.skill_dir.mkdir(parents=True, exist_ok=True)
    progress = _initial_run_progress(ctx, manifests, ordered, only_skill=only_skill)
    _write_run_progress(ctx, progress)
    (ctx.out_dir / "run_progress.jsonl").write_text("")
    for skill_id in ordered:
        _mark_progress_step(ctx, progress, skill_id, "running")
        _emit_run_event(ctx, progress, progress_callback, "skill_started", skill_id)
        results[skill_id] = run_skill_step(manifests, ctx, skill_id, results)
        _mark_progress_step(ctx, progress, skill_id, results[skill_id].status, results[skill_id])
        _emit_run_event(ctx, progress, progress_callback, "skill_completed", skill_id, results[skill_id])

    progress["status"] = "complete_with_failures" if any(result.status == "fail" for result in results.values()) else "complete"
    progress["finished_at"] = _utc_now()
    progress["updated_at"] = progress["finished_at"]
    _write_run_progress(ctx, progress)

    return write_harness_artifacts(ctx, manifests, results)


def prepare_harness(
    root: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path,
    include_real: bool = False,
    skill_dirs: list[str | Path] | None = None,
    audience: str | None = None,
) -> tuple[HarnessContext, dict[str, SkillManifest]]:
    root_path = Path(root).resolve()
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_dataset_path = ensure_aligned_dataset(dataset_path, root=root_path).resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ctx = HarnessContext(
        root=root_path,
        config_path=resolved_config_path,
        config=config_with_ui_audience(load_config(resolved_config_path), audience),
        dataset=resolved_dataset_path,
        out_dir=out,
        include_real=include_real,
    )
    manifests = {m.skill_id: m for m in load_manifests(root_path, skill_dirs=skill_dirs)}
    for manifest in manifests.values():
        errors = validate_manifest(manifest)
        if errors:
            raise ValueError(f"{manifest.skill_id}: invalid manifest: {errors}")
    return ctx, manifests


def ordered_skill_ids(manifests: dict[str, SkillManifest]) -> list[str]:
    return _topological_order(manifests)


def run_skill_step(
    manifests: dict[str, SkillManifest],
    ctx: HarnessContext,
    skill_id: str,
    previous_results: dict[str, SkillResult],
) -> SkillResult:
    if skill_id not in manifests:
        raise ValueError(f"Unknown skill: {skill_id}")
    manifest = manifests[skill_id]
    skill_out = ctx.skill_dir / skill_id
    skill_out.mkdir(parents=True, exist_ok=True)

    if manifest.real_robot and not ctx.include_real:
        evidence = _write_json(
            skill_out / f"{skill_id}.json",
            {
                "status": "not_approved",
                "reason": "real robot skill was not included because human hardware approval is required",
                "safe_to_autorun_robot": False,
            },
        )
        result = SkillResult(
            skill_id=skill_id,
            status="not_approved",
            quality_score=0.0,
            confidence=1.0,
            warnings=["real robot skill is not approved; rerun with --include-real only after human approval"],
            evidence_files=[evidence],
            human_required=manifest.human_required,
            release_blocking=manifest.release_blocking,
            metrics={"safe_to_autorun_robot": False},
        )
    else:
        try:
            result = _run_one(manifest, ctx, skill_out, previous_results)
        except Exception as exc:
            result = _exception_result(manifest, skill_out, exc)
        result.human_required = manifest.human_required
        result.release_blocking = manifest.release_blocking
        result = _apply_quality_gate(manifest, result)

    (skill_out / "result.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    return result


def write_harness_artifacts(
    ctx: HarnessContext,
    manifests: dict[str, SkillManifest],
    results: dict[str, SkillResult],
    require_all_release_blocking: bool = False,
) -> dict[str, Any]:
    progress = _ensure_run_progress(ctx, manifests, results)
    scoreboard = _scoreboard(ctx.config, results, manifests, require_all_release_blocking=require_all_release_blocking)
    scoreboard["run_progress"] = progress
    (ctx.out_dir / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n")
    (ctx.out_dir / "release_candidate.json").write_text(
        json.dumps(_release_candidate(results, scoreboard), indent=2, sort_keys=True) + "\n"
    )
    artifact_paths = write_slide_contract_bundle(
        ctx.dataset,
        ctx.config,
        ctx.out_dir,
        results,
        scoreboard,
        config_path=ctx.config_path,
        skill_ids=sorted(manifests),
    )
    artifact_paths.update(write_pipeline_ui(ctx.out_dir, run_status="complete", audience=ctx.config.ui.get("audience")))
    artifact_paths["run_progress"] = str(ctx.out_dir / "run_progress.json")
    scoreboard["artifacts"] = artifact_paths
    (ctx.out_dir / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n")
    return scoreboard


def _exception_result(manifest: SkillManifest, skill_out: Path, exc: Exception) -> SkillResult:
    evidence = _write_json(
        skill_out / "unhandled_exception.json",
        {
            "status": "fail",
            "skill_id": manifest.skill_id,
            "exception_type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "pipeline_action": "captured_as_skill_failure",
        },
    )
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail",
        quality_score=0.0,
        confidence=1.0,
        blocking_failures=[f"{manifest.skill_id} raised {exc.__class__.__name__}: {exc}"],
        evidence_files=[evidence],
        metrics={
            "exception_type": exc.__class__.__name__,
            "run_reason": _skill_run_reason(manifest),
            "failure_mode": "unhandled_exception_captured",
        },
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_run_progress(
    ctx: HarnessContext,
    manifests: dict[str, SkillManifest],
    ordered: list[str],
    *,
    only_skill: str | None = None,
) -> dict[str, Any]:
    started = _utc_now()
    return {
        "schema": RUN_PROGRESS_SCHEMA,
        "status": "running",
        "started_at": started,
        "updated_at": started,
        "finished_at": None,
        "current_step_index": 0,
        "current_skill_id": None,
        "completed_steps": 0,
        "failed_steps": 0,
        "total_steps": len(ordered),
        "mode": "single_skill" if only_skill else "full_harness",
        "run": {
            "config_path": str(ctx.config_path),
            "dataset": str(ctx.dataset),
            "out_dir": str(ctx.out_dir),
            "include_real": ctx.include_real,
            "audience": ctx.config.ui.get("audience"),
        },
        "steps": [
            {
                "index": index,
                "skill_id": skill_id,
                "name": manifests[skill_id].data.get("name", skill_id),
                "owner_agent": manifests[skill_id].data.get("owner_agent", "pipeline"),
                "status": "pending",
                "dependencies": manifests[skill_id].depends_on,
                "release_blocking": manifests[skill_id].release_blocking,
                "human_required": manifests[skill_id].human_required,
                "real_robot": manifests[skill_id].real_robot,
                "run_reason": _skill_run_reason(manifests[skill_id], only_skill=only_skill),
                "started_at": None,
                "finished_at": None,
                "summary": "Waiting for dependencies and earlier steps.",
            }
            for index, skill_id in enumerate(ordered, start=1)
        ],
        "latest_event": None,
        "events": [],
    }


def _skill_run_reason(manifest: SkillManifest, *, only_skill: str | None = None) -> str:
    if only_skill:
        return f"Selected by --skill {only_skill}; dependencies are collapsed to the owning consolidated skill when needed."
    if manifest.skill_id == "project_preflight":
        return "Runs first to validate configuration, environment assumptions, task selection, ROS/Isaac setup, and policy artifacts."
    if manifest.skill_id == "real_data_evidence_gate":
        return "Runs after preflight to verify real logs, uploaded videos, pose repeatability, and data-readiness evidence."
    if manifest.skill_id == "physics_sysid":
        return "Runs because actuator, latency, contact, friction, and optional fitted-backend evidence drive simulator parameter proposals."
    if manifest.skill_id == "agentic_tuning_plan":
        return "Runs after evidence gates to write bounded sim-parameter and experiment proposals."
    if manifest.skill_id == "regression_evaluation":
        return "Runs after tuning proposals to check simulated regression and optional Isaac Lab rollout evidence."
    if manifest.skill_id == "release_candidate_gate":
        return "Runs after validation skills to decide whether the candidate can move to human review."
    if manifest.skill_id == "real_robot_gate":
        return "Runs last as an explicit human approval checkpoint; it never authorizes unattended robot motion."
    deps = ", ".join(manifest.depends_on) or "no declared dependencies"
    return f"Runs from the manifest order after {deps}."


def _progress_step(progress: dict[str, Any], skill_id: str) -> dict[str, Any] | None:
    for step in progress.get("steps", []):
        if step.get("skill_id") == skill_id:
            return step
    return None


def _mark_progress_step(
    ctx: HarnessContext,
    progress: dict[str, Any],
    skill_id: str,
    status: str,
    result: SkillResult | None = None,
) -> None:
    now = _utc_now()
    step = _progress_step(progress, skill_id)
    if step is None:
        return
    step["status"] = status
    progress["updated_at"] = now
    progress["current_skill_id"] = skill_id
    progress["current_step_index"] = step.get("index", 0)
    if status == "running":
        step["started_at"] = now
        step["summary"] = "Running now."
    else:
        step["finished_at"] = now
        if result:
            step["quality_score"] = round(float(result.quality_score), 3)
            step["confidence"] = round(float(result.confidence), 3)
            step["warnings_count"] = len(result.warnings)
            step["blocking_failure_count"] = len(result.blocking_failures)
            step["summary"] = _result_summary(result)
    finished = [item for item in progress.get("steps", []) if item.get("status") not in {"pending", "running"}]
    progress["completed_steps"] = len(finished)
    progress["failed_steps"] = sum(1 for item in finished if item.get("status") == "fail")
    _write_run_progress(ctx, progress)


def _result_summary(result: SkillResult) -> str:
    if result.blocking_failures:
        return str(result.blocking_failures[0])
    if result.warnings:
        return str(result.warnings[0])
    if result.status == "pass":
        return "Completed successfully."
    if result.status == "not_applicable":
        return str(result.metrics.get("skip_reason") or "Not applicable for this run.")
    if result.status == "evidence_missing":
        return str(result.metrics.get("skip_reason") or "Evidence was not configured or not found.")
    return f"Completed with status {result.status}."


def _emit_run_event(
    ctx: HarnessContext,
    progress: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None,
    event_name: str,
    skill_id: str,
    result: SkillResult | None = None,
) -> None:
    step = _progress_step(progress, skill_id) or {}
    event = {
        "timestamp": _utc_now(),
        "event": event_name,
        "index": step.get("index"),
        "total": progress.get("total_steps"),
        "skill_id": skill_id,
        "status": result.status if result else "running",
        "summary": _result_summary(result) if result else step.get("run_reason", ""),
    }
    progress["latest_event"] = event
    events = list(progress.get("events", []))
    events.append(event)
    progress["events"] = events[-50:]
    _write_run_progress(ctx, progress)
    with (ctx.out_dir / "run_progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    if progress_callback:
        progress_callback(event)


def _write_run_progress(ctx: HarnessContext, progress: dict[str, Any]) -> None:
    (ctx.out_dir / "run_progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")


def _ensure_run_progress(
    ctx: HarnessContext,
    manifests: dict[str, SkillManifest],
    results: dict[str, SkillResult],
) -> dict[str, Any]:
    path = ctx.out_dir / "run_progress.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    ordered = ordered_skill_ids(manifests)
    progress = _initial_run_progress(ctx, manifests, ordered)
    for skill_id, result in results.items():
        step = _progress_step(progress, skill_id)
        if not step:
            continue
        step["status"] = result.status
        step["finished_at"] = _utc_now()
        step["quality_score"] = round(float(result.quality_score), 3)
        step["confidence"] = round(float(result.confidence), 3)
        step["warnings_count"] = len(result.warnings)
        step["blocking_failure_count"] = len(result.blocking_failures)
        step["summary"] = _result_summary(result)
    pending_count = len(ordered) - len(results)
    if pending_count:
        progress["status"] = "stopped_before_all_steps"
    else:
        progress["status"] = "complete_with_failures" if any(result.status == "fail" for result in results.values()) else "complete"
    progress["completed_steps"] = len(results)
    progress["failed_steps"] = sum(1 for result in results.values() if result.status == "fail")
    progress["finished_at"] = _utc_now()
    progress["updated_at"] = progress["finished_at"]
    _write_run_progress(ctx, progress)
    return progress


def _run_one(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    if manifest.runner == "command":
        return _run_command_skill(manifest, ctx, skill_out, previous)
    impl = IMPLEMENTATIONS.get(manifest.implementation)
    if impl is None:
        return SkillResult(
            skill_id=manifest.skill_id,
            status="fail",
            quality_score=0.0,
            confidence=1.0,
            blocking_failures=[f"unknown implementation: {manifest.implementation}"],
        )
    return impl(manifest, ctx, skill_out, previous)


def _run_command_skill(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    command = [str(item) for item in manifest.data.get("command", [])]
    input_path = skill_out / "skill_input.json"
    output_path = skill_out / "skill_output.json"
    log_path = skill_out / "external_command_log.json"
    timeout_s = float(manifest.data.get("timeout_s", 300.0))
    input_payload = {
        "skill_id": manifest.skill_id,
        "manifest": manifest.data,
        "manifest_dir": str(manifest.path.parent),
        "root": str(ctx.root),
        "config_path": str(ctx.config_path),
        "config": ctx.config.merged(),
        "dataset": str(ctx.dataset),
        "out_dir": str(ctx.out_dir),
        "skill_out": str(skill_out),
        "previous_results": {skill_id: result.to_dict() for skill_id, result in previous.items()},
    }
    input_path.write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n")

    env = os.environ.copy()
    env.update(
        {
            "AGENTIC_SIM2REAL_SKILL_ID": manifest.skill_id,
            "AGENTIC_SIM2REAL_SKILL_INPUT_JSON": str(input_path),
            "AGENTIC_SIM2REAL_SKILL_OUTPUT_JSON": str(output_path),
            "AGENTIC_SIM2REAL_SKILL_OUT_DIR": str(skill_out),
            "AGENTIC_SIM2REAL_SKILL_MANIFEST_DIR": str(manifest.path.parent),
            "AGENTIC_SIM2REAL_ROOT": str(ctx.root),
            "AGENTIC_SIM2REAL_CONFIG_PATH": str(ctx.config_path),
            "AGENTIC_SIM2REAL_DATASET": str(ctx.dataset),
        }
    )
    try:
        completed = subprocess.run(
            command,
            cwd=ctx.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        evidence = _write_json(
            log_path,
            {"command": command, "timeout_s": timeout_s, "stdout": exc.stdout, "stderr": exc.stderr},
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status="fail",
            quality_score=0.0,
            confidence=1.0,
            blocking_failures=[f"external skill command timed out after {timeout_s} seconds"],
            evidence_files=[evidence],
        )
    except OSError as exc:
        evidence = _write_json(log_path, {"command": command, "error": str(exc)})
        return SkillResult(
            skill_id=manifest.skill_id,
            status="fail",
            quality_score=0.0,
            confidence=1.0,
            blocking_failures=[f"external skill command could not start: {exc}"],
            evidence_files=[evidence],
        )

    log_evidence = _write_json(
        log_path,
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "result_file": str(output_path),
        },
    )
    if completed.returncode != 0:
        return SkillResult(
            skill_id=manifest.skill_id,
            status="fail",
            quality_score=0.0,
            confidence=1.0,
            blocking_failures=[f"external skill command exited {completed.returncode}"],
            warnings=_stderr_warnings(completed.stderr),
            evidence_files=[log_evidence],
        )

    try:
        if output_path.exists():
            payload = json.loads(output_path.read_text())
        elif completed.stdout.strip():
            payload = json.loads(completed.stdout)
        else:
            raise ValueError("external skill did not write AGENTIC_SIM2REAL_SKILL_OUTPUT_JSON or JSON stdout")
    except Exception as exc:
        return SkillResult(
            skill_id=manifest.skill_id,
            status="fail",
            quality_score=0.0,
            confidence=1.0,
            blocking_failures=[f"external skill result could not be parsed: {exc}"],
            evidence_files=[log_evidence],
        )

    result = _skill_result_from_payload(manifest, payload, skill_out)
    result.evidence_files.append(log_evidence)
    return result


def _skill_result_from_payload(manifest: SkillManifest, payload: dict[str, Any], skill_out: Path) -> SkillResult:
    status = str(payload.get("status", "fail"))
    if status not in ALLOWED_SKILL_STATUSES:
        status = "fail"
        failures = [f"invalid external skill status: {payload.get('status')}"]
    else:
        failures = [str(item) for item in payload.get("blocking_failures", [])]
    evidence_files = [_normalize_evidence_path(path, skill_out) for path in payload.get("evidence_files", [])]
    return SkillResult(
        skill_id=manifest.skill_id,
        status=status,
        quality_score=float(payload.get("quality_score", 0.0)),
        confidence=float(payload.get("confidence", 0.0)),
        blocking_failures=failures,
        warnings=[str(item) for item in payload.get("warnings", [])],
        evidence_files=evidence_files,
        metrics=dict(payload.get("metrics", {})),
    )


def _normalize_evidence_path(path: Any, skill_out: Path) -> str:
    evidence = Path(str(path))
    if not evidence.is_absolute():
        evidence = skill_out / evidence
    return str(evidence)


def _stderr_warnings(stderr: str | None) -> list[str]:
    if not stderr:
        return []
    text = str(stderr).strip()
    return [text] if text else []


def _apply_quality_gate(manifest: SkillManifest, result: SkillResult) -> SkillResult:
    if result.status == "pass" and result.quality_score < manifest.min_score:
        result.status = "fail"
        result.blocking_failures.append(
            f"quality_score {result.quality_score:.3f} below skill gate {manifest.min_score:.3f}"
        )
    return result


def _topological_order(manifests: dict[str, SkillManifest]) -> list[str]:
    ordered: list[str] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(skill_id: str) -> None:
        if skill_id in permanent:
            return
        if skill_id in temporary:
            raise ValueError(f"Cycle in skill dependencies at {skill_id}")
        if skill_id not in manifests:
            raise ValueError(f"Missing skill dependency: {skill_id}")
        temporary.add(skill_id)
        for dep in manifests[skill_id].depends_on:
            visit(dep)
        temporary.remove(skill_id)
        permanent.add(skill_id)
        ordered.append(skill_id)

    for skill_id in sorted(manifests):
        visit(skill_id)
    return ordered


def _scoreboard(
    config: PipelineConfig,
    results: dict[str, SkillResult],
    manifests: dict[str, SkillManifest],
    require_all_release_blocking: bool = False,
) -> dict[str, Any]:
    blocking_failures = []
    scores = []
    status_counts: dict[str, int] = {}
    for skill_id, result in results.items():
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        if result.status == "pass":
            scores.append(result.quality_score)
        if manifests[skill_id].release_blocking and result.status in BLOCKING_SKILL_STATUSES:
            blocking_failures.append({"skill_id": skill_id, "failures": result.blocking_failures})
    if require_all_release_blocking:
        for skill_id, manifest in sorted(manifests.items()):
            if manifest.release_blocking and skill_id not in results:
                blocking_failures.append({"skill_id": skill_id, "failures": ["release-blocking skill did not run"]})
    release_gate = results.get("release_candidate_gate")
    real_gate = results.get("real_robot_gate")
    profile = release_profile(config)
    strict_profile = profile != "smoke"
    ready_for_human_review = bool(
        not blocking_failures
        and release_gate
        and release_gate.status == "pass"
    )
    readiness = "ready" if ready_for_human_review and strict_profile else "smoke_review_only" if ready_for_human_review else "not_ready"
    quality = sum(scores) / len(scores) if scores else 0.0
    subchecks = {
        subcheck_id: payload
        for result in results.values()
        for subcheck_id, payload in (
            result.metrics.get("subchecks", {}).items()
            if isinstance(result.metrics, dict) and isinstance(result.metrics.get("subchecks"), dict)
            else []
        )
    }
    return {
        "status": "pass" if not blocking_failures else "fail",
        "release_profile": profile,
        "review_scope": "release_candidate_review" if strict_profile else "smoke_offline_review",
        "offline_validation_status": "pass" if not blocking_failures else "fail",
        "human_review_readiness": readiness,
        "ready_for_human_review": ready_for_human_review,
        "release_candidate_ready": bool(ready_for_human_review and strict_profile),
        "hardware_approval_status": real_gate.status if real_gate else "not_requested",
        "safe_to_autorun_robot": False,
        "status_counts": status_counts,
        "quality_score": round(quality, 3),
        "blocking_failures": blocking_failures,
        "subchecks": subchecks,
        "skills": {skill_id: result.to_dict() for skill_id, result in results.items()},
    }


def _release_candidate(results: dict[str, SkillResult], scoreboard: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "promote_to_human_review" if scoreboard["status"] == "pass" else "blocked",
        "safe_to_autorun_robot": False,
        "offline_validation_status": scoreboard.get("offline_validation_status", scoreboard["status"]),
        "human_review_readiness": scoreboard.get("human_review_readiness", "not_ready"),
        "quality_score": scoreboard["quality_score"],
        "required_human_approvals": [
            skill_id for skill_id, result in results.items() if result.human_required
        ],
        "evidence_files": [
            evidence
            for result in results.values()
            for evidence in result.evidence_files
        ],
        "notes": [
            "AutoResearch can promote a candidate only to human review.",
            "Real robot motion is never automatically released by this harness.",
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return str(path)


def _optional_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _impl_env_preflight(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    report = run_preflight(ctx.config, root=ctx.root)
    failures = [str(item) for item in report.get("failures", [])]
    warnings = [str(item) for item in report.get("warnings", [])]
    evidence = _write_json(skill_out / "preflight.json", report)
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=1.0 if not failures else 0.0,
        confidence=1.0,
        blocking_failures=failures,
        warnings=warnings,
        evidence_files=[evidence],
        metrics={
            "required_commands_present": not any("python3" in failure for failure in failures),
            "newton_available": bool(report["sysid_backends"]["newton"]["available"]),
            "pace_available": bool(report["sysid_backends"]["pace"]["available"]),
            "local_sysid_available": True,
        },
    )


def _impl_isaaclab_task_check(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    task = choose_task(ctx.config)
    gripper = str(ctx.config.robot.get("gripper_type", ""))
    expected_action_scale = nominal_action_scale(ctx.config)
    configured_tasks = {
        str(ctx.config.isaac_lab.get("task_2f140", "")),
        str(ctx.config.isaac_lab.get("task_2f85", "")),
    }
    max_action_scale = float(ctx.config.robot.get("max_action_scale", 0.05))
    failures = []
    if not task or task not in configured_tasks:
        failures.append(f"selected Isaac Lab task is not in the configured task contract: {task}")
    if expected_action_scale <= 0 or expected_action_scale > max_action_scale:
        failures.append(
            f"nominal action scale {expected_action_scale} must be positive and <= configured max_action_scale {max_action_scale}"
        )
    obs = list(ctx.config.isaac_lab.get("observations", ["joint_pos", "joint_vel", "gear_shaft_pos", "gear_shaft_quat"]))
    evidence = _write_json(
        skill_out / "task_contract.json",
        {
            "task": task,
            "configured_tasks": sorted(item for item in configured_tasks if item),
            "gripper_type": gripper,
            "max_action_scale": max_action_scale,
            "nominal_action_scale": expected_action_scale,
            "observations": obs,
        },
    )
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=1.0 if not failures else 0.0,
        confidence=1.0,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics={"task": task, "nominal_action_scale": expected_action_scale},
    )


def _impl_policy_artifact_audit(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    artifact_cfg = str(ctx.config.policy.get("artifact_dir", "golden/sample_inputs/policy_artifacts")).strip()
    artifact_dir = Path(artifact_cfg).expanduser()
    if not artifact_dir.is_absolute():
        artifact_dir = ctx.root / artifact_dir
    required = ["agent.yaml", "env.yaml", "checkpoint.meta.json"]
    present = {name: (artifact_dir / name).exists() for name in required}
    failures = [f"missing policy artifact sample: {name}" for name, ok in present.items() if not ok]
    using_sample = is_sample_policy_artifact_path(ctx.config, str(artifact_dir))
    warnings = []
    status = "fail" if failures else "pass"
    if using_sample:
        warnings.append("sample policy artifacts are being used; configure policy.artifact_dir for a real release")
    if release_requires(ctx.config, "require_user_policy_artifacts_for_human_review") and using_sample:
        waiver_allowed = bool(ctx.config.policy.get("allow_policy_artifact_waiver", False))
        waiver_reason = str(ctx.config.policy.get("policy_artifact_waiver_reason", "")).strip()
        if waiver_allowed and waiver_reason:
            warnings.append(f"user policy artifact requirement waived: {waiver_reason}")
        else:
            status = "evidence_missing"
            failures.append("release-candidate profile requires user-provided policy artifacts, not sample fixtures")
    evidence = _write_json(
        skill_out / "policy_artifact_audit.json",
        {
            "artifact_dir": str(artifact_dir),
            "present": present,
            "using_sample_artifacts": using_sample,
            "release_profile": release_profile(ctx.config),
        },
    )
    score = sum(1 for ok in present.values() if ok) / len(present)
    return SkillResult(
        skill_id=manifest.skill_id,
        status=status,
        quality_score=score,
        confidence=0.8,
        blocking_failures=failures,
        warnings=warnings,
        evidence_files=[evidence],
        metrics={"artifact_completeness": round(score, 3), "using_sample_artifacts": using_sample},
    )


def _impl_ros_preflight(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    required_env = {
        "ROS_DOMAIN_ID": str(ctx.config.isaac_ros.get("ros_domain_id", "")),
        "RMW_IMPLEMENTATION": str(ctx.config.isaac_ros.get("rmw_implementation", "")),
        "workflow_type": str(ctx.config.isaac_ros.get("workflow_type", "")),
    }
    failures = []
    if required_env["workflow_type"] != "GEAR_ASSEMBLY":
        failures.append("workflow_type must be GEAR_ASSEMBLY")
    if required_env["RMW_IMPLEMENTATION"] != "rmw_cyclonedds_cpp":
        failures.append("Cyclone DDS is recommended for the real robot setup")
    evidence = _write_json(skill_out / "ros_preflight_contract.json", required_env)
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=1.0 if not failures else 0.5,
        confidence=0.9,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics=required_env,
    )


def _impl_pose_repeatability(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    records = load_records(ctx.dataset)
    gap = estimate_gap(records, ctx.config)
    pose = gap.get("object_pose_noise", gap["shaft_pose_noise"])
    gate = float(ctx.config.perception.get("pose_error_gate_m", 0.01))
    p95 = pose.get("position_error_p95_m")
    failures = []
    if p95 is None:
        failures.append("no object_pose_reference values available for pose repeatability")
    elif float(p95) > gate:
        failures.append(f"object pose p95 error {p95} m exceeds gate {gate} m")
    if int(pose.get("samples", 0) or 0) < 10:
        failures.append("pose repeatability needs at least 10 samples")
    evidence = _write_json(skill_out / "pose_repeatability.json", pose)
    score = 1.0 if not failures else max(0.0, 1.0 - len(failures) * 0.25)
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=score,
        confidence=min(1.0, int(pose.get("samples", 0) or 0) / 20.0),
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics=pose,
    )


def _impl_real_data_quality_gate(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    report = evaluate_real_data_quality(ctx.dataset, ctx.config, root=ctx.root)
    evidence = _write_json(skill_out / "real_data_quality.json", report)
    return SkillResult(
        skill_id=manifest.skill_id,
        status=report["status"],
        quality_score=float(report["quality_score"]),
        confidence=float(report["confidence"]),
        blocking_failures=[str(item) for item in report["blocking_failures"]],
        warnings=[str(item) for item in report["warnings"]],
        evidence_files=[evidence],
        metrics={**report["metrics"], "data_readiness": report.get("data_readiness", {})},
    )


def _impl_video_camera_tuning(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    video = collect_video_evidence(ctx.dataset, ctx.root, ctx.config, work_dir=skill_out)
    report = camera_tuning_from_video(video, ctx.config)
    evidence = _write_json(
        skill_out / "video_camera_tuning.json",
        {
            "video_evidence": video,
            "camera_tuning": report,
        },
    )
    return SkillResult(
        skill_id=manifest.skill_id,
        status=report["status"],
        quality_score=float(report["quality_score"]),
        confidence=float(report["confidence"]),
        blocking_failures=[str(item) for item in report["blocking_failures"]],
        warnings=[str(item) for item in report["warnings"]],
        evidence_files=[evidence],
        metrics=dict(report["metrics"]),
    )


def _impl_sysid_step_response(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    records = load_records(ctx.dataset)
    gap = estimate_gap(records, ctx.config)
    evidence = _write_json(skill_out / "gap_estimates.json", gap)
    summary = gap["summary"]
    failures = []
    if int(summary["episodes"]) < int(ctx.config.agent.get("min_real_episodes_for_gate", 3)):
        failures.append("not enough episodes for SysID gate")
    confidence = max(
        float(gap["delay"].get("confidence", 0.0)),
        float(gap["deadband_stiction_proxy"].get("confidence", 0.0)),
        0.5,
    )
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=0.75 if not failures else 0.45,
        confidence=confidence,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics={
            "episodes": summary["episodes"],
            "delay_steps": gap["delay"]["delay_steps"],
            "deadband_command_norm": gap["deadband_stiction_proxy"]["deadband_command_norm"],
            "run_reason": "The local log-based SysID estimator always runs as the offline baseline for actuator latency, deadband, and contact signals.",
        },
    )


def _impl_video_contact_friction(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    video = collect_video_evidence(ctx.dataset, ctx.root, ctx.config, work_dir=skill_out)
    report = friction_tuning_from_video(video, ctx.config)
    evidence = _write_json(
        skill_out / "video_contact_friction.json",
        {
            "video_evidence": video,
            "friction_tuning": report,
        },
    )
    return SkillResult(
        skill_id=manifest.skill_id,
        status=report["status"],
        quality_score=float(report["quality_score"]),
        confidence=float(report["confidence"]),
        blocking_failures=[str(item) for item in report["blocking_failures"]],
        warnings=[str(item) for item in report["warnings"]],
        evidence_files=[evidence],
        metrics={
            **dict(report["metrics"]),
            "run_reason": "Contact/friction video evidence is checked during physics SysID so uploaded videos can tune object and gripper friction; missing videos are recorded explicitly.",
        },
    )


def _impl_newton_sysid(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    sysid_cfg = ctx.config.sysid
    required = bool(sysid_cfg.get("require_newton", False))
    preference = [str(item).strip().lower() for item in sysid_cfg.get("sysid_backend_preference", ["pace", "newton", "local"])]
    pace = previous.get("pace_sysid")
    if not required and pace and pace.status == "pass" and _prefers_pace(preference):
        evidence = _write_json(
            skill_out / "newton_sysid.json",
            {
                "status": "not_applicable",
                "reason": "PACE SysID passed and is preferred ahead of Newton.",
                "fallback_skill": "pace_sysid",
            },
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status="not_applicable",
            quality_score=0.0,
            confidence=1.0,
            warnings=["Newton SysID skipped because PACE is the preferred fitted-physics backend"],
            evidence_files=[evidence],
            metrics={
                "newton_enabled": False,
                "fallback_skill": "pace_sysid",
                "run_reason": "Newton is checked after PACE so the run records whether a fallback fitted backend was needed.",
                "skip_reason": "PACE SysID passed and sysid_backend_preference puts PACE ahead of Newton.",
            },
        )
    command = [str(item) for item in sysid_cfg.get("newton_command", [])]
    newton_root = str(sysid_cfg.get("newton_root") or os.environ.get("ISAACLAB_NEWTON_ROOT", ""))
    enabled = bool(sysid_cfg.get("newton_enabled", False) or command or newton_root)
    input_path = skill_out / "newton_input.json"
    output_path = skill_out / "newton_output.json"
    log_path = skill_out / "newton_command_log.json"

    input_payload = {
        "dataset": str(ctx.dataset),
        "config": ctx.config.merged(),
        "root": str(ctx.root),
        "newton_root": newton_root,
        "previous_results": {skill_id: result.to_dict() for skill_id, result in previous.items()},
        "expected_output_json": str(output_path),
    }
    input_path.write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n")

    if not enabled:
        status = "fail" if required else "evidence_missing"
        message = "IsaacLab-Newton is required but is not enabled or configured." if required else "IsaacLab-Newton is not enabled; local log-based SysID remains the active fallback."
        evidence = _write_json(
            skill_out / "newton_sysid.json",
            {
                "status": status,
                "reason": message,
                "fallback_skill": "sysid_step_response",
                "input_file": str(input_path),
            },
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status=status,
            quality_score=0.0,
            confidence=1.0,
            blocking_failures=[message] if required else [],
            warnings=[] if required else ["Newton SysID skipped; set sysid.newton_enabled plus sysid.newton_root or sysid.newton_command to enable it"],
            evidence_files=[evidence],
            metrics={
                "newton_enabled": False,
                "fallback_skill": "sysid_step_response",
                "run_reason": "Newton is checked as the configured fallback fitted SysID backend and to make missing backend evidence explicit.",
                "skip_reason": "sysid.newton_enabled is false and neither sysid.newton_command nor sysid.newton_root is configured.",
            },
        )

    if not command:
        try:
            payload = run_newton_bridge(input_payload, work_dir=skill_out, output_path=output_path)
        except Exception as exc:
            message = f"Built-in Newton bridge could not complete: {exc}"
            evidence = _write_json(log_path, {"error": str(exc), "input_file": str(input_path)})
            return _newton_unavailable_result(manifest.skill_id, required, message, evidence)
        result = _sysid_backend_result_from_payload(
            manifest.skill_id,
            payload,
            sysid_cfg,
            required,
            skill_out,
            extra_evidence=[str(output_path)] if output_path.exists() else [],
        )
        result.metrics.setdefault("run_reason", "Newton ran because sysid.newton_enabled, sysid.newton_command, or sysid.newton_root is configured.")
        return result

    formatted_command = [
        item.format(input=input_path, output=output_path, dataset=ctx.dataset, root=ctx.root, newton_root=newton_root)
        for item in command
    ]
    env = os.environ.copy()
    env.update(
        {
            "AGENTIC_SIM2REAL_NEWTON_INPUT_JSON": str(input_path),
            "AGENTIC_SIM2REAL_NEWTON_OUTPUT_JSON": str(output_path),
            "AGENTIC_SIM2REAL_DATASET": str(ctx.dataset),
            "ISAACLAB_NEWTON_ROOT": newton_root,
        }
    )
    try:
        completed = subprocess.run(
            formatted_command,
            cwd=ctx.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=float(sysid_cfg.get("newton_timeout_s", 900)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        message = f"Newton SysID command could not complete: {exc}"
        evidence = _write_json(log_path, {"command": formatted_command, "error": str(exc)})
        return _newton_unavailable_result(manifest.skill_id, required, message, evidence)

    log_evidence = _write_json(
        log_path,
        {
            "command": formatted_command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "result_file": str(output_path),
        },
    )
    if completed.returncode != 0:
        message = f"Newton SysID command exited {completed.returncode}"
        return _newton_unavailable_result(manifest.skill_id, required, message, log_evidence)

    try:
        payload = json.loads(output_path.read_text()) if output_path.exists() else json.loads(completed.stdout)
    except Exception as exc:
        message = f"Newton SysID result could not be parsed: {exc}"
        return _newton_unavailable_result(manifest.skill_id, required, message, log_evidence)

    result = _sysid_backend_result_from_payload(
        manifest.skill_id,
        payload,
        sysid_cfg,
        required,
        skill_out,
        extra_evidence=[log_evidence],
    )
    result.metrics.setdefault("run_reason", "Newton ran because sysid.newton_command is configured.")
    return result


def _newton_unavailable_result(skill_id: str, required: bool, message: str, evidence: str) -> SkillResult:
    return _sysid_unavailable_result(skill_id, required, message, evidence, backend_key="newton_available")


def _sysid_unavailable_result(
    skill_id: str,
    required: bool,
    message: str,
    evidence: str,
    backend_key: str,
) -> SkillResult:
    return SkillResult(
        skill_id=skill_id,
        status="fail" if required else "evidence_missing",
        quality_score=0.0,
        confidence=1.0,
        blocking_failures=[message] if required else [],
        warnings=[] if required else [message],
        evidence_files=[evidence],
        metrics={
            backend_key: False,
            "fallback_skill": "sysid_step_response",
            "run_reason": "A configured fitted SysID backend was attempted and recorded as unavailable instead of stopping the whole pipeline.",
            "skip_reason": message,
        },
    )


def _sysid_backend_result_from_payload(
    skill_id: str,
    payload: dict[str, Any],
    sysid_cfg: dict[str, Any],
    required: bool,
    skill_out: Path,
    extra_evidence: list[str] | None = None,
    min_confidence_key: str = "min_newton_confidence",
    backend_label: str = "Newton SysID",
) -> SkillResult:
    confidence = float(payload.get("confidence", payload.get("metrics", {}).get("confidence", 0.0)))
    payload_status = str(payload.get("status", "pass")).lower()
    if payload_status not in ALLOWED_SKILL_STATUSES:
        payload_status = "pass"

    failures = [str(item) for item in payload.get("blocking_failures", [])]
    if payload_status == "fail" and not failures and payload.get("reason"):
        failures.append(str(payload["reason"]))
    if required and payload_status in {"skip", "evidence_missing", "not_applicable"}:
        failures.append(f"{backend_label} is required but the payload skipped")
    if payload_status not in {"skip", "evidence_missing", "not_applicable"}:
        min_confidence = float(sysid_cfg.get(min_confidence_key, 0.6))
        if confidence < min_confidence:
            failures.append(f"{backend_label} confidence {confidence:.3f} below required {min_confidence:.3f}")

    if failures or payload_status == "fail":
        status = "fail"
    elif payload_status in {"skip", "evidence_missing", "not_applicable"}:
        status = "evidence_missing" if payload_status != "not_applicable" else "not_applicable"
    else:
        status = "pass"

    evidence = _write_json(skill_out / f"{skill_id}.json", payload)
    evidence_files = [evidence]
    evidence_files.extend(extra_evidence or [])
    evidence_files.extend(str(item) for item in payload.get("evidence_files", []))
    deduped_evidence = list(dict.fromkeys(evidence_files))

    return SkillResult(
        skill_id=skill_id,
        status=status,
        quality_score=float(payload.get("quality_score", confidence)),
        confidence=confidence,
        blocking_failures=failures,
        warnings=[str(item) for item in payload.get("warnings", [])],
        evidence_files=deduped_evidence,
        metrics=dict(payload.get("metrics", {})),
    )


def _impl_pace_sysid(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    sysid_cfg = ctx.config.sysid
    required = bool(sysid_cfg.get("require_pace", False))
    preference = [str(item).strip().lower() for item in sysid_cfg.get("sysid_backend_preference", ["pace", "newton", "local"])]
    newton = previous.get("newton_sysid")
    if not required and newton and newton.status == "pass" and _prefers_newton(preference):
        evidence = _write_json(
            skill_out / "pace_sysid.json",
            {
                "status": "not_applicable",
                "reason": "Newton SysID passed and is preferred ahead of PACE.",
                "fallback_skill": "newton_sysid",
            },
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status="not_applicable",
            quality_score=0.0,
            confidence=1.0,
            warnings=["PACE SysID skipped because Newton is preferred ahead of PACE for this config"],
            evidence_files=[evidence],
            metrics={
                "pace_enabled": False,
                "fallback_skill": "newton_sysid",
                "run_reason": "PACE is checked so the run records whether the preferred fitted backend was needed.",
                "skip_reason": "Newton SysID passed and sysid_backend_preference puts Newton ahead of PACE.",
            },
        )

    command = [str(item) for item in sysid_cfg.get("pace_command", [])]
    pace_root = str(sysid_cfg.get("pace_root") or os.environ.get("PACE_SIM2REAL_ROOT", ""))
    enabled = bool(sysid_cfg.get("pace_enabled", False) or command or pace_root)
    input_path = skill_out / "pace_input.json"
    output_path = skill_out / "pace_output.json"
    input_payload = {
        "dataset": str(ctx.dataset),
        "config": ctx.config.merged(),
        "root": str(ctx.root),
        "pace_root": pace_root,
        "previous_results": {skill_id: result.to_dict() for skill_id, result in previous.items()},
        "expected_output_json": str(output_path),
    }
    input_path.write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n")

    if not enabled:
        status = "fail" if required else "evidence_missing"
        reason = "PACE SysID is required but is not enabled or configured." if required else "PACE SysID is not enabled; local log-based SysID remains the offline fallback."
        evidence = _write_json(
            skill_out / "pace_sysid.json",
            {"status": status, "reason": reason, "fallback_skill": "sysid_step_response", "input_file": str(input_path)},
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status=status,
            quality_score=0.0,
            confidence=1.0,
            blocking_failures=[reason] if required else [],
            warnings=[] if required else ["PACE SysID skipped; set sysid.pace_enabled plus sysid.pace_root or sysid.pace_command to enable fitted physics parameters"],
            evidence_files=[evidence],
            metrics={
                "pace_enabled": False,
                "fallback_skill": "sysid_step_response",
                "run_reason": "PACE is checked first because it is the default fitted SysID backend.",
                "skip_reason": "sysid.pace_enabled is false and neither sysid.pace_command nor sysid.pace_root is configured.",
            },
        )

    try:
        payload = run_pace_bridge(input_payload, work_dir=skill_out, output_path=output_path)
    except Exception as exc:
        message = f"PACE SysID bridge could not complete: {exc}"
        evidence = _write_json(skill_out / "pace_command_log.json", {"error": str(exc), "input_file": str(input_path)})
        return _sysid_unavailable_result(manifest.skill_id, required, message, evidence, backend_key="pace_available")
    result = _sysid_backend_result_from_payload(
        manifest.skill_id,
        payload,
        sysid_cfg,
        required,
        skill_out,
        extra_evidence=[str(output_path)] if output_path.exists() else [],
        min_confidence_key="min_pace_confidence",
        backend_label="PACE SysID",
    )
    result.metrics.setdefault("run_reason", "PACE ran because it is enabled or configured as the preferred fitted SysID backend.")
    return result


def _prefers_newton(preference: list[str]) -> bool:
    if "newton" not in preference:
        return False
    if "pace" not in preference:
        return True
    return preference.index("newton") < preference.index("pace")


def _prefers_pace(preference: list[str]) -> bool:
    if "pace" not in preference:
        return False
    if "newton" not in preference:
        return True
    return preference.index("pace") < preference.index("newton")


def _ordered_sysid_backend_subchecks(preference: list[str]) -> list[tuple[str, str, bool, bool]]:
    backend_specs = {
        "pace": ("pace_sysid", "pace_sysid", False, False),
        "newton": ("newton_sysid", "newton_sysid", False, False),
    }
    ordered: list[tuple[str, str, bool, bool]] = []
    seen: set[str] = set()
    for item in preference:
        backend = str(item).strip().lower()
        if backend in backend_specs and backend not in seen:
            ordered.append(backend_specs[backend])
            seen.add(backend)
    for backend in ("pace", "newton"):
        if backend not in seen:
            ordered.append(backend_specs[backend])
    return ordered


def _impl_domain_randomization_update(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    gap = estimate_gap(load_records(ctx.dataset), ctx.config)
    dr = gap["recommendations"]["domain_randomization"]
    video_friction = previous.get("video_contact_friction")
    video_friction_metrics = video_friction.metrics if video_friction and video_friction.status == "pass" else None
    dr = apply_video_friction_to_domain_randomization(dr, video_friction_metrics)
    failures = []
    pos_range = dr.get("object_pose_observation_noise", dr["shaft_pose_observation_noise"]).get("object_position_uniform_m", dr["shaft_pose_observation_noise"]["gear_shaft_pos_uniform_m"])
    if max(abs(float(v)) for v in pos_range) > 0.01:
        failures.append("object pose noise recommendation exceeds 1 cm safety cap")
    friction = dr["actuator_and_contact_randomization"]["friction_sweep_for_agent_experiments"]
    if friction[0] <= 0 or friction[1] > 1.5:
        failures.append("friction sweep moved outside conservative bounds")
    evidence = _write_json(skill_out / "domain_randomization_candidate.json", dr)
    metrics = {"object_pos_noise": pos_range, "friction_sweep": friction}
    if video_friction_metrics:
        metrics["video_friction_applied"] = True
        metrics["video_friction"] = video_friction_metrics.get("suggested_sim_params", {})
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=0.9 if not failures else 0.4,
        confidence=0.75,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics=metrics,
    )


def _impl_action_scale_sweep(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    gap = estimate_gap(load_records(ctx.dataset), ctx.config)
    action = gap["recommendations"]["action_scale"]
    nominal = float(action["nominal_from_tutorial_or_config"])
    candidates = sorted({round(max(0.005, nominal * factor), 5) for factor in [0.85, 1.0, 1.15]})
    suggested = float(action["suggested"])
    failures = []
    if suggested > float(ctx.config.robot.get("max_action_scale", 0.05)):
        failures.append("suggested action scale exceeds configured maximum")
    contact = gap["contact"]
    if float(contact.get("over_limit_ratio", 0.0)) > 0.05:
        failures.append("contact force over-limit ratio too high for action scale promotion")
    evidence = _write_json(
        skill_out / "action_scale_sweep.json",
        {"candidates": candidates, "suggested": suggested, "contact": contact, "action_scale": action},
    )
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=0.85 if not failures else 0.35,
        confidence=0.7,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics={"candidates": candidates, "suggested": suggested},
    )


def _impl_autoresearch_planner(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    gap = estimate_gap(load_records(ctx.dataset), ctx.config)
    plan = build_plan(gap, ctx.config)
    evidence = _write_json(skill_out / "autoresearch_plan.json", plan)
    failures = []
    if len(plan.get("experiments", [])) < 4:
        failures.append("AutoResearch plan must include at least four experiments")
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=0.9 if not failures else 0.3,
        confidence=0.8,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics={"experiment_count": len(plan.get("experiments", [])), "transfer_score": plan["transfer_score"]["score_0_to_1"]},
    )


def _impl_sim_eval_regression(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    baseline_path = ctx.root / "golden" / "sample_inputs" / "baseline_metrics.json"
    baseline = json.loads(baseline_path.read_text())
    gap = estimate_gap(load_records(ctx.dataset), ctx.config)
    candidate = {
        "success_rate": gap["summary"].get("success_rate") or 0.0,
        "peak_force_n": gap["contact"].get("peak_n") or 0.0,
        "pose_error_p95_m": gap.get("object_pose_noise", gap["shaft_pose_noise"]).get("position_error_p95_m") or 1.0,
    }
    failures = []
    if candidate["peak_force_n"] > float(ctx.config.safety.get("max_contact_force_n", 80.0)):
        failures.append("candidate peak force exceeds configured safety limit")
    if candidate["pose_error_p95_m"] > float(ctx.config.perception.get("pose_error_gate_m", 0.01)):
        failures.append("candidate pose p95 error exceeds calibration gate")
    evidence = _write_json(skill_out / "sim_eval_regression.json", {"baseline": baseline, "candidate": candidate})
    success_delta = candidate["success_rate"] - float(baseline.get("success_rate", 0.0))
    score = 0.8 + min(0.2, max(0.0, success_delta))
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=score if not failures else 0.4,
        confidence=0.65,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics={"success_delta": round(success_delta, 3), "candidate": candidate},
    )


def _impl_isaaclab_rollout_regression(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    required = release_requires(ctx.config, "require_isaaclab_rollout_for_human_review")
    waiver = release_waiver(ctx.config, "allow_isaaclab_rollout_waiver", "isaaclab_rollout_waiver_reason")
    metrics_path_cfg = str(ctx.config.isaac_lab.get("rollout_metrics_path", "")).strip()
    command = [str(item) for item in ctx.config.isaac_lab.get("rollout_command", [])]
    input_path = skill_out / "isaaclab_rollout_input.json"
    output_path = skill_out / "isaaclab_rollout_output.json"
    input_payload = {
        "dataset": str(ctx.dataset),
        "config": ctx.config.merged(),
        "root": str(ctx.root),
        "skill_out": str(skill_out),
        "expected_output_json": str(output_path),
        "previous_results": {skill_id: result.to_dict() for skill_id, result in previous.items()},
    }
    input_path.write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n")

    payload: dict[str, Any] | None = None
    evidence_files = [str(input_path)]
    source = "not_configured"
    failures: list[str] = []
    warnings: list[str] = []

    if metrics_path_cfg:
        metrics_path = Path(metrics_path_cfg).expanduser()
        if not metrics_path.is_absolute():
            metrics_path = ctx.root / metrics_path
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text())
            source = "metrics_file"
            evidence_files.append(str(metrics_path))
        else:
            failures.append(f"configured Isaac Lab rollout metrics file does not exist: {metrics_path}")
    elif command:
        formatted = [item.format(input=input_path, output=output_path, dataset=ctx.dataset, root=ctx.root) for item in command]
        env = os.environ.copy()
        env.update(
            {
                "AGENTIC_SIM2REAL_ROLLOUT_INPUT_JSON": str(input_path),
                "AGENTIC_SIM2REAL_ROLLOUT_OUTPUT_JSON": str(output_path),
                "AGENTIC_SIM2REAL_DATASET": str(ctx.dataset),
            }
        )
        try:
            completed = subprocess.run(
                formatted,
                cwd=ctx.root,
                env=env,
                text=True,
                capture_output=True,
                timeout=float(ctx.config.isaac_lab.get("rollout_timeout_s", 1800)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"Isaac Lab rollout command could not complete: {exc}")
            completed = None
        log = _write_json(
            skill_out / "isaaclab_rollout_command_log.json",
            {
                "command": formatted,
                "returncode": completed.returncode if completed else None,
                "stdout": completed.stdout if completed else "",
                "stderr": completed.stderr if completed else "",
                "result_file": str(output_path),
            },
        )
        evidence_files.append(log)
        if completed and completed.returncode != 0:
            failures.append(f"Isaac Lab rollout command exited {completed.returncode}")
        elif completed:
            try:
                payload = json.loads(output_path.read_text()) if output_path.exists() else json.loads(completed.stdout)
                source = "rollout_command"
            except Exception as exc:
                failures.append(f"Isaac Lab rollout result could not be parsed: {exc}")

    if payload is None:
        if failures:
            status = "evidence_missing" if required else "not_applicable"
            evidence = _write_json(
                skill_out / "isaaclab_rollout_regression.json",
                {"status": status, "source": source, "failures": failures, "release_profile": release_profile(ctx.config)},
            )
            return SkillResult(
                skill_id=manifest.skill_id,
                status=status,
                quality_score=0.0,
                confidence=0.0,
                blocking_failures=failures if required else [],
                warnings=[] if required else failures,
                evidence_files=[*evidence_files, evidence],
                metrics={"isaaclab_rollout_ran": False, "source": source},
            )
        if required and not waiver["allowed"]:
            failures.append("release-candidate profile requires true Isaac Lab rollout regression evidence")
            status = "evidence_missing"
        else:
            status = "not_applicable"
            if waiver["allowed"]:
                warnings.append(f"Isaac Lab rollout requirement waived: {waiver['reason']}")
            else:
                warnings.append("Isaac Lab rollout regression not configured; smoke profile treats this as not applicable")
        evidence = _write_json(
            skill_out / "isaaclab_rollout_regression.json",
            {"status": status, "source": source, "warnings": warnings, "failures": failures, "release_profile": release_profile(ctx.config)},
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status=status,
            quality_score=0.0,
            confidence=1.0 if status == "not_applicable" else 0.0,
            blocking_failures=failures,
            warnings=warnings,
            evidence_files=[*evidence_files, evidence],
            metrics={"isaaclab_rollout_ran": False, "source": source},
        )

    metrics = dict(payload.get("metrics", payload))
    episodes = int(metrics.get("episodes", metrics.get("num_episodes", metrics.get("rollout_episodes", 0))) or 0)
    success_rate = _optional_metric(metrics, "success_rate")
    peak_force = _optional_metric(metrics, "peak_force_n")
    pose_error = _optional_metric(metrics, "pose_error_p95_m")
    min_episodes = int(ctx.config.isaac_lab.get("rollout_min_episodes", 20))
    success_floor = float(ctx.config.isaac_lab.get("rollout_success_rate_floor", 0.7))
    if episodes < min_episodes:
        failures.append(f"Isaac Lab rollout episodes {episodes} below required {min_episodes}")
    if success_rate is None:
        failures.append("Isaac Lab rollout metrics must include success_rate")
    elif success_rate < success_floor:
        failures.append(f"Isaac Lab rollout success_rate {success_rate:.3f} below floor {success_floor:.3f}")
    if peak_force is not None and peak_force > float(ctx.config.safety.get("max_contact_force_n", 80.0)):
        failures.append("Isaac Lab rollout peak_force_n exceeds configured safety limit")
    if pose_error is not None and pose_error > float(ctx.config.perception.get("pose_error_gate_m", 0.01)):
        failures.append("Isaac Lab rollout pose_error_p95_m exceeds configured pose gate")

    evidence = _write_json(
        skill_out / "isaaclab_rollout_regression.json",
        {"status": "fail" if failures else "pass", "source": source, "metrics": metrics, "payload": payload},
    )
    confidence = min(1.0, episodes / max(float(min_episodes), 1.0))
    quality = confidence if not failures else min(0.4, confidence)
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=quality,
        confidence=confidence,
        blocking_failures=failures,
        warnings=warnings,
        evidence_files=[*evidence_files, evidence],
        metrics={**metrics, "isaaclab_rollout_ran": True, "source": source},
    )


def _impl_release_candidate_gate(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    flat_previous = _flatten_previous_results(previous)
    failures = []
    requirement_checks = {
        "release_profile": release_profile(ctx.config),
        "physics_sysid": "not_required",
        "camera_video_evidence": "not_required",
        "contact_friction_video": "not_required",
        "heldout_session": "not_required",
        "isaaclab_rollout": "not_required",
        "policy_artifacts": "not_required",
    }
    for skill_id, result in previous.items():
        if skill_id == manifest.skill_id:
            continue
        if result.release_blocking and result.status in BLOCKING_SKILL_STATUSES:
            failures.append(f"{skill_id} failed: {result.blocking_failures}")

    if release_requires(ctx.config, "require_physics_sysid_for_human_review"):
        newton = flat_previous.get("newton_sysid")
        pace = flat_previous.get("pace_sysid")
        physics_passed = bool((newton and newton.status == "pass") or (pace and pace.status == "pass"))
        waiver = release_waiver(ctx.config, "allow_sysid_waiver", "sysid_waiver_reason")
        if physics_passed:
            requirement_checks["physics_sysid"] = "pass"
        elif waiver["allowed"]:
            requirement_checks["physics_sysid"] = f"waived: {waiver['reason']}"
        else:
            requirement_checks["physics_sysid"] = "fail"
            failures.append("release-candidate profile requires PACE or Newton SysID evidence, or an explicit SysID waiver")

    if bool(ctx.config.release.get("require_camera_video_for_human_review", False)):
        camera_video = flat_previous.get("video_camera_tuning")
        waiver = release_waiver(ctx.config, "allow_camera_video_waiver", "camera_video_waiver_reason")
        if camera_video and camera_video.status == "pass":
            requirement_checks["camera_video_evidence"] = "pass"
        elif waiver["allowed"]:
            requirement_checks["camera_video_evidence"] = f"waived: {waiver['reason']}"
        else:
            requirement_checks["camera_video_evidence"] = "fail"
            failures.append("release profile requires uploaded camera video analysis for camera parameter tuning")

    if bool(ctx.config.release.get("require_contact_video_for_human_review", False)):
        friction_video = flat_previous.get("video_contact_friction")
        waiver = release_waiver(ctx.config, "allow_contact_video_waiver", "contact_video_waiver_reason")
        if friction_video and friction_video.status == "pass":
            requirement_checks["contact_friction_video"] = "pass"
        elif waiver["allowed"]:
            requirement_checks["contact_friction_video"] = f"waived: {waiver['reason']}"
        else:
            requirement_checks["contact_friction_video"] = "fail"
            failures.append("release profile requires uploaded contact/friction video analysis for object and gripper friction tuning")

    if release_requires(ctx.config, "require_heldout_session_for_human_review"):
        data_gate = flat_previous.get("real_data_quality_gate")
        metrics = data_gate.metrics if data_gate else {}
        heldout_episodes = int(metrics.get("heldout_episodes", 0) or 0)
        heldout_min = int(ctx.config.release.get("heldout_min_episodes", 1))
        waiver = release_waiver(ctx.config, "allow_heldout_waiver", "heldout_waiver_reason")
        if heldout_episodes >= heldout_min:
            requirement_checks["heldout_session"] = "pass"
        elif waiver["allowed"]:
            requirement_checks["heldout_session"] = f"waived: {waiver['reason']}"
        else:
            requirement_checks["heldout_session"] = "fail"
            failures.append(f"release-candidate profile requires at least {heldout_min} held-out episode(s)")

    if release_requires(ctx.config, "require_isaaclab_rollout_for_human_review"):
        rollout = flat_previous.get("isaaclab_rollout_regression")
        waiver = release_waiver(ctx.config, "allow_isaaclab_rollout_waiver", "isaaclab_rollout_waiver_reason")
        if rollout and rollout.status == "pass":
            requirement_checks["isaaclab_rollout"] = "pass"
        elif waiver["allowed"]:
            requirement_checks["isaaclab_rollout"] = f"waived: {waiver['reason']}"
        else:
            requirement_checks["isaaclab_rollout"] = "fail"
            failures.append("release-candidate profile requires true Isaac Lab rollout regression evidence")

    if release_requires(ctx.config, "require_user_policy_artifacts_for_human_review"):
        policy = flat_previous.get("policy_artifact_audit")
        using_sample = bool(policy and policy.metrics.get("using_sample_artifacts", False))
        if policy and policy.status == "pass" and not using_sample:
            requirement_checks["policy_artifacts"] = "pass"
        elif bool(ctx.config.policy.get("allow_policy_artifact_waiver", False)) and str(ctx.config.policy.get("policy_artifact_waiver_reason", "")).strip():
            requirement_checks["policy_artifacts"] = f"waived: {ctx.config.policy['policy_artifact_waiver_reason']}"
        else:
            requirement_checks["policy_artifacts"] = "fail"
            failures.append("release-candidate profile requires user-provided policy artifacts")
    evidence = _write_json(
        skill_out / "release_gate_review.json",
        {
            "previous_skills": {skill_id: result.to_dict() for skill_id, result in previous.items()},
            "subchecks": {
                skill_id: result.to_dict()
                for skill_id, result in sorted(flat_previous.items())
                if skill_id not in previous
            },
            "blocking_failures": failures,
            "requirement_checks": requirement_checks,
            "safe_to_autorun_robot": False,
        },
    )
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=0.95 if not failures else 0.0,
        confidence=1.0,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics={"safe_to_autorun_robot": False},
    )


def _impl_real_robot_gate(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    try:
        require_real_robot_gate(ctx.config)
    except SystemExit as exc:
        evidence = _write_json(skill_out / "real_robot_gate.json", {"blocked": True, "reason": str(exc)})
        return SkillResult(
            skill_id=manifest.skill_id,
            status="fail",
            quality_score=0.0,
            confidence=1.0,
            blocking_failures=[str(exc)],
            evidence_files=[evidence],
            metrics={"safe_to_autorun_robot": False},
        )
    evidence = _write_json(skill_out / "real_robot_gate.json", {"blocked": False, "safe_to_autorun_robot": False})
    return SkillResult(
        skill_id=manifest.skill_id,
        status="pass",
        quality_score=1.0,
        confidence=1.0,
        warnings=["human gate env var is present; this still does not authorize unattended robot motion"],
        evidence_files=[evidence],
        metrics={"safe_to_autorun_robot": False},
    )


def _subcheck_manifest(
    skill_id: str,
    implementation: str,
    *,
    release_blocking: bool = True,
    human_required: bool = False,
    real_robot: bool = False,
) -> SkillManifest:
    return SkillManifest(
        path=Path("skills") / skill_id / "skill.json",
        data={
            "id": skill_id,
            "name": skill_id,
            "owner_agent": "internal_subcheck",
            "description": f"Internal subcheck for consolidated skill: {skill_id}",
            "implementation": implementation,
            "inputs": [],
            "outputs": [],
            "validators": ["validated by consolidated parent skill"],
            "quality_gate": {"min_score": SUBCHECK_MIN_SCORES.get(skill_id, 0.0)},
            "human_required": human_required,
            "release_blocking": release_blocking,
            "real_robot": real_robot,
        },
    )


def _run_subcheck(
    skill_id: str,
    implementation: str,
    ctx: HarnessContext,
    parent_out: Path,
    previous: dict[str, SkillResult],
    *,
    release_blocking: bool = True,
    human_required: bool = False,
) -> SkillResult:
    impl = INTERNAL_IMPLEMENTATIONS[implementation]
    manifest = _subcheck_manifest(
        skill_id,
        implementation,
        release_blocking=release_blocking,
        human_required=human_required,
    )
    sub_out = parent_out / "subchecks" / skill_id
    sub_out.mkdir(parents=True, exist_ok=True)
    try:
        result = impl(manifest, ctx, sub_out, previous)
    except Exception as exc:
        result = _exception_result(manifest, sub_out, exc)
    result.release_blocking = release_blocking
    result.human_required = human_required
    result = _apply_quality_gate(manifest, result)
    (sub_out / "result.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    return result


def _aggregate_subchecks(
    manifest: SkillManifest,
    skill_out: Path,
    artifact_name: str,
    subchecks: dict[str, SkillResult],
    *,
    blocking_statuses: dict[str, set[str]] | None = None,
    metrics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> SkillResult:
    blocking_statuses = blocking_statuses or {}
    failures: list[str] = []
    inherited_warnings = list(warnings or [])
    for skill_id, result in subchecks.items():
        statuses = blocking_statuses.get(skill_id)
        if statuses is None:
            statuses = BLOCKING_SKILL_STATUSES if result.release_blocking else {"fail"}
        if result.status in statuses:
            details = result.blocking_failures or result.warnings or [result.status]
            failures.append(f"{skill_id}: {details}")
        inherited_warnings.extend(f"{skill_id}: {warning}" for warning in result.warnings)

    pass_scores = [result.quality_score for result in subchecks.values() if result.status == "pass"]
    confidences = [result.confidence for result in subchecks.values()]
    status = "fail" if failures else "pass"
    quality = sum(pass_scores) / len(pass_scores) if pass_scores else 0.0
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    subcheck_payload = {skill_id: result.to_dict() for skill_id, result in subchecks.items()}
    aggregate_metrics = {
        "subchecks": subcheck_payload,
        "subcheck_statuses": {skill_id: result.status for skill_id, result in subchecks.items()},
        **(metrics or {}),
    }
    evidence = _write_json(
        skill_out / artifact_name,
        {
            "status": status,
            "blocking_failures": failures,
            "warnings": inherited_warnings,
            "subchecks": subcheck_payload,
            "metrics": metrics or {},
        },
    )
    evidence_files = [evidence]
    for result in subchecks.values():
        evidence_files.extend(result.evidence_files)
    return SkillResult(
        skill_id=manifest.skill_id,
        status=status,
        quality_score=quality,
        confidence=confidence,
        blocking_failures=failures,
        warnings=inherited_warnings,
        evidence_files=list(dict.fromkeys(evidence_files)),
        metrics=aggregate_metrics,
    )


def _impl_project_preflight(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    subchecks: dict[str, SkillResult] = {}
    local_previous = dict(previous)
    for skill_id, implementation in [
        ("env_preflight", "env_preflight"),
        ("ros_preflight", "ros_preflight"),
        ("isaaclab_task_check", "isaaclab_task_check"),
        ("policy_artifact_audit", "policy_artifact_audit"),
    ]:
        result = _run_subcheck(skill_id, implementation, ctx, skill_out, local_previous)
        subchecks[skill_id] = result
        local_previous[skill_id] = result
    policy = subchecks["policy_artifact_audit"]
    env = subchecks["env_preflight"]
    return _aggregate_subchecks(
        manifest,
        skill_out,
        "project_preflight.json",
        subchecks,
        metrics={
            "required_commands_present": env.metrics.get("required_commands_present"),
            "newton_available": env.metrics.get("newton_available"),
            "pace_available": env.metrics.get("pace_available"),
            "local_sysid_available": env.metrics.get("local_sysid_available"),
            "using_sample_artifacts": policy.metrics.get("using_sample_artifacts"),
            "artifact_completeness": policy.metrics.get("artifact_completeness"),
        },
    )


def _impl_real_data_evidence_gate(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    subchecks: dict[str, SkillResult] = {}
    local_previous = _flatten_previous_results(previous)
    for skill_id, implementation, human_required in [
        ("real_data_quality_gate", "real_data_quality_gate", False),
        ("pose_repeatability", "pose_repeatability", True),
        ("video_camera_tuning", "video_camera_tuning", True),
    ]:
        result = _run_subcheck(
            skill_id,
            implementation,
            ctx,
            skill_out,
            local_previous,
            release_blocking=skill_id != "video_camera_tuning",
            human_required=human_required,
        )
        subchecks[skill_id] = result
        local_previous[skill_id] = result
    data_quality = subchecks["real_data_quality_gate"]
    pose = subchecks["pose_repeatability"]
    return _aggregate_subchecks(
        manifest,
        skill_out,
        "real_data_evidence_gate.json",
        subchecks,
        metrics={
            **data_quality.metrics,
            "pose_repeatability": pose.metrics,
            "video_camera_tuning": subchecks["video_camera_tuning"].metrics,
        },
    )


def _impl_physics_sysid(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    subchecks: dict[str, SkillResult] = {}
    local_previous = _flatten_previous_results(previous)
    preference = [str(item).strip().lower() for item in ctx.config.sysid.get("sysid_backend_preference", ["pace", "newton", "local"])]
    check_order = [
        ("sysid_step_response", "sysid_step_response", True, True),
        ("video_contact_friction", "video_contact_friction", False, True),
    ]
    check_order.extend(_ordered_sysid_backend_subchecks(preference))
    for skill_id, implementation, release_blocking, human_required in check_order:
        result = _run_subcheck(
            skill_id,
            implementation,
            ctx,
            skill_out,
            local_previous,
            release_blocking=release_blocking,
            human_required=human_required,
        )
        subchecks[skill_id] = result
        local_previous[skill_id] = result
    local = subchecks["sysid_step_response"]
    newton = subchecks["newton_sysid"]
    pace = subchecks["pace_sysid"]
    physics_required = release_requires(ctx.config, "require_physics_sysid_for_human_review")
    waiver = release_waiver(ctx.config, "allow_sysid_waiver", "sysid_waiver_reason")
    backend_passed = newton.status == "pass" or pace.status == "pass"
    result = _aggregate_subchecks(
        manifest,
        skill_out,
        "physics_sysid.json",
        subchecks,
        blocking_statuses={
            "newton_sysid": {"fail"},
            "pace_sysid": {"fail"},
        },
        metrics={
            **local.metrics,
            "run_reason": "Physics SysID runs in characterization because actuator, contact, friction, and fitted backend signals feed simulator parameter patches.",
            "sysid_backend_preference": preference,
            "sysid_backend_order": [skill_id for skill_id, *_ in check_order if skill_id in {"pace_sysid", "newton_sysid"}],
            "local_log_estimator": "used",
            "video_contact_friction": subchecks["video_contact_friction"].status,
            "video_contact_friction_metrics": subchecks["video_contact_friction"].metrics,
            "newton_sysid": newton.status,
            "pace_sysid": pace.status,
            "physics_backend_passed": backend_passed,
            "physics_required": physics_required,
            "sysid_waiver": waiver,
        },
    )
    if physics_required and not backend_passed and not waiver["allowed"]:
        result.status = "fail"
        result.blocking_failures.append(
            "physics profile requires PACE or Newton SysID evidence, but neither backend passed"
        )
    if bool(ctx.config.sysid.get("require_newton", False)) and newton.status != "pass":
        result.status = "fail"
        result.blocking_failures.append("config.sysid.require_newton is true, but Newton SysID did not pass")
    if bool(ctx.config.sysid.get("require_pace", False)) and pace.status != "pass":
        result.status = "fail"
        result.blocking_failures.append("config.sysid.require_pace is true, but PACE SysID did not pass")
    return result


def _impl_agentic_tuning_plan(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    subchecks: dict[str, SkillResult] = {}
    local_previous = _flatten_previous_results(previous)
    for skill_id, implementation in [
        ("domain_randomization_update", "domain_randomization_update"),
        ("action_scale_sweep", "action_scale_sweep"),
        ("autoresearch_planner", "autoresearch_planner"),
    ]:
        result = _run_subcheck(skill_id, implementation, ctx, skill_out, local_previous)
        subchecks[skill_id] = result
        local_previous[skill_id] = result
    dr = subchecks["domain_randomization_update"]
    action = subchecks["action_scale_sweep"]
    plan = subchecks["autoresearch_planner"]
    camera = local_previous.get("video_camera_tuning")
    friction = local_previous.get("video_contact_friction")
    gap = estimate_gap(load_records(ctx.dataset), ctx.config)
    patch_dr = gap["recommendations"]["domain_randomization"]
    if friction and friction.status == "pass":
        patch_dr = apply_video_friction_to_domain_randomization(patch_dr, friction.metrics)
    patch_action = dict(gap["recommendations"]["action_scale"])
    if action.metrics.get("candidates") is not None:
        patch_action["candidates"] = action.metrics["candidates"]
    if action.metrics.get("suggested") is not None:
        patch_action["suggested"] = action.metrics["suggested"]
    sim_patch = build_sim_params_patch(
        config=ctx.config,
        dataset_path=ctx.dataset,
        domain_randomization=patch_dr,
        action_scale=patch_action,
        camera_metrics=camera.metrics if camera else {},
        friction_metrics=friction.metrics if friction else {},
    )
    sim_patch_yaml = write_sim_params_patch(skill_out / "sim_params_patch.yaml", sim_patch)
    sim_patch_json = write_sim_params_patch_json(skill_out / "sim_params_patch.json", sim_patch)
    result = _aggregate_subchecks(
        manifest,
        skill_out,
        "agentic_tuning_plan.json",
        subchecks,
        metrics={
            "object_pos_noise": dr.metrics.get("object_pos_noise"),
            "friction_sweep": dr.metrics.get("friction_sweep"),
            "video_friction_applied": dr.metrics.get("video_friction_applied", False),
            "video_friction": dr.metrics.get("video_friction", {}),
            "candidates": action.metrics.get("candidates"),
            "suggested": action.metrics.get("suggested"),
            "experiment_count": plan.metrics.get("experiment_count"),
            "transfer_score": plan.metrics.get("transfer_score"),
            "sim_params_patch": sim_patch_yaml,
            "sim_params_patch_json": sim_patch_json,
            "sim_params_patch_operation_count": len(sim_patch.get("operations", [])),
        },
    )
    result.evidence_files.extend([sim_patch_yaml, sim_patch_json])
    result.evidence_files = list(dict.fromkeys(result.evidence_files))
    return result


def _impl_regression_evaluation(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    subchecks: dict[str, SkillResult] = {}
    local_previous = _flatten_previous_results(previous)
    for skill_id, implementation in [
        ("sim_eval_regression", "sim_eval_regression"),
        ("isaaclab_rollout_regression", "isaaclab_rollout_regression"),
    ]:
        result = _run_subcheck(skill_id, implementation, ctx, skill_out, local_previous)
        subchecks[skill_id] = result
        local_previous[skill_id] = result
    sim_eval = subchecks["sim_eval_regression"]
    rollout = subchecks["isaaclab_rollout_regression"]
    rollout_metrics = rollout.metrics if isinstance(rollout.metrics, dict) else {}
    return _aggregate_subchecks(
        manifest,
        skill_out,
        "regression_evaluation.json",
        subchecks,
        metrics={
            **sim_eval.metrics,
            "isaaclab_rollout": rollout_metrics,
            "isaaclab_rollout_status": rollout.status,
        },
    )


def _flatten_previous_results(previous: dict[str, SkillResult]) -> dict[str, SkillResult]:
    flattened = dict(previous)
    for result in previous.values():
        subchecks = result.metrics.get("subchecks", {}) if isinstance(result.metrics, dict) else {}
        if not isinstance(subchecks, dict):
            continue
        for skill_id, payload in subchecks.items():
            if skill_id not in flattened and isinstance(payload, dict):
                flattened[skill_id] = _result_from_dict(skill_id, payload)
    return flattened


def _result_from_dict(skill_id: str, payload: dict[str, Any]) -> SkillResult:
    return SkillResult(
        skill_id=skill_id,
        status=str(payload.get("status", "unknown")),
        quality_score=float(payload.get("quality_score", 0.0) or 0.0),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        blocking_failures=[str(item) for item in payload.get("blocking_failures", [])],
        warnings=[str(item) for item in payload.get("warnings", [])],
        evidence_files=[str(item) for item in payload.get("evidence_files", [])],
        metrics=dict(payload.get("metrics", {})),
        human_required=bool(payload.get("human_required", False)),
        release_blocking=bool(payload.get("release_blocking", False)),
    )


INTERNAL_IMPLEMENTATIONS: dict[str, Callable[[SkillManifest, HarnessContext, Path, dict[str, SkillResult]], SkillResult]] = {
    "env_preflight": _impl_env_preflight,
    "isaaclab_task_check": _impl_isaaclab_task_check,
    "policy_artifact_audit": _impl_policy_artifact_audit,
    "ros_preflight": _impl_ros_preflight,
    "real_data_quality_gate": _impl_real_data_quality_gate,
    "pose_repeatability": _impl_pose_repeatability,
    "video_camera_tuning": _impl_video_camera_tuning,
    "sysid_step_response": _impl_sysid_step_response,
    "video_contact_friction": _impl_video_contact_friction,
    "newton_sysid": _impl_newton_sysid,
    "pace_sysid": _impl_pace_sysid,
    "domain_randomization_update": _impl_domain_randomization_update,
    "action_scale_sweep": _impl_action_scale_sweep,
    "autoresearch_planner": _impl_autoresearch_planner,
    "sim_eval_regression": _impl_sim_eval_regression,
    "isaaclab_rollout_regression": _impl_isaaclab_rollout_regression,
}


IMPLEMENTATIONS: dict[str, Callable[[SkillManifest, HarnessContext, Path, dict[str, SkillResult]], SkillResult]] = {
    "project_preflight": _impl_project_preflight,
    "real_data_evidence_gate": _impl_real_data_evidence_gate,
    "physics_sysid": _impl_physics_sysid,
    "agentic_tuning_plan": _impl_agentic_tuning_plan,
    "regression_evaluation": _impl_regression_evaluation,
    "env_preflight": _impl_env_preflight,
    "isaaclab_task_check": _impl_isaaclab_task_check,
    "policy_artifact_audit": _impl_policy_artifact_audit,
    "ros_preflight": _impl_ros_preflight,
    "real_data_quality_gate": _impl_real_data_quality_gate,
    "pose_repeatability": _impl_pose_repeatability,
    "video_camera_tuning": _impl_video_camera_tuning,
    "sysid_step_response": _impl_sysid_step_response,
    "video_contact_friction": _impl_video_contact_friction,
    "newton_sysid": _impl_newton_sysid,
    "pace_sysid": _impl_pace_sysid,
    "domain_randomization_update": _impl_domain_randomization_update,
    "action_scale_sweep": _impl_action_scale_sweep,
    "autoresearch_planner": _impl_autoresearch_planner,
    "sim_eval_regression": _impl_sim_eval_regression,
    "isaaclab_rollout_regression": _impl_isaaclab_rollout_regression,
    "release_candidate_gate": _impl_release_candidate_gate,
    "real_robot_gate": _impl_real_robot_gate,
}
