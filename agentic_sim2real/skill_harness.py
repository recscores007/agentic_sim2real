from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .autoresearch import build_plan
from .config import PipelineConfig, choose_task, load_config, nominal_action_scale
from .data_quality import evaluate_real_data_quality
from .dataset import load_records
from .newton_bridge import run_newton_bridge
from .pace_bridge import run_pace_bridge
from .preflight import run_preflight
from .real_data import ensure_aligned_dataset
from .safety import require_real_robot_gate
from .sysid import estimate_gap


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
) -> dict[str, Any]:
    ctx, manifests = prepare_harness(
        root=root,
        config_path=config_path,
        dataset_path=dataset_path,
        out_dir=out_dir,
        include_real=include_real,
        skill_dirs=skill_dirs,
    )
    ordered = ordered_skill_ids(manifests)
    if only_skill:
        ordered = [skill_id for skill_id in ordered if skill_id == only_skill]
        if not ordered:
            raise ValueError(f"Unknown skill: {only_skill}")

    results: dict[str, SkillResult] = {}
    ctx.skill_dir.mkdir(parents=True, exist_ok=True)
    for skill_id in ordered:
        results[skill_id] = run_skill_step(manifests, ctx, skill_id, results)

    return write_harness_artifacts(ctx, manifests, results)


def prepare_harness(
    root: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path,
    include_real: bool = False,
    skill_dirs: list[str | Path] | None = None,
) -> tuple[HarnessContext, dict[str, SkillManifest]]:
    root_path = Path(root).resolve()
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_dataset_path = ensure_aligned_dataset(dataset_path, root=root_path).resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ctx = HarnessContext(
        root=root_path,
        config_path=resolved_config_path,
        config=load_config(resolved_config_path),
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
        result = SkillResult(
            skill_id=skill_id,
            status="skip",
            quality_score=1.0,
            confidence=1.0,
            warnings=["real robot skill skipped by default harness; run with --include-real after human approval"],
            human_required=manifest.human_required,
            release_blocking=manifest.release_blocking,
        )
    else:
        result = _run_one(manifest, ctx, skill_out, previous_results)
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
    scoreboard = _scoreboard(results, manifests, require_all_release_blocking=require_all_release_blocking)
    (ctx.out_dir / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n")
    (ctx.out_dir / "release_candidate.json").write_text(
        json.dumps(_release_candidate(results, scoreboard), indent=2, sort_keys=True) + "\n"
    )
    return scoreboard


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
    if status not in {"pass", "fail", "skip"}:
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
    results: dict[str, SkillResult],
    manifests: dict[str, SkillManifest],
    require_all_release_blocking: bool = False,
) -> dict[str, Any]:
    blocking_failures = []
    scores = []
    for skill_id, result in results.items():
        if result.status == "pass":
            scores.append(result.quality_score)
        if manifests[skill_id].release_blocking and result.status == "fail":
            blocking_failures.append({"skill_id": skill_id, "failures": result.blocking_failures})
    if require_all_release_blocking:
        for skill_id, manifest in sorted(manifests.items()):
            if manifest.release_blocking and skill_id not in results:
                blocking_failures.append({"skill_id": skill_id, "failures": ["release-blocking skill did not run"]})
    quality = sum(scores) / len(scores) if scores else 0.0
    return {
        "status": "pass" if not blocking_failures else "fail",
        "quality_score": round(quality, 3),
        "blocking_failures": blocking_failures,
        "skills": {skill_id: result.to_dict() for skill_id, result in results.items()},
    }


def _release_candidate(results: dict[str, SkillResult], scoreboard: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "promote_to_human_review" if scoreboard["status"] == "pass" else "blocked",
        "safe_to_autorun_robot": False,
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
    artifact_dir = ctx.root / "golden" / "sample_inputs" / "policy_artifacts"
    required = ["agent.yaml", "env.yaml", "checkpoint.meta.json"]
    present = {name: (artifact_dir / name).exists() for name in required}
    failures = [f"missing policy artifact sample: {name}" for name, ok in present.items() if not ok]
    evidence = _write_json(skill_out / "policy_artifact_audit.json", {"artifact_dir": str(artifact_dir), "present": present})
    score = sum(1 for ok in present.values() if ok) / len(present)
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=score,
        confidence=0.8,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics={"artifact_completeness": round(score, 3)},
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
        metrics=report["metrics"],
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
        evidence = _write_json(
            skill_out / "newton_sysid.json",
            {
                "status": "skip",
                "reason": "IsaacLab-Newton is not enabled; local log-based SysID remains the active fallback.",
                "fallback_skill": "sysid_step_response",
                "input_file": str(input_path),
            },
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status="skip",
            quality_score=1.0,
            confidence=1.0,
            warnings=["Newton SysID skipped; set sysid.newton_enabled plus sysid.newton_root or sysid.newton_command to enable it"],
            evidence_files=[evidence],
            metrics={"newton_enabled": False, "fallback_skill": "sysid_step_response"},
        )

    if not command:
        try:
            payload = run_newton_bridge(input_payload, work_dir=skill_out, output_path=output_path)
        except Exception as exc:
            message = f"Built-in Newton bridge could not complete: {exc}"
            evidence = _write_json(log_path, {"error": str(exc), "input_file": str(input_path)})
            return _newton_unavailable_result(manifest.skill_id, required, message, evidence)
        return _sysid_backend_result_from_payload(
            manifest.skill_id,
            payload,
            sysid_cfg,
            required,
            skill_out,
            extra_evidence=[str(output_path)] if output_path.exists() else [],
        )

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

    return _sysid_backend_result_from_payload(
        manifest.skill_id,
        payload,
        sysid_cfg,
        required,
        skill_out,
        extra_evidence=[log_evidence],
    )


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
        status="fail" if required else "skip",
        quality_score=0.0 if required else 1.0,
        confidence=1.0,
        blocking_failures=[message] if required else [],
        warnings=[] if required else [message],
        evidence_files=[evidence],
        metrics={backend_key: False, "fallback_skill": "sysid_step_response"},
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
    if payload_status not in {"pass", "fail", "skip"}:
        payload_status = "pass"

    failures = [str(item) for item in payload.get("blocking_failures", [])]
    if payload_status == "fail" and not failures and payload.get("reason"):
        failures.append(str(payload["reason"]))
    if required and payload_status == "skip":
        failures.append(f"{backend_label} is required but the payload skipped")
    if payload_status != "skip":
        min_confidence = float(sysid_cfg.get(min_confidence_key, 0.6))
        if confidence < min_confidence:
            failures.append(f"{backend_label} confidence {confidence:.3f} below required {min_confidence:.3f}")

    if failures or payload_status == "fail":
        status = "fail"
    elif payload_status == "skip":
        status = "skip"
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
    preference = [str(item) for item in sysid_cfg.get("sysid_backend_preference", ["newton", "pace", "local"])]
    newton = previous.get("newton_sysid")
    if newton and newton.status == "pass" and _prefers_newton(preference):
        evidence = _write_json(
            skill_out / "pace_sysid.json",
            {
                "status": "skip",
                "reason": "Newton SysID passed and is preferred ahead of PACE.",
                "fallback_skill": "newton_sysid",
            },
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status="skip",
            quality_score=1.0,
            confidence=1.0,
            warnings=["PACE backup skipped because Newton SysID passed"],
            evidence_files=[evidence],
            metrics={"pace_enabled": False, "fallback_skill": "newton_sysid"},
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
        reason = "PACE backup SysID is not enabled; local log-based SysID remains the fallback when Newton is unavailable."
        evidence = _write_json(
            skill_out / "pace_sysid.json",
            {"status": "skip", "reason": reason, "fallback_skill": "sysid_step_response", "input_file": str(input_path)},
        )
        return SkillResult(
            skill_id=manifest.skill_id,
            status="skip",
            quality_score=1.0,
            confidence=1.0,
            warnings=["PACE backup skipped; set sysid.pace_enabled plus sysid.pace_root or sysid.pace_command to enable it"],
            evidence_files=[evidence],
            metrics={"pace_enabled": False, "fallback_skill": "sysid_step_response"},
        )

    try:
        payload = run_pace_bridge(input_payload, work_dir=skill_out, output_path=output_path)
    except Exception as exc:
        message = f"PACE backup bridge could not complete: {exc}"
        evidence = _write_json(skill_out / "pace_command_log.json", {"error": str(exc), "input_file": str(input_path)})
        return _sysid_unavailable_result(manifest.skill_id, required, message, evidence, backend_key="pace_available")
    return _sysid_backend_result_from_payload(
        manifest.skill_id,
        payload,
        sysid_cfg,
        required,
        skill_out,
        extra_evidence=[str(output_path)] if output_path.exists() else [],
        min_confidence_key="min_pace_confidence",
        backend_label="PACE SysID",
    )


def _prefers_newton(preference: list[str]) -> bool:
    if "newton" not in preference:
        return False
    if "pace" not in preference:
        return True
    return preference.index("newton") < preference.index("pace")


def _impl_domain_randomization_update(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    gap = estimate_gap(load_records(ctx.dataset), ctx.config)
    dr = gap["recommendations"]["domain_randomization"]
    failures = []
    pos_range = dr.get("object_pose_observation_noise", dr["shaft_pose_observation_noise"]).get("object_position_uniform_m", dr["shaft_pose_observation_noise"]["gear_shaft_pos_uniform_m"])
    if max(abs(float(v)) for v in pos_range) > 0.01:
        failures.append("object pose noise recommendation exceeds 1 cm safety cap")
    friction = dr["actuator_and_contact_randomization"]["friction_sweep_for_agent_experiments"]
    if friction[0] <= 0 or friction[1] > 1.5:
        failures.append("friction sweep moved outside conservative bounds")
    evidence = _write_json(skill_out / "domain_randomization_candidate.json", dr)
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=0.9 if not failures else 0.4,
        confidence=0.75,
        blocking_failures=failures,
        evidence_files=[evidence],
        metrics={"object_pos_noise": pos_range, "friction_sweep": friction},
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


def _impl_release_candidate_gate(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    failures = []
    for skill_id, result in previous.items():
        if skill_id == manifest.skill_id:
            continue
        if result.release_blocking and result.status == "fail":
            failures.append(f"{skill_id} failed: {result.blocking_failures}")
    evidence = _write_json(
        skill_out / "release_gate_review.json",
        {
            "previous_skills": {skill_id: result.to_dict() for skill_id, result in previous.items()},
            "blocking_failures": failures,
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


IMPLEMENTATIONS: dict[str, Callable[[SkillManifest, HarnessContext, Path, dict[str, SkillResult]], SkillResult]] = {
    "env_preflight": _impl_env_preflight,
    "isaaclab_task_check": _impl_isaaclab_task_check,
    "policy_artifact_audit": _impl_policy_artifact_audit,
    "ros_preflight": _impl_ros_preflight,
    "real_data_quality_gate": _impl_real_data_quality_gate,
    "pose_repeatability": _impl_pose_repeatability,
    "sysid_step_response": _impl_sysid_step_response,
    "newton_sysid": _impl_newton_sysid,
    "pace_sysid": _impl_pace_sysid,
    "domain_randomization_update": _impl_domain_randomization_update,
    "action_scale_sweep": _impl_action_scale_sweep,
    "autoresearch_planner": _impl_autoresearch_planner,
    "sim_eval_regression": _impl_sim_eval_regression,
    "release_candidate_gate": _impl_release_candidate_gate,
    "real_robot_gate": _impl_real_robot_gate,
}
