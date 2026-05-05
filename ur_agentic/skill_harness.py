from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .autoresearch import build_plan
from .config import PipelineConfig, choose_task, load_config, nominal_action_scale
from .dataset import load_records
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
    config: PipelineConfig
    dataset: Path
    out_dir: Path
    include_real: bool = False

    @property
    def skill_dir(self) -> Path:
        return self.out_dir / "skills"


def load_manifests(root: str | Path) -> list[SkillManifest]:
    base = Path(root)
    manifests = []
    for path in sorted((base / "skills").glob("*/skill.json")):
        data = json.loads(path.read_text())
        manifests.append(SkillManifest(path=path, data=data))
    if not manifests:
        raise ValueError(f"No skill manifests found under {base / 'skills'}")
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
    return errors


def validate_all_manifests(root: str | Path) -> dict[str, Any]:
    manifests = load_manifests(root)
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
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ctx = HarnessContext(
        root=root_path,
        config=load_config(config_path),
        dataset=Path(dataset_path).expanduser().resolve(),
        out_dir=out,
        include_real=include_real,
    )
    manifests = {m.skill_id: m for m in load_manifests(root_path)}
    for manifest in manifests.values():
        errors = validate_manifest(manifest)
        if errors:
            raise ValueError(f"{manifest.skill_id}: invalid manifest: {errors}")

    ordered = _topological_order(manifests)
    if only_skill:
        ordered = [skill_id for skill_id in ordered if skill_id == only_skill]
        if not ordered:
            raise ValueError(f"Unknown skill: {only_skill}")

    results: dict[str, SkillResult] = {}
    ctx.skill_dir.mkdir(parents=True, exist_ok=True)
    for skill_id in ordered:
        manifest = manifests[skill_id]
        skill_out = ctx.skill_dir / skill_id
        skill_out.mkdir(parents=True, exist_ok=True)

        if manifest.real_robot and not include_real:
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
            result = _run_one(manifest, ctx, skill_out, results)
            result.human_required = manifest.human_required
            result.release_blocking = manifest.release_blocking
            result = _apply_quality_gate(manifest, result)

        (skill_out / "result.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
        results[skill_id] = result

    scoreboard = _scoreboard(results, manifests)
    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2, sort_keys=True) + "\n")
    (out / "release_candidate.json").write_text(
        json.dumps(_release_candidate(results, scoreboard), indent=2, sort_keys=True) + "\n"
    )
    return scoreboard


def _run_one(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
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


def _scoreboard(results: dict[str, SkillResult], manifests: dict[str, SkillManifest]) -> dict[str, Any]:
    blocking_failures = []
    scores = []
    for skill_id, result in results.items():
        if result.status == "pass":
            scores.append(result.quality_score)
        if manifests[skill_id].release_blocking and result.status == "fail":
            blocking_failures.append({"skill_id": skill_id, "failures": result.blocking_failures})
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
    checks = {}
    for command in ["python3", "ros2", "launch_test", "rqt_image_view"]:
        checks[command] = shutil.which(command)
    failures = []
    if not checks["python3"]:
        failures.append("python3 is required")
    warnings = [f"{cmd} not found; hardware/runtime step will need Isaac ROS environment" for cmd, path in checks.items() if not path and cmd != "python3"]
    evidence = _write_json(skill_out / "preflight.json", {"commands": checks, "warnings": warnings})
    return SkillResult(
        skill_id=manifest.skill_id,
        status="fail" if failures else "pass",
        quality_score=1.0 if not failures else 0.0,
        confidence=1.0,
        blocking_failures=failures,
        warnings=warnings,
        evidence_files=[evidence],
        metrics={"required_commands_present": not failures},
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
    pose = gap["shaft_pose_noise"]
    gate = float(ctx.config.perception.get("pose_error_gate_m", 0.01))
    p95 = pose.get("position_error_p95_m")
    failures = []
    if p95 is None:
        failures.append("no shaft_pose_reference values available for pose repeatability")
    elif float(p95) > gate:
        failures.append(f"shaft pose p95 error {p95} m exceeds gate {gate} m")
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


def _impl_domain_randomization_update(
    manifest: SkillManifest,
    ctx: HarnessContext,
    skill_out: Path,
    previous: dict[str, SkillResult],
) -> SkillResult:
    gap = estimate_gap(load_records(ctx.dataset), ctx.config)
    dr = gap["recommendations"]["domain_randomization"]
    failures = []
    pos_range = dr["shaft_pose_observation_noise"]["gear_shaft_pos_uniform_m"]
    if max(abs(float(v)) for v in pos_range) > 0.01:
        failures.append("shaft pose noise recommendation exceeds 1 cm safety cap")
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
        metrics={"shaft_pos_noise": pos_range, "friction_sweep": friction},
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
        "pose_error_p95_m": gap["shaft_pose_noise"].get("position_error_p95_m") or 1.0,
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
    "pose_repeatability": _impl_pose_repeatability,
    "sysid_step_response": _impl_sysid_step_response,
    "domain_randomization_update": _impl_domain_randomization_update,
    "action_scale_sweep": _impl_action_scale_sweep,
    "autoresearch_planner": _impl_autoresearch_planner,
    "sim_eval_regression": _impl_sim_eval_regression,
    "release_candidate_gate": _impl_release_candidate_gate,
    "real_robot_gate": _impl_real_robot_gate,
}
