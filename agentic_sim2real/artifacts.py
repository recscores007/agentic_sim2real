from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autoresearch import build_plan
from .config import PipelineConfig, choose_task, normalize_ui_audience
from .data_quality import evaluate_data_readiness
from .dataset import Record, load_records
from .metrics import summarize_records
from .sysid import estimate_gap


SCHEMA_VERSION = "agentic_sim2real.slide_contract.v1"


def write_rollout_data_artifacts(
    dataset_path: str | Path,
    config: PipelineConfig,
    out_dir: str | Path,
) -> dict[str, str]:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(dataset_path)
    payload = build_rollout_data(dataset_path, config, records)
    json_path = out / "rollout_data.json"
    md_path = out / "rollout_data.md"
    _write_json(json_path, payload)
    md_path.write_text(render_rollout_data_markdown(payload) + "\n")
    return {"rollout_data": str(json_path), "rollout_data_view": str(md_path)}


def write_pipeline_input_artifacts(
    dataset_path: str | Path,
    config: PipelineConfig,
    out_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    skill_ids: list[str] | None = None,
) -> dict[str, str]:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(dataset_path)
    payload = build_pipeline_input(dataset_path, config, records, config_path=config_path, skill_ids=skill_ids)
    json_path = out / "pipeline_input.json"
    md_path = out / "pipeline_input.md"
    _write_json(json_path, payload)
    md_path.write_text(render_pipeline_input_markdown(payload) + "\n")
    return {"pipeline_input": str(json_path), "pipeline_input_view": str(md_path)}


def write_scorecard_artifacts(
    dataset_path: str | Path,
    config: PipelineConfig,
    out_dir: str | Path,
    results: dict[str, Any],
    scoreboard: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    run_id: str | None = None,
    previous_scorecard: dict[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(dataset_path)
    payload = build_scorecard(
        dataset_path,
        config,
        records,
        _result_dicts(results),
        scoreboard,
        config_path=config_path,
        run_id=run_id,
        previous_scorecard=previous_scorecard,
    )
    json_path = out / "scorecard.json"
    md_path = out / "scorecard.md"
    _write_json(json_path, payload)
    md_path.write_text(render_scorecard_markdown(payload) + "\n")
    return {"scorecard": str(json_path), "scorecard_view": str(md_path)}


def write_pipeline_output_artifacts(
    dataset_path: str | Path,
    config: PipelineConfig,
    out_dir: str | Path,
    results: dict[str, Any],
    scoreboard: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    scorecard: dict[str, Any] | None = None,
    pipeline_input: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, str]:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(dataset_path)
    if scorecard is None:
        scorecard = build_scorecard(
            dataset_path,
            config,
            records,
            _result_dicts(results),
            scoreboard,
            config_path=config_path,
            run_id=run_id,
        )
    if pipeline_input is None:
        pipeline_input = build_pipeline_input(dataset_path, config, records, config_path=config_path)
    payload = build_pipeline_output(
        dataset_path,
        config,
        records,
        _result_dicts(results),
        scoreboard,
        scorecard,
        pipeline_input,
        config_path=config_path,
        trace=trace,
    )
    json_path = out / "pipeline_output.json"
    md_path = out / "pipeline_output.md"
    release_alias = out / "release_artifact.json"
    _write_json(json_path, payload)
    _write_json(release_alias, payload)
    md_path.write_text(render_pipeline_output_markdown(payload) + "\n")
    return {
        "pipeline_output": str(json_path),
        "pipeline_output_view": str(md_path),
        "release_artifact": str(release_alias),
    }


def write_run_record_artifacts(
    dataset_path: str | Path,
    config: PipelineConfig,
    out_dir: str | Path,
    results: dict[str, Any],
    scoreboard: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    pipeline_input: dict[str, Any],
    scorecard: dict[str, Any],
    pipeline_output: dict[str, Any],
    artifact_paths: dict[str, str],
    trace: dict[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(dataset_path)
    real_data_manifest = build_real_data_manifest(dataset_path, config, records)
    manifest_path = out / "real_data_manifest.json"
    json_path = out / "run_record.json"
    md_path = out / "run_record.md"
    _write_json(manifest_path, real_data_manifest)

    payload = build_run_record(
        dataset_path,
        config,
        records,
        _result_dicts(results),
        scoreboard,
        pipeline_input=pipeline_input,
        scorecard=scorecard,
        pipeline_output=pipeline_output,
        real_data_manifest=real_data_manifest,
        artifact_paths={
            **artifact_paths,
            "real_data_manifest": str(manifest_path),
            "run_record": str(json_path),
            "run_record_view": str(md_path),
        },
        config_path=config_path,
        trace=trace,
    )
    _write_json(json_path, payload)
    md_path.write_text(render_run_record_markdown(payload) + "\n")
    return {
        "run_record": str(json_path),
        "run_record_view": str(md_path),
        "real_data_manifest": str(manifest_path),
    }


def write_slide_contract_bundle(
    dataset_path: str | Path,
    config: PipelineConfig,
    out_dir: str | Path,
    results: dict[str, Any],
    scoreboard: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    skill_ids: list[str] | None = None,
    run_id: str | None = None,
    trace: dict[str, Any] | None = None,
) -> dict[str, str]:
    paths = {}
    paths.update(write_rollout_data_artifacts(dataset_path, config, out_dir))
    pipeline_input_paths = write_pipeline_input_artifacts(
        dataset_path,
        config,
        out_dir,
        config_path=config_path,
        skill_ids=skill_ids,
    )
    paths.update(pipeline_input_paths)
    scorecard_paths = write_scorecard_artifacts(
        dataset_path,
        config,
        out_dir,
        results,
        scoreboard,
        config_path=config_path,
        run_id=run_id,
    )
    paths.update(scorecard_paths)
    pipeline_input = json.loads(Path(pipeline_input_paths["pipeline_input"]).read_text())
    scorecard = json.loads(Path(scorecard_paths["scorecard"]).read_text())
    paths.update(
        write_pipeline_output_artifacts(
            dataset_path,
            config,
            out_dir,
            results,
            scoreboard,
            config_path=config_path,
            scorecard=scorecard,
            pipeline_input=pipeline_input,
            trace=trace,
            run_id=run_id,
        )
    )
    pipeline_output = json.loads(Path(paths["pipeline_output"]).read_text())
    paths.update(
        write_run_record_artifacts(
            dataset_path,
            config,
            out_dir,
            results,
            scoreboard,
            config_path=config_path,
            pipeline_input=pipeline_input,
            scorecard=scorecard,
            pipeline_output=pipeline_output,
            artifact_paths=paths,
            trace=trace,
        )
    )
    return paths


def build_real_data_manifest(
    dataset_path: str | Path,
    config: PipelineConfig,
    records: list[Record],
) -> dict[str, Any]:
    source = Path(dataset_path).expanduser()
    canonical = _canonical_records_path(source)
    rollout_data = build_rollout_data(dataset_path, config, records)
    data_readiness = evaluate_data_readiness(records, config, dataset_path=source)
    return {
        "schema": f"{SCHEMA_VERSION}.real_data_manifest",
        "description": "Complete file-and-hash manifest for the real data fed to this sim2real run.",
        "source_path": str(source),
        "source_exists": source.exists(),
        "source_type": "directory" if source.is_dir() else "file" if source.exists() else "missing",
        "source_sha256": _dataset_hash(source),
        "canonical_records": {
            "path": str(canonical) if canonical else None,
            "sha256": _path_hash(canonical) if canonical else None,
            "records": len(records),
            "episodes": len({record.episode_index for record in records}),
        },
        "record_contract": {
            "required_fields": ["episode_index", "timestamp", "action", "joint_state"],
            "optional_fields": ["joint_velocity", "ee_pose", "object_pose_estimate", "object_pose_reference", "contact_force", "success", "failure_mode"],
        },
        "data_readiness": data_readiness,
        "files": _file_manifest(source),
        "rollouts": [
            {
                "rollout_id": rollout["rollout_id"],
                "episode_index": rollout["episode_index"],
                "record_count": rollout["record_count"],
                "sha256": rollout["sha256"],
                "streams": sorted(rollout.get("streams", {})),
                "outcome": rollout.get("outcome", {}),
                "calibration": rollout.get("calibration"),
            }
            for rollout in rollout_data.get("rollouts", [])
        ],
    }


def build_run_record(
    dataset_path: str | Path,
    config: PipelineConfig,
    records: list[Record],
    results: dict[str, dict[str, Any]],
    scoreboard: dict[str, Any],
    *,
    pipeline_input: dict[str, Any],
    scorecard: dict[str, Any],
    pipeline_output: dict[str, Any],
    real_data_manifest: dict[str, Any],
    artifact_paths: dict[str, str],
    config_path: str | Path | None = None,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = str(scorecard.get("run_id") or _run_id(dataset_path, config_path))
    run_version = str(scorecard.get("run_version") or _run_version(run_id))
    return {
        "schema": f"{SCHEMA_VERSION}.run_record",
        "description": "Versioned audit record for one sim2real run: inputs, lineage, score breakdowns, outputs, and gates.",
        "run": {
            "run_id": run_id,
            "run_version": run_version,
            "created_at_utc": _utc_now(),
            "task": _task_name(config),
            "mode": _pipeline_mode(config),
            "release_profile": scoreboard.get("release_profile"),
            "review_scope": scoreboard.get("review_scope"),
            "git_sha": scorecard.get("git_sha"),
            "output_status": pipeline_output.get("status"),
            "safe_to_autorun_robot": False,
        },
        "lineage": {
            "config": {
                "path": str(Path(config_path).expanduser()) if config_path else None,
                "sha256": _path_hash(config_path) if config_path else None,
            },
            "real_data_fed": {
                "manifest": "real_data_manifest.json",
                "source_path": real_data_manifest.get("source_path"),
                "source_sha256": real_data_manifest.get("source_sha256"),
                "canonical_records": real_data_manifest.get("canonical_records"),
                "file_count": len(real_data_manifest.get("files", [])),
                "rollout_count": len(real_data_manifest.get("rollouts", [])),
                "rollout_hashes": [
                    {"rollout_id": item.get("rollout_id"), "sha256": item.get("sha256")}
                    for item in real_data_manifest.get("rollouts", [])
                ],
            },
            "policy_checkpoint": _policy_checkpoint_record(config, config_path),
            "retraining": _retraining_record(config, config_path),
        },
        "score_breakdown": {
            "transfer_readiness": scorecard.get("transfer_readiness_breakdown", {}),
            "release_gap": scorecard.get("release_gap_breakdown", {}),
            "split_scores": scorecard.get("split_scores", {}),
            "data_readiness": scorecard.get("data_readiness", {}),
            "characterization": scorecard.get("characterization", {}),
            "policy_release": scorecard.get("policy_release", {}),
        },
        "pipeline_contract": {
            "input_hash": pipeline_input.get("input_hash"),
            "pipeline_input": "pipeline_input.json",
            "scorecard": "scorecard.json",
            "pipeline_output": "pipeline_output.json",
            "real_data_manifest": "real_data_manifest.json",
        },
        "artifacts": dict(sorted(artifact_paths.items())),
        "skills": {
            skill_id: {
                "status": result.get("status"),
                "quality_score": result.get("quality_score"),
                "confidence": result.get("confidence"),
                "evidence_files": result.get("evidence_files", []),
            }
            for skill_id, result in sorted(results.items())
        },
        "release": {
            "offline_validation_status": scoreboard.get("offline_validation_status", scoreboard.get("status")),
            "human_review_readiness": scoreboard.get("human_review_readiness"),
            "release_candidate_ready": bool(scoreboard.get("release_candidate_ready", False)),
            "hardware_approval_status": scoreboard.get("hardware_approval_status", "not_requested"),
            "safe_to_autorun_robot": False,
            "blocking_failures": scoreboard.get("blocking_failures", []),
        },
        "trace": {
            "present": bool(trace),
            "release_gate_status": (trace or {}).get("release_gate_decides", {}).get("status") if trace else None,
            "human_gate_status": (trace or {}).get("human_approves_hardware", {}).get("status") if trace else None,
        },
    }


def build_rollout_data(dataset_path: str | Path, config: PipelineConfig, records: list[Record]) -> dict[str, Any]:
    by_episode = _records_by_episode(records)
    task = _task_name(config)
    rollouts = []
    for episode, episode_records in sorted(by_episode.items()):
        rollout_id = _rollout_id(episode, episode_records)
        duration_s = _duration_s(episode_records)
        success, failure_mode = _episode_outcome(episode_records)
        rollouts.append(
            {
                "rollout_id": rollout_id,
                "task": task,
                "scenario": _episode_raw_value(episode_records, "scenario", default="default"),
                "seed": _episode_raw_value(episode_records, "seed", "sim_seed", default=None),
                "episode_index": episode,
                "duration_s": duration_s,
                "streams": _episode_streams(episode_records, config),
                "labels": _episode_labels(episode_records),
                "outcome": {"success": success, "failure_mode": failure_mode},
                "calibration": _calibration_reference(dataset_path, config),
                "sha256": _episode_hash(episode_records),
                "record_count": len(episode_records),
                "source_dataset": str(Path(dataset_path).expanduser()),
            }
        )
    return {
        "schema": f"{SCHEMA_VERSION}.rollout_data",
        "description": "One rollout record per real episode. This is the slide-21 contract.",
        "dataset": str(Path(dataset_path).expanduser()),
        "task": task,
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
    }


def build_pipeline_input(
    dataset_path: str | Path,
    config: PipelineConfig,
    records: list[Record],
    *,
    config_path: str | Path | None = None,
    skill_ids: list[str] | None = None,
) -> dict[str, Any]:
    task_cfg = config.task_spec
    summary = summarize_records(records)
    task = _task_name(config)
    scenarios = _scenarios(records) or [str(item) for item in task_cfg.get("scenarios", [])]
    if not scenarios:
        scenarios = ["default"]
    goal_cfg = dict(task_cfg.get("goal", {}))
    real_success = goal_cfg.get("real_success", config.agent.get("target_real_success", 0.8))
    gap_target = goal_cfg.get("gap_target", config.agent.get("gap_target", 0.1))
    release_gap_target = goal_cfg.get("release_gap_target", gap_target)
    mode = _pipeline_mode(config)
    input_payload = {
        "schema": f"{SCHEMA_VERSION}.pipeline_input",
        "description": "Task spec consumed by the agent and harness. This is the slide-22 contract.",
        "task": task,
        "mode": mode,
        "goal": {
            "real_success": _round_or_none(real_success),
            "gap_target": _round_or_none(gap_target),
            "release_gap_target": _round_or_none(release_gap_target),
            "characterization": {
                "primary": "fit sim parameters from trajectory, camera, pose, and contact data before policy training",
                "success_labels_required": False,
            },
            "policy_release": {
                "primary": "validate a trained policy candidate before human-supervised hardware release",
                "success_labels_required": True,
            },
        },
        "scenarios": scenarios,
        "policy_ckpt": _policy_checkpoint(config),
        "sim_config": _sim_config(config, config_path),
        "real_data": {
            "rollouts": int(task_cfg.get("real_data", {}).get("rollouts") or summary["episodes"]),
            "source": str(task_cfg.get("real_data", {}).get("source") or _dataset_source(dataset_path)),
            "dataset": str(Path(dataset_path).expanduser()),
            "records": summary["records"],
            "sha256": _dataset_hash(dataset_path),
        },
        "skills_allowed": skill_ids or [str(item) for item in task_cfg.get("skills_allowed", [])],
        "ui": {
            "audience": normalize_ui_audience(config.ui.get("audience")),
        },
        "budget": dict(task_cfg.get("budget", {})),
        "kill_criteria": {
            **dict(task_cfg.get("kill_criteria", {})),
            "max_iters": int(task_cfg.get("kill_criteria", {}).get("max_iters") or config.llm_orchestrator.get("max_steps", 32)),
        },
        "owner": str(task_cfg.get("owner") or "unassigned"),
        "submitted": str(task_cfg.get("submitted") or "not_set"),
    }
    input_payload["input_hash"] = _hash_payload(input_payload)
    return input_payload


def build_scorecard(
    dataset_path: str | Path,
    config: PipelineConfig,
    records: list[Record],
    results: dict[str, dict[str, Any]],
    scoreboard: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    run_id: str | None = None,
    previous_scorecard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gap = estimate_gap(records, config)
    plan = build_plan(gap, config)
    summary = gap["summary"]
    data_readiness = evaluate_data_readiness(records, config, dataset_path=dataset_path)
    split_scores = _split_transfer_scores(records, config)
    transfer_score = float(plan["transfer_score"]["score_0_to_1"])
    release_gap_score = round(max(0.0, 1.0 - transfer_score), 3)
    previous_gap = _previous_gap(previous_scorecard)
    release_gap_target = _release_gap_target(config)
    real_success = summary.get("success_rate")
    sim_success = _sim_success_rate(results)
    regression_pp = _regression_pp(results)
    per_skill = {skill_id: float(result.get("quality_score", 0.0)) for skill_id, result in sorted(results.items())}
    failure_modes = _failure_modes(summary)
    scorecard = {
        "schema": f"{SCHEMA_VERSION}.scorecard",
        "description": "Unified per-run scorecard consumed by the agent and human reviewer. This is the slide-23 contract.",
        "task": _task_name(config),
        "mode": _pipeline_mode(config),
        "run_id": run_id or _run_id(dataset_path, config_path),
        "run_version": _run_version(run_id or _run_id(dataset_path, config_path)),
        "git_sha": _git_sha(Path(config_path).parent if config_path else Path.cwd()),
        "transfer_readiness_score": transfer_score,
        "transfer_readiness_breakdown": _transfer_readiness_breakdown(plan),
        "release_gap_score": release_gap_score,
        "release_gap_score_delta": None if previous_gap is None else round(release_gap_score - previous_gap, 3),
        "release_gap_target": release_gap_target,
        "release_gap_breakdown": _release_gap_breakdown(transfer_score, release_gap_score, release_gap_target),
        "sim2real_gap": release_gap_score,
        "sim2real_gap_delta": None if previous_gap is None else round(release_gap_score - previous_gap, 3),
        "score_meaning": {
            "transfer_readiness_score": "Normalized evidence/readiness score from AutoResearch components. Higher is better.",
            "release_gap_score": "Normalized remaining readiness gap, computed as max(0, 1 - transfer_readiness_score). Lower is better.",
            "sim2real_gap": "Backward-compatible alias for release_gap_score, not a physical distance or standard industry metric.",
            "target_source": "Configured by the user or release policy via task_spec.goal.release_gap_target, task_spec.goal.gap_target, or agent.gap_target.",
            "formula": plan["transfer_score"].get("score_policy", {}).get("formula", "weighted average of included evidence components")
            + "; release_gap_score = max(0, 1 - transfer_readiness_score)",
        },
        "data_readiness": data_readiness,
        "split_scores": split_scores,
        "success_rate": {"sim": sim_success, "real": real_success},
        "regression_pp": regression_pp,
        "per_skill": per_skill,
        "per_skill_detail": results,
        "characterization": _characterization_metrics(gap, plan, results),
        "policy_release": _policy_release_metrics(config, scoreboard, real_success, sim_success, release_gap_score),
        "failure_modes": failure_modes,
        "cost": _cost(config, results),
        "verdict": _scorecard_verdict(scoreboard),
        "recommended_skill": _recommended_skill(scoreboard, results),
        "patch": _collect_patches(results),
        "provenance": {
            "dataset": str(Path(dataset_path).expanduser()),
            "dataset_sha256": _dataset_hash(dataset_path),
            "config_sha256": _path_hash(config_path) if config_path else None,
            "scoreboard_status": scoreboard.get("status"),
            "human_review_readiness": scoreboard.get("human_review_readiness"),
            "safe_to_autorun_robot": False,
        },
    }
    return scorecard


def build_pipeline_output(
    dataset_path: str | Path,
    config: PipelineConfig,
    records: list[Record],
    results: dict[str, dict[str, Any]],
    scoreboard: dict[str, Any],
    scorecard: dict[str, Any],
    pipeline_input: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_cfg = config.task_spec
    baseline_gap = task_cfg.get("baseline_gap")
    baseline_success = task_cfg.get("baseline_real_success")
    if baseline_success is None:
        baseline_success = _baseline_success_from_results(results)
    status = _pipeline_output_status(scoreboard, trace)
    release_id = scorecard["run_id"]
    changes = _collect_patches(results)
    output = {
        "schema": f"{SCHEMA_VERSION}.pipeline_output",
        "description": "Release artifact tying policy, sim config, changes, scorecard, and deploy command. This is the slide-24 contract.",
        "task": _task_name(config),
        "mode": _pipeline_mode(config),
        "release_id": release_id,
        "run_version": scorecard.get("run_version"),
        "status": status,
        "policy_ckpt": _policy_checkpoint(config),
        "sim_config": {
            **_sim_config(config, config_path),
            "patches": changes,
        },
        "release_gap_score": {
            "before": _round_or_none(baseline_gap),
            "after": scorecard.get("release_gap_score"),
            "target": scorecard.get("release_gap_target"),
            "meaning": "normalized readiness gap; lower is better; not a physical sim2real distance",
        },
        "transfer_readiness_score": scorecard.get("transfer_readiness_score"),
        "sim2real_gap": {
            "before": _round_or_none(baseline_gap),
            "after": scorecard.get("sim2real_gap"),
            "deprecated_alias_for": "release_gap_score",
        },
        "success_real": {
            "before": _round_or_none(baseline_success),
            "after": scorecard.get("success_rate", {}).get("real"),
        },
        "changes": changes,
        "used": {
            "iters": len(results),
            "gpu_hr": _cost(config, results).get("gpu_hr", 0.0),
        },
        "provenance": {
            "input_hash": pipeline_input["input_hash"],
            "dataset_sha256": _dataset_hash(dataset_path),
            "config_sha256": _path_hash(config_path) if config_path else None,
            "scorecard": "scorecard.json",
            "scoreboard_status": scoreboard.get("status"),
        },
        "deploy": _deploy_command(release_id, status),
        "safe_to_autorun_robot": False,
        "characterization": scorecard.get("characterization", {}),
        "policy_release": scorecard.get("policy_release", {}),
        "human_approval": {
            "hardware_approval_status": scoreboard.get("hardware_approval_status", "not_requested"),
            "required": True,
        },
    }
    return output


def render_rollout_data_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Rollout Data",
        "",
        f"- Task: {payload['task']}",
        f"- Rollouts: {payload['rollout_count']}",
        f"- Dataset: `{payload['dataset']}`",
        "",
    ]
    for rollout in payload["rollouts"]:
        outcome = "success" if rollout["outcome"]["success"] is True else "failed" if rollout["outcome"]["success"] is False else "unlabeled"
        lines.extend(
            [
                f"## {rollout['rollout_id']} - {rollout['scenario']}",
                "",
                f"- Outcome: {outcome}",
                f"- Duration: {rollout['duration_s']} s",
                f"- Records: {rollout['record_count']}",
                f"- Calibration: `{rollout['calibration']}`",
                f"- SHA256: `{rollout['sha256']}`",
                "- Streams:",
            ]
        )
        for name, stream in rollout["streams"].items():
            lines.append(f"  - {name}: {stream}")
        lines.append("")
    return "\n".join(lines)


def render_pipeline_input_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Pipeline Input",
            "",
            f"- Task: {payload['task']}",
            f"- Mode: {payload.get('mode', 'characterization')}",
            f"- UI audience: {payload.get('ui', {}).get('audience', 'customer')}",
            "- Characterization goal: tune sim parameters from trajectory/camera/contact evidence",
            f"- Policy release goal: real_success >= {payload['goal']['real_success']}, release_gap_score <= {payload['goal']['release_gap_target']}",
            f"- Scenarios: {', '.join(payload['scenarios'])}",
            f"- Policy checkpoint: `{payload['policy_ckpt']}`",
            f"- Sim config: {payload['sim_config']['engine']} `{payload['sim_config']['hash']}`",
            f"- Real rollouts: {payload['real_data']['rollouts']} from {payload['real_data']['source']}",
            f"- Budget: {payload['budget']}",
            f"- Kill criteria: {payload['kill_criteria']}",
            f"- Input hash: `{payload['input_hash']}`",
        ]
    )


def render_scorecard_markdown(payload: dict[str, Any]) -> str:
    delta = payload.get("release_gap_score_delta", payload.get("sim2real_gap_delta"))
    delta_text = "n/a" if delta is None else f"{delta:+.3f}"
    lines = [
        "# Scorecard",
        "",
        f"- Task: {payload['task']}",
        f"- Mode: {payload.get('mode', 'characterization')}",
        f"- Run: `{payload['run_id']}`",
        f"- Verdict: {payload['verdict']}",
        f"- Transfer readiness score: {payload.get('transfer_readiness_score')} (higher is better)",
        f"- Release gap score: {payload.get('release_gap_score', payload['sim2real_gap'])} ({delta_text}, lower is better)",
        "- Score note: release_gap_score is a normalized readiness gap, not a physical sim2real distance.",
        f"- Success rate: sim={payload['success_rate']['sim']} real={payload['success_rate']['real']}",
        f"- Regression: {payload['regression_pp']} pp",
        f"- Recommended next skill: {payload['recommended_skill']}",
        "",
        "## Data Readiness",
        "",
        f"- Status: {payload.get('data_readiness', {}).get('status')}",
        f"- Frame link coverage: {payload.get('data_readiness', {}).get('frame_link_coverage')}",
        f"- Heldout frame link coverage: {payload.get('data_readiness', {}).get('heldout_frame_link_coverage')}",
        f"- Delay observability: {payload.get('data_readiness', {}).get('delay_observability_status')}",
        f"- Pose validation source: {payload.get('data_readiness', {}).get('pose_validation', {}).get('validation_source')}",
        f"- Action items: {len(payload.get('data_readiness', {}).get('action_items', []))}",
        "",
        "## Split Scores",
        "",
        f"- Train transfer readiness: {payload.get('split_scores', {}).get('train', {}).get('transfer_readiness_score')}",
        f"- Heldout transfer readiness: {payload.get('split_scores', {}).get('heldout', {}).get('transfer_readiness_score')}",
        f"- Heldout minus train: {payload.get('split_scores', {}).get('heldout_vs_train_delta')}",
        "",
        "## Characterization",
        "",
    ]
    char = payload.get("characterization", {})
    for group in ["trajectory_data", "actuator_latency", "camera_pose_noise", "contact", "tuning_outputs"]:
        if group in char:
            lines.append(f"- {group}: {char[group]}")
    policy = payload.get("policy_release", {})
    lines.extend(
        [
            "",
            "## Policy Release",
            "",
            f"- Real success: {policy.get('success', {}).get('real')}",
            f"- Target real success: {policy.get('success', {}).get('real_target')}",
            f"- Hardware approval: {policy.get('hardware_approval_status')}",
            "",
            "## Per Skill",
            "",
        ]
    )
    for skill, score in payload["per_skill"].items():
        detail = payload["per_skill_detail"].get(skill, {})
        lines.append(f"- {skill}: {score:.3f} ({detail.get('status', 'unknown')})")
    lines.extend(["", "## Failure Modes", ""])
    if payload["failure_modes"]:
        for mode in payload["failure_modes"]:
            lines.append(f"- {mode['mode']}: {mode['count']}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def render_pipeline_output_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Pipeline Output",
        "",
        f"- Release: `{payload['release_id']}`",
        f"- Mode: {payload.get('mode', 'characterization')}",
        f"- Status: {payload['status']}",
        f"- Policy: `{payload['policy_ckpt']}`",
        f"- Sim config: `{payload['sim_config']['hash']}`",
        f"- Release gap score: {payload['release_gap_score']['before']} -> {payload['release_gap_score']['after']} (target {payload['release_gap_score']['target']})",
        "- Score note: release_gap_score is a normalized readiness gap, not a physical sim2real distance.",
        f"- Real success: {payload['success_real']['before']} -> {payload['success_real']['after']}",
        f"- Iterations: {payload['used']['iters']}",
        f"- GPU hours: {payload['used']['gpu_hr']}",
        f"- Deploy: `{payload['deploy']}`",
        f"- Safe to autorun robot: {payload['safe_to_autorun_robot']}",
        "",
        "## Changes",
        "",
    ]
    if payload["changes"]:
        for change in payload["changes"]:
            lines.append(f"- {change['skill']}: {change['patch']}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def render_run_record_markdown(payload: dict[str, Any]) -> str:
    run = payload.get("run", {})
    lineage = payload.get("lineage", {})
    real_data = lineage.get("real_data_fed", {})
    policy = lineage.get("policy_checkpoint", {})
    retraining = lineage.get("retraining", {})
    transfer = payload.get("score_breakdown", {}).get("transfer_readiness", {})
    release_gap = payload.get("score_breakdown", {}).get("release_gap", {})
    lines = [
        "# Versioned Run Record",
        "",
        f"- Run ID: `{run.get('run_id')}`",
        f"- Run version: `{run.get('run_version')}`",
        f"- Created: {run.get('created_at_utc')}",
        f"- Task: {run.get('task')}",
        f"- Mode: {run.get('mode')}",
        f"- Output status: {run.get('output_status')}",
        f"- Safe to autorun robot: {run.get('safe_to_autorun_robot')}",
        "",
        "## Real Data Fed",
        "",
        f"- Source: `{real_data.get('source_path')}`",
        f"- Source SHA256: `{real_data.get('source_sha256')}`",
        f"- Canonical records: {real_data.get('canonical_records')}",
        f"- Files recorded: {real_data.get('file_count')}",
        f"- Rollouts recorded: {real_data.get('rollout_count')}",
        "",
        "## Policy / Retraining",
        "",
        f"- Policy checkpoint: `{policy.get('configured')}`",
        f"- Policy SHA256: `{policy.get('sha256')}`",
        f"- Retraining requested: {retraining.get('requested')}",
        f"- Source checkpoint: `{retraining.get('source_checkpoint')}`",
        f"- Target checkpoint: `{retraining.get('target_checkpoint')}`",
        "",
        "## Score Breakdown",
        "",
        f"- Transfer readiness score: {transfer.get('score_0_to_1')}",
        f"- Release gap score: {release_gap.get('release_gap_score')}",
        f"- Release gap target: {release_gap.get('target')}",
        f"- Release gap status: {release_gap.get('status')}",
        "",
        "## Artifacts",
        "",
    ]
    for key, path in payload.get("artifacts", {}).items():
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines)


def _records_by_episode(records: list[Record]) -> dict[int, list[Record]]:
    grouped: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.episode_index].append(record)
    for values in grouped.values():
        values.sort(key=lambda item: item.timestamp)
    return grouped


def _task_name(config: PipelineConfig) -> str:
    return str(config.task_spec.get("task") or choose_task(config))


def _pipeline_mode(config: PipelineConfig) -> str:
    raw = str(config.task_spec.get("mode") or "characterization").strip().lower()
    aliases = {
        "characterize": "characterization",
        "characterization_mode": "characterization",
        "tuning": "characterization",
        "policy": "policy_release",
        "release": "policy_release",
        "policy-release": "policy_release",
        "policy_release_mode": "policy_release",
    }
    mode = aliases.get(raw, raw)
    return mode if mode in {"characterization", "policy_release"} else "characterization"


def _rollout_id(episode: int, records: list[Record]) -> str:
    value = _episode_raw_value(records, "rollout_id", "episode_uid", default=None)
    return str(value) if value not in (None, "") else f"rollout_{episode:04d}"


def _episode_raw_value(records: list[Record], *keys: str, default: Any) -> Any:
    for record in records:
        raw = record.raw or {}
        for key in keys:
            value = raw.get(key)
            if value not in (None, ""):
                return value
    return default


def _duration_s(records: list[Record]) -> float:
    if not records:
        return 0.0
    return round(max(item.timestamp for item in records) - min(item.timestamp for item in records), 3)


def _episode_rate(records: list[Record], fallback: float) -> float:
    timestamps = sorted(item.timestamp for item in records)
    dts = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not dts:
        return round(float(fallback), 3)
    return round(1.0 / statistics.median(dts), 3)


def _episode_streams(records: list[Record], config: PipelineConfig) -> dict[str, Any]:
    rate = _episode_rate(records, float(config.robot.get("control_rate_hz", 30.0)))
    streams: dict[str, Any] = {
        "joint_state": {"hz": rate, "dim": len(records[0].joint_state)},
        "actions": {"hz": rate, "dim": len(records[0].action)},
    }
    if any(_camera_field(record, "color_image") for record in records):
        streams["rgb"] = {"hz": rate, "source": "camera.color_image"}
    if any(_camera_field(record, "depth_image") for record in records):
        streams["depth"] = {"hz": rate, "source": "camera.depth_image"}
    if any(record.contact_force is not None for record in records):
        streams["contact"] = {"hz": rate, "source": "contact_force"}
    if any(record.ee_pose for record in records):
        streams["tcp_pose"] = {"hz": rate, "dim": len(next(record.ee_pose for record in records if record.ee_pose))}
    return streams


def _camera_field(record: Record, key: str) -> Any:
    camera = (record.raw or {}).get("camera")
    if isinstance(camera, dict):
        return camera.get(key)
    return None


def _episode_labels(records: list[Record]) -> dict[str, Any]:
    pose_source = "reference" if any(record.object_pose_reference for record in records) else "estimate_only" if any(record.object_pose_estimate for record in records) else "missing"
    event_time = None
    for record in records:
        if record.contact_force is not None and float(record.contact_force) > 0:
            event_time = record.timestamp
            break
    return {
        "object_pose": pose_source,
        "first_contact_evt": None if event_time is None else f"t={event_time:.3f}s",
    }


def _episode_outcome(records: list[Record]) -> tuple[bool | None, str | None]:
    success_values = [record.success for record in records if record.success is not None]
    success = success_values[-1] if success_values else None
    failures = Counter(record.failure_mode for record in records if record.failure_mode)
    failure_mode = failures.most_common(1)[0][0] if failures else None
    return success, failure_mode


def _calibration_reference(dataset_path: str | Path, config: PipelineConfig) -> str:
    source = Path(dataset_path).expanduser()
    session_dir = source if source.is_dir() else source.parent.parent if source.parent.name == "aligned" else source.parent
    calibration = session_dir / "calibration" / "calibration.json"
    if calibration.exists():
        return str(calibration)
    return str(config.task_spec.get("calibration") or "not_provided")


def _episode_hash(records: list[Record]) -> str:
    rows = [json.dumps(record.raw or record.__dict__, sort_keys=True, separators=(",", ":")) for record in records]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _dataset_hash(dataset_path: str | Path) -> str:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        candidates = [path / "aligned" / "records.jsonl", path / "records.jsonl"]
        for candidate in candidates:
            if candidate.exists():
                return _path_hash(candidate)
        chunks = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            chunks.append(child.relative_to(path).as_posix())
            chunks.append(_path_hash(child))
        return hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()
    return _path_hash(path)


def _path_hash(path: str | Path | None) -> str | None:
    if not path:
        return None
    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        return None
    h = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _scenarios(records: list[Record]) -> list[str]:
    values = []
    for record in records:
        value = (record.raw or {}).get("scenario")
        if value not in (None, "") and str(value) not in values:
            values.append(str(value))
    return values


def _policy_checkpoint(config: PipelineConfig) -> str:
    return str(config.task_spec.get("policy_ckpt") or config.policy.get("artifact_dir") or "not_configured")


def _sim_config(config: PipelineConfig, config_path: str | Path | None) -> dict[str, Any]:
    sim_cfg = dict(config.task_spec.get("sim_config", {}))
    sim_cfg.setdefault("engine", "isaac")
    sim_cfg["hash"] = str(sim_cfg.get("hash") or _path_hash(config_path) or "not_hashed")
    return sim_cfg


def _dataset_source(dataset_path: str | Path) -> str:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        return path.name
    return path.parent.name


def _result_dicts(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized = {}
    for skill_id, result in results.items():
        if hasattr(result, "to_dict"):
            normalized[skill_id] = result.to_dict()
        elif isinstance(result, dict):
            normalized[skill_id] = dict(result)
        else:
            normalized[skill_id] = {"status": "unknown", "value": str(result)}
    for result in list(normalized.values()):
        subchecks = result.get("metrics", {}).get("subchecks", {}) if isinstance(result.get("metrics"), dict) else {}
        if not isinstance(subchecks, dict):
            continue
        for skill_id, payload in subchecks.items():
            if isinstance(payload, dict):
                normalized.setdefault(str(skill_id), dict(payload))
    return normalized


def _sim_success_rate(results: dict[str, dict[str, Any]]) -> float | None:
    rollout = results.get("isaaclab_rollout_regression", {}).get("metrics", {})
    if rollout.get("success_rate") is not None:
        return _round_or_none(rollout.get("success_rate"))
    candidate = results.get("sim_eval_regression", {}).get("metrics", {}).get("candidate", {})
    if isinstance(candidate, dict) and candidate.get("success_rate") is not None:
        return _round_or_none(candidate.get("success_rate"))
    return None


def _regression_pp(results: dict[str, dict[str, Any]]) -> float | None:
    value = results.get("sim_eval_regression", {}).get("metrics", {}).get("success_delta")
    if value is None:
        return None
    return round(float(value) * 100.0, 1)


def _previous_gap(previous_scorecard: dict[str, Any] | None) -> float | None:
    if not previous_scorecard:
        return None
    value = previous_scorecard.get("sim2real_gap")
    return None if value is None else float(value)


def _failure_modes(summary: dict[str, Any]) -> list[dict[str, Any]]:
    modes = summary.get("failure_modes", {}) or {}
    return [
        {"mode": str(mode), "count": int(count)}
        for mode, count in sorted(modes.items(), key=lambda item: (-int(item[1]), str(item[0])))
    ]


def _cost(config: PipelineConfig, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gpu_hr = 0.0
    wall_hr = 0.0
    for result in results.values():
        metrics = result.get("metrics", {})
        gpu_hr += float(metrics.get("gpu_hr", 0.0) or 0.0)
        wall_hr += float(metrics.get("wall_hr", 0.0) or 0.0)
    budget = dict(config.task_spec.get("budget", {}))
    return {
        "gpu_hr": round(gpu_hr, 3),
        "wall_hr": round(wall_hr, 3),
        "budget_gpu_hr": budget.get("gpu_hr"),
        "budget_wall_hr": budget.get("wall_hr"),
    }


def _scorecard_verdict(scoreboard: dict[str, Any]) -> str:
    if scoreboard.get("status") != "pass":
        return "blocked"
    if scoreboard.get("release_candidate_ready"):
        return "human_review"
    if scoreboard.get("human_review_readiness") == "smoke_review_only":
        return "iterate"
    return "iterate"


def _recommended_skill(scoreboard: dict[str, Any], results: dict[str, dict[str, Any]]) -> str:
    for item in scoreboard.get("blocking_failures", []):
        skill_id = item.get("skill_id")
        if skill_id:
            return str(skill_id)
    for skill_id, result in results.items():
        if result.get("status") in {"fail", "evidence_missing"}:
            return str(skill_id)
    if scoreboard.get("release_candidate_ready"):
        return "human_review"
    return "continue_or_collect_more_evidence"


def _collect_patches(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    patches = []
    action = results.get("action_scale_sweep", {})
    if action:
        suggested = action.get("metrics", {}).get("suggested")
        if suggested is not None:
            patches.append({"skill": "action_scale_sweep", "patch": f"action_scale -> {suggested}", "status": action.get("status")})
    dr = results.get("domain_randomization_update", {})
    if dr:
        metrics = dr.get("metrics", {})
        patch_bits = []
        if metrics.get("object_pos_noise") is not None:
            patch_bits.append(f"object_pos_noise {metrics['object_pos_noise']}")
        if metrics.get("friction_sweep") is not None:
            patch_bits.append(f"friction_sweep {metrics['friction_sweep']}")
        if metrics.get("video_friction_applied"):
            patch_bits.append("video friction -> object/gripper material params")
        if patch_bits:
            patches.append({"skill": "domain_randomization_update", "patch": "; ".join(patch_bits), "status": dr.get("status")})
    for skill in ("newton_sysid", "pace_sysid"):
        result = results.get(skill, {})
        metrics = result.get("metrics", {})
        if result and result.get("status") == "pass":
            patches.append({"skill": skill, "patch": "fitted physics parameters available", "status": result.get("status"), "metrics": metrics})
    return patches


def _release_gap_target(config: PipelineConfig) -> float | None:
    goal_cfg = dict(config.task_spec.get("goal", {}))
    return _round_or_none(
        goal_cfg.get("release_gap_target", goal_cfg.get("gap_target", config.agent.get("gap_target", 0.1)))
    )


def _transfer_readiness_breakdown(plan: dict[str, Any]) -> dict[str, Any]:
    transfer = dict(plan.get("transfer_score", {}))
    if transfer.get("components"):
        return {
            "score_0_to_1": transfer.get("score_0_to_1"),
            "interpretation": transfer.get("interpretation"),
            "formula": transfer.get("score_policy", {}).get(
                "formula",
                "weighted average of included evidence components",
            ),
            "components": transfer.get("components", []),
            "excluded_components": transfer.get("excluded_components", []),
            "score_policy": transfer.get("score_policy", {}),
            "note": "This is an internal evidence/readiness score, not an industry-standard physical sim2real metric.",
        }
    weights = {
        "episode_score": 0.20,
        "success_component": 0.20,
        "delay_confidence": 0.15,
        "deadband_confidence": 0.15,
        "pose_score": 0.15,
        "contact_score": 0.15,
    }
    components = []
    for key, weight in weights.items():
        value = float(transfer.get(key, 0.0) or 0.0)
        components.append(
            {
                "component": key,
                "weight": weight,
                "value": round(value, 3),
                "weighted_contribution": round(weight * value, 3),
            }
        )
    return {
        "score_0_to_1": transfer.get("score_0_to_1"),
        "interpretation": transfer.get("interpretation"),
        "formula": "sum(weight * component_value)",
        "components": components,
        "note": "This is an internal evidence/readiness score, not an industry-standard physical sim2real metric.",
    }


def _release_gap_breakdown(
    transfer_readiness_score: float,
    release_gap_score: float,
    target: float | None,
) -> dict[str, Any]:
    margin = None if target is None else round(float(target) - release_gap_score, 3)
    return {
        "formula": "release_gap_score = max(0, 1 - transfer_readiness_score)",
        "transfer_readiness_score": round(float(transfer_readiness_score), 3),
        "release_gap_score": round(float(release_gap_score), 3),
        "target": target,
        "margin_to_target": margin,
        "status": "no_target" if target is None else "meets_target" if release_gap_score <= float(target) else "above_target",
        "note": "Lower is better. This is a release-readiness gap and not a physical distance.",
    }


def _split_transfer_scores(records: list[Record], config: PipelineConfig) -> dict[str, Any]:
    train_records = [record for record in records if not _record_is_heldout(record)]
    heldout_records = [record for record in records if _record_is_heldout(record)]
    splits: dict[str, Any] = {}
    if train_records:
        splits["train"] = _score_record_subset(train_records, config)
    if heldout_records:
        splits["heldout"] = _score_record_subset(heldout_records, config)
    train_score = splits.get("train", {}).get("transfer_readiness_score")
    heldout_score = splits.get("heldout", {}).get("transfer_readiness_score")
    splits["heldout_vs_train_delta"] = (
        None
        if train_score is None or heldout_score is None
        else round(float(heldout_score) - float(train_score), 3)
    )
    return splits


def _score_record_subset(records: list[Record], config: PipelineConfig) -> dict[str, Any]:
    gap = estimate_gap(records, config)
    plan = build_plan(gap, config)
    transfer = plan.get("transfer_score", {})
    return {
        "records": len(records),
        "episodes": gap.get("summary", {}).get("episodes"),
        "transfer_readiness_score": transfer.get("score_0_to_1"),
        "release_gap_score": None
        if transfer.get("score_0_to_1") is None
        else round(max(0.0, 1.0 - float(transfer["score_0_to_1"])), 3),
        "excluded_components": transfer.get("excluded_components", []),
    }


def _record_is_heldout(record: Record) -> bool:
    raw = record.raw or {}
    split_value = raw.get("split") or raw.get("dataset_split") or raw.get("eval_split") or raw.get("partition")
    split = str(split_value).strip().lower() if split_value not in (None, "") else ""
    flag = raw.get("heldout") or raw.get("is_holdout") or raw.get("holdout")
    if isinstance(flag, bool):
        return flag
    if flag not in (None, ""):
        return str(flag).strip().lower() in {"1", "true", "yes", "heldout", "holdout"}
    return split in {"heldout", "holdout", "test", "validation", "val"}


def _characterization_metrics(gap: dict[str, Any], plan: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = gap.get("summary", {})
    delay = gap.get("delay", {})
    deadband = gap.get("deadband_stiction_proxy", {})
    pose = gap.get("object_pose_noise", {})
    contact = gap.get("contact", {})
    recommendations = gap.get("recommendations", {})
    real_data_quality = results.get("real_data_quality_gate", {}).get("metrics", {})
    video_camera = results.get("video_camera_tuning", {}).get("metrics", {})
    video_friction = results.get("video_contact_friction", {}).get("metrics", {})
    return {
        "purpose": "Use real trajectory, uploaded video, camera, pose, and contact data to tune sim parameters before policy training.",
        "trajectory_data": {
            "episodes": summary.get("episodes"),
            "records": summary.get("records"),
            "estimated_rate_hz": summary.get("estimated_rate_hz"),
            "joint_velocity_coverage": real_data_quality.get("joint_velocity_coverage"),
            "success_labels_required": False,
        },
        "actuator_latency": {
            "delay_steps": delay.get("delay_steps"),
            "delay_seconds": delay.get("delay_seconds"),
            "delay_confidence": delay.get("confidence"),
            "sample_rate_hz": delay.get("sample_rate_hz"),
            "min_sample_rate_hz": delay.get("min_sample_rate_hz"),
            "observability_status": delay.get("observability_status"),
            "deadband_command_norm": deadband.get("deadband_command_norm"),
            "swallowed_command_ratio": deadband.get("swallowed_command_ratio"),
            "deadband_confidence": deadband.get("confidence"),
        },
        "camera_pose_noise": {
            "samples": pose.get("samples"),
            "position_error_mean_m": pose.get("position_error_mean_m"),
            "position_error_p95_m": pose.get("position_error_p95_m"),
            "orientation_error_mean_deg": pose.get("orientation_error_mean_deg"),
            "orientation_error_p95_deg": pose.get("orientation_error_p95_deg"),
            "validation_source": pose.get("validation_source"),
        },
        "camera_video_tuning": {
            "video_count": video_camera.get("camera_video_count"),
            "analysis_available": video_camera.get("analysis_available"),
            "reprojection_error_px": video_camera.get("reprojection_error_px"),
            "suggested_camera_parameters": video_camera.get("suggested_camera_parameters", {}),
        },
        "contact": {
            "samples": contact.get("samples"),
            "mean_force_n": contact.get("mean_force_n", contact.get("mean_n")),
            "p95_force_n": contact.get("p95_force_n", contact.get("p95_n")),
            "peak_force_n": contact.get("peak_force_n", contact.get("peak_n")),
            "over_limit_ratio": contact.get("over_limit_ratio"),
            "force_limit_n": contact.get("force_limit_n"),
        },
        "contact_friction_video": {
            "video_count": video_friction.get("friction_video_count"),
            "analysis_available": video_friction.get("analysis_available"),
            "object_static_friction": video_friction.get("object_static_friction"),
            "object_dynamic_friction": video_friction.get("object_dynamic_friction"),
            "gripper_pad_static_friction": video_friction.get("gripper_pad_static_friction"),
            "gripper_pad_dynamic_friction": video_friction.get("gripper_pad_dynamic_friction"),
            "slip_ratio": video_friction.get("slip_ratio"),
            "suggested_sim_params": video_friction.get("suggested_sim_params", {}),
        },
        "sysid_backends": {
            "local_log_estimator": "used",
            "newton_sysid": _skill_status(results, "newton_sysid"),
            "pace_sysid": _skill_status(results, "pace_sysid"),
        },
        "tuning_outputs": {
            "action_scale_candidate": recommendations.get("action_scale", {}).get("suggested"),
            "domain_randomization_families": sorted((recommendations.get("domain_randomization") or {}).keys()),
            "sysid_target_count": len(recommendations.get("sysid_targets", [])),
            "autoresearch_experiments": [item.get("id") for item in plan.get("experiments", [])],
        },
    }


def _policy_release_metrics(
    config: PipelineConfig,
    scoreboard: dict[str, Any],
    real_success: float | None,
    sim_success: float | None,
    release_gap_score: float,
) -> dict[str, Any]:
    target_success = dict(config.task_spec.get("goal", {})).get("real_success", config.agent.get("target_real_success", 0.8))
    return {
        "purpose": "Use this lane only after a trained policy candidate and release evidence exist.",
        "success": {
            "sim": sim_success,
            "real": real_success,
            "real_target": _round_or_none(target_success),
            "real_success_labels_present": real_success is not None,
        },
        "release_gap_score": {
            "value": release_gap_score,
            "target": _release_gap_target(config),
            "target_decided_by": "user config or release policy, not the LLM",
            "meaning": "normalized release-readiness gap; lower is better; not a physical distance",
        },
        "release_gate_status": scoreboard.get("status"),
        "human_review_readiness": scoreboard.get("human_review_readiness", "not_ready"),
        "release_candidate_ready": bool(scoreboard.get("release_candidate_ready", False)),
        "hardware_approval_status": scoreboard.get("hardware_approval_status", "not_requested"),
        "safe_to_autorun_robot": False,
    }


def _skill_status(results: dict[str, dict[str, Any]], skill_id: str) -> str:
    if skill_id not in results:
        return "not_run"
    return str(results.get(skill_id, {}).get("status", "unknown"))


def _policy_checkpoint_record(config: PipelineConfig, config_path: str | Path | None) -> dict[str, Any]:
    configured = _policy_checkpoint(config)
    resolved = _resolve_artifact_path(configured, config_path)
    exists = resolved.exists() if resolved else False
    return {
        "configured": configured,
        "resolved_path": str(resolved) if resolved else None,
        "exists": exists,
        "path_type": "directory" if exists and resolved.is_dir() else "file" if exists else "missing",
        "sha256": _dataset_hash(resolved) if exists else None,
        "file_count": len(_file_manifest(resolved)) if exists else 0,
        "files": _file_manifest(resolved) if exists else [],
        "using_sample_artifacts": "golden/sample_inputs" in configured.replace("\\", "/"),
    }


def _retraining_record(config: PipelineConfig, config_path: str | Path | None) -> dict[str, Any]:
    policy_cfg = dict(config.policy)
    retraining_cfg = policy_cfg.get("retraining", {})
    if not isinstance(retraining_cfg, dict):
        retraining_cfg = {}
    task_cfg = dict(config.task_spec)
    requested = bool(
        retraining_cfg.get("enabled")
        or policy_cfg.get("retrain")
        or task_cfg.get("retrain")
        or policy_cfg.get("training_command")
        or policy_cfg.get("retrained_checkpoint")
    )
    source_checkpoint = str(
        retraining_cfg.get("source_checkpoint")
        or policy_cfg.get("source_checkpoint")
        or policy_cfg.get("artifact_dir")
        or task_cfg.get("policy_ckpt")
        or ""
    )
    target_checkpoint = str(
        retraining_cfg.get("target_checkpoint")
        or policy_cfg.get("retrained_checkpoint")
        or task_cfg.get("retrained_policy_ckpt")
        or ""
    )
    target_path = _resolve_artifact_path(target_checkpoint, config_path) if target_checkpoint else None
    return {
        "requested": requested,
        "source_checkpoint": source_checkpoint,
        "target_checkpoint": target_checkpoint or None,
        "target_checkpoint_resolved": str(target_path) if target_path else None,
        "target_checkpoint_exists": bool(target_path and target_path.exists()),
        "training_command": retraining_cfg.get("command") or policy_cfg.get("training_command") or config.isaac_lab.get("train_command"),
        "training_run_id": retraining_cfg.get("training_run_id") or policy_cfg.get("training_run_id"),
        "notes": "When retraining is enabled, this record captures the input checkpoint and expected retrained checkpoint lineage.",
    }


def _resolve_artifact_path(value: str | Path | None, config_path: str | Path | None) -> Path | None:
    if value in (None, "", "not_configured"):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    root = _repo_root(Path(config_path).expanduser().parent if config_path else Path.cwd())
    candidate = root / path
    if candidate.exists():
        return candidate
    config_parent = Path(config_path).expanduser().parent if config_path else Path.cwd()
    fallback = config_parent / path
    return fallback if fallback.exists() else candidate


def _repo_root(start: Path) -> Path:
    path = start.expanduser().resolve()
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return path


def _canonical_records_path(source: Path) -> Path | None:
    if source.is_dir():
        for candidate in [source / "aligned" / "records.jsonl", source / "records.jsonl"]:
            if candidate.exists():
                return candidate
        return None
    return source if source.exists() else None


def _file_manifest(source: Path | None) -> list[dict[str, Any]]:
    if not source or not source.exists():
        return []
    if source.is_file():
        files = [source]
        root = source.parent
    else:
        files = sorted(item for item in source.rglob("*") if item.is_file())
        root = source
    manifest = []
    for file_path in files:
        manifest.append(
            {
                "relative_path": file_path.relative_to(root).as_posix(),
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "sha256": _path_hash(file_path),
            }
        )
    return manifest


def _baseline_success_from_results(results: dict[str, dict[str, Any]]) -> float | None:
    baseline = results.get("sim_eval_regression", {}).get("metrics", {}).get("baseline", {})
    if isinstance(baseline, dict) and baseline.get("success_rate") is not None:
        return _round_or_none(baseline.get("success_rate"))
    return None


def _pipeline_output_status(scoreboard: dict[str, Any], trace: dict[str, Any] | None) -> str:
    if trace:
        gate = trace.get("release_gate_decides", {})
        human = trace.get("human_approves_hardware", {})
        if gate.get("status") == "blocked":
            return "blocked"
        if human.get("status") == "approved_for_supervised_run":
            return "human_approved_for_supervised_run"
    if scoreboard.get("release_candidate_ready"):
        return "ready_for_human_review"
    if scoreboard.get("human_review_readiness") == "smoke_review_only":
        return "smoke_review_only"
    if scoreboard.get("status") == "pass":
        return "offline_validated"
    return "blocked"


def _deploy_command(release_id: str, status: str) -> str:
    if status in {"ready_for_human_review", "human_approved_for_supervised_run"}:
        return f"coach deploy {release_id} --supervised"
    return "blocked until release gate and human hardware approval"


def _git_sha(start: Path) -> str:
    path = start.expanduser().resolve()
    for candidate in [path, *path.parents]:
        git_dir = candidate / ".git"
        head = git_dir / "HEAD"
        if not head.exists():
            continue
        text = head.read_text().strip()
        if text.startswith("ref:"):
            ref = git_dir / text.split(" ", 1)[1]
            return ref.read_text().strip() if ref.exists() else "unknown"
        return text
    env_sha = os.environ.get("GIT_COMMIT") or os.environ.get("CI_COMMIT_SHA")
    return env_sha or "unknown"


def _run_id(dataset_path: str | Path, config_path: str | Path | None) -> str:
    seed = f"{Path(dataset_path).expanduser()}::{config_path or ''}::{_dataset_hash(dataset_path)}"
    return "run_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]


def _run_version(run_id: str) -> str:
    suffix = str(run_id).replace("run_", "")[:8] or "unknown"
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{suffix}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return round(float(value), 3)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
