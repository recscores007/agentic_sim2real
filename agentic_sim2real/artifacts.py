from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autoresearch import build_plan
from .config import PipelineConfig, choose_task
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
    return paths


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
    input_payload = {
        "schema": f"{SCHEMA_VERSION}.pipeline_input",
        "description": "Task spec consumed by the agent and harness. This is the slide-22 contract.",
        "task": task,
        "goal": {"real_success": _round_or_none(real_success), "gap_target": _round_or_none(gap_target)},
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
    transfer_score = float(plan["transfer_score"]["score_0_to_1"])
    sim2real_gap = round(max(0.0, 1.0 - transfer_score), 3)
    previous_gap = _previous_gap(previous_scorecard)
    real_success = summary.get("success_rate")
    sim_success = _sim_success_rate(results)
    regression_pp = _regression_pp(results)
    per_skill = {skill_id: float(result.get("quality_score", 0.0)) for skill_id, result in sorted(results.items())}
    failure_modes = _failure_modes(summary)
    scorecard = {
        "schema": f"{SCHEMA_VERSION}.scorecard",
        "description": "Unified per-run scorecard consumed by the agent and human reviewer. This is the slide-23 contract.",
        "task": _task_name(config),
        "run_id": run_id or _run_id(dataset_path, config_path),
        "git_sha": _git_sha(Path(config_path).parent if config_path else Path.cwd()),
        "sim2real_gap": sim2real_gap,
        "sim2real_gap_delta": None if previous_gap is None else round(sim2real_gap - previous_gap, 3),
        "success_rate": {"sim": sim_success, "real": real_success},
        "regression_pp": regression_pp,
        "per_skill": per_skill,
        "per_skill_detail": results,
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
        "release_id": release_id,
        "status": status,
        "policy_ckpt": _policy_checkpoint(config),
        "sim_config": {
            **_sim_config(config, config_path),
            "patches": changes,
        },
        "sim2real_gap": {
            "before": _round_or_none(baseline_gap),
            "after": scorecard.get("sim2real_gap"),
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
            f"- Goal: real_success >= {payload['goal']['real_success']}, gap <= {payload['goal']['gap_target']}",
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
    delta = payload.get("sim2real_gap_delta")
    delta_text = "n/a" if delta is None else f"{delta:+.3f}"
    lines = [
        "# Scorecard",
        "",
        f"- Task: {payload['task']}",
        f"- Run: `{payload['run_id']}`",
        f"- Verdict: {payload['verdict']}",
        f"- Sim2real gap: {payload['sim2real_gap']} ({delta_text})",
        f"- Success rate: sim={payload['success_rate']['sim']} real={payload['success_rate']['real']}",
        f"- Regression: {payload['regression_pp']} pp",
        f"- Recommended next skill: {payload['recommended_skill']}",
        "",
        "## Per Skill",
        "",
    ]
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
        f"- Status: {payload['status']}",
        f"- Policy: `{payload['policy_ckpt']}`",
        f"- Sim config: `{payload['sim_config']['hash']}`",
        f"- Sim2real gap: {payload['sim2real_gap']['before']} -> {payload['sim2real_gap']['after']}",
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


def _records_by_episode(records: list[Record]) -> dict[int, list[Record]]:
    grouped: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.episode_index].append(record)
    for values in grouped.values():
        values.sort(key=lambda item: item.timestamp)
    return grouped


def _task_name(config: PipelineConfig) -> str:
    return str(config.task_spec.get("task") or choose_task(config))


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
        if patch_bits:
            patches.append({"skill": "domain_randomization_update", "patch": "; ".join(patch_bits), "status": dr.get("status")})
    for skill in ("newton_sysid", "pace_sysid"):
        result = results.get(skill, {})
        metrics = result.get("metrics", {})
        if result and result.get("status") == "pass":
            patches.append({"skill": skill, "patch": "fitted physics parameters available", "status": result.get("status"), "metrics": metrics})
    return patches


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


def _round_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return round(float(value), 3)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
