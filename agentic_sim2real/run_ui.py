from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


UI_SCHEMA_VERSION = "agentic_sim2real.pipeline_ui.v1"


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
        "pipeline_output": pipeline_output,
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
        "skills": [
            {
                "skill_id": skill_id,
                "status": result.get("status"),
                "quality_score": result.get("quality_score"),
                "confidence": result.get("confidence"),
                "release_blocking": result.get("release_blocking"),
                "human_required": result.get("human_required"),
            }
            for skill_id, result in sorted(skills.items())
        ],
    }


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
.note {{ color:var(--muted); line-height:1.45; }}
@media (max-width:1050px) {{
  main,.lane-grid {{ grid-template-columns:1fr; }}
  .metrics,.workflow,.journal {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width:640px) {{
  header {{ padding:14px; }}
  main {{ padding:12px; }}
  .metrics,.workflow,.journal,.subgrid {{ grid-template-columns:1fr; }}
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
const input = state.pipeline_input || {{}};
const out = state.pipeline_output || {{}};
const rollout = state.rollout_summary || {{}};
const board = state.scoreboard || {{}};
const char = state.characterization || {{}};
const policy = state.policy_release || {{}};
const gap = score.release_gap_score ?? score.sim2real_gap;
document.getElementById("header-pills").innerHTML = [
  pill(state.mode),
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
    <div class="metric"><span>Real data</span><b>${{fmt(rollout.rollout_count)}} rollouts</b><div class="note">${{fmt(rollout.streams?.join(", "))}}</div></div>
    <div class="metric"><span>Real success</span><b>${{pct(rollout.success_rate)}}</b><div class="note">policy-release signal only</div></div>
  </div>
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
    <div class="kv"><label>Delay</label><strong>${{fmt(char.actuator_latency?.delay_steps)}} steps</strong></div>
    <div class="kv"><label>Deadband proxy</label><strong>${{num(char.actuator_latency?.deadband_command_norm)}}</strong></div>
    <div class="kv"><label>Pose p95 error</label><strong>${{fmt(char.camera_pose_noise?.position_error_p95_m)}} m</strong></div>
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

const skills = (board.skills || []).map(row => `<tr>
  <td><code>${{row.skill_id}}</code></td>
  <td>${{pill(row.status)}}</td>
  <td>${{num(row.quality_score)}}</td>
  <td>${{num(row.confidence)}}</td>
  <td>${{row.release_blocking ? "yes" : "no"}}</td>
  <td>${{row.human_required ? "yes" : "no"}}</td>
</tr>`).join("");

const journal = (state.journal || []).slice(-9).map(row => `<div class="journal-item">
  <b>#${{fmt(row.step)}} ${{fmt(row.skill_id || row.action)}}</b>
  <div style="margin-top:6px">${{pill(row.status)}}</div>
  <div class="note" style="margin-top:6px">${{fmt(row.rationale).slice(0, 120)}}</div>
</div>`).join("");

document.getElementById("app").innerHTML = [
  metricHtml,
  workflowHtml,
  `<div class="lane-grid">${{charHtml}}${{releaseHtml}}</div>`,
  `<section class="panel wide"><h2>Skill Validation</h2><table><thead><tr><th>Skill</th><th>Status</th><th>Score</th><th>Confidence</th><th>Release blocking</th><th>Human required</th></tr></thead><tbody>${{skills}}</tbody></table></section>`,
  `<section class="panel wide"><h2>LLM Orchestrator Journal</h2><div class="journal">${{journal || "<span class='note'>No LLM journal found for this run.</span>"}}</div></section>`,
  `<section class="panel wide"><h2>Score Meaning</h2><p class="note">${{fmt(score.score_meaning?.release_gap_score)}} ${{fmt(score.score_meaning?.target_source)}} Formula: ${{fmt(score.score_meaning?.formula)}}</p></section>`
].join("");
</script>
</body>
</html>"""
