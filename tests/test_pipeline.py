from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from agentic_sim2real.cli import main
from agentic_sim2real.config import choose_task, load_config, nominal_action_scale
from agentic_sim2real.data_quality import evaluate_real_data_quality
from agentic_sim2real.dataset import load_records
from agentic_sim2real.embodiments import validate_embodiments
from agentic_sim2real.evaluation_loop import run_evaluation_loop
from agentic_sim2real.llm_orchestrator import LLMProvider, ScriptedLLMProvider, run_llm_orchestrated_loop
from agentic_sim2real.preflight import run_preflight
from agentic_sim2real.real_data import inspect_real_session, prepare_real_session
from agentic_sim2real.skill_harness import run_harness, validate_all_manifests
from agentic_sim2real.sysid import estimate_gap


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_load_sample_records(self) -> None:
        records = load_records(ROOT / "sample_data" / "real_log_demo.jsonl")
        self.assertEqual(len(records), 15)
        self.assertEqual(records[0].episode_index, 0)
        self.assertEqual(len(records[0].action), 6)

    def test_config_selects_ur_2f140_task(self) -> None:
        cfg = load_config(ROOT / "configs" / "ur10e_gear_assembly.example.json")
        self.assertEqual(choose_task(cfg), "Isaac-Deploy-GearAssembly-UR10e-2F140-v0")
        self.assertEqual(nominal_action_scale(cfg), 0.0325)

    def test_estimate_gap_contains_tutorial_parameters(self) -> None:
        cfg = load_config(ROOT / "configs" / "ur10e_gear_assembly.example.json")
        records = load_records(ROOT / "sample_data" / "real_log_demo.jsonl")
        gap = estimate_gap(records, cfg)
        dr = gap["recommendations"]["domain_randomization"]
        self.assertIn("object_pose_observation_noise", dr)
        self.assertIn("shaft_pose_observation_noise", dr)
        self.assertIn("object_and_base_pose_randomization", dr)
        self.assertIn("base_and_gear_pose_randomization", dr)
        self.assertEqual(dr["actuator_and_contact_randomization"]["stiffness_scale_log_uniform"], [0.75, 1.5])
        self.assertGreaterEqual(gap["recommendations"]["action_scale"]["suggested"], 0.005)

    def test_cli_analyze_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--config",
                    str(ROOT / "configs" / "ur10e_gear_assembly.example.json"),
                    "analyze",
                    "--dataset",
                    str(ROOT / "sample_data" / "real_log_demo.jsonl"),
                    "--out",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            report = Path(tmp) / "report.md"
            score = Path(tmp) / "transfer_score.json"
            params = Path(tmp) / "agentic_params.yaml"
            self.assertTrue(report.exists())
            self.assertTrue(params.exists())
            self.assertIn("score_0_to_1", json.loads(score.read_text()))

    def test_skill_manifests_validate(self) -> None:
        result = validate_all_manifests(ROOT)
        self.assertEqual(result["status"], "pass")
        self.assertIn("autoresearch_planner", result["skills"])
        self.assertIn("real_data_quality_gate", result["skills"])
        self.assertIn("newton_sysid", result["skills"])
        self.assertIn("pace_sysid", result["skills"])
        self.assertIn("isaaclab_rollout_regression", result["skills"])

    def test_harness_writes_scoreboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scoreboard = run_harness(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=tmp,
            )
            self.assertEqual(scoreboard["status"], "pass")
            self.assertIn("real_data_quality_gate", scoreboard["skills"])
            self.assertEqual(scoreboard["skills"]["newton_sysid"]["status"], "evidence_missing")
            self.assertEqual(scoreboard["skills"]["pace_sysid"]["status"], "evidence_missing")
            self.assertEqual(scoreboard["skills"]["isaaclab_rollout_regression"]["status"], "not_applicable")
            self.assertEqual(scoreboard["skills"]["real_robot_gate"]["status"], "not_approved")
            self.assertEqual(scoreboard["human_review_readiness"], "smoke_review_only")
            self.assertFalse(scoreboard["release_candidate_ready"])
            self.assertIn("release_candidate_gate", scoreboard["skills"])
            self.assertTrue((Path(tmp) / "scoreboard.json").exists())
            self.assertTrue((Path(tmp) / "rollout_data.json").exists())
            self.assertTrue((Path(tmp) / "pipeline_input.json").exists())
            self.assertTrue((Path(tmp) / "scorecard.json").exists())
            self.assertTrue((Path(tmp) / "pipeline_output.json").exists())
            scorecard = json.loads((Path(tmp) / "scorecard.json").read_text())
            self.assertEqual(scorecard["schema"], "agentic_sim2real.slide_contract.v1.scorecard")
            self.assertIn("sim2real_gap", scorecard)
            self.assertEqual(scorecard["mode"], "characterization")
            self.assertIn("release_gap_score", scorecard)
            self.assertIn("transfer_readiness_score", scorecard)
            self.assertIn("run_version", scorecard)
            self.assertIn("transfer_readiness_breakdown", scorecard)
            self.assertIn("release_gap_breakdown", scorecard)
            self.assertIn("characterization", scorecard)
            self.assertIn("policy_release", scorecard)
            pipeline_output = json.loads((Path(tmp) / "pipeline_output.json").read_text())
            self.assertEqual(pipeline_output["schema"], "agentic_sim2real.slide_contract.v1.pipeline_output")
            self.assertIn("release_gap_score", pipeline_output)
            self.assertEqual(pipeline_output["run_version"], scorecard["run_version"])
            self.assertFalse(pipeline_output["safe_to_autorun_robot"])
            artifacts = scoreboard["artifacts"]
            for key in ["ui", "state", "run_record", "real_data_manifest"]:
                self.assertTrue(Path(artifacts[key]).exists())
            ui_state = json.loads(Path(artifacts["state"]).read_text())
            self.assertEqual(ui_state["schema_version"], "agentic_sim2real.pipeline_ui.v1")
            self.assertEqual(ui_state["mode"], "characterization")
            self.assertIn("workflow", ui_state)
            self.assertIn("run_record", ui_state)
            run_record = json.loads(Path(artifacts["run_record"]).read_text())
            self.assertEqual(run_record["schema"], "agentic_sim2real.slide_contract.v1.run_record")
            self.assertEqual(run_record["run"]["run_version"], scorecard["run_version"])
            self.assertGreater(run_record["lineage"]["real_data_fed"]["file_count"], 0)
            self.assertIn("policy_checkpoint", run_record["lineage"])
            self.assertIn("transfer_readiness", run_record["score_breakdown"])
            real_data_manifest = json.loads(Path(artifacts["real_data_manifest"]).read_text())
            self.assertEqual(real_data_manifest["schema"], "agentic_sim2real.slide_contract.v1.real_data_manifest")
            self.assertGreater(len(real_data_manifest["files"]), 0)

    def test_stronger_preflight_reports_sysid_backends(self) -> None:
        cfg = load_config(ROOT / "configs" / "ur10e_gear_assembly.example.json")
        report = run_preflight(cfg, root=ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertIn("newton", report["sysid_backends"])
        self.assertIn("pace", report["sysid_backends"])
        self.assertTrue(report["sysid_backends"]["local"]["available"])

        cfg_with_pace_command = load_config(ROOT / "configs" / "ur10e_gear_assembly.example.json")
        cfg_with_pace_command.sysid["pace_enabled"] = True
        cfg_with_pace_command.sysid["pace_command"] = ["python3", "custom_pace_adapter.py"]
        command_report = run_preflight(cfg_with_pace_command, root=ROOT)
        self.assertEqual(command_report["status"], "pass")
        self.assertTrue(command_report["sysid_backends"]["pace"]["available"])
        self.assertTrue(command_report["sysid_backends"]["pace"]["command_configured"])

    def test_custom_command_skill_overrides_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            custom_root = tmp_path / "custom_skills"
            custom_skill = custom_root / "env_preflight"
            custom_skill.mkdir(parents=True)
            runner = tmp_path / "custom_env_preflight.py"
            runner.write_text(
                "\n".join(
                    [
                        "import json",
                        "import os",
                        "from pathlib import Path",
                        "out = Path(os.environ['AGENTIC_SIM2REAL_SKILL_OUTPUT_JSON'])",
                        "skill_out = Path(os.environ['AGENTIC_SIM2REAL_SKILL_OUT_DIR'])",
                        "evidence = skill_out / 'custom_evidence.json'",
                        "evidence.write_text(json.dumps({'custom': True}) + '\\n')",
                        "out.write_text(json.dumps({",
                        "    'status': 'pass',",
                        "    'quality_score': 0.93,",
                        "    'confidence': 0.88,",
                        "    'blocking_failures': [],",
                        "    'warnings': ['custom override used'],",
                        "    'evidence_files': [str(evidence)],",
                        "    'metrics': {'custom_skill_used': True},",
                        "}) + '\\n')",
                    ]
                )
                + "\n"
            )
            (custom_skill / "skill.json").write_text(
                json.dumps(
                    {
                        "id": "env_preflight",
                        "name": "Test Custom Environment Preflight",
                        "owner_agent": "orchestrator_agent",
                        "description": "Test override for env preflight.",
                        "implementation": "external_command",
                        "runner": "command",
                        "command": ["python3", str(runner)],
                        "timeout_s": 30,
                        "depends_on": [],
                        "inputs": ["config"],
                        "outputs": ["custom_evidence.json", "result.json"],
                        "validators": ["custom command writes a skill result"],
                        "quality_gate": {"min_score": 0.7},
                        "human_required": False,
                        "release_blocking": True,
                        "real_robot": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            out_dir = tmp_path / "out"
            scoreboard = run_harness(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=out_dir,
                only_skill="env_preflight",
                skill_dirs=[custom_root],
            )
            result = scoreboard["skills"]["env_preflight"]
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["metrics"]["custom_skill_used"])
            self.assertIn("external_command_log.json", result["evidence_files"][-1])

    def test_evaluation_loop_writes_five_stage_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_evaluation_loop(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=tmp,
            )
            self.assertIn("agent_proposes", trace)
            self.assertIn("evaluator_measures", trace)
            self.assertIn("critic_challenges", trace)
            self.assertIn("release_gate_decides", trace)
            self.assertIn("human_approves_hardware", trace)
            self.assertIn("llm_orchestrator", trace)
            self.assertEqual(trace["llm_orchestrator"]["provider"], "scripted")
            self.assertFalse(trace["release_gate_decides"]["safe_to_autorun_robot"])
            self.assertEqual(trace["offline_validation_status"], "pass")
            self.assertEqual(trace["release_gate_decides"]["human_review_readiness"], "smoke_review_only")
            self.assertFalse(trace["release_gate_decides"]["release_candidate_ready"])
            self.assertTrue((Path(tmp) / "evaluation_trace.md").exists())
            self.assertTrue((Path(tmp) / "rollout_data.json").exists())
            self.assertTrue((Path(tmp) / "pipeline_input.md").exists())
            self.assertTrue((Path(tmp) / "scorecard.json").exists())
            self.assertTrue((Path(tmp) / "pipeline_output.md").exists())
            self.assertIn("slide_contract_artifacts", trace)
            self.assertIn("ui_artifacts", trace)
            self.assertTrue(Path(trace["ui_artifacts"]["ui"]).exists())
            self.assertTrue(Path(trace["ui_artifacts"]["state"]).exists())

    def test_llm_orchestrator_runs_skills_and_writes_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_llm_orchestrated_loop(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=tmp,
            )
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["provider"], "scripted")
            self.assertIn("release_candidate_gate", summary["completed_skill_ids"])
            self.assertIn("real_robot_gate", summary["completed_skill_ids"])
            self.assertIn("isaaclab_rollout_regression", summary["completed_skill_ids"])
            self.assertEqual(summary["scoreboard"]["skills"]["real_robot_gate"]["status"], "not_approved")
            journal = Path(summary["journal"])
            self.assertTrue(journal.exists())
            lines = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertGreaterEqual(len(lines), 15)
            self.assertEqual(lines[0]["status"], "skill_completed")
            self.assertIn("scorecard", lines[0])
            self.assertTrue(Path(lines[0]["scorecard"]["scorecard"]).exists())
            self.assertIn("ui_artifacts", summary)
            for key in ["ui", "state"]:
                self.assertTrue(Path(summary["ui_artifacts"][key]).exists())
            state = json.loads(Path(summary["ui_artifacts"]["state"]).read_text())
            self.assertEqual(state["schema_version"], "agentic_sim2real.pipeline_ui.v1")
            self.assertEqual(state["run"]["status"], "complete")
            self.assertFalse(state["run"]["safe_to_autorun_robot"])
            self.assertFalse(state["scoreboard"]["safe_to_autorun_robot"])
            self.assertIn("pipeline_input", state)
            self.assertIn("rollout_summary", state)
            self.assertGreaterEqual(len(state["journal"]), 15)
            self.assertIn("characterization", state)
            self.assertIn("policy_release", state)
            artifacts = summary["scoreboard"]["artifacts"]
            for key in ["rollout_data", "pipeline_input", "scorecard", "pipeline_output", "run_record", "real_data_manifest", "ui", "state"]:
                self.assertTrue(Path(artifacts[key]).exists())
            final_output = json.loads(Path(artifacts["pipeline_output"]).read_text())
            self.assertEqual(final_output["schema"], "agentic_sim2real.slide_contract.v1.pipeline_output")
            self.assertFalse(final_output["safe_to_autorun_robot"])

    def test_llm_orchestrator_uses_user_gap_hint_for_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_llm_orchestrated_loop(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=tmp,
                gap_hints=["contact"],
            )
            self.assertEqual(summary["gap_hints"][0]["normalized"], "contact")
            completed = summary["completed_skill_ids"]
            self.assertLess(completed.index("action_scale_sweep"), completed.index("newton_sysid"))
            first_context = Path(tmp) / "llm_orchestrator" / "steps" / "step_001_context.json"
            context = json.loads(first_context.read_text())
            self.assertEqual(context["task"]["user_gap_hints"][0]["normalized"], "contact")
            self.assertIn("action_scale_sweep", context["task"]["gap_hint_priority_skill_ids"])

    def test_llm_orchestrator_rejects_invalid_release_gate_call(self) -> None:
        class BadReleaseThenScriptedProvider(LLMProvider):
            name = "bad_release_then_scripted"

            def __init__(self) -> None:
                self.scripted = ScriptedLLMProvider()

            def decide(self, context: dict) -> dict:
                if int(context["step"]) == 1:
                    return {
                        "action": "run_skill",
                        "skill_id": "release_candidate_gate",
                        "rationale": "bad early release decision",
                        "confidence": 0.9,
                    }
                return self.scripted.decide(context)

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_llm_orchestrated_loop(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=tmp,
                provider=BadReleaseThenScriptedProvider(),
            )
            self.assertEqual(summary["status"], "pass")
            lines = [json.loads(line) for line in Path(summary["journal"]).read_text().splitlines()]
            self.assertEqual(lines[0]["status"], "rejected")
            self.assertTrue(
                any("release_candidate_gate cannot run" in reason for reason in lines[0]["guardrail"]["reasons"])
            )
            self.assertIn("ui_artifacts", summary)
            self.assertTrue(Path(summary["ui_artifacts"]["ui"]).exists())
            first_completed = next(entry for entry in lines if entry["status"] == "skill_completed")
            self.assertTrue(Path(first_completed["scorecard"]["scorecard"]).exists())

    def test_llm_orchestrator_command_provider_runs_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runner = tmp_path / "fake_llm.py"
            runner.write_text(
                "\n".join(
                    [
                        "import json",
                        "import os",
                        "from pathlib import Path",
                        "ctx = json.loads(Path(os.environ['AGENTIC_SIM2REAL_LLM_INPUT_JSON']).read_text())",
                        "out = Path(os.environ['AGENTIC_SIM2REAL_LLM_OUTPUT_JSON'])",
                        "runnable = ctx.get('runnable_skills', [])",
                        "if runnable:",
                        "    decision = {'action': 'run_skill', 'skill_id': runnable[0], 'rationale': 'fake llm picked first runnable', 'confidence': 0.91}",
                        "elif 'release_candidate_gate' in ctx.get('completed_skill_ids', []) and 'real_robot_gate' in ctx.get('completed_skill_ids', []):",
                        "    decision = {'action': 'request_human_review', 'rationale': 'fake llm sees release evidence', 'confidence': 0.91}",
                        "else:",
                        "    decision = {'action': 'stop', 'rationale': 'nothing runnable', 'confidence': 0.5}",
                        "out.write_text(json.dumps(decision) + '\\n')",
                    ]
                )
                + "\n"
            )
            summary = run_llm_orchestrated_loop(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=tmp_path / "out",
                provider_name="command",
                provider_command=["python3", str(runner)],
            )
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["provider"], "command")

    def test_prepare_real_data_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "records.jsonl"
            session = ROOT / "embodiments" / "manipulator" / "ur10e_gear_assembly" / "real_data" / "example_session"
            summary = prepare_real_session(session, out_path=out)
            self.assertEqual(summary["records"], 12)
            self.assertTrue(out.exists())
            records = load_records(out)
            self.assertEqual(len(records), 12)
            self.assertEqual(len(records[0].action), 6)
            self.assertEqual(len(records[0].raw.get("joint_command", [])), 6)
            self.assertGreater(len(records[0].shaft_pose_estimate), 0)

    def test_inspect_real_data_uses_ur_adapter(self) -> None:
        session = ROOT / "embodiments" / "manipulator" / "ur10e_gear_assembly" / "real_data" / "example_session"
        report = inspect_real_session(session, root=ROOT)
        self.assertEqual(report["status"], "aligned_records_ready")
        self.assertEqual(report["adapter"]["id"], "manipulator/ur10e_gear_assembly")
        self.assertIn("shaft_pose.csv", report["accepted_pose_files"])
        self.assertTrue(report["quality_inputs"]["calibration_present"])

    def test_data_quality_gate_reports_canonical_records(self) -> None:
        cfg = load_config(ROOT / "configs" / "ur10e_gear_assembly.example.json")
        session = ROOT / "embodiments" / "manipulator" / "ur10e_gear_assembly" / "real_data" / "example_session"
        report = evaluate_real_data_quality(session, cfg, root=ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["metrics"]["episodes"], 3)
        self.assertEqual(report["metrics"]["missing_object_pose_estimate"], 0)

    def test_harness_auto_prepares_raw_csv_session(self) -> None:
        source = ROOT / "embodiments" / "manipulator" / "ur10e_gear_assembly" / "real_data" / "example_session"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = tmp_path / "real_data" / "session"
            shutil.copytree(source, session)
            shutil.rmtree(session / "aligned")
            (session / "aligned").mkdir()
            scoreboard = run_harness(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=session,
                out_dir=tmp_path / "out",
            )
            self.assertEqual(scoreboard["status"], "pass")
            self.assertTrue((session / "aligned" / "records.jsonl").exists())

    def test_newton_sysid_command_adapter_runs_from_canonical_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runner = tmp_path / "fake_newton.py"
            runner.write_text(
                "\n".join(
                    [
                        "import json",
                        "import os",
                        "from pathlib import Path",
                        "payload = json.loads(Path(os.environ['AGENTIC_SIM2REAL_NEWTON_INPUT_JSON']).read_text())",
                        "assert Path(payload['dataset']).exists()",
                        "Path(os.environ['AGENTIC_SIM2REAL_NEWTON_OUTPUT_JSON']).write_text(json.dumps({",
                        "    'confidence': 0.82,",
                        "    'quality_score': 0.88,",
                        "    'metrics': {'newton_fit_used': True, 'records': payload['dataset']},",
                        "    'fitted_parameters': {'joint_friction_scale': 1.1},",
                        "}) + '\\n')",
                    ]
                )
                + "\n"
            )
            config = json.loads((ROOT / "configs" / "ur10e_gear_assembly.example.json").read_text())
            config["sysid"]["newton_enabled"] = True
            config["sysid"]["newton_command"] = ["python3", str(runner)]
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config) + "\n")
            scoreboard = run_harness(
                root=ROOT,
                config_path=config_path,
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=tmp_path / "out",
                only_skill="newton_sysid",
            )
            result = scoreboard["skills"]["newton_sysid"]
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["metrics"]["newton_fit_used"])

    def test_pace_sysid_command_adapter_runs_as_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runner = tmp_path / "fake_pace.py"
            runner.write_text(
                "\n".join(
                    [
                        "import json",
                        "import os",
                        "from pathlib import Path",
                        "summary = json.loads(Path(os.environ['AGENTIC_SIM2REAL_PACE_INPUT_JSON']).read_text())",
                        "assert summary['records'] >= 3",
                        "Path(os.environ.get('AGENTIC_SIM2REAL_SKILL_OUTPUT_JSON', os.environ.get('AGENTIC_SIM2REAL_PACE_OUTPUT_JSON', 'pace_output.json'))).write_text(json.dumps({",
                        "    'status': 'pass',",
                        "    'confidence': 0.81,",
                        "    'quality_score': 0.84,",
                        "    'metrics': {'pace_fit_used': True, 'records': summary['records']},",
                        "    'fitted_parameters': {'pace_delay_steps': 2},",
                        "}) + '\\n')",
                    ]
                )
                + "\n"
            )
            config = json.loads((ROOT / "configs" / "ur10e_gear_assembly.example.json").read_text())
            config["sysid"]["pace_enabled"] = True
            config["sysid"]["pace_command"] = ["python3", str(runner)]
            config["sysid"]["min_pace_records"] = 3
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config) + "\n")
            scoreboard = run_harness(
                root=ROOT,
                config_path=config_path,
                dataset_path=ROOT / "embodiments" / "manipulator" / "ur10e_gear_assembly" / "real_data" / "example_session",
                out_dir=tmp_path / "out",
                only_skill="pace_sysid",
            )
            result = scoreboard["skills"]["pace_sysid"]
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["metrics"]["pace_fit_used"])

    def test_release_candidate_profile_blocks_missing_strong_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = json.loads((ROOT / "configs" / "ur10e_gear_assembly.example.json").read_text())
            config["release"]["profile"] = "release_candidate"
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config) + "\n")
            trace = run_evaluation_loop(
                root=ROOT,
                config_path=config_path,
                dataset_path=ROOT / "embodiments" / "manipulator" / "ur10e_gear_assembly" / "real_data" / "example_session",
                out_dir=tmp_path / "out",
            )
            self.assertEqual(trace["release_gate_decides"]["status"], "blocked")
            joined = json.dumps(trace["release_gate_decides"]["blocking_failures"])
            self.assertIn("Newton or PACE SysID", joined)
            self.assertIn("held-out", joined)
            self.assertIn("Isaac Lab rollout", joined)

    def test_release_candidate_profile_accepts_explicit_waivers_and_user_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_dir = tmp_path / "policy"
            artifact_dir.mkdir()
            for name in ["agent.yaml", "env.yaml", "checkpoint.meta.json"]:
                (artifact_dir / name).write_text("{}\n")
            rollout_metrics = tmp_path / "rollout_metrics.json"
            rollout_metrics.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "episodes": 20,
                            "success_rate": 0.8,
                            "peak_force_n": 42.0,
                            "pose_error_p95_m": 0.004,
                        }
                    }
                )
                + "\n"
            )
            config = json.loads((ROOT / "configs" / "ur10e_gear_assembly.example.json").read_text())
            config["release"]["profile"] = "release_candidate"
            config["release"]["allow_sysid_waiver"] = True
            config["release"]["sysid_waiver_reason"] = "bench has validated local SysID for this smoke fixture"
            config["release"]["allow_heldout_waiver"] = True
            config["release"]["heldout_waiver_reason"] = "sample fixture has no held-out split"
            config["policy"]["artifact_dir"] = str(artifact_dir)
            config["isaac_lab"]["rollout_metrics_path"] = str(rollout_metrics)
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config) + "\n")
            trace = run_evaluation_loop(
                root=ROOT,
                config_path=config_path,
                dataset_path=ROOT / "embodiments" / "manipulator" / "ur10e_gear_assembly" / "real_data" / "example_session",
                out_dir=tmp_path / "out",
            )
            self.assertEqual(trace["release_gate_decides"]["status"], "promote_to_human_review")
            self.assertEqual(trace["human_review_readiness"], "ready")

    def test_prepare_generic_object_pose_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "records.jsonl"
            session = ROOT / "embodiments" / "manipulator" / "generic_manipulator" / "real_data" / "example_session"
            summary = prepare_real_session(session, out_path=out)
            self.assertEqual(summary["records"], 12)
            self.assertIn("object_pose.csv", summary["pose_file"])
            records = load_records(out)
            self.assertEqual(len(records), 12)
            self.assertEqual(len(records[0].action), 6)
            self.assertGreater(len(records[0].shaft_pose_estimate), 0)
            self.assertIn("object_pose_estimate", records[0].raw or {})
            self.assertNotIn("shaft_pose_estimate", records[0].raw or {})

    def test_embodiment_scaffolds_validate(self) -> None:
        result = validate_embodiments(ROOT)
        self.assertEqual(result["status"], "pass")
        self.assertIn("manipulator/ur10e_gear_assembly", result["embodiments"])
        self.assertIn("manipulator/generic_manipulator", result["embodiments"])
        self.assertIn("humanoid/generic_humanoid", result["embodiments"])
        self.assertIn("mobile_manipulator/generic_mobile_manipulator", result["embodiments"])


if __name__ == "__main__":
    unittest.main()
