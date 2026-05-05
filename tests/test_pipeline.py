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
            self.assertEqual(scoreboard["skills"]["newton_sysid"]["status"], "skip")
            self.assertIn("release_candidate_gate", scoreboard["skills"])
            self.assertTrue((Path(tmp) / "scoreboard.json").exists())

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
            self.assertFalse(trace["release_gate_decides"]["safe_to_autorun_robot"])
            self.assertTrue((Path(tmp) / "evaluation_trace.md").exists())

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
