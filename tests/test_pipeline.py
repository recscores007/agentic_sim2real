from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ur_agentic.cli import main
from ur_agentic.config import choose_task, load_config, nominal_action_scale
from ur_agentic.dataset import load_records
from ur_agentic.evaluation_loop import run_evaluation_loop
from ur_agentic.real_data import prepare_real_session
from ur_agentic.skill_harness import run_harness, validate_all_manifests
from ur_agentic.sysid import estimate_gap


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
        self.assertIn("shaft_pose_observation_noise", dr)
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

    def test_harness_writes_scoreboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scoreboard = run_harness(
                root=ROOT,
                config_path=ROOT / "configs" / "ur10e_gear_assembly.example.json",
                dataset_path=ROOT / "sample_data" / "real_log_demo.jsonl",
                out_dir=tmp,
            )
            self.assertEqual(scoreboard["status"], "pass")
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
                        "out = Path(os.environ['UR_SKILL_OUTPUT_JSON'])",
                        "skill_out = Path(os.environ['UR_SKILL_OUT_DIR'])",
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
            self.assertGreater(len(records[0].shaft_pose_estimate), 0)


if __name__ == "__main__":
    unittest.main()
