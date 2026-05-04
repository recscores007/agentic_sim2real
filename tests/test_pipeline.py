from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ur_agentic.cli import main
from ur_agentic.config import choose_task, load_config, nominal_action_scale
from ur_agentic.dataset import load_records
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


if __name__ == "__main__":
    unittest.main()
