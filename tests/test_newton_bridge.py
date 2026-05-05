from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from agentic_sim2real.newton_bridge import parse_newton_outputs, prepare_newton_input, run_newton_bridge


def _write_records(path: Path) -> None:
    rows = []
    for idx in range(6):
        command = [0.10 + idx * 0.001, -1.20, 1.30, -1.60, -1.50, 0.02]
        measured = [command[0] - 0.0005, -1.20, 1.30, -1.60, -1.50, 0.02]
        rows.append(
            {
                "episode_index": 0,
                "timestamp": idx * 0.002,
                "action": [0.0] * 6,
                "joint_command": command,
                "joint_state": measured,
                "joint_velocity": [0.0] * 6,
                "object_pose_estimate": [0.5, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0],
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _config() -> dict:
    return {
        "robot": {"arm": "ur10e"},
        "sysid": {
            "newton_robot_name": "ur10e",
            "newton_joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "newton_joint_types": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
            "newton_command_source": "auto",
        },
    }


class NewtonBridgeTests(unittest.TestCase):
    def test_prepare_newton_input_writes_sage_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            records = tmp_path / "records.jsonl"
            _write_records(records)

            result = prepare_newton_input(records, _config(), tmp_path / "newton_input")

            self.assertEqual(result["metrics"]["newton_input_records"], 6)
            self.assertEqual(result["metrics"]["newton_joint_count"], 6)
            self.assertTrue((tmp_path / "newton_input" / "joint_list.txt").exists())
            with (tmp_path / "newton_input" / "control.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["timestamp"], "0")
            self.assertEqual(len(json.loads(rows[0]["positions"])), 6)

    def test_parse_newton_outputs_collects_params_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "best_params.yaml").write_text(
                "\n".join(
                    [
                        "damping:",
                        "  shoulder_pan_joint: 0.11",
                        "  shoulder_lift_joint: 0.12",
                        "damping_mean: 0.115",
                        "armature_mean: 0.01",
                        "best_mse: 0.00042",
                        "robot_name: ur10e",
                        "mode: all_joints",
                    ]
                )
                + "\n"
            )
            (out / "run_summary.yaml").write_text(
                "\n".join(
                    [
                        "sysid:",
                        "  num_envs: 64",
                        "  max_iter: 100",
                        "  control_freq: 500",
                    ]
                )
                + "\n"
            )
            (out / "optimization_log.csv").write_text("generation,best_mse,mean_mse,min_mse\n0,0.01,0.02,0.01\n1,0.00042,0.01,0.00042\n")

            parsed = parse_newton_outputs(out)

            self.assertEqual(parsed["metrics"]["robot_name"], "ur10e")
            self.assertAlmostEqual(parsed["metrics"]["best_mse"], 0.00042)
            self.assertIn("damping", parsed["fitted_parameters"])

    def test_bridge_prepare_only_does_not_launch_newton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            records = tmp_path / "records.jsonl"
            output = tmp_path / "newton_output.json"
            _write_records(records)
            config = _config()
            config["sysid"]["newton_run_mode"] = "prepare_only"

            result = run_newton_bridge(
                {"dataset": str(records), "config": config},
                work_dir=tmp_path / "work",
                output_path=output,
            )

            self.assertEqual(result["status"], "not_applicable")
            self.assertTrue(result["metrics"]["newton_input_prepared"])
            self.assertFalse(result["metrics"]["newton_ran"])
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
