from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .autoresearch import build_plan
from .config import PipelineConfig, choose_task, command_env, load_config
from .dataset import load_records
from .evaluation_loop import run_evaluation_loop
from .real_data import prepare_real_session
from .report import write_outputs
from .safety import require_real_robot_gate
from .skill_harness import load_manifests, run_harness, validate_all_manifests
from .sysid import estimate_gap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic sim2real helper for the configured robot task")
    parser.add_argument("--config", default="configs/ur10e_gear_assembly.example.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preflight", help="Check local command availability")
    sub.add_parser("commands", help="Print tutorial-aligned Isaac Lab and Isaac ROS commands")
    sub.add_parser("list-skills", help="List atomic skills and owning agents")

    validate_skills = sub.add_parser("validate-skills", help="Validate every skill manifest")
    validate_skills.add_argument("--root", default=".")

    analyze = sub.add_parser("analyze", help="Analyze recorded real logs offline")
    analyze.add_argument("--dataset", required=True)
    analyze.add_argument("--out", required=True)

    prepare = sub.add_parser("prepare-real-data", help="Merge a real_data session into pipeline records.jsonl")
    prepare.add_argument("--session", required=True)
    prepare.add_argument("--out", default=None)
    prepare.add_argument("--tolerance-s", type=float, default=0.05)

    harness = sub.add_parser("run-harness", help="Run the skill validation harness")
    harness.add_argument("--root", default=".")
    harness.add_argument("--dataset", default="sample_data/real_log_demo.jsonl")
    harness.add_argument("--out", required=True)
    harness.add_argument("--skill", default=None, help="Run one skill by id")
    harness.add_argument("--include-real", action="store_true", help="Include real-robot skills after human approval")

    eval_loop = sub.add_parser("run-evaluation-loop", help="Run Agent/Evaluator/Critic/Release/Human trace")
    eval_loop.add_argument("--root", default=".")
    eval_loop.add_argument("--dataset", default="sample_data/real_log_demo.jsonl")
    eval_loop.add_argument("--out", required=True)
    eval_loop.add_argument("--threshold-policy", default=None)
    eval_loop.add_argument("--include-real", action="store_true")

    sub.add_parser("check-real-gate", help="Fail unless the real-robot human gate env var is set")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.cmd == "preflight":
        return cmd_preflight(config)
    if args.cmd == "commands":
        return cmd_commands(config)
    if args.cmd == "list-skills":
        return cmd_list_skills(Path("."))
    if args.cmd == "validate-skills":
        return cmd_validate_skills(args.root)
    if args.cmd == "analyze":
        return cmd_analyze(config, args.dataset, args.out)
    if args.cmd == "prepare-real-data":
        return cmd_prepare_real_data(args.session, args.out, args.tolerance_s)
    if args.cmd == "run-harness":
        return cmd_run_harness(args.root, args.config, args.dataset, args.out, args.include_real, args.skill)
    if args.cmd == "run-evaluation-loop":
        return cmd_run_evaluation_loop(
            args.root,
            args.config,
            args.dataset,
            args.out,
            args.threshold_policy,
            args.include_real,
        )
    if args.cmd == "check-real-gate":
        require_real_robot_gate(config)
        print("Human gate env var present. Continue only with active supervision.")
        return 0
    raise AssertionError(args.cmd)


def cmd_preflight(config: PipelineConfig) -> int:
    commands = ["python3", "ros2", "launch_test", "rqt_image_view"]
    print("Preflight checks")
    failed = False
    for command in commands:
        found = shutil.which(command)
        status = "ok" if found else "missing"
        print(f"- {command}: {status}{' (' + found + ')' if found else ''}")
        if command == "python3":
            failed = failed or not found
    print(f"- selected Isaac Lab task: {choose_task(config)}")
    print(f"- ROS_DOMAIN_ID: {config.isaac_ros['ros_domain_id']}")
    print(f"- RMW_IMPLEMENTATION: {config.isaac_ros['rmw_implementation']}")
    return 1 if failed else 0


def cmd_commands(config: PipelineConfig) -> int:
    env = command_env(config)
    print("# Export these first")
    for key, value in env.items():
        print(f"export {key}={_shell_quote(value)}")
    print()
    print("# Isaac Lab: visualize the training environment")
    print("cd $ISAAC_LAB_ROOT")
    print("python scripts/reinforcement_learning/rsl_rl/train.py --task $GEAR_TASK --num_envs 4")
    print()
    print("# Isaac Lab: full training with video recording")
    print(
        "python scripts/reinforcement_learning/rsl_rl/train.py "
        "--task $GEAR_TASK --headless --num_envs 256 --video --video_length 800 --video_interval 5000"
    )
    print()
    print("# Isaac ROS: validate pose estimation and calibration")
    print("export ENABLE_MANIPULATOR_TESTING=manual_on_robot")
    print("launch_test $(ros2 pkg prefix --share isaac_ros_manipulation_bringup)/test/test_pose_estimation_error_test.py")
    print("bash ${ISAAC_ROS_WS}/src/isaac_ros_manipulation/isaac_ros_manipulation_bringup/test/compare_pose_estimation_results.sh")
    print()
    print("# Isaac ROS: deploy the gear assembly workflow after the human gate")
    print("ros2 launch isaac_ros_manipulation_bringup workflows.launch.py manipulator_workflow_config:=$GEAR_MANIPULATOR_CONFIG")
    print("ros2 action send_goal $GEAR_ACTION isaac_ros_manipulation_interfaces/action/GearAssembly {}")
    return 0


def cmd_analyze(config: PipelineConfig, dataset_path: str, out_dir: str) -> int:
    records = load_records(dataset_path)
    gap = estimate_gap(records, config)
    plan = build_plan(gap, config)
    paths = write_outputs(out_dir, gap, plan)
    print(f"Analyzed {gap['summary']['records']} records across {gap['summary']['episodes']} episodes.")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    print(f"Transfer score: {plan['transfer_score']['score_0_to_1']} ({plan['transfer_score']['interpretation']})")
    return 0


def cmd_list_skills(root: Path) -> int:
    manifests = load_manifests(root)
    for manifest in manifests:
        data = manifest.data
        print(f"{data['id']}: {data['owner_agent']} - {data['name']}")
    return 0


def cmd_validate_skills(root: str) -> int:
    result = validate_all_manifests(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def cmd_run_harness(
    root: str,
    config_path: str,
    dataset_path: str,
    out_dir: str,
    include_real: bool,
    only_skill: str | None,
) -> int:
    scoreboard = run_harness(
        root=root,
        config_path=config_path,
        dataset_path=dataset_path,
        out_dir=out_dir,
        include_real=include_real,
        only_skill=only_skill,
    )
    print(json.dumps(scoreboard, indent=2, sort_keys=True))
    return 0 if scoreboard["status"] == "pass" else 1


def cmd_prepare_real_data(session: str, out: str | None, tolerance_s: float) -> int:
    summary = prepare_real_session(session, out_path=out, tolerance_s=tolerance_s)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_run_evaluation_loop(
    root: str,
    config_path: str,
    dataset_path: str,
    out_dir: str,
    threshold_policy_path: str | None,
    include_real: bool,
) -> int:
    trace = run_evaluation_loop(
        root=root,
        config_path=config_path,
        dataset_path=dataset_path,
        out_dir=out_dir,
        threshold_policy_path=threshold_policy_path,
        include_real=include_real,
    )
    print(json.dumps(trace, indent=2, sort_keys=True))
    return 0 if trace["release_gate_decides"]["status"] == "promote_to_human_review" else 1


def _shell_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:@-{}$()")
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    sys.exit(main())
