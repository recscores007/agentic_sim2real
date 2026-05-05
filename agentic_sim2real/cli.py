from __future__ import annotations

import argparse
import json
import sys

from .autoresearch import build_plan
from .config import PipelineConfig, choose_task, command_env, load_config
from .dataset import load_records
from .embodiments import validate_embodiments
from .evaluation_loop import run_evaluation_loop
from .golden_mutator import run_mutation_suite
from .llm_orchestrator import load_provider_command_json, run_llm_orchestrated_loop
from .preflight import run_preflight
from .real_data import ensure_aligned_dataset, inspect_real_session, prepare_real_session
from .report import write_outputs
from .safety import require_real_robot_gate
from .skill_harness import load_manifests, run_harness, validate_all_manifests
from .sysid import estimate_gap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic sim2real helper for the configured robot task")
    parser.add_argument("--config", default="configs/ur10e_gear_assembly.example.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    preflight = sub.add_parser("preflight", help="Check local command availability and runtime configuration")
    preflight.add_argument("--root", default=".")
    sub.add_parser("commands", help="Print tutorial-aligned Isaac Lab and Isaac ROS commands")
    list_skills = sub.add_parser("list-skills", help="List atomic skills and owning agents")
    list_skills.add_argument("--root", default=".")
    list_skills.add_argument("--skill-dir", action="append", default=[], help="Overlay directory with replacement skills")

    validate_skills = sub.add_parser("validate-skills", help="Validate every skill manifest")
    validate_skills.add_argument("--root", default=".")
    validate_skills.add_argument("--skill-dir", action="append", default=[], help="Overlay directory with replacement skills")

    validate_embodiments_cmd = sub.add_parser("validate-embodiments", help="Validate embodiment manifests and real-data scaffold structure")
    validate_embodiments_cmd.add_argument("--root", default=".")

    analyze = sub.add_parser("analyze", help="Analyze recorded real logs offline")
    analyze.add_argument("--dataset", required=True)
    analyze.add_argument("--out", required=True)

    golden_mutations = sub.add_parser("run-golden-mutations", help="Generate temporary golden dataset mutations and verify data-readiness alerts")
    golden_mutations.add_argument("--root", default=".")
    golden_mutations.add_argument("--dataset", default="golden/real_datasets/data_readiness_stress")
    golden_mutations.add_argument("--out", required=True)

    inspect = sub.add_parser("inspect-real-data", help="Inspect an embodiment real_data session before alignment")
    inspect.add_argument("--session", required=True)
    inspect.add_argument("--root", default=".")
    inspect.add_argument("--embodiment", default=None)

    prepare = sub.add_parser("prepare-real-data", help="Merge a real_data session into pipeline records.jsonl")
    prepare.add_argument("--session", required=True)
    prepare.add_argument("--out", default=None)
    prepare.add_argument("--tolerance-s", type=float, default=0.05)
    prepare.add_argument("--root", default=".")
    prepare.add_argument("--embodiment", default=None)

    harness = sub.add_parser("run-harness", help="Run the skill validation harness")
    harness.add_argument("--root", default=".")
    harness.add_argument("--dataset", default="sample_data/real_log_demo.jsonl")
    harness.add_argument("--out", required=True)
    harness.add_argument("--skill", default=None, help="Run one skill by id")
    harness.add_argument("--skill-dir", action="append", default=[], help="Overlay directory with replacement skills")
    harness.add_argument("--include-real", action="store_true", help="Include real-robot skills after human approval")
    harness.add_argument("--audience", choices=["customer", "developer"], default=None, help="Hub UI audience")

    eval_loop = sub.add_parser("run-evaluation-loop", help="Run LLM/Agent/Evaluator/Critic/Release/Human trace")
    eval_loop.add_argument("--root", default=".")
    eval_loop.add_argument("--dataset", default="sample_data/real_log_demo.jsonl")
    eval_loop.add_argument("--out", required=True)
    eval_loop.add_argument("--threshold-policy", default=None)
    eval_loop.add_argument("--skill-dir", action="append", default=[], help="Overlay directory with replacement skills")
    eval_loop.add_argument("--include-real", action="store_true")
    eval_loop.add_argument("--llm-provider", default=None, help="LLM provider: scripted or command")
    eval_loop.add_argument("--llm-command-json", default=None, help="JSON list command for command-backed LLM provider")
    eval_loop.add_argument("--max-steps", type=int, default=None)
    eval_loop.add_argument("--gap-hint", action="append", default=[], help="Initial sim2real gap focus, e.g. perception, actuator, contact, latency, domain_randomization, deployment, policy")
    eval_loop.add_argument("--audience", choices=["customer", "developer"], default=None, help="Hub UI audience")

    llm_loop = sub.add_parser("run-llm-loop", help="Run the LLM-orchestrated skill loop")
    llm_loop.add_argument("--root", default=".")
    llm_loop.add_argument("--dataset", default="sample_data/real_log_demo.jsonl")
    llm_loop.add_argument("--out", required=True)
    llm_loop.add_argument("--skill-dir", action="append", default=[], help="Overlay directory with replacement skills")
    llm_loop.add_argument("--include-real", action="store_true")
    llm_loop.add_argument("--llm-provider", default=None, help="LLM provider: scripted or command")
    llm_loop.add_argument("--llm-command-json", default=None, help="JSON list command for command-backed LLM provider")
    llm_loop.add_argument("--max-steps", type=int, default=None)
    llm_loop.add_argument("--gap-hint", action="append", default=[], help="Initial sim2real gap focus, e.g. perception, actuator, contact, latency, domain_randomization, deployment, policy")
    llm_loop.add_argument("--audience", choices=["customer", "developer"], default=None, help="Hub UI audience")

    sub.add_parser("check-real-gate", help="Fail unless the real-robot human gate env var is set")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.cmd == "preflight":
        return cmd_preflight(config, args.root)
    if args.cmd == "commands":
        return cmd_commands(config)
    if args.cmd == "list-skills":
        return cmd_list_skills(args.root, args.skill_dir)
    if args.cmd == "validate-skills":
        return cmd_validate_skills(args.root, args.skill_dir)
    if args.cmd == "validate-embodiments":
        return cmd_validate_embodiments(args.root)
    if args.cmd == "analyze":
        return cmd_analyze(config, args.dataset, args.out)
    if args.cmd == "run-golden-mutations":
        return cmd_run_golden_mutations(config, args.root, args.dataset, args.out)
    if args.cmd == "inspect-real-data":
        return cmd_inspect_real_data(args.session, args.root, args.embodiment)
    if args.cmd == "prepare-real-data":
        return cmd_prepare_real_data(args.session, args.out, args.tolerance_s, args.root, args.embodiment)
    if args.cmd == "run-harness":
        return cmd_run_harness(args.root, args.config, args.dataset, args.out, args.include_real, args.skill, args.skill_dir, args.audience)
    if args.cmd == "run-evaluation-loop":
        return cmd_run_evaluation_loop(
            args.root,
            args.config,
            args.dataset,
            args.out,
            args.threshold_policy,
            args.include_real,
            args.skill_dir,
            args.llm_provider,
            args.llm_command_json,
            args.max_steps,
            args.gap_hint,
            args.audience,
        )
    if args.cmd == "run-llm-loop":
        return cmd_run_llm_loop(
            args.root,
            args.config,
            args.dataset,
            args.out,
            args.include_real,
            args.skill_dir,
            args.llm_provider,
            args.llm_command_json,
            args.max_steps,
            args.gap_hint,
            args.audience,
        )
    if args.cmd == "check-real-gate":
        require_real_robot_gate(config)
        print("Human gate env var present. Continue only with active supervision.")
        return 0
    raise AssertionError(args.cmd)


def cmd_preflight(config: PipelineConfig, root: str = ".") -> int:
    report = run_preflight(config, root=root)
    print(f"Preflight checks: {report['status']}")
    print(f"- selected Isaac Lab task: {report['selected_isaac_lab_task']}")
    print(f"- release profile: {report['required_physics']['release_profile']}")
    print(f"- physics required: {report['required_physics']['physics_required']}")
    print(f"- ROS_DOMAIN_ID: {report['isaac_ros']['ros_domain_id']}")
    print(f"- RMW_IMPLEMENTATION: {report['isaac_ros']['rmw_implementation']}")
    print()
    for check in report["checks"]:
        label = {"pass": "ok", "warn": "warn", "fail": "fail"}[check["status"]]
        suffix = f" ({check['path']})" if check.get("path") else ""
        print(f"- [{label}] {check['category']}.{check['name']}: {check['message']}{suffix}")
    if report["recommendations"]:
        print()
        print("Recommendations:")
        for item in report["recommendations"]:
            print(f"- {item}")
    return 0 if report["status"] == "pass" else 1


def cmd_commands(config: PipelineConfig) -> int:
    env = command_env(config)
    print("# Export these first")
    for key, value in env.items():
        print(f"export {key}={_shell_quote(value)}")
    print()
    print("# Isaac Lab: visualize the training environment")
    print("cd $ISAAC_LAB_ROOT")
    print("python scripts/reinforcement_learning/rsl_rl/train.py --task $AGENTIC_SIM2REAL_TASK --num_envs 4")
    print()
    print("# Isaac Lab: full training with video recording")
    print(
        "python scripts/reinforcement_learning/rsl_rl/train.py "
        "--task $AGENTIC_SIM2REAL_TASK --headless --num_envs 256 --video --video_length 800 --video_interval 5000"
    )
    print()
    print("# Isaac ROS: validate pose estimation and calibration")
    print("export ENABLE_MANIPULATOR_TESTING=manual_on_robot")
    print("launch_test $(ros2 pkg prefix --share isaac_ros_manipulation_bringup)/test/test_pose_estimation_error_test.py")
    print("bash ${ISAAC_ROS_WS}/src/isaac_ros_manipulation/isaac_ros_manipulation_bringup/test/compare_pose_estimation_results.sh")
    print()
    print("# Isaac ROS: deploy the gear assembly workflow after the human gate")
    print("ros2 launch isaac_ros_manipulation_bringup workflows.launch.py manipulator_workflow_config:=$AGENTIC_SIM2REAL_DEPLOYMENT_CONFIG")
    print("ros2 action send_goal $AGENTIC_SIM2REAL_ACTION isaac_ros_manipulation_interfaces/action/GearAssembly {}")
    return 0


def cmd_analyze(config: PipelineConfig, dataset_path: str, out_dir: str) -> int:
    records = load_records(ensure_aligned_dataset(dataset_path))
    gap = estimate_gap(records, config)
    plan = build_plan(gap, config)
    paths = write_outputs(out_dir, gap, plan)
    print(f"Analyzed {gap['summary']['records']} records across {gap['summary']['episodes']} episodes.")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    print(f"Transfer score: {plan['transfer_score']['score_0_to_1']} ({plan['transfer_score']['interpretation']})")
    return 0


def cmd_run_golden_mutations(config: PipelineConfig, root: str, dataset_path: str, out_dir: str) -> int:
    report = run_mutation_suite(dataset_path, out_dir, config, root=root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def cmd_inspect_real_data(session: str, root: str, embodiment_id: str | None) -> int:
    report = inspect_real_session(session, root=root, embodiment_id=embodiment_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"aligned_records_ready", "csv_session_ready"} else 1


def cmd_list_skills(root: str, skill_dirs: list[str]) -> int:
    manifests = load_manifests(root, skill_dirs=skill_dirs)
    for manifest in manifests:
        data = manifest.data
        print(f"{data['id']}: {data['owner_agent']} - {data['name']} ({manifest.runner}, {manifest.path})")
    return 0


def cmd_validate_skills(root: str, skill_dirs: list[str]) -> int:
    result = validate_all_manifests(root, skill_dirs=skill_dirs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def cmd_validate_embodiments(root: str) -> int:
    result = validate_embodiments(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def cmd_run_harness(
    root: str,
    config_path: str,
    dataset_path: str,
    out_dir: str,
    include_real: bool,
    only_skill: str | None,
    skill_dirs: list[str],
    audience: str | None,
) -> int:
    scoreboard = run_harness(
        root=root,
        config_path=config_path,
        dataset_path=dataset_path,
        out_dir=out_dir,
        include_real=include_real,
        only_skill=only_skill,
        skill_dirs=skill_dirs,
        audience=audience,
    )
    print(json.dumps(scoreboard, indent=2, sort_keys=True))
    return 0 if scoreboard["status"] == "pass" else 1


def cmd_prepare_real_data(
    session: str,
    out: str | None,
    tolerance_s: float,
    root: str,
    embodiment_id: str | None,
) -> int:
    summary = prepare_real_session(
        session,
        out_path=out,
        tolerance_s=tolerance_s,
        root=root,
        embodiment_id=embodiment_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_run_evaluation_loop(
    root: str,
    config_path: str,
    dataset_path: str,
    out_dir: str,
    threshold_policy_path: str | None,
    include_real: bool,
    skill_dirs: list[str],
    llm_provider: str | None,
    llm_command_json: str | None,
    max_steps: int | None,
    gap_hints: list[str],
    audience: str | None,
) -> int:
    trace = run_evaluation_loop(
        root=root,
        config_path=config_path,
        dataset_path=dataset_path,
        out_dir=out_dir,
        threshold_policy_path=threshold_policy_path,
        include_real=include_real,
        skill_dirs=skill_dirs,
        llm_provider_name=llm_provider,
        llm_command=load_provider_command_json(llm_command_json),
        max_steps=max_steps,
        gap_hints=gap_hints,
        audience=audience,
    )
    print(json.dumps(trace, indent=2, sort_keys=True))
    return 0 if trace["release_gate_decides"]["status"] == "promote_to_human_review" else 1


def cmd_run_llm_loop(
    root: str,
    config_path: str,
    dataset_path: str,
    out_dir: str,
    include_real: bool,
    skill_dirs: list[str],
    llm_provider: str | None,
    llm_command_json: str | None,
    max_steps: int | None,
    gap_hints: list[str],
    audience: str | None,
) -> int:
    summary = run_llm_orchestrated_loop(
        root=root,
        config_path=config_path,
        dataset_path=dataset_path,
        out_dir=out_dir,
        include_real=include_real,
        skill_dirs=skill_dirs,
        provider_name=llm_provider,
        provider_command=load_provider_command_json(llm_command_json),
        max_steps=max_steps,
        gap_hints=gap_hints,
        audience=audience,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


def _shell_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:@-{}$()")
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    sys.exit(main())
