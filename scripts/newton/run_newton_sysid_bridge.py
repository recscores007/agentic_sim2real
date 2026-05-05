#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_sim2real.newton_bridge import run_newton_bridge_from_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert agentic_sim2real records to IsaacLab-Newton SysID inputs.")
    parser.add_argument("--input", default=os.environ.get("AGENTIC_SIM2REAL_NEWTON_INPUT_JSON"))
    parser.add_argument("--output", default=os.environ.get("AGENTIC_SIM2REAL_NEWTON_OUTPUT_JSON"))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write Newton control.csv/state_motor.csv/joint_list.txt; do not launch IsaacLab-Newton.",
    )
    args = parser.parse_args()

    if not args.input:
        parser.error("--input or AGENTIC_SIM2REAL_NEWTON_INPUT_JSON is required")

    work_dir = args.work_dir
    if work_dir is None:
        if args.output:
            work_dir = str(Path(args.output).expanduser().parent)
        else:
            work_dir = "outputs/newton_bridge"

    run_newton_bridge_from_file(
        input_path=args.input,
        output_path=args.output,
        work_dir=work_dir,
        force_prepare_only=args.prepare_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
