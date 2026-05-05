from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic_sim2real.config import load_config
from agentic_sim2real.golden_mutator import run_mutation_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dynamic golden dataset mutation checks.")
    parser.add_argument("--config", default="configs/ur10e_gear_assembly.example.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", default="golden/real_datasets/data_readiness_stress")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    config = load_config(root / args.config if not Path(args.config).is_absolute() else args.config)
    dataset = root / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    out = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    report = run_mutation_suite(dataset, out, config, root=root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
