#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

DATASET="${DATASET:-${SCRIPT_DIR}/../sample_data/real_log_demo.jsonl}"
OUT="${OUT:-${SCRIPT_DIR}/../outputs/autoresearch_demo}"

echo "Running skill validation harness with AutoResearch planner enabled."
python3 -m ur_agentic.cli --config "${UR_GEAR_CONFIG}" run-harness \
  --root "${SCRIPT_DIR}/.." \
  --dataset "${DATASET}" \
  --out "${OUT}"

echo
echo "AutoResearch evidence:"
echo "  ${OUT}/skills/autoresearch_planner/autoresearch_plan.json"
echo "Release scoreboard:"
echo "  ${OUT}/scoreboard.json"
