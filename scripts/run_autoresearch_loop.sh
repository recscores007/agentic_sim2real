#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

DATASET="${DATASET:-${SCRIPT_DIR}/../sample_data/real_log_demo.jsonl}"
OUT="${OUT:-${SCRIPT_DIR}/../outputs/autoresearch_demo}"

echo "Running LLM-orchestrated skill loop with AutoResearch planner enabled."
python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" run-llm-loop \
  --root "${SCRIPT_DIR}/.." \
  --dataset "${DATASET}" \
  --out "${OUT}" \
  "$@"

echo
echo "AutoResearch evidence:"
echo "  ${OUT}/skills/agentic_tuning_plan/subchecks/autoresearch_planner/autoresearch_plan.json"
echo "LLM journal:"
echo "  ${OUT}/llm_orchestrator/journal.jsonl"
echo "Release scoreboard:"
echo "  ${OUT}/scoreboard.json"
