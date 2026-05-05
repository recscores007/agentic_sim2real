#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

DATASET="${DATASET:-${SCRIPT_DIR}/../sample_data/real_log_demo.jsonl}"
OUT="${OUT:-${SCRIPT_DIR}/../outputs/developer_harness}"

python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" run-evaluation-loop \
  --root "${SCRIPT_DIR}/.." \
  --dataset "${DATASET}" \
  --out "${OUT}" \
  --audience developer \
  "$@"

echo
echo "Developer harness:"
echo "  ${OUT}/ui/index.html"
echo "  ${OUT}/ui/state.json"
echo "Developer audit artifacts:"
echo "  ${OUT}/evaluation_trace.json"
echo "  ${OUT}/harness/llm_orchestrator/journal.jsonl"
echo "  ${OUT}/scoreboard.json"
