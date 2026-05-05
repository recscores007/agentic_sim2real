#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

DATASET="${DATASET:-${SCRIPT_DIR}/../sample_data/real_log_demo.jsonl}"
OUT="${OUT:-${SCRIPT_DIR}/../outputs/evaluation_demo}"

python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" run-evaluation-loop \
  --root "${SCRIPT_DIR}/.." \
  --dataset "${DATASET}" \
  --out "${OUT}" \
  "$@"

echo
echo "Evaluation trace:"
echo "  ${OUT}/evaluation_trace.md"
echo "  ${OUT}/harness/llm_orchestrator/journal.jsonl"
echo "  ${OUT}/ui/index.html"
echo "  ${OUT}/ui/state.json"
echo "  ${OUT}/run_record.json"
echo "  ${OUT}/real_data_manifest.json"
echo "Slide-style artifacts:"
echo "  ${OUT}/rollout_data.json"
echo "  ${OUT}/pipeline_input.json"
echo "  ${OUT}/scorecard.json"
echo "  ${OUT}/pipeline_output.json"
