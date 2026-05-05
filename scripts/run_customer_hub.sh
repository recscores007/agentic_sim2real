#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

DATASET="${DATASET:-${SCRIPT_DIR}/../sample_data/real_log_demo.jsonl}"
OUT="${OUT:-${SCRIPT_DIR}/../outputs/customer_hub}"

python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" run-evaluation-loop \
  --root "${SCRIPT_DIR}/.." \
  --dataset "${DATASET}" \
  --out "${OUT}" \
  --audience customer \
  "$@"

echo
echo "Customer hub:"
echo "  ${OUT}/ui/index.html"
echo "  ${OUT}/ui/state.json"
echo "Customer evidence pack:"
echo "  ${OUT}/scorecard.json"
echo "  ${OUT}/pipeline_output.json"
echo "  ${OUT}/run_record.json"
