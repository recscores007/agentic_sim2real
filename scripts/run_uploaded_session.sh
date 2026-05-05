#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <session_dir> [out_dir]" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

export AGENTIC_SIM2REAL_CONFIG="${AGENTIC_SIM2REAL_CONFIG:-${ROOT_DIR}/configs/ur10e_gear_assembly.production_upload.example.json}"
source "${SCRIPT_DIR}/_env.sh"

SESSION_DIR="$1"
STAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_NAME=$(basename "${SESSION_DIR%/}")
OUT="${2:-${ROOT_DIR}/outputs/upload_runs/${SESSION_NAME}_${STAMP}}"
PREPARED_RECORDS="${OUT}/prepared_records/records.jsonl"

mkdir -p "${OUT}"

echo "Production preflight:"
python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" preflight --root "${ROOT_DIR}"

echo
echo "Preparing uploaded session:"
echo "  Video evidence, when present, is read from ${SESSION_DIR}/video_data/index.csv and ${SESSION_DIR}/video_data/analysis.json"
python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" prepare-real-data \
  --root "${ROOT_DIR}" \
  --session "${SESSION_DIR}" \
  --out "${PREPARED_RECORDS}"

echo
echo "Running strict evaluation loop:"
python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" run-evaluation-loop \
  --root "${ROOT_DIR}" \
  --dataset "${PREPARED_RECORDS}" \
  --out "${OUT}" \
  --audience customer

echo
echo "Upload run artifacts:"
echo "  ${OUT}/ui/index.html"
echo "  ${OUT}/ui/state.json"
echo "  ${OUT}/scoreboard.json"
echo "  ${OUT}/evaluation_trace.json"
