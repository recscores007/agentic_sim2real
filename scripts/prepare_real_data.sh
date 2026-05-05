#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

SESSION="${1:-${SCRIPT_DIR}/../real_data/example_session}"
OUT="${2:-}"

if [[ -n "${OUT}" ]]; then
  python3 -m ur_agentic.cli --config "${UR_GEAR_CONFIG}" prepare-real-data \
    --session "${SESSION}" \
    --out "${OUT}"
else
  python3 -m ur_agentic.cli --config "${UR_GEAR_CONFIG}" prepare-real-data \
    --session "${SESSION}"
fi
