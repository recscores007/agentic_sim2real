#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" commands
