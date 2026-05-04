#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 /path/to/model.pt"
  exit 2
fi

ISAAC_LAB_ROOT_EXPANDED="${ISAAC_LAB_ROOT/#\~/${HOME}}"
cd "${ISAAC_LAB_ROOT_EXPANDED}"

python scripts/reinforcement_learning/rsl_rl/play.py \
  --task "${GEAR_TASK}" \
  --checkpoint "${CHECKPOINT}"
