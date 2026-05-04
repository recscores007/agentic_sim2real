#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

MODE="${1:-full}"
ISAAC_LAB_ROOT_EXPANDED="${ISAAC_LAB_ROOT/#\~/${HOME}}"

if [[ ! -d "${ISAAC_LAB_ROOT_EXPANDED}" ]]; then
  echo "Isaac Lab root not found: ${ISAAC_LAB_ROOT_EXPANDED}"
  echo "Edit UR_GEAR_CONFIG or configs/ur10e_gear_assembly.local.json."
  exit 2
fi

cd "${ISAAC_LAB_ROOT_EXPANDED}"

if [[ "${MODE}" == "visualize" ]]; then
  python scripts/reinforcement_learning/rsl_rl/train.py \
    --task "${GEAR_TASK}" \
    --num_envs 4
elif [[ "${MODE}" == "full" ]]; then
  python scripts/reinforcement_learning/rsl_rl/train.py \
    --task "${GEAR_TASK}" \
    --headless \
    --num_envs 256 \
    --video --video_length 800 --video_interval 5000
else
  echo "Usage: $0 [visualize|full]"
  exit 2
fi
