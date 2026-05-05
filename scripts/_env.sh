#!/usr/bin/env bash

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export AGENTIC_SIM2REAL_CONFIG="${AGENTIC_SIM2REAL_CONFIG:-${ROOT_DIR}/configs/ur10e_gear_assembly.example.json}"

eval "$(
python3 - "$AGENTIC_SIM2REAL_CONFIG" <<'PY'
import shlex
import sys
from agentic_sim2real.config import command_env, load_config

cfg = load_config(sys.argv[1])
for key, value in command_env(cfg).items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Agentic sim2real environment loaded from ${AGENTIC_SIM2REAL_CONFIG}"
  echo "ISAAC_LAB_ROOT=${ISAAC_LAB_ROOT}"
  echo "ISAAC_ROS_WS=${ISAAC_ROS_WS}"
  echo "AGENTIC_SIM2REAL_TASK=${AGENTIC_SIM2REAL_TASK}"
fi
