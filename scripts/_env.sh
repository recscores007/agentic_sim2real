#!/usr/bin/env bash

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export UR_GEAR_CONFIG="${UR_GEAR_CONFIG:-${ROOT_DIR}/configs/ur10e_gear_assembly.example.json}"

eval "$(
python3 - "$UR_GEAR_CONFIG" <<'PY'
import shlex
import sys
from ur_agentic.config import command_env, load_config

cfg = load_config(sys.argv[1])
for key, value in command_env(cfg).items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "UR gear environment loaded from ${UR_GEAR_CONFIG}"
  echo "ISAAC_LAB_ROOT=${ISAAC_LAB_ROOT}"
  echo "ISAAC_ROS_WS=${ISAAC_ROS_WS}"
  echo "GEAR_TASK=${GEAR_TASK}"
fi
