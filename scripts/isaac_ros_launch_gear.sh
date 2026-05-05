#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

export ROS_DOMAIN_ID
export RMW_IMPLEMENTATION

python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" check-real-gate

if [[ "${AGENTIC_SIM2REAL_DEPLOYMENT_CONFIG}" == \$\(* ]]; then
  MANIPULATOR_CONFIG="$(ros2 pkg prefix --share isaac_ros_manipulation_bringup)/params/ur10e_robotiq_2f_140_gear_assembly.yaml"
else
  MANIPULATOR_CONFIG="${AGENTIC_SIM2REAL_DEPLOYMENT_CONFIG}"
fi

ros2 launch isaac_ros_manipulation_bringup workflows.launch.py \
  manipulator_workflow_config:="${MANIPULATOR_CONFIG}"
