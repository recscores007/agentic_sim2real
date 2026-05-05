#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

export ROS_DOMAIN_ID
export RMW_IMPLEMENTATION

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo
echo "Checking key ROS tools:"
python3 -m agentic_sim2real.cli --config "${AGENTIC_SIM2REAL_CONFIG}" preflight
echo
echo "Manual-on-robot tests recommended by the tutorial:"
echo "  export ENABLE_MANIPULATOR_TESTING=manual_on_robot"
echo "  launch_test \$(ros2 pkg prefix --share isaac_ros_manipulation_bringup)/test/test_pose_estimation_error_test.py"
echo "  bash \${ISAAC_ROS_WS}/src/isaac_ros_manipulation/isaac_ros_manipulation_bringup/test/compare_pose_estimation_results.sh"
echo
echo "Gate: do not run the full workflow until calibration and pose error are under 1 cm."
