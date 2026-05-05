#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_env.sh"

OUT="${1:-rosbags/agentic_sim2real_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname -- "${OUT}")"

export ROS_DOMAIN_ID
export RMW_IMPLEMENTATION

echo "Recording ROS bag to ${OUT}"
echo "This records observations only. It does not command robot motion."
ros2 bag record -o "${OUT}" \
  /joint_states \
  /tf \
  /tf_static \
  /camera_1/color/image_raw \
  /camera_1/aligned_depth_to_color/image_raw \
  /input_points_debug
