#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -f install/setup.bash ]]; then
  bash scripts/build.sh
fi
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID="${AVQA_ROS_DOMAIN_ID:-67}"
unset ROS_STATIC_PEERS
export GZ_PARTITION="${AVQA_GZ_PARTITION:-activevisionqa_${ROS_DOMAIN_ID}}"
export ROS_LOG_DIR="$project_dir/outputs/ros_logs"
export GZ_SIM_LOG_PATH="$project_dir/outputs/gazebo_logs"
mkdir -p "$ROS_LOG_DIR" "$GZ_SIM_LOG_PATH"
exec ros2 launch activevisionqa workcell.launch.py "$@"
