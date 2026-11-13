#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID="${AVQA_ROS_DOMAIN_ID:-69}"
unset ROS_STATIC_PEERS
export ROS_LOG_DIR="$project_dir/outputs/ros_logs"
mkdir -p "$ROS_LOG_DIR"
exec /usr/bin/python3 -m activevisionqa.dataset_cli capture-case "$@"
