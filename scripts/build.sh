#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source /opt/ros/jazzy/setup.bash
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo 'Build ROS with system Python: deactivate the Python venv first.' >&2
  exit 1
fi
/usr/bin/colcon build --symlink-install --packages-select activevisionqa
