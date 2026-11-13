#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source /opt/ros/jazzy/setup.bash
exec /usr/bin/python3 -m pytest -q "$@"
