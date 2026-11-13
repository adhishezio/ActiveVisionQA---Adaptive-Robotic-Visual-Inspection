#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source /opt/ros/jazzy/setup.bash
if [[ -f install/setup.bash ]]; then source install/setup.bash; fi
exec /usr/bin/python3 -m activevisionqa.dataset_cli "$@"
