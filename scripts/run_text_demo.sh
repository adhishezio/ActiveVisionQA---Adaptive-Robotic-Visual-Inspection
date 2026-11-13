#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec bash scripts/validate_simulation.sh --config "$project_dir/config/text_inspection.json" \
  --scenarios occluded --gui "$@"
