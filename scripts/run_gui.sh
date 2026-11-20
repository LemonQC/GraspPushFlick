#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
test -n "${DISPLAY:-}" || { echo 'No desktop DISPLAY inherited. Run this from the existing graphical session.' >&2; exit 2; }
exec "$PROJECT_ROOT/scripts/python.sh" -m apfg.simulation.run --enable_cameras "$@"
